"""run_so101.py

Minimal SO-101 inference demo for an AsyncVLA checkpoint trained with
``vla-scripts/train_so101.py``.

Loads:
- the OpenVLA-7B base + LoRA adapter saved under
  ``runs_so101/.../<step>_chkpt/lora_adapter``
- ``pose_projector--<step>_checkpoint.pt`` and
  ``action_head--<step>_checkpoint.pt`` from the same directory

Then pulls one frame from the SO-101 LeRobot dataset (or any source you
plug in), runs the model, and prints the 8-step predicted action chunk
alongside the ground-truth chunk (denormalised to joint-position units).

Usage:
    ASYNCVLA_PLATFORM=so101 PYTHONPATH=$PWD python inference/run_so101.py \
        --checkpoint_dir runs_so101/openvla-7b+so101+b1+lr-5e-05+lora-r32--long-run--10000_chkpt \
        --step 10000 \
        --frame_index 0
"""
from __future__ import annotations

import os

os.environ.setdefault("ASYNCVLA_PLATFORM", "so101")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from peft import PeftModel
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForVision2Seq,
    AutoProcessor,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction_MMNv1
from prismatic.extern.hf.processing_prismatic import (
    PrismaticImageProcessor,
    PrismaticProcessor,
)
from prismatic.models.action_heads import L1RegressionActionHead_idcat
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.models.projectors import ProprioProjector
from prismatic.training.train_utils import (
    get_current_action_mask,
    get_next_actions_mask,
)
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK, POSE_DIM


def _remove_ddp_prefix(state_dict: dict) -> dict:
    return {k[len("module.") :] if k.startswith("module.") else k: v for k, v in state_dict.items()}


def _denormalize(action_norm: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    """Inverse of (x - low) / (high - low) * 2 - 1, clamped to [-1, 1]."""
    return ((action_norm.clamp(-1.0, 1.0) + 1.0) * 0.5) * (high - low) + low


def load_inference_modules(
    checkpoint_dir: Path,
    step: int,
    base_vla_path: str = "openvla/openvla-7b",
    num_images_in_input: int = 2,
    attn_implementation: str = "eager",
    device: Optional[str] = None,
):
    """Restore VLA + LoRA + pose_projector + action_head for inference."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction_MMNv1)

    print(f"[so101-infer] base VLA  = {base_vla_path}")
    base = AutoModelForVision2Seq.from_pretrained(
        base_vla_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation=attn_implementation,
    ).to(device)
    base.vision_backbone.set_num_images_in_input(num_images_in_input)
    base.to(dtype=torch.bfloat16, device=device)

    adapter_dir = Path(checkpoint_dir) / "lora_adapter"
    print(f"[so101-infer] LoRA       = {adapter_dir}")
    vla = PeftModel.from_pretrained(base, adapter_dir)
    vla = vla.merge_and_unload()  # collapse LoRA into base weights for faster inference
    vla.eval()

    print(f"[so101-infer] tokenizer  = {checkpoint_dir} (processor)")
    processor = AutoProcessor.from_pretrained(checkpoint_dir, trust_remote_code=True)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    llm_dim = vla.llm_dim
    pose_projector = ProprioProjector(llm_dim=llm_dim, proprio_dim=POSE_DIM)
    pose_state = torch.load(checkpoint_dir / f"pose_projector--{step}_checkpoint.pt", map_location="cpu", weights_only=True)
    pose_projector.load_state_dict(_remove_ddp_prefix(pose_state), strict=False)
    pose_projector = pose_projector.to(torch.bfloat16).to(device).eval()

    action_head = L1RegressionActionHead_idcat(input_dim=llm_dim, hidden_dim=llm_dim, action_dim=ACTION_DIM)
    head_state = torch.load(checkpoint_dir / f"action_head--{step}_checkpoint.pt", map_location="cpu", weights_only=True)
    action_head.load_state_dict(_remove_ddp_prefix(head_state), strict=False)
    action_head = action_head.to(torch.bfloat16).to(device).eval()

    num_patches = vla.vision_backbone.get_num_patches() * vla.vision_backbone.get_num_images_in_input() + 1
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    return vla, action_head, pose_projector, processor, action_tokenizer, num_patches, device


def build_inputs(
    processor,
    action_tokenizer,
    main_image: Image.Image,
    secondary_image: Image.Image,
    state_norm: np.ndarray,
    instruction: str,
    device: torch.device,
):
    """Construct the VLA input batch for a single observation."""
    # Placeholder action chunk (zeros) — used only to make the prompt the
    # correct length so the model emits hidden states at the action positions.
    placeholder_chunk = np.zeros((NUM_ACTIONS_CHUNK, ACTION_DIM), dtype=np.float32)
    current_str = action_tokenizer(placeholder_chunk[0])
    future_str = "".join(action_tokenizer(placeholder_chunk[1:]))
    chunk_string = current_str + future_str

    pb = PurePromptBuilder("openvla")
    pb.add_turn("human", f"What action should the robot take to {instruction}")
    pb.add_turn("gpt", chunk_string)

    input_ids = torch.as_tensor(
        processor.tokenizer(pb.get_prompt(), add_special_tokens=True).input_ids
    ).unsqueeze(0).to(device)
    labels = input_ids.clone()
    # Mark only the action region as supervised so the same masks used at
    # training time still work to slice out the action hidden states.
    labels[:, : -(len(chunk_string) + 1)] = -100

    pixel_main = processor.image_processor.apply_transform(main_image).unsqueeze(0)
    pixel_sec = processor.image_processor.apply_transform(secondary_image).unsqueeze(0)
    pixel_values = torch.cat([pixel_main, pixel_sec], dim=1).to(torch.bfloat16).to(device)

    proprio = torch.as_tensor(state_norm, dtype=torch.bfloat16).unsqueeze(0).to(device)
    modality_id = torch.tensor([5.0], dtype=torch.bfloat16, device=device)

    attention_mask = input_ids.ne(processor.tokenizer.pad_token_id)
    attention_mask_label = labels.ne(-100)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "attention_mask_label": attention_mask_label,
        "pixel_values": pixel_values,
        "proprio": proprio,
        "modality_id": modality_id,
    }


@torch.inference_mode()
def predict_action_chunk(
    vla,
    action_head,
    pose_projector,
    batch: dict,
    num_patches: int,
) -> torch.Tensor:
    output: CausalLMOutputWithPast = vla(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        attention_mask_label=batch["attention_mask_label"],
        pixel_values=batch["pixel_values"],
        modality_id=batch["modality_id"],
        labels=batch["labels"],
        proprio=batch["proprio"],
        proprio_projector=pose_projector,
        output_hidden_states=True,
        use_film=False,
    )

    ground_truth_token_ids = batch["labels"][:, 1:]
    current_mask = get_current_action_mask(ground_truth_token_ids)
    next_mask = get_next_actions_mask(ground_truth_token_ids)

    last_hidden = output.hidden_states[-1]
    text_hidden = last_hidden[:, num_patches:-1]
    bsz = batch["input_ids"].shape[0]
    actions_hidden = (
        text_hidden[current_mask | next_mask]
        .reshape(bsz, NUM_ACTIONS_CHUNK * ACTION_DIM, -1)
        .to(torch.bfloat16)
    )
    return action_head.predict_action(actions_hidden, batch["modality_id"])  # (B, chunk, action_dim)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=Path, required=True,
                        help="Path to a runs_so101/.../<step>_chkpt directory")
    parser.add_argument("--step", type=int, required=True, help="Step suffix of the .pt files")
    parser.add_argument("--base_vla_path", default="openvla/openvla-7b")
    parser.add_argument("--frame_index", type=int, default=0,
                        help="Frame index inside the dataset's first episode")
    parser.add_argument("--dataset_repo_id", default="k1000dai/so101_pick_candy_clean")
    parser.add_argument("--main_camera", default="observation.images.top")
    parser.add_argument("--secondary_camera", default="observation.images.wrist")
    parser.add_argument("--instruction", default="pick up candy and put it in the bowl")
    args = parser.parse_args()

    vla, action_head, pose_projector, processor, action_tokenizer, num_patches, device = (
        load_inference_modules(args.checkpoint_dir, args.step, base_vla_path=args.base_vla_path)
    )

    # Load one frame using the same SO101_Dataset normalisation logic.
    from prismatic.vla.datasets.so101_dataset import SO101_Dataset

    ds = SO101_Dataset(
        repo_id=args.dataset_repo_id,
        action_tokenizer=action_tokenizer,
        base_tokenizer=processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        episodes=[0],
        main_camera=args.main_camera,
        secondary_camera=args.secondary_camera,
    )
    sample = ds[args.frame_index]

    state_norm = sample["proprio"].numpy().astype(np.float32)
    actions_norm_gt = np.asarray(sample["actions"], dtype=np.float32)
    main_pil = sample["img_PIL"]
    secondary_pil = sample["gimg_PIL"]

    batch = build_inputs(
        processor=processor,
        action_tokenizer=action_tokenizer,
        main_image=main_pil,
        secondary_image=secondary_pil,
        state_norm=state_norm,
        instruction=args.instruction,
        device=device,
    )
    pred_norm = predict_action_chunk(vla, action_head, pose_projector, batch, num_patches)
    pred_norm = pred_norm.float().squeeze(0).cpu()
    gt_norm = torch.from_numpy(actions_norm_gt)

    # Denormalise both with the dataset's q01/q99 bounds for human-readable
    # joint positions (degrees).
    low = ds.action_low
    high = ds.action_high
    pred_joints = _denormalize(pred_norm, low, high).numpy()
    gt_joints = _denormalize(gt_norm, low, high).numpy()

    np.set_printoptions(precision=3, suppress=True)
    print(f"\n[so101-infer] instruction: {args.instruction!r}")
    print(f"[so101-infer] frame_index : {args.frame_index} / {len(ds)}")
    print(f"[so101-infer] proprio (deg): {_denormalize(torch.from_numpy(state_norm), ds.state_low, ds.state_high).numpy()}")

    print("\n[predicted joint targets, 8 x 6]")
    print(pred_joints)
    print("\n[ground-truth joint targets, 8 x 6]")
    print(gt_joints)

    l1 = np.mean(np.abs(pred_norm.numpy() - gt_norm.numpy()))
    l2 = np.mean((pred_norm.numpy() - gt_norm.numpy()) ** 2)
    print(f"\n[normalised metrics on this frame]  L1={l1:.4f}  L2={l2:.4f}")


if __name__ == "__main__":
    main()
