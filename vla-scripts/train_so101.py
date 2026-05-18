"""train_so101.py

LoRA finetune the AsyncVLA / OmniVLA base on a LeRobot SO-101 dataset
(e.g. ``k1000dai/so101_pick_candy_clean``).

This script reuses the AsyncVLA model surface (vision backbone + LLM + proprio
projector + L1 regression action head) but strips the navigation-specific
multi-modal trajectory losses and the MBRA edge adapter. The platform
constants (``ACTION_DIM=6``, ``POSE_DIM=6``) are selected automatically when
``ASYNCVLA_PLATFORM=so101`` is exported – the script enforces that at start-up.

Example:

```
ASYNCVLA_PLATFORM=so101 torchrun --standalone --nnodes 1 --nproc-per-node 1 \
    vla-scripts/train_so101.py --vla_path openvla/openvla-7b \
    --dataset_repo_id k1000dai/so101_pick_candy_clean \
    --wandb_entity my-entity --wandb_project asyncvla-so101
```
"""

# ==============================
# Path Setup
# ==============================
import os
import sys

os.environ.setdefault("ASYNCVLA_PLATFORM", "so101")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Type

import draccus
import torch
import torch.distributed as dist
import torch.nn as nn
import tqdm
import wandb
import yaml
from accelerate import PartialState
from huggingface_hub import snapshot_download
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForVision2Seq,
    AutoProcessor,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

# ==============================
# Repo imports
# ==============================
from experiments.robot.openvla_utils import (
    check_model_logic_mismatch,
    model_is_on_hf_hub,
    update_auto_map,
)
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction_MMNv1
from prismatic.extern.hf.processing_prismatic import (
    PrismaticImageProcessor,
    PrismaticProcessor,
)
from prismatic.models.action_heads import L1RegressionActionHead_idcat
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.models.projectors import ProprioProjector
from prismatic.training.train_utils import get_current_action_mask, get_next_actions_mask
from prismatic.util.data_utils import PaddedCollatorForActionPrediction_SO101
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import (
    ACTION_DIM,
    NUM_ACTIONS_CHUNK,
    POSE_DIM,
    ROBOT_PLATFORM,
)
from prismatic.vla.datasets.so101_dataset import SO101_Dataset


# ==============================
# Config
# ==============================
@dataclass
class SO101Config:
    # Base VLA checkpoint (HF Hub id or local dir).
    vla_path: str = "openvla/openvla-7b"

    # Dataset configuration overrides. The actual defaults live in
    # ``config_nav/so101_config.yaml`` – CLI flags override the YAML values.
    dataset_config: Path = Path("config_nav/so101_config.yaml")
    dataset_repo_id: Optional[str] = None
    dataset_root: Optional[str] = None
    episodes: Optional[List[int]] = None

    # Run management.
    run_root_dir: Path = Path("runs_so101")
    run_id_note: Optional[str] = None
    run_id_override: Optional[str] = None

    # Training hyperparameters.
    num_images_in_input: int = 2
    batch_size: int = 1
    learning_rate: float = 5e-5
    lr_warmup_steps: int = 0
    num_steps_before_decay: int = 50_000
    grad_accumulation_steps: int = 1
    max_steps: int = 100_000
    save_freq: int = 5_000
    save_latest_checkpoint_only: bool = False

    # Loss weighting.
    action_loss_weight: float = 1.0
    proprio_token_loss_weight: float = 0.0  # disabled by default

    # LoRA.
    use_lora: bool = True
    lora_rank: int = 32
    lora_dropout: float = 0.0
    merge_lora_during_training: bool = False

    # Backbone attention implementation. Default "auto" lets transformers pick
    # flash_attention_2 when installed and fall back otherwise. Override to
    # "eager" or "sdpa" if flash-attn isn't available on your system.
    attn_implementation: str = "eager"

    # Optional resume from a previous AsyncVLA/OmniVLA checkpoint directory.
    resume: bool = False
    resume_step: Optional[int] = None

    # Data loading.
    num_workers: int = 4

    # Logging.
    wandb_entity: str = "your-wandb-entity"
    wandb_project: str = "asyncvla-so101"
    wandb_log_freq: int = 10

    def load_dataset_yaml(self) -> dict:
        with open(self.dataset_config, "r") as f:
            return yaml.safe_load(f)


# ==============================
# Helpers
# ==============================
def remove_ddp_prefix(state_dict: dict) -> dict:
    return {k[len("module."):] if k.startswith("module.") else k: v for k, v in state_dict.items()}


def wrap_ddp(module: nn.Module, device_id: int, find_unused: bool = False) -> DDP:
    return DDP(
        module,
        device_ids=[device_id],
        find_unused_parameters=find_unused,
        gradient_as_bucket_view=True,
    )


def maybe_load_state(module: nn.Module, ckpt_path: Path) -> None:
    if not ckpt_path.exists():
        print(f"[so101] No checkpoint at {ckpt_path}; using random init for module.")
        return
    print(f"[so101] Loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    module.load_state_dict(remove_ddp_prefix(state), strict=False)


def init_module(
    module_class: Type[nn.Module],
    module_name: str,
    cfg: SO101Config,
    device_id: int,
    module_args: dict,
    to_bf16: bool = False,
    find_unused_params: bool = False,
) -> DDP:
    module = module_class(**module_args)
    num_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"[so101] # trainable params in {module_name}: {num_trainable}")
    if cfg.resume and cfg.resume_step is not None:
        ckpt = Path(cfg.vla_path) / f"{module_name}--{cfg.resume_step}_checkpoint.pt"
        maybe_load_state(module, ckpt)
    if to_bf16:
        module = module.to(torch.bfloat16)
    module = module.to(device_id)
    return wrap_ddp(module, device_id, find_unused_params)


def get_run_id(cfg: SO101Config) -> str:
    if cfg.run_id_override:
        return cfg.run_id_override
    base = cfg.vla_path.rstrip("/").split("/")[-1]
    run_id = (
        f"{base}+so101+b{cfg.batch_size * cfg.grad_accumulation_steps}"
        f"+lr-{cfg.learning_rate}"
    )
    if cfg.use_lora:
        run_id += f"+lora-r{cfg.lora_rank}"
    if cfg.run_id_note:
        run_id += f"--{cfg.run_id_note}"
    return run_id


# ==============================
# Forward pass
# ==============================
def run_forward_pass(
    vla,
    action_head,
    pose_projector,
    batch,
    device_id: int,
    num_patches: int,
) -> Tuple[torch.Tensor, dict]:
    ground_truth_actions = batch["actions"].to(device_id).to(torch.bfloat16)
    modality_id = batch["goal_mask_select"].to(torch.bfloat16).to(device_id)

    output: CausalLMOutputWithPast = vla(
        input_ids=batch["input_ids"].to(device_id),
        attention_mask=batch["attention_mask"].to(device_id),
        attention_mask_label=batch["attention_mask_label"].to(device_id),
        pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
        modality_id=modality_id,
        labels=batch["labels"],
        output_hidden_states=True,
        proprio=batch["proprio"].to(torch.bfloat16).to(device_id),
        proprio_projector=pose_projector,
        use_film=False,
    )

    ground_truth_token_ids = batch["labels"][:, 1:].to(device_id)
    current_action_mask = get_current_action_mask(ground_truth_token_ids)
    next_actions_mask = get_next_actions_mask(ground_truth_token_ids)

    last_hidden_states = output.hidden_states[-1]
    text_hidden_states = last_hidden_states[:, num_patches:-1]
    batch_size = batch["input_ids"].shape[0]
    actions_hidden_states = (
        text_hidden_states[current_action_mask | next_actions_mask]
        .reshape(batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, -1)
        .to(torch.bfloat16)
    )

    predicted_actions = action_head.module.predict_action(actions_hidden_states, modality_id)
    l1_loss = torch.nn.functional.l1_loss(predicted_actions, ground_truth_actions)
    mse_loss = torch.nn.functional.mse_loss(
        predicted_actions.float(), ground_truth_actions.float()
    )
    loss = l1_loss
    metrics = {
        "loss_value": loss.item(),
        "L1_action_value": l1_loss.item(),
        "L2_action_value": mse_loss.item(),
    }
    return loss, metrics


# ==============================
# Checkpointing
# ==============================
def save_training_checkpoint(
    cfg: SO101Config,
    run_dir: Path,
    log_step: int,
    vla,
    processor,
    pose_projector,
    action_head,
    distributed_state,
) -> None:
    if cfg.save_latest_checkpoint_only:
        checkpoint_dir = run_dir
        checkpoint_name = "latest_checkpoint.pt"
    else:
        checkpoint_dir = Path(str(run_dir) + f"--{log_step}_chkpt")
        checkpoint_name = f"{log_step}_checkpoint.pt"
    adapter_dir = checkpoint_dir / "lora_adapter"

    if distributed_state.is_main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(adapter_dir, exist_ok=True)
    dist.barrier()

    if distributed_state.is_main_process:
        processor.save_pretrained(checkpoint_dir)
        vla.module.save_pretrained(adapter_dir)
        torch.save(pose_projector.state_dict(), checkpoint_dir / f"pose_projector--{checkpoint_name}")
        torch.save(action_head.state_dict(), checkpoint_dir / f"action_head--{checkpoint_name}")

    dist.barrier()

    if cfg.use_lora and cfg.merge_lora_during_training and distributed_state.is_main_process:
        base_vla = AutoModelForVision2Seq.from_pretrained(
            cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
        )
        merged = PeftModel.from_pretrained(base_vla, adapter_dir).merge_and_unload()
        merged.save_pretrained(checkpoint_dir)
    dist.barrier()


# ==============================
# Training entrypoint
# ==============================
@draccus.wrap()
def train_so101(cfg: SO101Config) -> None:
    if ROBOT_PLATFORM != "so101":
        raise RuntimeError(
            "ASYNCVLA_PLATFORM must be set to 'so101' before importing prismatic. "
            "Either export ASYNCVLA_PLATFORM=so101 or run via `ASYNCVLA_PLATFORM=so101 python ...`."
        )
    assert cfg.use_lora, "Only LoRA fine-tuning is supported. Please set --use_lora=True!"
    cfg.vla_path = cfg.vla_path.rstrip("/")

    yaml_cfg = cfg.load_dataset_yaml()
    dataset_cfg = yaml_cfg["dataset"]
    if cfg.dataset_repo_id:
        dataset_cfg["repo_id"] = cfg.dataset_repo_id
    if cfg.dataset_root:
        dataset_cfg["root"] = cfg.dataset_root
    if cfg.episodes is not None:
        dataset_cfg["episodes"] = list(cfg.episodes)

    run_id = get_run_id(cfg)
    run_dir = cfg.run_root_dir / run_id
    os.makedirs(run_dir, exist_ok=True)
    print(f"[so101] run_dir = {run_dir}")

    distributed_state = PartialState()
    device_id = distributed_state.local_process_index
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()
    print(f"[so101] world size={world_size}, rank={device_id}")

    if distributed_state.is_main_process:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"ft+{run_id}")

    # Resolve VLA path / register HF auto classes.
    on_hub = model_is_on_hf_hub(cfg.vla_path)
    if on_hub:
        cfg.vla_path = snapshot_download(repo_id=cfg.vla_path)
    else:
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction_MMNv1)

    if distributed_state.is_main_process:
        update_auto_map(cfg.vla_path)
        check_model_logic_mismatch(cfg.vla_path)
    dist.barrier()

    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    if processor.tokenizer.pad_token_id is None:
        # OpenVLA ships a LLaMA tokenizer without a pad token; reuse EOS so the
        # collator can pad input_ids and attention masks.
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    if on_hub:
        import json

        from safetensors.torch import load_file

        index_file = os.path.join(cfg.vla_path, "model.safetensors.index.json")
        if os.path.exists(index_file):
            with open(index_file, "r") as f:
                index = json.load(f)
            filenames = set(index["weight_map"].values())
        else:
            filenames = {"model.safetensors"}
        state_dict = {}
        for fname in filenames:
            state_dict.update(load_file(os.path.join(cfg.vla_path, fname)))
        config_openvla = AutoConfig.from_pretrained(cfg.vla_path, trust_remote_code=True)
        if cfg.attn_implementation:
            config_openvla._attn_implementation = cfg.attn_implementation
        vla = OpenVLAForActionPrediction_MMNv1(config_openvla)
        vla.load_state_dict(state_dict, strict=False)
    else:
        vla = AutoModelForVision2Seq.from_pretrained(
            cfg.vla_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation=cfg.attn_implementation or "eager",
        ).to(device_id)

    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)
    vla.to(dtype=torch.bfloat16, device=device_id)

    target_modules = [name for name, mod in vla.named_modules() if isinstance(mod, torch.nn.Linear)]
    lora_config = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=min(cfg.lora_rank, 16),
        lora_dropout=cfg.lora_dropout,
        target_modules=target_modules,
        init_lora_weights="gaussian",
    )
    vla = get_peft_model(vla, lora_config)
    vla.print_trainable_parameters()
    vla = wrap_ddp(vla, device_id, find_unused=True)

    pose_projector = init_module(
        ProprioProjector,
        "pose_projector",
        cfg,
        device_id,
        {"llm_dim": vla.module.llm_dim, "proprio_dim": POSE_DIM},
    )
    action_head = init_module(
        L1RegressionActionHead_idcat,
        "action_head",
        cfg,
        device_id,
        {"input_dim": vla.module.llm_dim, "hidden_dim": vla.module.llm_dim, "action_dim": ACTION_DIM},
        to_bf16=True,
    )

    num_patches = (
        vla.module.vision_backbone.get_num_patches()
        * vla.module.vision_backbone.get_num_images_in_input()
        + 1  # proprio token
    )

    trainable_params = [p for p in vla.parameters() if p.requires_grad]
    trainable_params += [p for p in pose_projector.parameters() if p.requires_grad]
    trainable_params += [p for p in action_head.parameters() if p.requires_grad]
    print(f"[so101] total trainable params: {sum(p.numel() for p in trainable_params)}")
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)
    original_lr = optimizer.param_groups[0]["lr"]
    scheduler = MultiStepLR(optimizer, milestones=[cfg.num_steps_before_decay], gamma=0.1)

    action_tokenizer = ActionTokenizer(processor.tokenizer)
    cameras_cfg = yaml_cfg.get("cameras", {})
    features_cfg = yaml_cfg.get("features", {})

    dataset = SO101_Dataset(
        repo_id=dataset_cfg["repo_id"],
        action_tokenizer=action_tokenizer,
        base_tokenizer=processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        root=dataset_cfg.get("root"),
        episodes=dataset_cfg.get("episodes"),
        main_camera=cameras_cfg.get("main_camera"),
        secondary_camera=cameras_cfg.get("secondary_camera"),
        state_key=features_cfg.get("state", "observation.state"),
        action_key=features_cfg.get("action", "action"),
        task_key=features_cfg.get("task", "task"),
        action_chunk_size=int(yaml_cfg.get("action_chunk_size", NUM_ACTIONS_CHUNK)),
        image_size=tuple(yaml_cfg.get("image_size", [96, 96])),
        modality_id=int(yaml_cfg.get("modality_id", 5)),
        default_prompt=yaml_cfg.get("default_prompt", "Perform the demonstrated manipulation task."),
        video_backend=dataset_cfg.get("video_backend", "pyav"),
    )

    collator = PaddedCollatorForActionPrediction_SO101(
        model_max_length=processor.tokenizer.model_max_length,
        pad_token_id=processor.tokenizer.pad_token_id,
        padding_side="right",
        num_img=cfg.num_images_in_input,
    )

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=device_id, shuffle=True)
    train_loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=cfg.num_workers,
        drop_last=True,
        persistent_workers=cfg.num_workers > 0,
        sampler=sampler,
    )

    recent_metrics = {
        "loss_value": deque(maxlen=cfg.grad_accumulation_steps),
        "L1_action_value": deque(maxlen=cfg.grad_accumulation_steps),
        "L2_action_value": deque(maxlen=cfg.grad_accumulation_steps),
    }

    log_count = 0
    train_iter = iter(train_loader)
    vla.train()
    pose_projector.train()
    action_head.train()
    optimizer.zero_grad()

    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        epoch = 0
        for batch_idx in range(cfg.max_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                epoch += 1
                sampler.set_epoch(epoch)
                train_iter = iter(train_loader)
                batch = next(train_iter)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, metrics = run_forward_pass(
                    vla=vla,
                    action_head=action_head,
                    pose_projector=pose_projector,
                    batch=batch,
                    device_id=device_id,
                    num_patches=num_patches,
                )
            normalized_loss = loss / cfg.grad_accumulation_steps
            normalized_loss.backward()

            for name, value in metrics.items():
                if name in recent_metrics:
                    recent_metrics[name].append(value)

            gradient_step_idx = log_count // cfg.grad_accumulation_steps
            log_count += 1
            log_step = gradient_step_idx if not cfg.resume else (cfg.resume_step or 0) + gradient_step_idx

            if cfg.lr_warmup_steps > 0:
                lr_progress = min((gradient_step_idx + 1) / cfg.lr_warmup_steps, 1.0)
                current_lr = original_lr * (0.1 + 0.9 * lr_progress)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = current_lr

            if distributed_state.is_main_process and log_step % cfg.wandb_log_freq == 0:
                wandb_payload = {f"VLA Train/{k}": (sum(v) / max(len(v), 1)) for k, v in recent_metrics.items() if v}
                wandb_payload["VLA Train/Learning Rate"] = scheduler.get_last_lr()[0]
                wandb.log(wandb_payload, step=log_step)
                print(
                    f"[so101] step {log_step:>6d}  loss={metrics.get('loss_value', float('nan')):.4f}  "
                    f"L1={metrics.get('L1_action_value', float('nan')):.4f}  "
                    f"L2={metrics.get('L2_action_value', float('nan')):.4f}  "
                    f"lr={scheduler.get_last_lr()[0]:.2e}",
                    flush=True,
                )

            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                progress.update()

            if gradient_step_idx > 0 and log_step % cfg.save_freq == 0:
                save_training_checkpoint(
                    cfg=cfg,
                    run_dir=run_dir,
                    log_step=log_step,
                    vla=vla,
                    processor=processor,
                    pose_projector=pose_projector,
                    action_head=action_head,
                    distributed_state=distributed_state,
                )

    save_training_checkpoint(
        cfg=cfg,
        run_dir=run_dir,
        log_step=cfg.max_steps,
        vla=vla,
        processor=processor,
        pose_projector=pose_projector,
        action_head=action_head,
        distributed_state=distributed_state,
    )


if __name__ == "__main__":
    train_so101()
