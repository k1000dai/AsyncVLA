"""train_so101_async.py

Full AsyncVLA two-stage finetune for SO-101 manipulation.

Stages (mirrors the navigation recipe in train_asyncvla.py):

  Base VLA (LoRA)
    -> hidden states for action tokens
       -> Proj_Actiontokens (action_proj):    (B, NUM_ACTIONS_CHUNK, 1024)
          -> Edge_adapter (shead) using current + past 96x96 images
             -> predicted_djoints (B, 8, 6)
                -> cumulative sum
                   -> predicted_joints (B, 8, 6) in [-1, 1]

The first stage is slow / large (BaseVLA on the workstation) and the second
stage is fast / small (Edge_adapter on the robot edge controller).

Loss is the joint-space equivalent of the nav loss:

    L = 0.5 * MSE(action,       predicted_joints)        # absolute target
      + 0.5 * 15 * MSE(daction, predicted_djoints)        # smoothness
      + 0.1 * MSE(sm_ref,       predicted_joints)         # frame-to-frame

where ``daction[t] = action[t] - action[t-1]`` for t>=1 and ``daction[0] =
action[0]`` (matches the convention in train_asyncvla.py's
``pose_to_delta``). The 15x scaling on the delta term is the same one used by
the nav script; deltas are much smaller in magnitude than absolute joint
targets so this keeps both terms in the same ballpark.

Example:

    ASYNCVLA_PLATFORM=so101 PYTHONPATH=$PWD \
    torchrun --standalone --nnodes 1 --nproc-per-node 1 \
        vla-scripts/train_so101_async.py \
        --vla_path openvla/openvla-7b \
        --dataset_repo_id k1000dai/so101_pick_candy_clean \
        --wandb_entity weblabot --wandb_project asyncvla-so101 \
        --max_steps 10000 --batch_size 1 --lora_rank 32
"""
# ==============================
# Path Setup
# ==============================
import os
import sys

os.environ.setdefault("ASYNCVLA_PLATFORM", "so101")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Type

import draccus
import torch
import torch.distributed as dist
import torch.nn as nn
import torchvision.transforms as transforms
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
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.models.projectors import ProprioProjector
from prismatic.models.small_head import Edge_adapter, Proj_Actiontokens
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


# Image normalization for the Edge_adapter input branch (matches train_asyncvla.py).
_IMG_NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)


# ==============================
# Config
# ==============================
@dataclass
class SO101AsyncConfig:
    # Base VLA / dataset
    vla_path: str = "openvla/openvla-7b"
    dataset_config: Path = Path("config_nav/so101_config.yaml")
    dataset_repo_id: Optional[str] = None
    dataset_root: Optional[str] = None
    episodes: Optional[List[int]] = None
    past_offset: int = 4

    # Run management
    run_root_dir: Path = Path("runs_so101_async")
    run_id_note: Optional[str] = None
    run_id_override: Optional[str] = None

    # Architecture
    num_images_in_input: int = 2
    proj_action_dim: int = 1024  # output dim of Proj_Actiontokens, matches Edge_adapter's embed dim
    attn_implementation: str = "eager"

    # Training stages (mirrors the TRAIN_BASE / TRAIN_HEAD flags in train_asyncvla.py)
    train_base: bool = True   # if True, LoRA-finetune the VLA + pose_projector too
    train_head: bool = True   # if True, train action_proj + shead

    # Hyperparameters
    batch_size: int = 1
    learning_rate: float = 5e-5
    lr_warmup_steps: int = 200
    num_steps_before_decay: int = 8000
    grad_accumulation_steps: int = 1
    max_steps: int = 10000
    save_freq: int = 2000
    save_latest_checkpoint_only: bool = False

    # Loss weights (defaults match train_asyncvla.py within the manipulation-applicable terms)
    w_action: float = 0.5
    w_daction: float = 0.5 * 15.0
    w_smooth: float = 0.1

    # LoRA
    use_lora: bool = True
    lora_rank: int = 32
    lora_dropout: float = 0.0
    merge_lora_during_training: bool = False

    # Resume
    resume: bool = False
    resume_step: Optional[int] = None

    # Data loading
    num_workers: int = 4

    # Logging
    wandb_entity: str = "your-wandb-entity"
    wandb_project: str = "asyncvla-so101"
    wandb_log_freq: int = 25
    print_freq: int = 25

    def load_dataset_yaml(self) -> dict:
        with open(self.dataset_config, "r") as f:
            return yaml.safe_load(f)


# ==============================
# Helpers
# ==============================
def remove_ddp_prefix(state_dict: dict) -> dict:
    return {
        k[len("module."):] if k.startswith("module.") else k: v
        for k, v in state_dict.items()
    }


def wrap_ddp(module: nn.Module, device_id: int, find_unused: bool = False) -> DDP:
    return DDP(
        module,
        device_ids=[device_id],
        find_unused_parameters=find_unused,
        gradient_as_bucket_view=True,
    )


def maybe_load_state(module: nn.Module, ckpt_path: Path) -> None:
    if not ckpt_path.exists():
        print(f"[so101-async] no checkpoint at {ckpt_path}; random init.")
        return
    print(f"[so101-async] loading {ckpt_path.name}")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    module.load_state_dict(remove_ddp_prefix(state), strict=False)


def init_module(
    module_class: Type[nn.Module],
    module_name: str,
    cfg: SO101AsyncConfig,
    device_id: int,
    module_args: dict,
    to_bf16: bool = False,
    find_unused_params: bool = False,
) -> DDP:
    module = module_class(**module_args)
    n_train = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"[so101-async] # trainable params in {module_name}: {n_train:,}")
    if cfg.resume and cfg.resume_step is not None:
        ckpt = Path(cfg.vla_path) / f"{module_name}--{cfg.resume_step}_checkpoint.pt"
        maybe_load_state(module, ckpt)
    if to_bf16:
        module = module.to(torch.bfloat16)
    module = module.to(device_id)
    return wrap_ddp(module, device_id, find_unused_params)


def get_run_id(cfg: SO101AsyncConfig) -> str:
    if cfg.run_id_override:
        return cfg.run_id_override
    base = cfg.vla_path.rstrip("/").split("/")[-1]
    run_id = (
        f"{base}+so101-async+b{cfg.batch_size * cfg.grad_accumulation_steps}"
        f"+lr-{cfg.learning_rate}"
    )
    if cfg.use_lora:
        run_id += f"+lora-r{cfg.lora_rank}"
    if cfg.run_id_note:
        run_id += f"--{cfg.run_id_note}"
    return run_id


# ==============================
# Joint-space delta / accumulate (cumsum-based, fully differentiable)
# ==============================
def joints_to_delta(joints: torch.Tensor) -> torch.Tensor:
    """``joints`` shape (B, T, D) -> delta with delta[0]=joints[0], delta[t]=joints[t]-joints[t-1]."""
    if joints.size(1) == 0:
        return joints
    first = joints[:, :1, :]
    rest = joints[:, 1:, :] - joints[:, :-1, :]
    return torch.cat([first, rest], dim=1)


def delta_to_joints(delta: torch.Tensor) -> torch.Tensor:
    """Inverse of ``joints_to_delta``: cumulative sum along the chunk axis."""
    return torch.cumsum(delta, dim=1)


# ==============================
# Forward pass (the heart of the AsyncVLA stack)
# ==============================
def run_forward_pass(
    cfg: SO101AsyncConfig,
    vla,
    action_proj,
    shead,
    pose_projector,
    batch,
    device_id: int,
    num_patches: int,
) -> Tuple[torch.Tensor, dict]:
    ground_truth_actions = batch["actions"].to(device_id).to(torch.bfloat16)        # (B, 8, 6)
    modality_id = batch["goal_mask_select"].to(torch.bfloat16).to(device_id)

    # Edge adapter image inputs (Normalize them inline to match the nav script).
    img_cur = _IMG_NORMALIZE(batch["c_image"]).to(device_id).to(torch.bfloat16)
    img_past = _IMG_NORMALIZE(batch["p_image"]).to(device_id).to(torch.bfloat16)

    # ----- Stage 1: Base VLA -----
    def _vla_forward():
        return vla(
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

    if cfg.train_base:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output: CausalLMOutputWithPast = _vla_forward()
    else:
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = _vla_forward()

    # Extract action-token hidden states (identical slicing to train_asyncvla.py).
    ground_truth_token_ids = batch["labels"][:, 1:].to(device_id)
    current_mask = get_current_action_mask(ground_truth_token_ids)
    next_mask = get_next_actions_mask(ground_truth_token_ids)

    last_hidden = output.hidden_states[-1]                     # (B, seq, llm_dim)
    text_hidden = last_hidden[:, num_patches:-1]               # skip vision+proprio, drop last position
    bsz = batch["input_ids"].shape[0]
    actions_hidden = (
        text_hidden[current_mask | next_mask]
        .reshape(bsz, NUM_ACTIONS_CHUNK * ACTION_DIM, -1)
        .to(torch.bfloat16)
    )

    # ----- Stage 2: action_proj -> Edge_adapter -----
    if cfg.train_head:
        projected = action_proj.module.predict_action(actions_hidden, modality_id)    # (B, 8, 1024)
        predicted_djoints = shead(img_cur, img_past, projected)                       # (B, 8, 6)
    else:
        with torch.no_grad():
            projected = action_proj.module.predict_action(actions_hidden.detach(), modality_id)
            predicted_djoints = shead(img_cur, img_past, projected)

    predicted_joints = delta_to_joints(predicted_djoints)                             # (B, 8, 6)
    djoints_ref = joints_to_delta(ground_truth_actions)                               # (B, 8, 6)

    # Smoothness reference: previous predicted joints, prepended with proprio of t=-1.
    proprio = batch["proprio"].to(torch.bfloat16).to(device_id).unsqueeze(1)          # (B, 1, 6)
    sm_ref = torch.cat([proprio, predicted_joints[:, :-1].detach()], dim=1)           # (B, 8, 6)

    L2_action = nn.functional.mse_loss(predicted_joints, ground_truth_actions)
    L2_daction = nn.functional.mse_loss(predicted_djoints, djoints_ref)
    L2_smooth = nn.functional.mse_loss(predicted_joints, sm_ref)
    L1_action = nn.functional.l1_loss(predicted_joints, ground_truth_actions)

    loss = (
        cfg.w_action * L2_action
        + cfg.w_daction * L2_daction
        + cfg.w_smooth * L2_smooth
    )

    metrics = {
        "loss_value": loss.item(),
        "L1_action_value": L1_action.item(),
        "L2_action_value": L2_action.item(),
        "L2_daction_value": L2_daction.item(),
        "L2_smooth_value": L2_smooth.item(),
    }
    return loss, metrics


# ==============================
# Checkpointing
# ==============================
def save_training_checkpoint(
    cfg: SO101AsyncConfig,
    run_dir: Path,
    log_step: int,
    vla,
    processor,
    pose_projector,
    action_proj,
    shead,
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
        torch.save(action_proj.state_dict(), checkpoint_dir / f"action_proj--{checkpoint_name}")
        torch.save(shead.state_dict(), checkpoint_dir / f"shead--{checkpoint_name}")

    dist.barrier()

    if cfg.use_lora and cfg.merge_lora_during_training and distributed_state.is_main_process:
        base_vla = AutoModelForVision2Seq.from_pretrained(
            cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
        )
        merged = PeftModel.from_pretrained(base_vla, adapter_dir).merge_and_unload()
        merged.save_pretrained(checkpoint_dir)
    dist.barrier()


# ==============================
# Entrypoint
# ==============================
@draccus.wrap()
def train_so101_async(cfg: SO101AsyncConfig) -> None:
    if ROBOT_PLATFORM != "so101":
        raise RuntimeError(
            "ASYNCVLA_PLATFORM must be set to 'so101' before importing prismatic. "
            "Export it (e.g. `export ASYNCVLA_PLATFORM=so101`) or run via "
            "`ASYNCVLA_PLATFORM=so101 python ...`."
        )
    assert cfg.use_lora, "Only LoRA fine-tuning is supported; pass --use_lora=True."
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
    print(f"[so101-async] run_dir = {run_dir}")

    distributed_state = PartialState()
    device_id = distributed_state.local_process_index
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()
    print(f"[so101-async] world size={world_size}, rank={device_id}")

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
    action_proj = init_module(
        Proj_Actiontokens,
        "action_proj",
        cfg,
        device_id,
        {
            "input_dim": vla.module.llm_dim,
            "hidden_dim": vla.module.llm_dim,
            "action_dim": cfg.proj_action_dim,
        },
        to_bf16=True,
    )
    shead = init_module(
        Edge_adapter,
        "shead",
        cfg,
        device_id,
        {
            "obs_encoding_size": cfg.proj_action_dim,
            "mha_num_attention_heads": 4,
            "mha_num_attention_layers": 4,
            "mha_ff_dim_factor": 4,
        },
        to_bf16=True,
        find_unused_params=True,
    )

    num_patches = (
        vla.module.vision_backbone.get_num_patches()
        * vla.module.vision_backbone.get_num_images_in_input()
        + 1
    )

    # Freeze base VLA / proprio projector when only training the head.
    if not cfg.train_base:
        for p in vla.parameters():
            p.requires_grad = False
        for p in pose_projector.parameters():
            p.requires_grad = False
    if not cfg.train_head:
        for p in action_proj.parameters():
            p.requires_grad = False
        for p in shead.parameters():
            p.requires_grad = False

    trainable_params = []
    if cfg.train_base:
        trainable_params += [p for p in vla.parameters() if p.requires_grad]
        trainable_params += [p for p in pose_projector.parameters() if p.requires_grad]
    if cfg.train_head:
        trainable_params += [p for p in action_proj.parameters() if p.requires_grad]
        trainable_params += [p for p in shead.parameters() if p.requires_grad]
    print(f"[so101-async] total trainable params: {sum(p.numel() for p in trainable_params):,}")
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
        past_offset=cfg.past_offset,
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
        k: deque(maxlen=cfg.grad_accumulation_steps)
        for k in (
            "loss_value",
            "L1_action_value",
            "L2_action_value",
            "L2_daction_value",
            "L2_smooth_value",
        )
    }

    log_count = 0
    train_iter = iter(train_loader)

    # Match the eval/train mode policy from train_asyncvla.py.
    if cfg.train_base:
        vla.train()
        pose_projector.train()
    else:
        vla.eval()
        pose_projector.eval()
    if cfg.train_head:
        action_proj.train()
        shead.train()
    else:
        action_proj.eval()
        shead.eval()
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

            loss, metrics = run_forward_pass(
                cfg=cfg,
                vla=vla,
                action_proj=action_proj,
                shead=shead,
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
                payload = {
                    f"VLA Train/{k}": (sum(v) / max(len(v), 1))
                    for k, v in recent_metrics.items() if v
                }
                payload["VLA Train/Learning Rate"] = scheduler.get_last_lr()[0]
                wandb.log(payload, step=log_step)

            if distributed_state.is_main_process and log_step % cfg.print_freq == 0:
                print(
                    f"[so101-async] step {log_step:>6d}  "
                    f"loss={metrics['loss_value']:.4f}  "
                    f"L1={metrics['L1_action_value']:.4f}  "
                    f"L2_a={metrics['L2_action_value']:.4f}  "
                    f"L2_d={metrics['L2_daction_value']:.4f}  "
                    f"L2_s={metrics['L2_smooth_value']:.4f}  "
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
                    action_proj=action_proj,
                    shead=shead,
                    distributed_state=distributed_state,
                )

    save_training_checkpoint(
        cfg=cfg,
        run_dir=run_dir,
        log_step=cfg.max_steps,
        vla=vla,
        processor=processor,
        pose_projector=pose_projector,
        action_proj=action_proj,
        shead=shead,
        distributed_state=distributed_state,
    )


if __name__ == "__main__":
    train_so101_async()
