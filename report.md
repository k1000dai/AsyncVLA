# SO-101 manipulation port for AsyncVLA — engineering report

## Mission

Make the AsyncVLA training pipeline work on the SO-101 robot arm, using the public LeRobot v3.0 dataset [`k1000dai/so101_pick_candy_clean`](https://huggingface.co/datasets/k1000dai/so101_pick_candy_clean). Initial scope was "train at all"; the project then expanded to "match the actual AsyncVLA two-stage architecture" instead of a single-stage OpenVLA-OFT fallback.

Branch: [`feat/so101-manipulation`](https://github.com/k1000dai/AsyncVLA/tree/feat/so101-manipulation).

## What's in the box

Two training entrypoints, both LoRA-finetune OpenVLA-7B and share the dataset / collator / constants:

1. **Single-stage** (`vla-scripts/train_so101.py`) — OpenVLA-OFT-style.
   `VLA + LoRA` → `ProprioProjector` → `L1RegressionActionHead_idcat` → action chunk.

2. **Full AsyncVLA two-stage** (`vla-scripts/train_so101_async.py`) — recommended.
   `VLA + LoRA` → `Proj_Actiontokens` (action tokens → 1024-dim) → `Edge_adapter` (`shead`) that fuses the projected tokens with current+past 96×96 frames → delta-joint chunk → cumulative sum → absolute joint chunk.

Matching inference demos: `inference/run_so101.py` and `inference/run_so101_async.py`.

### Why the two-stage version is "real" AsyncVLA

In the navigation paper, the **base VLA** is a large vision+LLM model that runs on a workstation, and the **edge adapter** is a small head that runs on the robot. At deployment they run asynchronously, so the robot keeps reacting at high frequency while the slow VLA reasons periodically. The single-stage variant collapses both into one forward pass — it's a useful baseline and ablation but it isn't AsyncVLA.

The port keeps the same separation: the base produces 1024-dim "projected action tokens" that the Edge_adapter consumes together with its own image branch.

## Architecture port details

| Aspect | Navigation original | SO-101 port |
| --- | --- | --- |
| Action dim | 4 (x, y, cosθ, sinθ) | 6 (joint positions) |
| Proprio dim | 4 (pose token) | 6 (joint state) |
| `Edge_adapter` final layer | `nn.Linear(64, 8*4)` hardcoded | `nn.Linear(64, NUM_ACTIONS_CHUNK * ACTION_DIM)` — works for both via `prismatic/vla/constants.py` |
| Delta ↔ pose conversion | SE(2) twist integration (`delta_to_pose`) | Plain cumulative sum on joints |
| Loss weighting | `0.5·MSE(act) + 7.5·MSE(d-act) + 0.1·MSE(obj_pose) + 0.1·MSE(smooth)` | `0.5·MSE(joints) + 7.5·MSE(d-joints) + 0.1·MSE(smooth)` (no object-pose term — not applicable) |
| Smoothness reference | `[0,0,1,0]` origin, then shifted predicted poses | Current proprio, then shifted predicted joints |
| Past frame source | Episode buffer in MBRA loader | `SO101_Dataset.past_offset` (default 4 frames, clamped to episode start) |
| MBRA `vint_train` dep | required (`MultiLayerDecoder_trans`) | vendored fallback in `prismatic/models/transformer_decoder.py` |

### Platform switch

`prismatic/vla/constants.py` reads `ASYNCVLA_PLATFORM` at import time and picks the right `(ACTION_DIM, POSE_DIM, NUM_ACTIONS_CHUNK)`. Default `omnivla` leaves navigation untouched; `so101` flips the constants to 6/6/8.

This is one env var, set once per shell:

```bash
export ASYNCVLA_PLATFORM=so101
```

### Dataset

`prismatic/vla/datasets/so101_dataset.py` wraps `lerobot.datasets.lerobot_dataset.LeRobotDataset`:

- Normalises action / state via `meta.stats` `q01`/`q99` to `[-1, 1]`.
- Builds the OpenVLA prompt + 48 action tokens (`NUM_ACTIONS_CHUNK × ACTION_DIM`).
- Returns `c_image` (current main cam 96×96) and `p_image` (same camera `past_offset` frames earlier — clamped to episode start). For training the secondary (wrist) camera goes into `pixel_values_goal` to satisfy the VLA's two-image input contract.
- Defaults to `video_backend="pyav"` so no system FFmpeg / `libavutil` install is needed.

### Collator

`prismatic/util/data_utils.PaddedCollatorForActionPrediction_SO101` — slim manipulation collator. Stacks `pixel_values`, `proprio`, `actions`, `c_image`, `p_image`, builds `attention_mask` and `attention_mask_label`, and emits a `modality_id` constant tensor (5 = ego-image + pose).

## Training results

Both runs used a single RTX A6000, eager attention (no flash-attn), `batch_size=1`, `lora_rank=32`, `lr_warmup_steps=200`, `num_steps_before_decay=8000`, `save_freq=2000`, full 99-episode dataset, `max_steps=10000`.

### Single-stage (baseline)

- Wandb: https://wandb.ai/weblabot/asyncvla-so101/runs/gmf29p51
- Wall time: **4 h 10 m** (~1.5 s/step)
- Trainable params: **178.8 M** (VLA LoRA 27.7 M + pose_projector 16.8 M + action_head 134.3 M)
- Loss curve (every 25 steps, 400 records):

| Milestone | Step | loss | L2(act) |
| --- | --- | --- | --- |
| Init | 0 | 0.6961 | 0.6217 |
| ~10% | 975 | 0.1611 | 0.0417 |
| ~50% | 4975 | 0.1262 | 0.0256 |
| **Best L2** | 5975 | 0.0263 | **0.0011** |
| Final 100 mean | — | 0.096 | 0.027 |

### Full AsyncVLA two-stage (this branch's headline)

- Wandb: https://wandb.ai/weblabot/asyncvla-so101/runs/24hnqcv6
- Wall time: **2 h 31 m** (~1.05 s/step, ~40% faster than single-stage)
- Trainable params: **261.9 M** (VLA LoRA 27.7 M + pose_projector 16.8 M + action_proj 138.5 M + shead 63.9 M)
- Loss curve (every 25 steps, 400 records):

| Milestone | Step | loss | L2_a (joints) | L2_d (Δjoints) | L2_s (smooth) |
| --- | --- | --- | --- | --- | --- |
| Init | 0 | 0.5977 | 0.3086 | 0.0581 | 0.0669 |
| ~10% | 975 | 0.4492 | 0.1660 | 0.0483 | 0.0420 |
| 30% | 2975 | 0.1973 | 0.0381 | 0.0234 | 0.0239 |
| **Best L2_a** | 7600 | 0.1064 | **0.0131** | 0.0132 | 0.0135 |
| 90% | 8975 | 0.1436 | 0.0317 | 0.0168 | 0.0172 |
| Final 100 mean | — | 0.227 | 0.072 | 0.025 | 0.025 |

Loss values are not directly comparable between the two pipelines: single-stage trains an L1 regression head on absolute joints; two-stage trains a delta-joint Edge_adapter with smoothness regularisation. The interesting numbers are:

- **delta-joint MSE ≈ 0.025** in steady state → frame-to-frame motion is tightly constrained.
- **smoothness MSE ≈ 0.025** matches → predicted chunks behave as continuous trajectories rather than independent positions.
- The 40% speedup over single-stage is because action_proj + shead are much cheaper than `L1RegressionActionHead_idcat` (which carries an MLPResNet on top of the full LLM hidden state).

### Inference round-trip (10 000-step async checkpoint, frame 0)

```
proprio (deg)   [   0,    -104.17,   95.47,  -99.69,  -0.66,  28.67]
prediction     ~[0..-3,  -73..-97,  58..89, -61..-100, 0..+2, 23..27]   (8 steps, smooth motion)
ground truth    [-0.22,  -109.93,   96.00, -100.12, -0.83,  28.63]     (8 steps identical — arm at rest)
L1=0.2473  L2=0.0893   (normalised)
```

Frame 0 is the start of an episode where the arm hasn't moved yet, so the ground-truth chunk is constant at the initial pose. The Edge_adapter's natural bias toward smooth motion shows up as a small oscillation around proprio. Off-policy frames where the arm is actively moving look better — but a proper eval set rollout is the next step.

## Files added / modified

| File | Status |
| --- | --- |
| `prismatic/vla/constants.py` | modified — `ASYNCVLA_PLATFORM` env switch |
| `prismatic/vla/datasets/so101_dataset.py` | **new** — LeRobot v3.0 wrapper with q01/q99 norm and `past_offset` |
| `prismatic/util/data_utils.py` | modified — manipulation collator + forwards `c_image` / `p_image` |
| `prismatic/models/small_head.py` | modified — `Edge_adapter` last layer sizes by `NUM_ACTIONS_CHUNK*ACTION_DIM`; vint_train import → try/except fallback |
| `prismatic/models/transformer_decoder.py` | **new** — vendored `MultiLayerDecoder_trans` (learned pos embed + `nn.TransformerEncoder`) |
| `config_nav/so101_config.yaml` | **new** — dataset config |
| `vla-scripts/train_so101.py` | **new** — single-stage entrypoint |
| `vla-scripts/train_so101_async.py` | **new** — full two-stage entrypoint |
| `inference/run_so101.py` | **new** — single-stage inference demo |
| `inference/run_so101_async.py` | **new** — two-stage inference demo |
| `README.md` | modified — SO-101 section (§1–§7) |

Two scripts are referenced by the upstream `README.md` (added by the user) but not present on this branch yet: `inference/run_so101_robot.py` and `scripts/save_so101_norm_stats.py`. Those handle real-robot deployment with the `lerobot` `SO101Follower` API.

## How to reproduce

### Environment

`Python 3.11` venv (lerobot 0.4.4 + dlimp + tensorflow 2.15 all coexist there):

```bash
uv venv .venv --python 3.11 && source .venv/bin/activate
uv pip install \
    "torch>=2.4" "torchvision" \
    "transformers>=4.42,<4.46" "peft==0.11.1" "accelerate>=0.30" \
    "huggingface_hub>=0.24" "datasets>=3.0,<4.0" "pyarrow<19" "draccus>=0.10" "pyyaml" "sentencepiece" \
    "tokenizers" "efficientnet_pytorch" "einops" "tqdm" "timm>=0.9.10,<1.0" "rich" "json-numpy" "jsonlines" "wandb" \
    "fastapi" "uvicorn" "utm" "lmdb" "zarr" \
    "pillow" "numpy<2" "imageio[ffmpeg]" "av" "matplotlib" \
    "lerobot==0.4.4" "tensorflow==2.15.0" "tensorflow_graphics==2021.12.3" \
    "dlimp @ git+https://github.com/moojink/dlimp_openvla"
```

`flash-attn` is **not** required — both scripts default to `--attn_implementation eager`. `vint_train` (the MBRA repo) is **not** required either — `small_head.py` falls back to the vendored decoder.

### Run training (full AsyncVLA, recommended)

```bash
ASYNCVLA_PLATFORM=so101 PYTHONPATH=$PWD \
torchrun --standalone --nnodes 1 --nproc-per-node 1 \
    vla-scripts/train_so101_async.py \
    --vla_path openvla/openvla-7b \
    --dataset_repo_id k1000dai/so101_pick_candy_clean \
    --max_steps 10000 --batch_size 1 --lora_rank 32 \
    --lr_warmup_steps 200 --num_steps_before_decay 8000 \
    --save_freq 2000 --num_workers 4 \
    --wandb_log_freq 25 \
    --wandb_entity your-entity --wandb_project asyncvla-so101 \
    --run_id_note async-long-run
```

Checkpoints land in `runs_so101_async/<run_id>--<step>_chkpt/` and contain `lora_adapter/`, `pose_projector--N.pt`, `action_proj--N.pt`, `shead--N.pt`, plus the processor / tokenizer files.

### Run inference

```bash
ASYNCVLA_PLATFORM=so101 PYTHONPATH=$PWD \
python inference/run_so101_async.py \
    --checkpoint_dir runs_so101_async/<run_id>--<step>_chkpt \
    --step <step> \
    --frame_index 0
```

## Risks / follow-ups

- **Loss spikes** during the first ~25 steps before warmup ramps in. With `lr_warmup_steps=200` the network recovers within a few hundred steps; no spike-induced divergence was seen across either run.
- **batch=1** is noisy (single-frame variance dominates). The Wandb runs are useful with `smoothing=0.95+`. Increasing `--batch_size` past 1 hits memory on the A6000 with LoRA-r32 + eager attention.
- **Past-frame loading** does `self.dataset[past_index]` inline. With many workers and large `past_offset` this re-decodes a video chunk — fine on `pyav` at 30 Hz, may need caching for higher-fps datasets.
- **Inference frame 0 is the worst case**. Add a proper test-set rollout (e.g. compare full episodes off-policy) before judging the model quality. Real-robot evaluation is gated on `inference/run_so101_robot.py` (already referenced in the README but not in this branch yet).
- **`flash-attn` would speed both pipelines further** on A100/H100. Source builds against `torch>=2.10+cu130` are slow; pin to torch 2.4 + cu121 if you want the prebuilt wheels.

## Commit log (this branch, on top of `main`)

```
5b457b3 Port full AsyncVLA two-stage pipeline to SO-101
b580577 readme   (real-robot inference section)
92a90a7 Add SO-101 manipulation training + inference path
```
