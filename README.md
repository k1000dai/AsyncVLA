# AsyncVLA: An Asynchronous VLA for Fast and Robust Navigation on the Edge
[![Python](https://img.shields.io/badge/python-3.10-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Static Badge](https://img.shields.io/badge/Project-Page-a)](https://asyncvla.github.io)


[Noriaki Hirose](https://sites.google.com/view/noriaki-hirose/)<sup>1, 2</sup>, [Catherine Glossop](https://catglossop.github.io/)<sup>1</sup>, [Dhruv Shah](https://robodhruv.github.io/)<sup>3</sup>, [Sergey Levine](https://people.eecs.berkeley.edu/~svlevine/)<sup>1</sup>

<sup>1</sup> UC Berkeley (_Berkeley AI Research_),  <sup>2</sup> Toyota Motor North America, ,  <sup>3</sup> Princeton University

### Installation
Please set up a conda environment (see instructions in [SETUP.md](SETUP.md)).

### Inference
1. Download our repositories. (You need to download another repository of our previous project to define our entire model.)
    ```
    git clone https://github.com/NHirose/AsyncVLA.git
    git clone https://github.com/NHirose/Learning-to-Drive-Anywhere-with-MBRA.git
    ```

2. Download our checkpoints and place them in our directory, AsyncVLA. 
    ```
    cd AsyncVLA
    git clone https://huggingface.co/NHirose/AsyncVLA_release
    ```
3. Run AsyncVLA using sample current images and 2D goal pose. You can view the generated trajectories in the output figure visualization_asyncvla.jpg. (Run BaseVLA and Edge adapter in same PC)
    ```
    cd ..
    python inference/run_asyncvla.py
    ```   
4. Run AsyncVLA to control the real robot. We split the AsyncVLA into the base VLA and the edge adapter. Then we run the base VLA in the remote workstation and run the edge adapter in the robot edge controller with ROS1. Details are shown in the paper appendix. 

### Datasets
We provide training code that supports multiple public datasets. Before following the full training process, please first ensure that you can run the example training with the sample dataloader.

1. Downloading all datasets from the original website. ([GNM](https://github.com/robodhruv/visualnav-transformer), [LeLaN](https://github.com/NHirose/learning-language-navigation), [SACSoN(HuRoN)](https://sites.google.com/view/sacson-review/home)) Please verify that the downloaded datasets work properly in their original codebase.

2. Downloading the lerobot code base for the Frodobots dataset dataloader:
    ```
    git clone https://github.com/huggingface/lerobot.git 
    ```
3. Edit the data path in config_nav/dataset_config.yaml:
       
In our training setup, we use 5 Nvidia H200 GPUs (140 GB each) across 5 nodes. The batch sizes are configured as [LeLaN, GNM, SACSoN] = [6, 6, 6], with gradient accumulation set to 2 steps. 

### Training
We provide the training code along with a sample dataloader to help you quickly understand the required data loading structure. Since preparing the full training dataset is resource-intensive, we include this simplified code base for convenience.

1. You can set the training mode at line 10 and 11 in vla-scripts/train_asyncvla.py.

2. You can configure visualization at line 12 in vla-scripts/train_asyncvla.py. During training, it should be set to False.
    
3. Training our policy from AsyncVLA checkpoints (Please fill X):
    ```
    torchrun --standalone --nnodes 1 --nproc-per-node X vla-scripts/train_asyncvla.py  --vla_path ./AsyncVLA_release --dataset_name asyncvla --wandb_entity "X"   --wandb_project "asyncvla" --grad_accumulation_steps X
    ```
    
### SO-101 manipulation (LeRobot)

This fork adds a manipulation training + inference path that finetunes the OpenVLA-7B base (the AsyncVLA backbone) on a HuggingFace LeRobot v3.0 dataset such as [`k1000dai/so101_pick_candy_clean`](https://huggingface.co/datasets/k1000dai/so101_pick_candy_clean).

It reuses the AsyncVLA model surface (vision + LLM + proprio projector + L1 regression action head) and strips out the navigation-only multi-modal trajectory losses and the MBRA edge adapter — so no `Learning-to-Drive-Anywhere-with-MBRA` checkout is required.

#### 1. Environment

The original navigation environment doesn't load `k1000dai/so101_pick_candy_clean` (LeRobot v3) out of the box. A known-good combo:

```
python 3.11
torch 2.2.0  torchvision 0.17.1                # original pins also work
transformers 4.45.x  peft 0.11.1  accelerate>=0.30
huggingface_hub>=0.24  datasets>=3.0,<4.0  pyarrow<19  draccus>=0.10
timm>=0.9.10,<1.0.0                            # hard checked in modeling_prismatic.py
tensorflow==2.15.0  tensorflow_graphics==2021.12.3  dlimp@git+https://github.com/moojink/dlimp_openvla
numpy<2                                         # tensorflow 2.15 needs this; lerobot can bump it
lerobot==0.4.4                                  # reads LeRobot v3.0 datasets
av  imageio[ffmpeg]                             # pyav video backend, avoids torchcodec/libavutil
```

`flash-attn` is **not required** — `train_so101.py --attn_implementation eager` (default) runs on any CUDA GPU. Install flash-attn if you want extra throughput on A100/H100.

#### 2. Configure

Edit `config_nav/so101_config.yaml` to point at your dataset and cameras. Defaults match `k1000dai/so101_pick_candy_clean`:

```yaml
dataset:
  repo_id: k1000dai/so101_pick_candy_clean
  video_backend: pyav
cameras:
  main_camera: observation.images.top
  secondary_camera: observation.images.wrist
modality_id: 5                # pose + ego-image (proprio token active)
default_prompt: "pick up candy and put it in the bowl"
```

The platform constants (`ACTION_DIM=6`, `POSE_DIM=6`) are selected by an env var — export it once per shell, before any Python in this repo runs:

```
export ASYNCVLA_PLATFORM=so101
```

#### 3. Train

Single-GPU LoRA finetune of OpenVLA-7B:

```bash
ASYNCVLA_PLATFORM=so101 PYTHONPATH=$PWD \
torchrun --standalone --nnodes 1 --nproc-per-node 1 \
    vla-scripts/train_so101.py \
    --vla_path openvla/openvla-7b \
    --dataset_repo_id k1000dai/so101_pick_candy_clean \
    --max_steps 10000 --batch_size 1 --lora_rank 32 \
    --lr_warmup_steps 200 --num_steps_before_decay 8000 \
    --save_freq 2000 --num_workers 4 \
    --wandb_log_freq 25 \
    --wandb_entity your-entity --wandb_project asyncvla-so101 \
    --run_id_note long-run
```

Reference run on a single RTX A6000 / eager attention: **10 000 steps in ≈ 4 h 10 m**, action L2 from `0.62 → 0.001` (best), `0.027` running mean over the last 100 steps. Checkpoints land in `runs_so101/<run_id>--<step>_chkpt/` and contain:

- `lora_adapter/` (PEFT-style: `adapter_config.json`, `adapter_model.safetensors`) — LoRA delta on top of OpenVLA-7B
- `pose_projector--<step>_checkpoint.pt`
- `action_head--<step>_checkpoint.pt`
- HF processor/tokenizer files

To resume from an AsyncVLA checkpoint instead of OpenVLA, pass `--vla_path ./AsyncVLA_release --resume True --resume_step 750000`; the script will also pick up matching `pose_projector` / `action_head` checkpoints when present.

#### 4. Full AsyncVLA two-stage pipeline (recommended)

The "Async" in AsyncVLA refers to a two-stage architecture: a heavyweight **Base VLA** (vision + LLM + LoRA + proprio projector) and a lightweight **Edge adapter** (`shead`) that consumes a projected representation of the base's action tokens. At deployment the base can run at low frequency on a workstation while the edge adapter runs at high frequency on the robot edge controller.

`vla-scripts/train_so101_async.py` ports this two-stage training to manipulation. The action chunk is supervised in joint space using the same loss recipe as the navigation script, adapted to 6-DOF joint targets:

```
predicted_djoints = shead(c_image, p_image, action_proj(vla_hidden, modality_id))
predicted_joints  = cumsum(predicted_djoints)   # delta -> absolute, like delta_to_pose

L  = 0.5  * MSE(predicted_joints,  actions)        # absolute target
   + 7.5  * MSE(predicted_djoints, joints_to_delta(actions))   # smoothness via deltas
   + 0.1  * MSE(predicted_joints,  [proprio, predicted_joints[:-1]])  # consecutive-frame jump
```

Launch command (single A6000, eager attention):

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
    --run_id_note async
```

Verified smoke: 10 steps × LoRA r=8 produces 246.9 M trainable params (VLA LoRA 27.7 M + pose_projector 16.8 M + action_proj 138.5 M + shead 63.9 M), forward + backward closes cleanly, all four module checkpoints land under `runs_so101_async/<run_id>--<step>_chkpt/` (`lora_adapter/`, `pose_projector--N.pt`, `action_proj--N.pt`, `shead--N.pt`).

Inference uses the matching two-stage script:

```bash
ASYNCVLA_PLATFORM=so101 PYTHONPATH=$PWD \
python inference/run_so101_async.py \
    --checkpoint_dir runs_so101_async/<run_id>--<step>_chkpt \
    --step <step> \
    --frame_index 0
```

Notes:

- `vint_train` (the MBRA repo) is **not required** — `prismatic/models/transformer_decoder.py` provides a vendored `MultiLayerDecoder_trans` (learned positional embedding + standard `nn.TransformerEncoder`) that `prismatic/models/small_head.py` falls back to when `vint_train` is not importable. If you do have the MBRA checkout on `sys.path`, the original implementation is used unchanged so navigation checkpoints stay compatible.
- `Edge_adapter`'s final layer now sizes itself by `NUM_ACTIONS_CHUNK * ACTION_DIM`, so SO-101 (8×6) and navigation (8×4) both work from the same class without code forks.
- The `SO101_Dataset` now actually loads a past frame from the same episode (defaults to `past_offset=4`, clamped to the episode start) instead of repurposing the wrist camera as "past".

#### 5. Single-stage alternative

`vla-scripts/train_so101.py` is a simpler OpenVLA-OFT-style entrypoint kept around as a reference / minimal alternative. It trains only OpenVLA base + LoRA + `pose_projector` + `L1RegressionActionHead_idcat` (no Edge adapter). Use this if you want to ablate the two-stage architecture or save the action_proj / shead memory budget.

`inference/run_so101.py` loads a saved checkpoint, pulls one frame from the LeRobot dataset, runs a forward pass, and prints the predicted 8-step joint-position chunk next to the ground truth:

```bash
ASYNCVLA_PLATFORM=so101 PYTHONPATH=$PWD \
python inference/run_so101.py \
    --checkpoint_dir runs_so101/openvla-7b+so101+b1+lr-5e-05+lora-r32--long-run--10000_chkpt \
    --step 10000 \
    --frame_index 0
```

Example output (frame 0 of the candy-pick dataset):

```
proprio (deg)   [   0.,  -104.17,   95.47,  -99.69,   -0.66,  28.67]
prediction      ~[-1.5,  -108.,    90.,    -93.,    -1.1,  29.5 ]   (8 steps)
ground truth     [-0.22, -109.93,  96.00, -100.12,  -0.83, 28.63]   (all 8 identical)
L1=0.0591  L2=0.0044  (normalised; matches training-time L2)
```

To plug your own observation, replace the `SO101_Dataset` block at the bottom of `run_so101.py` with whatever produces `(main_image, secondary_image, state_norm, instruction)` from your robot. The `build_inputs(...)` + `predict_action_chunk(...)` helpers do the rest. Denormalise the model output with `dataset.action_low` / `dataset.action_high` (these are saved with the checkpoint in `lora_adapter/README.md` for the dataset you trained on, or recompute them from `LeRobotDatasetMetadata.stats`).

#### 6. Real-robot inference (SO-101 hardware)

`inference/run_so101_robot.py` runs the same model on a live SO-101 follower arm via [`lerobot`](https://github.com/huggingface/lerobot). It opens the cameras, reads the 6-DOF joint state, predicts an 8-step action chunk, denormalises it to degrees, and streams the joint targets back to the arm at the dataset fps (30 Hz on `k1000dai/so101_pick_candy_clean`). A `--max_relative_target` per-step clamp is on by default for safety.

**Extra deps on top of the training env above:**

```
pip install 'lerobot[feetech,intelrealsense]==0.4.4' pyrealsense2 opencv-python
```

`lerobot==0.4.4` requires `datasets>=4.0` and `numpy>=2`. If the strict training pins in §1 fight that, use a separate venv for inference, or pull the SO-101 motor + RealSense SDKs into your existing one — only the inference path needs the `lerobot` runtime.

**One-time setup:**

1. Persist the action / state normalisation bounds and fps next to the checkpoint (already shipped for the released checkpoint — re-run only if you retrain on a different dataset):
   ```bash
   ASYNCVLA_PLATFORM=so101 \
   python scripts/save_so101_norm_stats.py \
       --output openvla-7b+so101+b1+lr-5e-05+lora-r32--long-run--10000_chkpt/so101_norm_stats.json
   ```
2. Calibrate the SO-101 follower once (lerobot caches the result in `~/.cache/huggingface/lerobot/robots/so_follower/<robot_id>.json`):
   ```bash
   python -m lerobot.calibrate \
       --robot.type=so101_follower --robot.id=so101_follower \
       --robot.port=/dev/follower_arm
   ```
   After that you can pass `--no_calibrate` to skip the interactive prompt on subsequent runs.

**Run on the local hardware** (`/dev/follower_arm` + the wrist USB cam + the top RealSense at serial `135122073127`):

```bash
ASYNCVLA_PLATFORM=so101 PYTHONPATH=$PWD \
python inference/run_so101_robot.py \
    --checkpoint_dir openvla-7b+so101+b1+lr-5e-05+lora-r32--long-run--10000_chkpt \
    --step 10000 \
    --robot_port /dev/follower_arm \
    --robot_id so101_follower \
    --wrist_path /dev/v4l/by-id/usb-HD_USB_Camera_HD_USB_Camera_2020042508-video-index0 \
    --wrist_width 640 --wrist_height 480 \
    --top_serial 135122073127 \
    --top_width 848 --top_height 480 \
    --instruction "pick up candy and put it in the bowl" \
    --exec_horizon 8 \
    --max_relative_target 15 \
    --max_steps 600 \
    --no_calibrate
```

Useful flags:

- `--dry_run` runs the full perception + planning loop but never calls `robot.send_action`, so you can validate the predicted chunks against the live state before letting the arm move.
- `--exec_horizon N` (1–8) executes the first `N` predicted steps before re-planning. Smaller `N` = more reactive (re-plans more often), larger `N` = smoother (open-loop chunk).
- `--max_relative_target DEG` clamps every commanded joint to at most `DEG` away from the current position; set to `0` to disable.
- `--main_view {top,wrist}` swaps which camera is the model's primary view (default: `top`, matching training).
- `Ctrl+C` once cleanly finishes the current step, disables torque, and disconnects the arm.

**Safety:** keep the laptop's e-stop / power cut within reach the first time you run unclamped, especially with `--exec_horizon 8`. The candy-pick checkpoint reaches ~0.027 normalised L2 on the demo distribution but has no guarantees outside it. Start with `--dry_run`, then a small `--exec_horizon` + low `--max_relative_target` before opening it up.

#### 7. Summary of fork changes

| File | Status | Purpose |
| --- | --- | --- |
| `prismatic/vla/constants.py` | modified | `ASYNCVLA_PLATFORM` env var selects between `omnivla` (4-D nav action) and `so101` (6-D arm action). Navigation defaults unchanged. |
| `prismatic/vla/datasets/so101_dataset.py` | **new** | `SO101_Dataset` wraps `LeRobotDataset` (v3.0 ready), normalises action/state via q01-q99 stats, builds the AsyncVLA prompt + action chunk, loads a real past frame for the Edge adapter via `past_offset`. Defaults to `video_backend="pyav"`. |
| `prismatic/util/data_utils.py` | modified | Adds `PaddedCollatorForActionPrediction_SO101` and forwards `c_image` / `p_image` to the batch so the Edge adapter receives them. |
| `prismatic/models/small_head.py` | modified | `Edge_adapter` and `Edge_adapter_v0` final layer now sizes by `NUM_ACTIONS_CHUNK * ACTION_DIM`. Import of `vint_train` is wrapped in a try/except that falls back to the vendored transformer below — navigation users with MBRA on `sys.path` get the original implementation unchanged. |
| `prismatic/models/transformer_decoder.py` | **new** | Vendored `MultiLayerDecoder_trans` (learned positional embedding + `nn.TransformerEncoder`) so the SO-101 pipeline is self-contained and does not require the MBRA checkout. |
| `config_nav/so101_config.yaml` | **new** | Dataset config (repo_id, camera keys, modality_id, default prompt, video backend). |
| `vla-scripts/train_so101_async.py` | **new** | **Full AsyncVLA two-stage finetune**: VLA + LoRA + `ProprioProjector` + `Proj_Actiontokens` + `Edge_adapter`. Joint-space delta + smoothness loss matching the navigation recipe. |
| `vla-scripts/train_so101.py` | **new** | Single-stage OpenVLA-OFT-style alternative (no Edge adapter). |
| `inference/run_so101_async.py` | **new** | End-to-end two-stage inference: LoRA-merged VLA -> action_proj -> Edge_adapter -> joint chunk. |
| `inference/run_so101.py` | **new** | Inference for the single-stage script above. |
| `inference/run_so101_robot.py` | **new** | Real-robot control loop using `lerobot` `SO101Follower` + OpenCV wrist cam + Intel RealSense top cam. Reuses `load_inference_modules` / `build_inputs` / `predict_action_chunk` from `run_so101.py`. |
| `scripts/save_so101_norm_stats.py` | **new** | One-shot script that dumps the dataset's q01/q99 action and state bounds (plus joint names and fps) to `<checkpoint>/so101_norm_stats.json` so robot inference doesn't need the dataset at runtime. |
| `README.md` | modified | This section. |

### Acknowledgement
We implement our ideas and design choices on top of the pretrained checkpoints. Our work builds upon the [OpenVLA-OFT](https://openvla-oft.github.io/), [OmniVLA](https://github.com/NHirose/OmniVLA) and [ViNT](https://github.com/robodhruv/visualnav-transformer) codebases, with additional code added to create AsyncVLA. As such, our implementation leverages many components of these codebases. We sincerely appreciate the effort and contributions of the OpenVLA-OFT, OmniVLA and ViNT team!

## Citing
```
@misc{hirose2026asyncvla,
      title={AsyncVLA: An Asynchronous VLA for Fast and Robust Navigation on the Edge}, 
      author={Noriaki Hirose and Catherine Glossop and Dhruv Shah and Sergey Levine},
      year={2026},
      eprint={2602.13476},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2602.13476}, 
}
