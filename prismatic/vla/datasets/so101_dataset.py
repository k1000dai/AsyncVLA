"""so101_dataset.py

PyTorch dataset that wraps a LeRobot HuggingFace dataset (e.g.
``k1000dai/so101_pick_candy_clean``) so it can be fed into the AsyncVLA /
OmniVLA training loop with as few code changes as possible.

The original training pipeline targets 2-D navigation; here we reuse the same
batch contract (``input_ids``, ``labels``, ``pixel_values``,
``pixel_values_goal``, ``actions``, ``proprio``, ``modality_id``) but with
6-DOF arm actions and joint-state proprio instead of (x, y, cosθ, sinθ).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Type

import math

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import (
    ACTION_DIM,
    IGNORE_INDEX,
    NUM_ACTIONS_CHUNK,
    POSE_DIM,
)


def _load_lerobot_dataset(
    repo_id: str,
    root: Optional[str],
    episodes: Optional[List[int]],
    video_backend: Optional[str] = "pyav",
):
    """Import LeRobot lazily so the rest of the package works without it."""
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # type: ignore
    except ImportError:  # pragma: no cover - depends on lerobot version
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore

    kwargs: Dict[str, Any] = {"repo_id": repo_id}
    if root is not None:
        kwargs["root"] = root
    if episodes is not None:
        kwargs["episodes"] = list(episodes)
    if video_backend is not None:
        # LeRobot 0.4+ defaults to torchcodec which needs system FFmpeg; pyav
        # ships via the `av` Python wheel and avoids that requirement.
        try:
            return LeRobotDataset(video_backend=video_backend, **kwargs)
        except TypeError:
            pass  # older LeRobotDataset doesn't accept video_backend
    return LeRobotDataset(**kwargs)


def _to_chw_tensor(image: Any) -> torch.Tensor:
    """Convert a LeRobot image sample to a float32 CHW tensor in [0, 1]."""
    if isinstance(image, torch.Tensor):
        tensor = image.float()
        if tensor.ndim == 4:
            tensor = tensor[0]
        if tensor.shape[0] not in (1, 3) and tensor.shape[-1] in (1, 3):
            tensor = tensor.permute(2, 0, 1)
        if tensor.max() > 1.5:
            tensor = tensor / 255.0
        return tensor.clamp(0.0, 1.0)
    if isinstance(image, np.ndarray):
        return _to_chw_tensor(torch.from_numpy(image))
    if isinstance(image, Image.Image):
        return TF.to_tensor(image.convert("RGB"))
    raise TypeError(f"Unsupported image type from LeRobot dataset: {type(image)!r}")


def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    array = (tensor.clamp(0.0, 1.0) * 255.0).round().byte().cpu().numpy()
    array = np.transpose(array, (1, 2, 0))
    return Image.fromarray(array)


def _normalize_to_bounds(values: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    span = (high - low).clamp(min=1e-6)
    return ((values - low) / span * 2.0 - 1.0).clamp(-1.0, 1.0)


def _stats_to_bounds(stats: Dict[str, Any], key: str, dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    feature_stats = stats.get(key) if stats is not None else None
    if not feature_stats:
        return (torch.full((dim,), -1.0), torch.full((dim,), 1.0))

    def _resolve(name: str, fallback: float) -> torch.Tensor:
        value = feature_stats.get(name)
        if value is None:
            return torch.full((dim,), fallback)
        tensor = torch.as_tensor(np.asarray(value), dtype=torch.float32).reshape(-1)
        if tensor.numel() == 1:
            tensor = tensor.expand(dim).clone()
        return tensor[:dim]

    low = _resolve("q01", float("nan"))
    high = _resolve("q99", float("nan"))
    if torch.isnan(low).any() or torch.isnan(high).any():
        low = _resolve("min", -1.0)
        high = _resolve("max", 1.0)
    # Guard against degenerate ranges (mostly the gripper-open joint when a
    # demo never touched it).
    high = torch.where(high - low < 1e-3, low + 1.0, high)
    return low, high


class SO101_Dataset(Dataset):
    """LeRobot-backed dataset that emits AsyncVLA training samples."""

    def __init__(
        self,
        repo_id: str,
        action_tokenizer: ActionTokenizer,
        base_tokenizer: PreTrainedTokenizerBase,
        image_transform: ImageTransform,
        prompt_builder_fn: Type[PromptBuilder],
        root: Optional[str] = None,
        episodes: Optional[List[int]] = None,
        main_camera: Optional[str] = None,
        secondary_camera: Optional[str] = None,
        state_key: str = "observation.state",
        action_key: str = "action",
        task_key: str = "task",
        action_chunk_size: int = NUM_ACTIONS_CHUNK,
        image_size: Tuple[int, int] = (96, 96),
        modality_id: int = 5,
        default_prompt: str = "Perform the demonstrated manipulation task.",
        predict_stop_token: bool = True,
        video_backend: Optional[str] = "pyav",
    ) -> None:
        super().__init__()
        if action_chunk_size != NUM_ACTIONS_CHUNK:
            raise ValueError(
                f"action_chunk_size={action_chunk_size} must match prismatic.vla.constants.NUM_ACTIONS_CHUNK"
                f"={NUM_ACTIONS_CHUNK}; export ASYNCVLA_PLATFORM=so101 or adjust the constant."
            )

        self.dataset = _load_lerobot_dataset(repo_id, root, episodes, video_backend=video_backend)
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn
        self.action_chunk_size = action_chunk_size
        self.image_size = tuple(image_size)
        self.modality_id = int(modality_id)
        self.default_prompt = default_prompt
        self.predict_stop_token = predict_stop_token
        self.state_key = state_key
        self.action_key = action_key
        self.task_key = task_key

        camera_keys = self._discover_cameras()
        self.main_camera = main_camera or camera_keys[0]
        if self.main_camera not in camera_keys:
            raise KeyError(
                f"main_camera={self.main_camera!r} not found in dataset cameras {camera_keys}"
            )
        if secondary_camera is None:
            self.secondary_camera = camera_keys[1] if len(camera_keys) > 1 else self.main_camera
        else:
            if secondary_camera not in camera_keys:
                raise KeyError(
                    f"secondary_camera={secondary_camera!r} not found in dataset cameras {camera_keys}"
                )
            self.secondary_camera = secondary_camera

        stats = getattr(getattr(self.dataset, "meta", None), "stats", None)
        self.action_low, self.action_high = _stats_to_bounds(stats, self.action_key, ACTION_DIM)
        self.state_low, self.state_high = _stats_to_bounds(stats, self.state_key, POSE_DIM)

        self._episode_bounds = self._build_episode_bounds()

    # ------------------------------------------------------------------ helpers
    def _discover_cameras(self) -> List[str]:
        features = getattr(getattr(self.dataset, "meta", None), "features", None)
        if features:
            cams = [k for k in features if k.startswith("observation.images.")]
            if cams:
                return cams
        # Fallback: peek at the first sample.
        sample = self.dataset[0]
        return [k for k in sample if isinstance(k, str) and k.startswith("observation.images.")]

    def _build_episode_bounds(self) -> List[Tuple[int, int]]:
        meta = getattr(self.dataset, "meta", None)
        ep_data_index = getattr(meta, "episode_data_index", None) if meta is not None else None
        if ep_data_index is None:
            ep_data_index = getattr(self.dataset, "episode_data_index", None)
        if ep_data_index is None:
            return [(0, len(self.dataset))]
        starts = np.asarray(ep_data_index["from"]).reshape(-1).tolist()
        ends = np.asarray(ep_data_index["to"]).reshape(-1).tolist()
        return list(zip(starts, ends))

    def _episode_end(self, index: int) -> int:
        for start, end in self._episode_bounds:
            if start <= index < end:
                return end
        return len(self.dataset)

    def _resize_norm(self, image: torch.Tensor) -> torch.Tensor:
        return TF.resize(image, list(self.image_size))

    def _action_chunk(self, index: int) -> np.ndarray:
        end = self._episode_end(index)
        chunk = []
        for offset in range(self.action_chunk_size):
            idx = min(index + offset, end - 1)
            sample = self.dataset[idx]
            chunk.append(np.asarray(sample[self.action_key], dtype=np.float32).reshape(-1)[:ACTION_DIM])
        return np.stack(chunk, axis=0)

    # ------------------------------------------------------------------- Dataset
    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.dataset[index]

        main_image = _to_chw_tensor(sample[self.main_camera])
        if self.secondary_camera == self.main_camera:
            secondary_image = main_image.clone()
        else:
            secondary_image = _to_chw_tensor(sample[self.secondary_camera])

        current_image_pil = _tensor_to_pil(main_image)
        secondary_image_pil = _tensor_to_pil(secondary_image)

        pixel_values_current = self.image_transform(current_image_pil)
        pixel_values_secondary = self.image_transform(secondary_image_pil)

        state = np.asarray(sample[self.state_key], dtype=np.float32).reshape(-1)[:POSE_DIM]
        state_tensor = torch.from_numpy(state)
        proprio = _normalize_to_bounds(state_tensor, self.state_low, self.state_high)

        action_chunk = self._action_chunk(index)
        action_tensor = torch.from_numpy(action_chunk)
        action_norm = _normalize_to_bounds(action_tensor, self.action_low, self.action_high)
        # The action tokenizer clips to [-1, 1] and we feed normalized actions
        # into both the tokenizer and the regression loss.
        action_norm_np = action_norm.numpy().astype(np.float32)

        future_actions = action_norm_np[1:]
        current_action = action_norm_np[0]
        current_action_string = self.action_tokenizer(current_action)
        future_actions_string = "".join(self.action_tokenizer(future_actions))
        action_chunk_string = current_action_string + future_actions_string
        action_chunk_len = len(action_chunk_string)

        prompt_text = sample.get(self.task_key) if isinstance(sample, dict) else None
        if isinstance(prompt_text, (list, tuple)):
            prompt_text = prompt_text[0]
        if not isinstance(prompt_text, str) or not prompt_text.strip():
            prompt_text = self.default_prompt
        prompt_text = prompt_text.strip()

        conversation = [
            {"from": "human", "value": f"What action should the robot take to {prompt_text}"},
            {"from": "gpt", "value": action_chunk_string},
        ]
        prompt_builder = self.prompt_builder_fn("openvla")
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        input_ids = torch.as_tensor(
            self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        )
        labels = input_ids.clone()
        labels[: -(action_chunk_len + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX

        # MBRA-style image buffers are not used in manipulation training but the
        # collator still expects the keys.
        small_image = self._resize_norm(main_image)
        small_secondary = self._resize_norm(secondary_image)

        action_select_mask = torch.tensor(1.0)
        obj_pose_norm = proprio[:2].clone()

        return dict(
            pixel_values=pixel_values_current,
            pixel_values_goal=pixel_values_secondary,
            input_ids=input_ids,
            labels=labels,
            dataset_name="so101",
            modality_id=self.modality_id,
            actions=action_norm_np,
            action_select_mask=action_select_mask,
            proprio=proprio,
            goal_pose=proprio,
            obj_pose_norm=obj_pose_norm,
            p_image=small_secondary,
            c_image=small_image,
            cur_image=small_image,
            goal_image_8=small_secondary,
            temp_dist=torch.tensor(0.0),
            img_PIL=current_image_pil,
            gimg_PIL=secondary_image_pil,
            lan_prompt=prompt_text,
        )
