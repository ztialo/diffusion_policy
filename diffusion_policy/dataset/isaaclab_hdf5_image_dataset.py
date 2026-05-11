from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict

import h5py
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from diffusion_policy.common.normalize_util import get_image_range_normalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer


@dataclass(frozen=True)
class EpisodeSpan:
    rows: np.ndarray

    @property
    def length(self) -> int:
        return int(self.rows.shape[0])


def _read_h5_rows(dataset, rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64)
    start = int(rows.min())
    end = int(rows.max()) + 1
    block = np.asarray(dataset[start:end])
    return block[rows - start]


def _rotation_vector_to_matrix(rotvec: torch.Tensor) -> torch.Tensor:
    angle = torch.linalg.norm(rotvec, dim=-1, keepdim=True)
    axis = rotvec / torch.clamp(angle, min=1.0e-8)
    x, y, z = axis.unbind(dim=-1)
    zeros = torch.zeros_like(x)
    k = torch.stack(
        (
            zeros,
            -z,
            y,
            z,
            zeros,
            -x,
            -y,
            x,
            zeros,
        ),
        dim=-1,
    ).reshape(*rotvec.shape[:-1], 3, 3)
    ident = torch.eye(3, device=rotvec.device, dtype=rotvec.dtype).expand_as(k)
    sin_term = torch.sin(angle)[..., None] * k
    cos_term = (1.0 - torch.cos(angle))[..., None] * torch.matmul(k, k)
    rot = ident + sin_term + cos_term
    small_angle = angle.squeeze(-1) < 1.0e-8
    if torch.any(small_angle):
        rot = torch.where(small_angle[..., None, None], ident, rot)
    return rot


def _matrix_to_rot6d(rotmat: torch.Tensor) -> torch.Tensor:
    return rotmat[..., :, :2].reshape(*rotmat.shape[:-2], 6)


def _convert_action_repr_numpy(action: np.ndarray, rotation_rep: str) -> np.ndarray:
    if rotation_rep == "raw":
        return action.astype(np.float32)
    action_tensor = torch.from_numpy(action.astype(np.float32))
    dpos = action_tensor[..., :3]
    drot = action_tensor[..., 3:]
    rotmat = _rotation_vector_to_matrix(drot)
    if rotation_rep == "rot6d":
        rot_repr = _matrix_to_rot6d(rotmat)
    elif rotation_rep == "rot9d":
        rot_repr = rotmat.reshape(*rotmat.shape[:-2], 9)
    else:
        raise ValueError(f"Unsupported rotation_rep: {rotation_rep}")
    return torch.cat((dpos, rot_repr), dim=-1).cpu().numpy()


def _center_crop_numpy(images: np.ndarray, crop_size: int | None) -> np.ndarray:
    if crop_size is None:
        return images
    if crop_size <= 0:
        raise ValueError(f"image_crop_size must be positive when set, got {crop_size}")
    height, width = images.shape[1:3]
    if crop_size > height or crop_size > width:
        raise ValueError(
            f"image_crop_size={crop_size} exceeds image dimensions {(height, width)}."
        )
    top = (height - crop_size) // 2
    left = (width - crop_size) // 2
    return images[:, top : top + crop_size, left : left + crop_size, :]


def build_state_from_h5(h5_file: h5py.File, rows: np.ndarray, use_ft: bool = False) -> np.ndarray:
    eef_pos = np.asarray(_read_h5_rows(h5_file["eef_pos"], rows), dtype=np.float32)
    eef_quat = np.asarray(_read_h5_rows(h5_file["eef_quat"], rows), dtype=np.float32)
    gripper_pos = np.asarray(_read_h5_rows(h5_file["gripper_pos"], rows), dtype=np.float32)
    state_parts = [eef_pos, eef_quat, gripper_pos]
    if use_ft:
        if "left_ft_wrench" not in h5_file or "right_ft_wrench" not in h5_file:
            raise KeyError("Dataset is missing required FT keys: left_ft_wrench and/or right_ft_wrench")
        left_ft_wrench = np.asarray(_read_h5_rows(h5_file["left_ft_wrench"], rows), dtype=np.float32)
        right_ft_wrench = np.asarray(_read_h5_rows(h5_file["right_ft_wrench"], rows), dtype=np.float32)
        state_parts.extend((left_ft_wrench, right_ft_wrench))
    return np.concatenate(state_parts, axis=-1)


def load_episode_spans(h5_path: str) -> list[EpisodeSpan]:
    with h5py.File(h5_path, "r") as h5_file:
        if "done" not in h5_file:
            raise KeyError("Dataset is missing required key: done")
        done = np.asarray(h5_file["done"], dtype=np.bool_)
        spans: list[EpisodeSpan] = []
        if "env_id" in h5_file:
            env_ids = np.asarray(h5_file["env_id"], dtype=np.int64)
            for env_id in np.unique(env_ids):
                env_rows = np.nonzero(env_ids == env_id)[0]
                env_done = done[env_rows]
                start_idx = 0
                for i, is_done in enumerate(env_done):
                    if is_done:
                        rows = env_rows[start_idx : i + 1].astype(np.int64, copy=True)
                        spans.append(EpisodeSpan(rows=rows))
                        start_idx = i + 1
        else:
            start = 0
            for idx, is_done in enumerate(done):
                if is_done:
                    rows = np.arange(start, idx + 1, dtype=np.int64)
                    spans.append(EpisodeSpan(rows=rows))
                    start = idx + 1
    if not spans:
        raise ValueError("Dataset contains no complete episodes.")
    return spans


def split_episodes(
    spans: list[EpisodeSpan],
    val_ratio: float,
    seed: int,
    max_train_episodes: int | None,
    max_val_episodes: int | None,
) -> tuple[list[EpisodeSpan], list[EpisodeSpan]]:
    episode_ids = np.arange(len(spans))
    rng = np.random.default_rng(seed)
    rng.shuffle(episode_ids)

    if len(episode_ids) == 1 or val_ratio <= 0.0:
        train_ids = episode_ids
        val_ids = np.array([], dtype=np.int64)
    else:
        num_val = max(1, int(round(len(episode_ids) * val_ratio)))
        num_val = min(num_val, len(episode_ids) - 1)
        val_ids = episode_ids[:num_val]
        train_ids = episode_ids[num_val:]

    if max_train_episodes:
        train_ids = train_ids[: int(max_train_episodes)]
    if max_val_episodes:
        val_ids = val_ids[: int(max_val_episodes)]

    train_spans = [spans[int(idx)] for idx in train_ids]
    val_spans = [spans[int(idx)] for idx in val_ids]
    return train_spans, val_spans


class IsaacLabHdf5ImageDataset(BaseImageDataset):
    def __init__(
        self,
        shape_meta: dict,
        dataset_path: str,
        horizon: int = 1,
        pad_before: int = 0,
        pad_after: int = 0,
        n_obs_steps: int | None = None,
        val_ratio: float = 0.0,
        seed: int = 42,
        max_train_episodes: int | None = None,
        max_val_episodes: int | None = None,
        image_crop_size: int | None = None,
        use_ft: bool = False,
        rotation_rep: str = "raw",
        train: bool = True,
        train_spans: list[EpisodeSpan] | None = None,
        val_spans: list[EpisodeSpan] | None = None,
        normalizer: LinearNormalizer | None = None,
    ):
        self.shape_meta = shape_meta
        self.dataset_path = dataset_path
        self.horizon = int(horizon)
        self.pad_before = int(pad_before)
        self.pad_after = int(pad_after)
        self.n_obs_steps = int(n_obs_steps) if n_obs_steps is not None else self.horizon
        self.image_crop_size = image_crop_size
        self.use_ft = bool(use_ft)
        self.rotation_rep = rotation_rep
        self.train = bool(train)
        self._file = None

        spans = load_episode_spans(dataset_path)
        if train_spans is None or val_spans is None:
            train_spans, val_spans = split_episodes(
                spans=spans,
                val_ratio=float(val_ratio),
                seed=int(seed),
                max_train_episodes=max_train_episodes,
                max_val_episodes=max_val_episodes,
            )
        self.train_spans = train_spans
        self.val_spans = val_spans
        self.spans = self.train_spans if self.train else self.val_spans

        self.rgb_keys = [key for key, attr in shape_meta["obs"].items() if attr.get("type", "low_dim") == "rgb"]
        self.lowdim_keys = [key for key, attr in shape_meta["obs"].items() if attr.get("type", "low_dim") == "low_dim"]
        if self.rgb_keys != ["wrist"]:
            raise ValueError(f"Expected shape_meta rgb obs keys to be ['wrist'], got {self.rgb_keys}")
        if self.lowdim_keys != ["state"]:
            raise ValueError(f"Expected shape_meta low_dim obs keys to be ['state'], got {self.lowdim_keys}")

        self.samples: list[tuple[int, int]] = []
        for episode_idx, span in enumerate(self.spans):
            for step_idx in range(span.length):
                self.samples.append((episode_idx, step_idx))

        self.normalizer = normalizer if normalizer is not None else self._build_normalizer()

    def _ensure_open(self):
        if self._file is None:
            self._file = h5py.File(self.dataset_path, "r")

    def _build_normalizer(self) -> LinearNormalizer:
        with h5py.File(self.dataset_path, "r") as h5_file:
            state_chunks = []
            action_chunks = []
            for span in self.train_spans:
                rows = span.rows
                state_chunks.append(build_state_from_h5(h5_file, rows, use_ft=self.use_ft))
                raw_action = np.asarray(_read_h5_rows(h5_file["action"], rows), dtype=np.float32)
                action_chunks.append(_convert_action_repr_numpy(raw_action, self.rotation_rep))

        if not state_chunks or not action_chunks:
            raise ValueError("Training split is empty.")

        state = np.concatenate(state_chunks, axis=0)
        action = np.concatenate(action_chunks, axis=0)

        normalizer = LinearNormalizer()
        normalizer["state"] = SingleFieldLinearNormalizer.create_fit(state, mode="limits")
        normalizer["action"] = SingleFieldLinearNormalizer.create_fit(action, mode="limits")
        normalizer["wrist"] = get_image_range_normalizer()
        return normalizer

    def get_validation_dataset(self) -> "IsaacLabHdf5ImageDataset":
        return IsaacLabHdf5ImageDataset(
            shape_meta=self.shape_meta,
            dataset_path=self.dataset_path,
            horizon=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            n_obs_steps=self.n_obs_steps,
            image_crop_size=self.image_crop_size,
            use_ft=self.use_ft,
            rotation_rep=self.rotation_rep,
            train=False,
            train_spans=self.train_spans,
            val_spans=self.val_spans,
            normalizer=copy.deepcopy(self.normalizer),
        )

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        del kwargs
        return self.normalizer

    def get_all_actions(self) -> torch.Tensor:
        with h5py.File(self.dataset_path, "r") as h5_file:
            chunks = []
            for span in self.train_spans:
                rows = span.rows
                raw_action = np.asarray(_read_h5_rows(h5_file["action"], rows), dtype=np.float32)
                chunks.append(_convert_action_repr_numpy(raw_action, self.rotation_rep))
        return torch.from_numpy(np.concatenate(chunks, axis=0))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        self._ensure_open()
        episode_idx, step_idx = self.samples[idx]
        span = self.spans[episode_idx]
        episode_rows = span.rows

        obs_offsets = np.arange(step_idx - self.n_obs_steps + 1, step_idx + 1)
        obs_offsets = np.clip(obs_offsets, 0, span.length - 1)
        obs_rows = episode_rows[obs_offsets]

        action_offsets = np.arange(step_idx, step_idx + self.horizon)
        action_offsets = np.clip(action_offsets, 0, span.length - 1)
        action_rows = episode_rows[action_offsets]

        wrist = np.asarray(_read_h5_rows(self._file["wrist_rgb"], obs_rows), dtype=np.float32)
        wrist = _center_crop_numpy(wrist, self.image_crop_size) / 255.0
        wrist = np.moveaxis(wrist, -1, 1).astype(np.float32)

        state = build_state_from_h5(self._file, obs_rows, use_ft=self.use_ft).astype(np.float32)
        raw_action = np.asarray(_read_h5_rows(self._file["action"], action_rows), dtype=np.float32)
        action = _convert_action_repr_numpy(raw_action, self.rotation_rep).astype(np.float32)

        return {
            "obs": {
                "wrist": torch.from_numpy(wrist),
                "state": torch.from_numpy(state),
            },
            "action": torch.from_numpy(action),
        }
