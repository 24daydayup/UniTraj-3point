from __future__ import annotations

import math
import random
from typing import Optional

import numpy as np
import pandas as pd
import torch
from rdp import rdp
from torch.utils.data import Dataset

MIN_POINTS = 36
MAX_POINTS = 600
MIN_SAMPLING_RATIO = 0.35


class Normalize:
    """Normalize relative longitude/latitude coordinates with fixed statistics."""

    def __init__(self, mean=None, std=None):
        self.mean = torch.tensor(
            mean if mean is not None else [3.0675081728314398e-06, -5.4481778948893874e-05],
            dtype=torch.float32,
        )
        self.std = torch.tensor(
            std if std is not None else [0.03790469261263497, 0.030818511463853598],
            dtype=torch.float32,
        )

    def __call__(self, trajectory: torch.Tensor) -> torch.Tensor:
        return (trajectory - self.mean) / self.std

    def inverse(self, trajectory: torch.Tensor) -> torch.Tensor:
        return trajectory * self.std + self.mean


def logarithmic_sampling_ratio(
    length: int,
    min_points: int = MIN_POINTS,
    max_points: int = MAX_POINTS,
    min_ratio: float = MIN_SAMPLING_RATIO,
) -> float:
    if length <= min_points:
        return 1.0
    if length >= max_points:
        return min_ratio

    ratio = 1.0 - math.log(length - min_points + 1) / math.log(max_points - min_points + 1)
    ratio = ratio * (1.0 - min_ratio)
    return max(1.0 - ratio, min_ratio)


def build_observed_trajectory(
    trajectory: torch.Tensor,
    loss_mask: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    loss_mask = loss_mask.float()
    attention_mask = attention_mask.float()
    observed_mask = attention_mask * (1.0 - loss_mask)
    observed_trajectory = trajectory * observed_mask.unsqueeze(-1)
    return observed_trajectory, observed_mask


def build_completion_mask(
    attention_mask: torch.Tensor,
    mask_ratio: float,
    rng: Optional[np.random.Generator] = None,
) -> torch.Tensor:
    rng = rng or np.random.default_rng()
    valid_idx = torch.nonzero(attention_mask > 0.5, as_tuple=False).flatten().cpu().numpy()
    loss_mask = torch.zeros_like(attention_mask, dtype=torch.float32)
    if len(valid_idx) == 0:
        return loss_mask

    if mask_ratio <= 0.0:
        return loss_mask

    num_masked = max(1, int(len(valid_idx) * mask_ratio))
    chosen = rng.choice(valid_idx, size=min(num_masked, len(valid_idx)), replace=False)
    loss_mask[torch.as_tensor(chosen, dtype=torch.long)] = 1.0
    return loss_mask


def build_prediction_mask(
    attention_mask: torch.Tensor,
    predict_len: int,
) -> torch.Tensor:
    valid_idx = torch.nonzero(attention_mask > 0.5, as_tuple=False).flatten()
    loss_mask = torch.zeros_like(attention_mask, dtype=torch.float32)
    if len(valid_idx) == 0:
        return loss_mask

    n_real = min(max(1, predict_len), len(valid_idx))
    loss_mask[valid_idx[-n_real:]] = 1.0
    return loss_mask


def build_batch_completion_mask(
    attention_mask: torch.Tensor,
    mask_ratio: float,
    seed: Optional[int] = None,
) -> torch.Tensor:
    batch_masks = []
    for i in range(attention_mask.shape[0]):
        rng = np.random.default_rng(None if seed is None else seed + i)
        batch_masks.append(build_completion_mask(attention_mask[i], mask_ratio, rng))
    return torch.stack(batch_masks, dim=0)


def build_batch_prediction_mask(
    attention_mask: torch.Tensor,
    predict_len: int,
) -> torch.Tensor:
    return torch.stack(
        [build_prediction_mask(attention_mask[i], predict_len) for i in range(attention_mask.shape[0])],
        dim=0,
    )


class TrajectoryDataset(Dataset):
    def __init__(
        self,
        data_path,
        max_len: int = 200,
        transform=None,
        mask_ratio: float = 0.5,
        task_mode: str = "mixed",
        completion_prob: float = 0.7,
        prediction_horizon: int = 8,
        deterministic: bool = False,
        seed: int = 42,
    ):
        self.data_path = data_path
        self.transform = transform
        self.max_len = max_len
        self.mask_ratio = mask_ratio
        self.task_mode = task_mode
        self.completion_prob = completion_prob
        self.prediction_horizon = prediction_horizon
        self.deterministic = deterministic
        self.seed = seed
        if self.task_mode not in {"completion", "prediction", "mixed"}:
            raise ValueError("task_mode must be one of {'completion', 'prediction', 'mixed'}")
        if not 0.0 <= self.mask_ratio <= 1.0:
            raise ValueError("mask_ratio must be between 0.0 and 1.0")
        if not 0.0 <= self.completion_prob <= 1.0:
            raise ValueError("completion_prob must be between 0.0 and 1.0")
        self.sampling_ratios = [
            logarithmic_sampling_ratio(length)
            for length in np.arange(MIN_POINTS, MAX_POINTS + 1, 1)
        ]

        try:
            with open(self.data_path, "rb") as f:
                self.data = pd.read_pickle(f)
        except Exception as exc:
            raise FileNotFoundError(f"File not found: {self.data_path}") from exc

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        np_rng = np.random.default_rng(self.seed + idx) if self.deterministic else np.random.default_rng()
        py_rng = random.Random(self.seed + idx) if self.deterministic else random

        traj_df = self.resample_trajectory(idx, np_rng=np_rng, py_rng=py_rng)
        trajectory = torch.tensor(traj_df[["longitude", "latitude"]].values, dtype=torch.float32)
        intervals = torch.tensor(traj_df["interval"].values, dtype=torch.float32)

        original = trajectory[0].clone()
        centered_trajectory = trajectory - original
        if self.transform:
            centered_trajectory = self.transform(centered_trajectory)

        trajectory, attention_mask = self.pad_or_truncate(centered_trajectory)
        intervals, _ = self.pad_or_truncate(intervals)

        loss_mask = self.build_training_mask(
            raw_trajectory=centered_trajectory,
            attention_mask=attention_mask,
            np_rng=np_rng,
        )
        observed_trajectory, observed_mask = build_observed_trajectory(
            trajectory=trajectory,
            loss_mask=loss_mask,
            attention_mask=attention_mask,
        )

        inputs = {
            "trajectory": trajectory.transpose(0, 1),
            "observed_trajectory": observed_trajectory.transpose(0, 1),
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
            "observed_mask": observed_mask,
            "original": original,
            "intervals": intervals,
            "sample_id": torch.tensor(idx, dtype=torch.long),
        }
        return inputs

    def build_training_mask(
        self,
        raw_trajectory: torch.Tensor,
        attention_mask: torch.Tensor,
        np_rng: np.random.Generator,
    ) -> torch.Tensor:
        task_mode = self.task_mode
        if task_mode == "mixed":
            task_mode = "completion" if np_rng.random() < self.completion_prob else "prediction"

        valid_len = int(attention_mask.sum().item())
        if valid_len <= 0:
            return torch.zeros_like(attention_mask, dtype=torch.float32)

        if task_mode == "prediction":
            return build_prediction_mask(attention_mask, self.prediction_horizon)

        if self.mask_ratio <= 0.0:
            return torch.zeros_like(attention_mask, dtype=torch.float32)

        completion_selector = np_rng.random()
        if completion_selector < 0.6:
            mask = self.apply_random_mask(valid_len, np_rng)
        elif completion_selector < 0.8:
            mask = self.apply_rdp_mask(raw_trajectory[:valid_len], np_rng)
        else:
            mask = self.apply_block_mask(valid_len, np_rng)

        padded_mask = torch.zeros_like(attention_mask, dtype=torch.float32)
        padded_mask[:valid_len] = torch.as_tensor(mask.astype(np.float32))
        return padded_mask

    def apply_random_mask(self, trajectory_length: int, np_rng: np.random.Generator) -> np.ndarray:
        num_points = max(1, int(trajectory_length * self.mask_ratio))
        mask = np.zeros(trajectory_length, dtype=bool)
        indices = np_rng.choice(trajectory_length, size=min(num_points, trajectory_length), replace=False)
        mask[indices] = True
        return mask

    def apply_block_mask(self, trajectory_length: int, np_rng: np.random.Generator) -> np.ndarray:
        mask = np.zeros(trajectory_length, dtype=bool)
        block_size = min(int(np_rng.integers(5, 15)), trajectory_length)
        num_points = max(1, int(trajectory_length * self.mask_ratio))
        start_idx = int(np_rng.integers(0, max(1, trajectory_length - block_size + 1)))
        mask[start_idx:start_idx + block_size] = True

        additional_mask_points = max(0, min(num_points, trajectory_length) - int(mask.sum()))
        if additional_mask_points > 0:
            non_block_indices = np.where(~mask)[0]
            additional_indices = np_rng.choice(
                non_block_indices,
                size=min(additional_mask_points, len(non_block_indices)),
                replace=False,
            )
            mask[additional_indices] = True
        return mask

    def apply_rdp_mask(
        self,
        trajectory: torch.Tensor,
        np_rng: np.random.Generator,
        epsilon: float = 1e-4,
    ) -> np.ndarray:
        trajectory_np = trajectory[: self.max_len].cpu().numpy()
        trajectory_length = len(trajectory_np)
        num_points = max(1, int(trajectory_length * self.mask_ratio))

        rdp_mask = np.array(rdp(trajectory_np, epsilon=epsilon, return_mask=True))
        if trajectory_length > 1:
            rdp_mask[0] = False
            rdp_mask[-1] = False

        num_rdp_mask = int(rdp_mask.sum())
        if num_rdp_mask > num_points:
            indices = np.where(rdp_mask)[0]
            chosen = np_rng.choice(indices, size=num_points, replace=False)
            rdp_mask[:] = False
            rdp_mask[chosen] = True
        elif num_rdp_mask < num_points:
            non_rdp_indices = np.where(~rdp_mask)[0]
            additional = num_points - num_rdp_mask
            if additional > 0 and len(non_rdp_indices) > 0:
                chosen = np_rng.choice(
                    non_rdp_indices,
                    size=min(additional, len(non_rdp_indices)),
                    replace=False,
                )
                rdp_mask[chosen] = True
        return rdp_mask

    def resample_trajectory(
        self,
        idx: int,
        np_rng: np.random.Generator,
        py_rng,
    ) -> pd.DataFrame:
        sample = self.data.iloc[idx]
        full_df = pd.DataFrame(
            {
                "time": sample["time"],
                "longitude": [point[1] for point in sample["trajectory"]],
                "latitude": [point[0] for point in sample["trajectory"]],
            }
        )
        trajectory_length = len(full_df)

        if py_rng.random() < 0.3 and trajectory_length >= 360:
            if trajectory_length > 540:
                sampling_interval = py_rng.randint(8, 15)
            elif trajectory_length > 360:
                sampling_interval = py_rng.randint(6, 10)
            else:
                sampling_interval = py_rng.randint(3, 6)

            full_df["time"] = pd.to_datetime(full_df["time"])
            full_df.set_index("time", inplace=True)
            resampled_df = full_df.resample(f"{sampling_interval}s").mean().reset_index()
            resampled_df = resampled_df.dropna(subset=["longitude", "latitude"]).reset_index(drop=True)
            resampled_df["interval"] = (
                resampled_df["time"].diff().dt.total_seconds().fillna(0).astype("float32")
            )
        else:
            sampling_ratio = (
                1.0
                if trajectory_length <= MIN_POINTS
                else (
                    MIN_SAMPLING_RATIO
                    if trajectory_length >= MAX_POINTS
                    else self.sampling_ratios[trajectory_length - MIN_POINTS]
                )
            )

            num_sampled_points = max(2, int(trajectory_length * sampling_ratio))
            sampled_indices = np_rng.choice(full_df.index, size=min(num_sampled_points, trajectory_length), replace=False)
            resampled_df = full_df.loc[sampled_indices].sort_index().reset_index(drop=True)
            resampled_df["time"] = pd.to_datetime(resampled_df["time"])
            resampled_df["interval"] = (
                resampled_df["time"].diff().dt.total_seconds().fillna(0).astype("float32")
            )

        if resampled_df.empty:
            resampled_df = full_df.iloc[[0]].copy()
            resampled_df["time"] = pd.to_datetime(resampled_df["time"])
            resampled_df["interval"] = 0.0

        return resampled_df

    def pad_or_truncate(self, tensor: torch.Tensor):
        seq_len = len(tensor)
        if seq_len > self.max_len:
            tensor = tensor[: self.max_len]
            attention_mask = torch.ones(self.max_len, dtype=torch.float32)
            return tensor, attention_mask

        if tensor.dim() == 2:
            padded_tensor = torch.zeros((self.max_len, tensor.shape[-1]), dtype=tensor.dtype)
            attention_mask = torch.zeros(self.max_len, dtype=torch.float32)
            attention_mask[:seq_len] = 1.0
        else:
            padded_tensor = torch.zeros(self.max_len, dtype=tensor.dtype)
            attention_mask = None

        padded_tensor[:seq_len] = tensor
        return padded_tensor, attention_mask
