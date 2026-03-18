from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.dataset import TrajectoryDataset  # noqa: E402


def calculate_stats(dataloader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute dataset-wide mean and std over valid trajectory points."""

    total_sum = torch.zeros(2, dtype=torch.float64)
    total_sq_sum = torch.zeros(2, dtype=torch.float64)
    total_count = 0

    print("Starting calculation...")
    for batch in tqdm(dataloader, total=len(dataloader)):
        trajectory = batch["trajectory"].permute(0, 2, 1)
        attention_mask = batch["attention_mask"].reshape(-1) > 0.5
        valid_points = trajectory.reshape(-1, 2)[attention_mask]

        if valid_points.numel() == 0:
            continue

        valid_points = valid_points.double()
        total_sum += valid_points.sum(dim=0)
        total_sq_sum += (valid_points ** 2).sum(dim=0)
        total_count += valid_points.shape[0]

    if total_count == 0:
        raise ValueError("No valid points found in dataset.")

    mean = total_sum / total_count
    variance = (total_sq_sum / total_count) - mean ** 2
    std = torch.sqrt(torch.clamp(variance, min=0.0))
    return mean, std


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate normalization statistics for trajectories.")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--max_len", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_path)
    print(f"Loading dataset from {data_path}...")

    dataset = TrajectoryDataset(
        data_path=str(data_path),
        max_len=args.max_len,
        transform=None,
        mask_ratio=0.0,
        task_mode="completion",
        deterministic=True,
    )
    print(f"Dataset loaded. Size: {len(dataset)}")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    mean, std = calculate_stats(dataloader)

    print("\n" + "=" * 50)
    print("Calculation Complete")
    print("=" * 50)
    print(f"Mean (lon, lat): {mean.tolist()}")
    print(f"Std  (lon, lat): {std.tolist()}")
    print("-" * 50)
    print("Update utils/dataset.py Normalize class with:")
    print(f"self.mean = torch.tensor({mean.tolist()}, dtype=torch.float32)")
    print(f"self.std = torch.tensor({std.tolist()}, dtype=torch.float32)")
    print("=" * 50)


if __name__ == "__main__":
    main()
