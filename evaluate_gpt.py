#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import datetime
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from geopy.distance import geodesic
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.dataset import (
    Normalize,
    TrajectoryDataset,
    build_batch_completion_mask,
    build_batch_prediction_mask,
    build_observed_trajectory,
)
from utils.flow_matching import sample_trajectory_flow
from utils.project import (
    PATCH_STAT_KEYS,
    build_flow_model_from_config,
    find_config_snapshot,
    load_config_snapshot,
    load_default_config,
    resolve_device,
    set_random_seed,
)


def denormalise(
    pred: torch.Tensor,
    gt: torch.Tensor,
    origin: torch.Tensor,
    normaliser: Normalize,
) -> Tuple[np.ndarray, np.ndarray]:
    pred_np = pred.permute(0, 2, 1).detach().cpu().numpy()
    gt_np = gt.permute(0, 2, 1).detach().cpu().numpy()

    mean = normaliser.mean.numpy()
    std = normaliser.std.numpy()

    pred_np = pred_np * std + mean
    gt_np = gt_np * std + mean

    origin_np = origin.detach().cpu().numpy()
    pred_np += origin_np[:, None, :]
    gt_np += origin_np[:, None, :]
    return pred_np, gt_np


def geodesic_mae_rmse(real: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> Tuple[float, float]:
    distances = []
    distances_sq = []
    for r, p, m in zip(real, pred, mask):
        if not m:
            continue
        distance = geodesic((r[1], r[0]), (p[1], p[0])).m
        distances.append(distance)
        distances_sq.append(distance * distance)

    if not distances:
        return math.nan, math.nan
    return float(np.mean(distances)), float(np.sqrt(np.mean(distances_sq)))


def load_pretrained_flow_model(
    model_path: str,
    device: torch.device,
    config,
):
    model = build_flow_model_from_config(config)
    state = torch.load(model_path, map_location="cpu")
    state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    incompatible = model.load_state_dict(state, strict=False)
    print(
        "Loaded checkpoint with "
        f"{len(incompatible.missing_keys)} missing keys and "
        f"{len(incompatible.unexpected_keys)} unexpected keys"
    )
    model.to(device)
    model.eval()
    return model


def resolve_evaluation_config(args) -> Tuple[object, Path | None]:
    config = load_default_config()
    snapshot_path = find_config_snapshot(args.model_path)
    if snapshot_path is not None:
        snapshot_config = load_config_snapshot(snapshot_path)
        config.data = snapshot_config.data
        config.model = snapshot_config.model
        config.training = snapshot_config.training

    if args.max_len is not None:
        config.data.traj_length = args.max_len
    if args.hidden_size is not None:
        config.model.hidden_size = args.hidden_size
    if args.depth is not None:
        config.model.depth = args.depth
    if args.num_heads is not None:
        config.model.num_heads = args.num_heads
    if args.mlp_ratio is not None:
        config.model.mlp_ratio = args.mlp_ratio
    if args.use_moe is not None:
        config.model.use_moe = args.use_moe
    if args.num_experts is not None:
        config.model.num_experts = args.num_experts
    if args.top_k is not None:
        config.model.top_k = args.top_k

    return config, snapshot_path


def summarize_metrics(values: List[float]) -> Dict[str, float]:
    return pd.Series(values).describe().to_dict()


def summarize_patch_stats(values: Dict[str, List[float]]) -> Dict[str, float]:
    return {
        key: float(np.mean(metric_values))
        for key, metric_values in values.items()
        if metric_values
    }


@torch.no_grad()
def evaluate_reconstruction_tasks(args):
    config, snapshot_path = resolve_evaluation_config(args)
    set_random_seed(args.seed)
    device = resolve_device(gpu=args.gpu)
    sample_steps = args.sample_steps if args.sample_steps is not None else config.model.sample_steps
    print(f"Running on {device}")
    if snapshot_path is not None:
        print(f"Loaded config snapshot from: {snapshot_path}")
    else:
        print("Config snapshot not found, falling back to defaults / CLI overrides")

    normaliser = Normalize()
    dataset = TrajectoryDataset(
        data_path=args.test_data,
        max_len=config.data.traj_length,
        transform=normaliser,
        mask_ratio=0.0,
        task_mode="completion",
        deterministic=True,
        seed=args.seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = load_pretrained_flow_model(
        model_path=args.model_path,
        device=device,
        config=config,
    )

    do_completion = "completion" in args.task
    do_prediction = "prediction" in args.task

    completion_mae, completion_rmse = [], []
    prediction_mae, prediction_rmse = [], []
    patch_stats = {key: [] for key in PATCH_STAT_KEYS}

    for batch_idx, batch in enumerate(tqdm(loader, desc="Evaluating")):
        traj = batch["trajectory"].to(device)
        intervals = batch["intervals"].to(device)
        att_mask = batch["attention_mask"].to(device)
        origin = batch["original"].to(device)
        target = traj.transpose(1, 2)

        _, gt_dn = denormalise(traj, traj, origin, normaliser)

        if do_completion:
            completion_mask = build_batch_completion_mask(
                attention_mask=att_mask,
                mask_ratio=args.mask_ratio,
                seed=args.seed + batch_idx * args.batch_size,
            ).to(device)
            completion_observed, completion_observed_mask = build_observed_trajectory(
                trajectory=target,
                loss_mask=completion_mask,
                attention_mask=att_mask,
            )
            completion_pred = sample_trajectory_flow(
                model=model,
                observed_trajectory=completion_observed,
                intervals=intervals,
                attention_mask=att_mask,
                observed_mask=completion_observed_mask,
                steps=sample_steps,
            )
            if getattr(model, "_last_patch_stats", None):
                for key in PATCH_STAT_KEYS:
                    patch_stats[key].append(float(model._last_patch_stats[key]))
            completion_pred_dn, _ = denormalise(
                completion_pred.transpose(1, 2),
                traj,
                origin,
                normaliser,
            )
            completion_mask_np = completion_mask.cpu().numpy().astype(bool)

            for b in range(traj.shape[0]):
                mae, rmse = geodesic_mae_rmse(gt_dn[b], completion_pred_dn[b], completion_mask_np[b])
                if not math.isnan(mae):
                    completion_mae.append(mae)
                    completion_rmse.append(rmse)

        if do_prediction:
            prediction_mask = build_batch_prediction_mask(
                attention_mask=att_mask,
                predict_len=args.predict_len,
            ).to(device)
            prediction_observed, prediction_observed_mask = build_observed_trajectory(
                trajectory=target,
                loss_mask=prediction_mask,
                attention_mask=att_mask,
            )
            prediction_pred = sample_trajectory_flow(
                model=model,
                observed_trajectory=prediction_observed,
                intervals=intervals,
                attention_mask=att_mask,
                observed_mask=prediction_observed_mask,
                steps=sample_steps,
            )
            if getattr(model, "_last_patch_stats", None):
                for key in PATCH_STAT_KEYS:
                    patch_stats[key].append(float(model._last_patch_stats[key]))
            prediction_pred_dn, _ = denormalise(
                prediction_pred.transpose(1, 2),
                traj,
                origin,
                normaliser,
            )
            prediction_mask_np = prediction_mask.cpu().numpy().astype(bool)

            for b in range(traj.shape[0]):
                mae, rmse = geodesic_mae_rmse(gt_dn[b], prediction_pred_dn[b], prediction_mask_np[b])
                if not math.isnan(mae):
                    prediction_mae.append(mae)
                    prediction_rmse.append(rmse)

    results = {}
    if do_completion and completion_mae:
        print(f"\n=== Completion ({args.mask_ratio * 100:.0f}% masked real points) ===")
        print("MAE stats (m):\n", pd.Series(completion_mae).describe())
        print("\nRMSE stats (m):\n", pd.Series(completion_rmse).describe())
        results["completion"] = {
            "MAE": summarize_metrics(completion_mae),
            "RMSE": summarize_metrics(completion_rmse),
        }

    if do_prediction and prediction_mae:
        print(f"\n=== Prediction (last {args.predict_len} real points) ===")
        print("MAE stats (m):\n", pd.Series(prediction_mae).describe())
        print("\nRMSE stats (m):\n", pd.Series(prediction_rmse).describe())
        results["prediction"] = {
            "MAE": summarize_metrics(prediction_mae),
            "RMSE": summarize_metrics(prediction_rmse),
        }

    summarized_patch_stats = summarize_patch_stats(patch_stats)
    if summarized_patch_stats:
        print("\n=== Adaptive Patch Stats ===")
        for key, value in summarized_patch_stats.items():
            print(f"{key}: {value:.4f}")
        results["adaptive_patch"] = summarized_patch_stats

    if results:
        os.makedirs(args.results_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(args.results_dir, f"flow_reconstruction_{timestamp}.json")
        with open(output_path, "w", encoding="utf-8") as file_obj:
            json.dump(results, file_obj, indent=4)
        print(f"\nResults saved to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="completion,prediction")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--test_data", type=str, default="./data/worldtrace_sample.pkl")
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--mask_ratio", type=float, default=0.5)
    parser.add_argument("--predict_len", type=int, default=5)
    parser.add_argument("--sample_steps", type=int, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_len", type=int, default=None)
    parser.add_argument("--hidden_size", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--num_heads", type=int, default=None)
    parser.add_argument("--mlp_ratio", type=float, default=None)
    parser.add_argument(
        "--use_moe",
        type=lambda x: str(x).lower() in ("1", "true", "yes", "y"),
        default=None,
    )
    parser.add_argument("--num_experts", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=None)

    args = parser.parse_args()
    args.task = [task.strip() for task in args.task.split(",") if task.strip()]
    return args


if __name__ == "__main__":
    evaluate_reconstruction_tasks(parse_args())
