import datetime
import os
import shutil
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from utils.dataset import Normalize, TrajectoryDataset
from utils.flow_matching import (
    build_flow_path,
    build_flow_source,
    compute_spectral_loss,
    sample_trajectory_flow,
)
from utils.logger import Logger, log_info
from utils.project import (
    PATCH_STAT_KEYS,
    adaptive_patch_enabled,
    build_flow_model_from_config,
    checkpoint_filenames,
    load_default_config,
    resolve_device,
    save_config_snapshot,
    set_random_seed,
)


def move_batch_to_device(batch, device):
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def init_patch_metric_tracker():
    return {key: [] for key in PATCH_STAT_KEYS}


def update_patch_metric_tracker(tracker, metrics):
    for key in PATCH_STAT_KEYS:
        if key in metrics:
            tracker[key].append(float(metrics[key]))


def summarize_patch_metrics(tracker):
    return {
        key: float(np.mean(values))
        for key, values in tracker.items()
        if values
    }


def format_patch_metrics(metrics):
    if not metrics:
        return ""
    return (
        " patches={avg_num_patches:.2f}"
        " patch_len={avg_patch_len:.2f}"
        " patch_len_max={max_patch_len:.2f}"
        " patch_score={avg_patch_score:.3f}"
    ).format(**metrics)


def compute_training_loss(model, batch, config):
    target = batch["trajectory"].transpose(1, 2)
    observed = batch["observed_trajectory"].transpose(1, 2)
    attention_mask = batch["attention_mask"]
    observed_mask = batch["observed_mask"]
    loss_mask = batch["loss_mask"].unsqueeze(-1)
    intervals = batch["intervals"]

    x_0, unknown_mask = build_flow_source(
        observed_trajectory=observed,
        observed_mask=observed_mask,
        attention_mask=attention_mask,
    )
    t = torch.rand(target.shape[0], device=target.device)
    x_t, target_velocity = build_flow_path(x_0=x_0, x_1=target, t=t)
    pred_velocity = model(
        x_t=x_t,
        t=t,
        observed_trajectory=observed,
        intervals=intervals,
        attention_mask=attention_mask,
        observed_mask=observed_mask,
    )

    valid_loss_mask = loss_mask * unknown_mask
    denom = valid_loss_mask.sum().clamp_min(1.0)
    mse_loss = ((pred_velocity - target_velocity) ** 2 * valid_loss_mask).sum() / denom

    spectral_loss = torch.tensor(0.0, device=target.device)
    if config.model.use_spectral_loss and valid_loss_mask.sum().item() > 0:
        spectral_loss = compute_spectral_loss(
            prediction=pred_velocity,
            target=target_velocity,
            weight_mask=valid_loss_mask,
        )

    total_loss = mse_loss + config.model.spectral_loss_weight * spectral_loss
    metrics = {
        "loss": total_loss.detach().item(),
        "mse": mse_loss.detach().item(),
        "spectral": spectral_loss.detach().item(),
    }
    patch_stats = getattr(model, "_last_patch_stats", None)
    if patch_stats:
        metrics.update({key: float(value) for key, value in patch_stats.items()})
    return total_loss, metrics


@torch.no_grad()
def evaluate_epoch(model, dataloader, device, config):
    model.eval()
    loss_values = []
    mse_values = []
    spectral_values = []
    patch_metrics = init_patch_metric_tracker()

    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        loss, metrics = compute_training_loss(model, batch, config)
        loss_values.append(metrics["loss"])
        mse_values.append(metrics["mse"])
        spectral_values.append(metrics["spectral"])
        update_patch_metric_tracker(patch_metrics, metrics)

    summary = {
        "loss": float(np.mean(loss_values)) if loss_values else float("inf"),
        "mse": float(np.mean(mse_values)) if mse_values else float("inf"),
        "spectral": float(np.mean(spectral_values)) if spectral_values else 0.0,
    }
    summary.update(summarize_patch_metrics(patch_metrics))
    return summary


@torch.no_grad()
def run_quick_sample_check(model, dataloader, device, config):
    model.eval()
    try:
        batch = next(iter(dataloader))
    except StopIteration:
        return None

    batch = move_batch_to_device(batch, device)
    predicted = sample_trajectory_flow(
        model=model,
        observed_trajectory=batch["observed_trajectory"].transpose(1, 2),
        intervals=batch["intervals"],
        attention_mask=batch["attention_mask"],
        observed_mask=batch["observed_mask"],
        steps=config.model.sample_steps,
    )
    target = batch["trajectory"].transpose(1, 2)
    loss_mask = batch["loss_mask"].unsqueeze(-1)
    denom = loss_mask.sum().clamp_min(1.0)
    sample_error = ((predicted - target) ** 2 * loss_mask).sum() / denom
    return float(sample_error.item())


def train(config, logger, model_save):
    set_random_seed(config.training.seed)

    normalize_transform = Normalize()
    train_dataset = TrajectoryDataset(
        data_path=config.data.train_path,
        max_len=config.data.traj_length,
        transform=normalize_transform,
        mask_ratio=config.data.mask_ratio,
        task_mode=config.data.task_mode,
        completion_prob=config.data.completion_prob,
        prediction_horizon=config.data.predict_len,
        deterministic=False,
    )
    val_dataset = TrajectoryDataset(
        data_path=config.data.val_path,
        max_len=config.data.traj_length,
        transform=normalize_transform,
        mask_ratio=config.data.mask_ratio,
        task_mode=config.data.task_mode,
        completion_prob=config.data.completion_prob,
        prediction_horizon=config.data.predict_len,
        deterministic=True,
        seed=config.training.seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=max(1, config.data.num_workers // 2),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    device = resolve_device(config.training.device)
    model = build_flow_model_from_config(config).to(device)
    logger.info(f"Training single-stage flow model on {device}")
    if adaptive_patch_enabled(config.model):
        logger.info(f"Adaptive patch config: {config.model.adaptive_patch}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.training.lr_decay_factor,
        patience=config.training.lr_patience,
    )

    best_val_loss = float("inf")
    trigger_times = 0
    for epoch in range(config.training.n_epochs):
        model.train()
        epoch_losses = []
        epoch_mse = []
        epoch_spectral = []
        epoch_patch_metrics = init_patch_metric_tracker()

        logger.info(f"<----- Epoch {epoch} Training ----->")
        for batch in train_loader:
            batch = move_batch_to_device(batch, device)
            loss, metrics = compute_training_loss(model, batch, config)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)
            optimizer.step()

            epoch_losses.append(metrics["loss"])
            epoch_mse.append(metrics["mse"])
            epoch_spectral.append(metrics["spectral"])
            update_patch_metric_tracker(epoch_patch_metrics, metrics)

        train_stats = {
            "loss": float(np.mean(epoch_losses)),
            "mse": float(np.mean(epoch_mse)),
            "spectral": float(np.mean(epoch_spectral)),
        }
        train_patch_stats = summarize_patch_metrics(epoch_patch_metrics)
        logger.info(
            "Train loss={loss:.6f} mse={mse:.6f} spectral={spectral:.6f}{patch}".format(
                patch=format_patch_metrics(train_patch_stats),
                **train_stats,
            )
        )

        val_stats = evaluate_epoch(model, val_loader, device, config)
        scheduler.step(val_stats["loss"])
        val_patch_stats = {key: val_stats[key] for key in PATCH_STAT_KEYS if key in val_stats}
        logger.info(
            "Val loss={loss:.6f} mse={mse:.6f} spectral={spectral:.6f}{patch}".format(
                patch=format_patch_metrics(val_patch_stats),
                **val_stats,
            )
        )

        sample_error = run_quick_sample_check(model, val_loader, device, config)
        if sample_error is not None:
            logger.info(f"Val sampled endpoint mse={sample_error:.6f}")

        if val_stats["loss"] < best_val_loss:
            best_val_loss = val_stats["loss"]
            trigger_times = 0
            best_path = model_save / checkpoint_filenames(config.model)[0]
            torch.save(model.state_dict(), best_path)
            logger.info(f"Validation improved, saved checkpoint to {best_path}")
        else:
            trigger_times += 1
            logger.info(f"Validation did not improve for {trigger_times} epochs")
            if trigger_times >= config.training.patience:
                logger.info("Early stopping triggered")
                break

    final_path = model_save / checkpoint_filenames(config.model)[1]
    torch.save(model.state_dict(), final_path)
    logger.info(f"Training done, final checkpoint saved to {final_path}")


def setup_experiment_directories(config, exp_name="UniTraj"):
    root_dir = Path(__file__).resolve().parent
    result_name = f"{config.data.dataset}_bs={config.training.batch_size}"
    exp_dir = root_dir / exp_name / result_name
    timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M-%S")
    exp_time_dir = exp_dir / timestamp
    files_save = exp_time_dir / "Files"
    result_save = exp_time_dir / "Results"
    model_save = exp_time_dir / "models"

    for directory in [files_save, result_save, model_save]:
        directory.mkdir(parents=True, exist_ok=True)

    for filename in os.listdir(root_dir / "utils"):
        if filename.endswith(".py"):
            shutil.copy(root_dir / "utils" / filename, files_save)
    shutil.copy(Path(__file__), files_save)
    save_config_snapshot(config, exp_time_dir / "config_snapshot.json")

    logger = Logger(
        __name__,
        log_path=exp_dir / (timestamp + "/out.log"),
        colorize=True,
    )
    print("All files saved path ---->>", exp_time_dir)
    return logger, files_save, result_save, model_save


if __name__ == "__main__":
    config = load_default_config()

    logger, _, _, model_save = setup_experiment_directories(config, exp_name="UniTraj")
    log_info(config, logger)
    train(config, logger, model_save)
