import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PatchSpec:
    """Variable-length patch boundaries for one trajectory."""

    starts: List[int]
    ends: List[int]
    scores: List[float]


def masked_diff(x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """Compute first-order differences while zeroing invalid transitions."""

    if x.ndim not in (2, 3):
        raise ValueError(f"masked_diff expects a 2D or 3D tensor, got shape {tuple(x.shape)}")

    diff = torch.zeros_like(x)
    pair_mask = (valid_mask[:, 1:] > 0.5) & (valid_mask[:, :-1] > 0.5)
    if x.ndim == 3:
        diff[:, 1:] = (x[:, 1:] - x[:, :-1]) * pair_mask.unsqueeze(-1).to(x.dtype)
    else:
        diff[:, 1:] = (x[:, 1:] - x[:, :-1]) * pair_mask.to(x.dtype)
    return diff


class LocalComplexityScorer(nn.Module):
    """Lightweight convolutional scorer for local trajectory complexity."""

    def __init__(self, in_dim: int, hidden_dim: int = 64, kernel_size: int = 5):
        super().__init__()
        kernel_size = max(3, kernel_size)
        if kernel_size % 2 == 0:
            kernel_size += 1
        padding = kernel_size // 2

        self.network = nn.Sequential(
            nn.Conv1d(in_dim, hidden_dim, kernel_size=kernel_size, padding=padding),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=padding),
            nn.GELU(),
            nn.Conv1d(hidden_dim, 1, kernel_size=1),
        )

    def forward(self, feats: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        scores = torch.sigmoid(self.network(feats.transpose(1, 2)).squeeze(1))
        return scores * attention_mask


class AdaptiveTrajectoryPatcher(nn.Module):
    """Detect variable patch boundaries from point-level trajectory content."""

    def __init__(
        self,
        coord_dim: int = 2,
        hidden_dim: int = 64,
        score_mode: str = "hybrid",
        threshold_global: float = 0.55,
        threshold_relative: float = 0.10,
        min_patch_len: int = 4,
        max_patch_len: int = 24,
        smooth_kernel: int = 5,
        learned_weight: float = 0.5,
    ):
        super().__init__()
        self.coord_dim = coord_dim
        self.score_mode = score_mode
        self.threshold_global = threshold_global
        self.threshold_relative = threshold_relative
        self.min_patch_len = max(1, min_patch_len)
        self.max_patch_len = max(self.min_patch_len, max_patch_len)
        self.smooth_kernel = max(1, smooth_kernel)
        if self.smooth_kernel % 2 == 0:
            self.smooth_kernel += 1
        self.learned_weight = learned_weight
        self.detach_patch_scores = False

        scorer_in_dim = coord_dim * 3 + 4
        self.learned_scorer = LocalComplexityScorer(
            in_dim=scorer_in_dim,
            hidden_dim=hidden_dim,
            kernel_size=self.smooth_kernel,
        )

    def _robust_normalize(self, values: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        normalized = torch.zeros_like(values)
        for batch_idx in range(values.shape[0]):
            valid_values = values[batch_idx][attention_mask[batch_idx] > 0.5]
            if valid_values.numel() == 0:
                continue

            sorted_values = torch.sort(valid_values)[0]
            percentile_index = max(1, int(math.ceil(sorted_values.numel() * 0.95))) - 1
            scale = sorted_values[percentile_index].abs()
            if scale.item() < 1e-6:
                scale = sorted_values.abs().max()
            normalized[batch_idx] = values[batch_idx] / scale.clamp_min(1e-6)

        return normalized * attention_mask

    def _smooth_scores(self, scores: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.smooth_kernel <= 1:
            return scores * attention_mask
        smoothed = F.avg_pool1d(
            scores.unsqueeze(1),
            kernel_size=self.smooth_kernel,
            stride=1,
            padding=self.smooth_kernel // 2,
        ).squeeze(1)
        return smoothed * attention_mask

    def build_features(
        self,
        trajectory: torch.Tensor,
        intervals: Optional[torch.Tensor],
        attention_mask: torch.Tensor,
        observed_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Build local geometric and masking features for patch scoring."""

        valid_mask = attention_mask > 0.5
        masked_trajectory = trajectory * attention_mask.unsqueeze(-1)

        coord_diff = masked_diff(masked_trajectory, valid_mask)
        if intervals is None:
            delta_t = torch.ones_like(coord_diff[..., :1])
            interval_feat = torch.zeros_like(attention_mask)
        else:
            delta_t = intervals.unsqueeze(-1).clamp_min(1e-3)
            interval_feat = intervals * attention_mask

        velocity = coord_diff / delta_t
        velocity = velocity * attention_mask.unsqueeze(-1)

        acceleration = masked_diff(velocity, valid_mask) / delta_t
        acceleration = acceleration * attention_mask.unsqueeze(-1)

        speed = velocity.norm(dim=-1) * attention_mask
        speed_change = masked_diff(speed, valid_mask).abs() * attention_mask
        acceleration_norm = acceleration.norm(dim=-1) * attention_mask

        heading = torch.atan2(velocity[..., 1], velocity[..., 0])
        heading_change = torch.zeros_like(speed)
        valid_pairs = ((valid_mask[:, 1:]) & (valid_mask[:, :-1])).to(speed.dtype)
        raw_heading_change = heading[:, 1:] - heading[:, :-1]
        raw_heading_change = torch.atan2(
            torch.sin(raw_heading_change),
            torch.cos(raw_heading_change),
        ).abs()
        heading_change[:, 1:] = raw_heading_change * valid_pairs
        heading_change = heading_change * attention_mask

        observed_transition = torch.zeros_like(speed)
        observed_feat = torch.zeros_like(attention_mask)
        if observed_mask is not None:
            observed_feat = observed_mask * attention_mask
            observed_transition[:, 1:] = (
                (observed_mask[:, 1:] - observed_mask[:, :-1]).abs() * valid_pairs
            )
            observed_transition = observed_transition * attention_mask

        scorer_features = torch.cat(
            [
                masked_trajectory,
                velocity,
                acceleration,
                speed.unsqueeze(-1),
                heading_change.unsqueeze(-1),
                interval_feat.unsqueeze(-1),
                observed_feat.unsqueeze(-1),
            ],
            dim=-1,
        )
        scorer_features = scorer_features * attention_mask.unsqueeze(-1)

        return {
            "trajectory": masked_trajectory,
            "velocity": velocity,
            "acceleration": acceleration,
            "speed": speed,
            "speed_change": speed_change,
            "acceleration_norm": acceleration_norm,
            "heading_change": heading_change,
            "observed_transition": observed_transition,
            "intervals": interval_feat,
            "observed_mask": observed_feat,
            "scorer_features": scorer_features,
        }

    def _compute_rule_scores_from_features(
        self,
        features: Dict[str, torch.Tensor],
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        heading_change = self._robust_normalize(features["heading_change"], attention_mask)
        acceleration_norm = self._robust_normalize(features["acceleration_norm"], attention_mask)
        speed_change = self._robust_normalize(features["speed_change"], attention_mask)
        observed_transition = features["observed_transition"].clamp(0.0, 1.0) * attention_mask

        scores = (
            0.35 * heading_change
            + 0.30 * acceleration_norm
            + 0.25 * speed_change
            + 0.10 * observed_transition
        )
        scores = self._smooth_scores(scores, attention_mask)
        scores = self._robust_normalize(scores, attention_mask)
        return scores.clamp(0.0, 1.0) * attention_mask

    def compute_rule_scores(
        self,
        trajectory: torch.Tensor,
        intervals: Optional[torch.Tensor],
        attention_mask: torch.Tensor,
        observed_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        features = self.build_features(
            trajectory=trajectory,
            intervals=intervals,
            attention_mask=attention_mask,
            observed_mask=observed_mask,
        )
        return self._compute_rule_scores_from_features(features, attention_mask)

    def compute_scores(
        self,
        trajectory: torch.Tensor,
        intervals: Optional[torch.Tensor],
        attention_mask: torch.Tensor,
        observed_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        features = self.build_features(
            trajectory=trajectory,
            intervals=intervals,
            attention_mask=attention_mask,
            observed_mask=observed_mask,
        )
        rule_scores = self._compute_rule_scores_from_features(features, attention_mask)

        if self.score_mode == "rule":
            return rule_scores

        learned_scores = self.learned_scorer(features["scorer_features"], attention_mask)
        learned_scores = self._smooth_scores(learned_scores, attention_mask).clamp(0.0, 1.0)

        if self.score_mode == "learned":
            return learned_scores * attention_mask

        scores = (1.0 - self.learned_weight) * rule_scores + self.learned_weight * learned_scores
        return scores.clamp(0.0, 1.0) * attention_mask

    def detect_boundaries_single(
        self,
        scores: torch.Tensor,
        valid_len: int,
    ) -> PatchSpec:
        """Split one valid trajectory prefix into variable-length patches."""

        if valid_len <= 0:
            return PatchSpec(starts=[], ends=[], scores=[])

        starts: List[int] = []
        ends: List[int] = []
        start = 0
        valid_scores = scores[:valid_len]

        while start < valid_len:
            current_end = valid_len - 1
            for idx in range(start, valid_len):
                current_len = idx - start + 1
                if current_len >= self.max_patch_len:
                    current_end = idx
                    break
                if idx == valid_len - 1:
                    current_end = idx
                    break
                if current_len < self.min_patch_len:
                    continue

                local_mean = float(valid_scores[start:idx + 1].mean().item())
                score_value = float(valid_scores[idx].item())
                if (
                    score_value >= self.threshold_global
                    and score_value >= local_mean + self.threshold_relative
                ):
                    current_end = idx
                    break

            starts.append(start)
            ends.append(current_end)
            start = current_end + 1

        if len(starts) > 1 and (ends[0] - starts[0] + 1) < self.min_patch_len:
            starts[1] = starts[0]
            del starts[0]
            del ends[0]

        patch_idx = 1
        while patch_idx < len(starts):
            if (ends[patch_idx] - starts[patch_idx] + 1) < self.min_patch_len:
                ends[patch_idx - 1] = ends[patch_idx]
                del starts[patch_idx]
                del ends[patch_idx]
                continue
            patch_idx += 1

        patch_scores = [
            float(valid_scores[start:end + 1].mean().item())
            for start, end in zip(starts, ends)
        ]
        return PatchSpec(starts=starts, ends=ends, scores=patch_scores)

    def forward(
        self,
        trajectory: torch.Tensor,
        attention_mask: torch.Tensor,
        intervals: Optional[torch.Tensor] = None,
        observed_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, object]:
        scores = self.compute_scores(
            trajectory=trajectory,
            intervals=intervals,
            attention_mask=attention_mask,
            observed_mask=observed_mask,
        )
        detection_scores = scores.detach() if self.detach_patch_scores else scores

        batch_size, seq_len = attention_mask.shape
        patch_specs: List[PatchSpec] = []
        num_patches: List[int] = []
        for batch_idx in range(batch_size):
            valid_len = int(attention_mask[batch_idx].sum().item())
            spec = self.detect_boundaries_single(detection_scores[batch_idx], valid_len)
            patch_specs.append(spec)
            num_patches.append(len(spec.starts))

        max_patches = max(max(num_patches, default=0), 1)
        patch2point_mask = torch.zeros(
            batch_size,
            max_patches,
            seq_len,
            device=trajectory.device,
            dtype=trajectory.dtype,
        )
        patch_lengths = torch.zeros(
            batch_size,
            max_patches,
            device=trajectory.device,
            dtype=torch.long,
        )

        for batch_idx, spec in enumerate(patch_specs):
            for patch_idx, (start, end) in enumerate(zip(spec.starts, spec.ends)):
                patch2point_mask[batch_idx, patch_idx, start:end + 1] = 1.0
                patch_lengths[batch_idx, patch_idx] = end - start + 1

        patch2point_mask = patch2point_mask * attention_mask.unsqueeze(1)
        return {
            "scores": scores * attention_mask,
            "patch_specs": patch_specs,
            "patch2point_mask": patch2point_mask,
            "patch_lengths": patch_lengths,
        }
