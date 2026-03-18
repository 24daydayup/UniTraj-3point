from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn


class PatchCrossAttention(nn.Module):
    """Refine one patch token with the point tokens inside that patch."""

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.kv_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attn_output, _ = self.attn(
            self.query_norm(query),
            self.kv_norm(key_value),
            self.kv_norm(key_value),
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        hidden = query + attn_output
        return hidden + self.ffn(self.output_norm(hidden))


class AdaptivePatchEncoder(nn.Module):
    """Encode variable-length trajectory patches into fixed-size patch tokens."""

    def __init__(
        self,
        point_dim: int = 2,
        model_dim: int = 128,
        interval_dim: int = 1,
        use_observed_flag: bool = True,
        num_heads: int = 4,
        dropout: float = 0.1,
        num_refine_layers: int = 2,
        max_patch_len_embed: int = 64,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.interval_dim = interval_dim
        self.use_observed_flag = use_observed_flag
        self.max_patch_len_embed = max_patch_len_embed
        self.use_patch_length_embedding = True

        feature_dim = point_dim + interval_dim + (1 if use_observed_flag else 0)
        self.point_mlp = nn.Sequential(
            nn.Linear(feature_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )
        self.length_embedding = nn.Embedding(max_patch_len_embed + 1, model_dim)
        self.refine_layers = nn.ModuleList(
            [PatchCrossAttention(model_dim, num_heads=num_heads, dropout=dropout) for _ in range(num_refine_layers)]
        )

    def build_point_features(
        self,
        trajectory: torch.Tensor,
        intervals: Optional[torch.Tensor],
        observed_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Build point-level patch encoder inputs."""

        feature_list = [trajectory]
        if self.interval_dim > 0:
            if intervals is None:
                interval_feat = torch.zeros(
                    trajectory.shape[0],
                    trajectory.shape[1],
                    self.interval_dim,
                    device=trajectory.device,
                    dtype=trajectory.dtype,
                )
            else:
                interval_feat = intervals.unsqueeze(-1)
            feature_list.append(interval_feat)

        if self.use_observed_flag:
            if observed_mask is None:
                observed_feat = torch.zeros_like(trajectory[..., :1])
            else:
                observed_feat = observed_mask.unsqueeze(-1)
            feature_list.append(observed_feat)

        return torch.cat(feature_list, dim=-1)

    def masked_patch_pool(
        self,
        point_tokens: torch.Tensor,
        patch2point_mask: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        valid_patch_mask = patch2point_mask * attention_mask.unsqueeze(1)
        patch_lengths = valid_patch_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        return torch.matmul(valid_patch_mask, point_tokens) / patch_lengths

    def extract_patch_sequences(
        self,
        point_tokens: torch.Tensor,
        patch2point_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[int, int]]]:
        batch_size, num_patches, _ = patch2point_mask.shape
        lengths: List[int] = []
        index_map: List[Tuple[int, int]] = []

        for batch_idx in range(batch_size):
            for patch_idx in range(num_patches):
                patch_indices = torch.nonzero(
                    patch2point_mask[batch_idx, patch_idx] > 0.5,
                    as_tuple=False,
                ).flatten()
                lengths.append(int(patch_indices.numel()))
                index_map.append((batch_idx, patch_idx))

        max_tokens = max(max(lengths, default=0), 1)
        patch_sequences = torch.zeros(
            batch_size * num_patches,
            max_tokens,
            point_tokens.shape[-1],
            device=point_tokens.device,
            dtype=point_tokens.dtype,
        )
        key_padding_mask = torch.ones(
            batch_size * num_patches,
            max_tokens,
            device=point_tokens.device,
            dtype=torch.bool,
        )

        flat_idx = 0
        for batch_idx in range(batch_size):
            for patch_idx in range(num_patches):
                patch_indices = torch.nonzero(
                    patch2point_mask[batch_idx, patch_idx] > 0.5,
                    as_tuple=False,
                ).flatten()
                if patch_indices.numel() > 0:
                    patch_sequences[flat_idx, : patch_indices.numel()] = point_tokens[batch_idx, patch_indices]
                    key_padding_mask[flat_idx, : patch_indices.numel()] = False
                flat_idx += 1

        return patch_sequences, key_padding_mask, index_map

    def forward(
        self,
        trajectory: torch.Tensor,
        attention_mask: torch.Tensor,
        patch2point_mask: torch.Tensor,
        intervals: Optional[torch.Tensor] = None,
        observed_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        point_features = self.build_point_features(
            trajectory=trajectory,
            intervals=intervals,
            observed_mask=observed_mask,
        )
        point_tokens = self.point_mlp(point_features) * attention_mask.unsqueeze(-1)

        valid_patch_mask = patch2point_mask * attention_mask.unsqueeze(1)
        patch_lengths = valid_patch_mask.sum(dim=-1)
        patch_padding_mask = patch_lengths <= 0.5

        patch_tokens = self.masked_patch_pool(
            point_tokens=point_tokens,
            patch2point_mask=patch2point_mask,
            attention_mask=attention_mask,
        )

        if self.use_patch_length_embedding:
            clipped_lengths = patch_lengths.long().clamp(0, self.max_patch_len_embed)
            patch_tokens = patch_tokens + self.length_embedding(clipped_lengths)

        patch_sequences, key_padding_mask, _ = self.extract_patch_sequences(
            point_tokens=point_tokens,
            patch2point_mask=valid_patch_mask,
        )

        flat_patch_tokens = patch_tokens.reshape(-1, 1, self.model_dim)
        valid_rows = ~patch_padding_mask.reshape(-1)
        if valid_rows.any():
            refined_tokens = flat_patch_tokens[valid_rows]
            refined_sequences = patch_sequences[valid_rows]
            refined_padding_mask = key_padding_mask[valid_rows]
            for layer in self.refine_layers:
                refined_tokens = layer(
                    query=refined_tokens,
                    key_value=refined_sequences,
                    key_padding_mask=refined_padding_mask,
                )
            flat_patch_tokens[valid_rows] = refined_tokens

        patch_tokens = flat_patch_tokens.reshape_as(patch_tokens)
        patch_tokens = patch_tokens * (~patch_padding_mask).unsqueeze(-1).to(patch_tokens.dtype)

        return {
            "patch_tokens": patch_tokens,
            "patch_padding_mask": patch_padding_mask,
            "point_tokens": point_tokens,
            "patch_lengths": patch_lengths.long(),
        }
