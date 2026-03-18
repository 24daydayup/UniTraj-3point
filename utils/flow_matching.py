from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.adaptive_patch_encoder import AdaptivePatchEncoder
from utils.adaptive_patcher import AdaptiveTrajectoryPatcher


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class SparseMoE(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_experts: int = 4,
        top_k: int = 2,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)

        self.gate = nn.Linear(hidden_size, num_experts)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_size, mlp_hidden_dim),
                    nn.GELU(),
                    nn.Linear(mlp_hidden_dim, hidden_size),
                )
                for _ in range(num_experts)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_size = x.shape
        flat_x = x.reshape(-1, hidden_size)

        logits = self.gate(flat_x)
        weights = F.softmax(logits, dim=-1)
        top_k_weights, top_k_indices = torch.topk(weights, self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        output = torch.zeros_like(flat_x)
        for k in range(self.top_k):
            expert_ids = top_k_indices[:, k]
            expert_weights = top_k_weights[:, k].unsqueeze(-1)
            for expert_idx in range(self.num_experts):
                token_mask = expert_ids == expert_idx
                if token_mask.any():
                    expert_out = self.experts[expert_idx](flat_x[token_mask])
                    output[token_mask] += expert_out * expert_weights[token_mask]

        return output.reshape(batch_size, seq_len, hidden_size)


class DiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_moe: bool = False,
        num_experts: int = 4,
        top_k: int = 2,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        if use_moe:
            self.mlp = SparseMoE(
                hidden_size=hidden_size,
                num_experts=num_experts,
                top_k=top_k,
                mlp_ratio=mlp_ratio,
            )
        else:
            mlp_hidden_dim = int(hidden_size * mlp_ratio)
            self.mlp = nn.Sequential(
                nn.Linear(hidden_size, mlp_hidden_dim),
                nn.GELU(),
                nn.Linear(mlp_hidden_dim, hidden_size),
            )
        self.ada_ln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = self.ada_ln(condition).chunk(6, dim=-1)

        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask <= 0.5

        attn_input = modulate(self.norm1(x), shift_attn, scale_attn)
        attn_output, _ = self.attn(
            attn_input,
            attn_input,
            attn_input,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + gate_attn.unsqueeze(1) * attn_output

        mlp_input = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(mlp_input)
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.ada_ln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        shift, scale = self.ada_ln(condition).chunk(2, dim=-1)
        return self.linear(modulate(self.norm(x), shift, scale))


class PointPatchFusion(nn.Module):
    """Fuse patch-level context back into point-level condition tokens."""

    def __init__(self, dim: int = 256, num_heads: int = 4, dropout: float = 0.1):
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
        point_q: torch.Tensor,
        patch_kv: torch.Tensor,
        patch_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if patch_kv.numel() == 0 or patch_kv.shape[1] == 0:
            return point_q

        if patch_padding_mask is None:
            patch_padding_mask = torch.zeros(
                point_q.shape[0],
                patch_kv.shape[1],
                device=point_q.device,
                dtype=torch.bool,
            )

        valid_batches = ~patch_padding_mask.all(dim=1)
        if not valid_batches.any():
            return point_q

        fused = point_q.clone()
        attn_output, _ = self.attn(
            self.query_norm(point_q[valid_batches]),
            self.kv_norm(patch_kv[valid_batches]),
            self.kv_norm(patch_kv[valid_batches]),
            key_padding_mask=patch_padding_mask[valid_batches],
            need_weights=False,
        )
        refined = point_q[valid_batches] + attn_output
        refined = refined + self.ffn(self.output_norm(refined))
        fused[valid_batches] = refined
        return fused


class SingleStageTrajectoryFlow(nn.Module):
    def __init__(
        self,
        input_dim: int = 2,
        hidden_size: int = 256,
        depth: int = 6,
        num_heads: int = 4,
        max_len: int = 200,
        mlp_ratio: float = 4.0,
        use_moe: bool = False,
        num_experts: int = 4,
        top_k: int = 2,
        adaptive_patch_cfg: Optional[dict] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.max_len = max_len
        self.adaptive_patch_cfg = dict(adaptive_patch_cfg or {})
        self.adaptive_patch_cfg.setdefault("enabled", False)
        self.adaptive_patch_cfg.setdefault("score_mode", "rule")
        self.adaptive_patch_cfg.setdefault("hidden_dim", 64)
        self.adaptive_patch_cfg.setdefault("threshold_global", 0.55)
        self.adaptive_patch_cfg.setdefault("threshold_relative", 0.10)
        self.adaptive_patch_cfg.setdefault("min_patch_len", 4)
        self.adaptive_patch_cfg.setdefault("max_patch_len", 24)
        self.adaptive_patch_cfg.setdefault("smooth_kernel", 5)
        self.adaptive_patch_cfg.setdefault("learned_weight", 0.5)
        self.adaptive_patch_cfg.setdefault("patch_encoder_dim", hidden_size)
        self.adaptive_patch_cfg.setdefault("patch_encoder_heads", num_heads)
        self.adaptive_patch_cfg.setdefault("patch_encoder_layers", 2)
        self.adaptive_patch_cfg.setdefault("patch_dropout", 0.1)
        self.adaptive_patch_cfg.setdefault("use_patch_length_embedding", True)
        self.adaptive_patch_cfg.setdefault("use_point_patch_fusion", True)
        self.adaptive_patch_cfg.setdefault("fusion_heads", num_heads)
        self.adaptive_patch_cfg.setdefault("fusion_dropout", 0.1)
        self.adaptive_patch_cfg.setdefault("detach_patch_scores", False)
        self.use_adaptive_patch = bool(self.adaptive_patch_cfg["enabled"])
        self.use_point_patch_fusion = bool(self.adaptive_patch_cfg["use_point_patch_fusion"])

        self.x_embedder = nn.Linear(input_dim, hidden_size)
        self.cond_embedder = nn.Linear(input_dim, hidden_size)
        self.interval_embedder = nn.Linear(1, hidden_size)
        self.mask_embedder = nn.Linear(2, hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, hidden_size))

        self.t_embedder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.context_projector = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    use_moe=use_moe,
                    num_experts=num_experts,
                    top_k=top_k,
                )
                for _ in range(depth)
            ]
        )
        self.final_layer = FinalLayer(hidden_size, input_dim)

        if self.use_adaptive_patch:
            patch_dim = int(self.adaptive_patch_cfg["patch_encoder_dim"])
            self.adaptive_patcher = AdaptiveTrajectoryPatcher(
                coord_dim=input_dim,
                hidden_dim=int(self.adaptive_patch_cfg["hidden_dim"]),
                score_mode=str(self.adaptive_patch_cfg["score_mode"]),
                threshold_global=float(self.adaptive_patch_cfg["threshold_global"]),
                threshold_relative=float(self.adaptive_patch_cfg["threshold_relative"]),
                min_patch_len=int(self.adaptive_patch_cfg["min_patch_len"]),
                max_patch_len=int(self.adaptive_patch_cfg["max_patch_len"]),
                smooth_kernel=int(self.adaptive_patch_cfg["smooth_kernel"]),
                learned_weight=float(self.adaptive_patch_cfg["learned_weight"]),
            )
            self.adaptive_patcher.detach_patch_scores = bool(
                self.adaptive_patch_cfg["detach_patch_scores"]
            )
            self.patch_encoder = AdaptivePatchEncoder(
                point_dim=input_dim,
                model_dim=patch_dim,
                interval_dim=1,
                use_observed_flag=True,
                num_heads=int(self.adaptive_patch_cfg["patch_encoder_heads"]),
                dropout=float(self.adaptive_patch_cfg["patch_dropout"]),
                num_refine_layers=int(self.adaptive_patch_cfg["patch_encoder_layers"]),
                max_patch_len_embed=max(
                    int(self.adaptive_patch_cfg["max_patch_len"]),
                    64,
                ),
            )
            self.patch_encoder.use_patch_length_embedding = bool(
                self.adaptive_patch_cfg["use_patch_length_embedding"]
            )
            if patch_dim != hidden_size:
                self.patch_proj = nn.Linear(patch_dim, hidden_size)
            if self.use_point_patch_fusion:
                self.point_patch_fusion = PointPatchFusion(
                    dim=hidden_size,
                    num_heads=int(self.adaptive_patch_cfg["fusion_heads"]),
                    dropout=float(self.adaptive_patch_cfg["fusion_dropout"]),
                )
            else:
                self.point_patch_fusion = None
        else:
            self.point_patch_fusion = None

        self._last_patch_scores: Optional[torch.Tensor] = None
        self._last_patch_lengths: Optional[torch.Tensor] = None
        self._last_num_patches: Optional[torch.Tensor] = None
        self._last_patch_stats: Optional[Dict[str, float]] = None
        self._condition_cache_key: Optional[Tuple[int, ...]] = None
        self._condition_cache: Optional[Dict[str, Any]] = None
        self.initialize_weights()

    def initialize_weights(self):
        def _init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_init)
        nn.init.normal_(self.pos_embed, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.ada_ln[-1].weight, 0)
            nn.init.constant_(block.ada_ln[-1].bias, 0)

        nn.init.constant_(self.final_layer.ada_ln[-1].weight, 0)
        nn.init.constant_(self.final_layer.ada_ln[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def train(self, mode: bool = True):
        self.clear_condition_cache()
        return super().train(mode)

    def clear_condition_cache(self) -> None:
        self._condition_cache_key = None
        self._condition_cache = None

    def _clear_patch_debug(self) -> None:
        self._last_patch_scores = None
        self._last_patch_lengths = None
        self._last_num_patches = None
        self._last_patch_stats = None

    def _build_condition_cache_key(
        self,
        observed_trajectory: torch.Tensor,
        intervals: torch.Tensor,
        attention_mask: torch.Tensor,
        observed_mask: torch.Tensor,
    ) -> Tuple[int, ...]:
        return (
            observed_trajectory.data_ptr(),
            intervals.data_ptr(),
            attention_mask.data_ptr(),
            observed_mask.data_ptr(),
            observed_trajectory.shape[0],
            observed_trajectory.shape[1],
            observed_trajectory.device.index if observed_trajectory.device.index is not None else -1,
        )

    def _restore_patch_debug(self, debug_cache: Optional[Dict[str, Any]]) -> None:
        if debug_cache is None:
            self._clear_patch_debug()
            return

        self._last_patch_stats = dict(debug_cache["stats"]) if debug_cache["stats"] is not None else None
        self._last_patch_scores = debug_cache["scores"]
        self._last_patch_lengths = debug_cache["lengths"]
        self._last_num_patches = debug_cache["num_patches"]

    def _load_condition_cache(
        self,
        observed_trajectory: torch.Tensor,
        intervals: torch.Tensor,
        attention_mask: torch.Tensor,
        observed_mask: torch.Tensor,
    ) -> Optional[Dict[str, torch.Tensor]]:
        if not self.use_adaptive_patch or self.training or torch.is_grad_enabled():
            return None

        cache_key = self._build_condition_cache_key(
            observed_trajectory=observed_trajectory,
            intervals=intervals,
            attention_mask=attention_mask,
            observed_mask=observed_mask,
        )
        if self._condition_cache is None or self._condition_cache_key != cache_key:
            return None

        self._restore_patch_debug(self._condition_cache.get("debug"))
        return self._condition_cache

    def _store_condition_cache(
        self,
        observed_trajectory: torch.Tensor,
        intervals: torch.Tensor,
        attention_mask: torch.Tensor,
        observed_mask: torch.Tensor,
        cond_tokens: torch.Tensor,
        pooled: torch.Tensor,
    ) -> None:
        if not self.use_adaptive_patch or self.training or torch.is_grad_enabled():
            return

        self._condition_cache_key = self._build_condition_cache_key(
            observed_trajectory=observed_trajectory,
            intervals=intervals,
            attention_mask=attention_mask,
            observed_mask=observed_mask,
        )
        debug_cache = None
        if self._last_patch_stats is not None:
            debug_cache = {
                "stats": dict(self._last_patch_stats),
                "scores": self._last_patch_scores,
                "lengths": self._last_patch_lengths,
                "num_patches": self._last_num_patches,
            }
        self._condition_cache = {
            "cond_tokens": cond_tokens.detach(),
            "pooled": pooled.detach(),
            "debug": debug_cache,
        }

    def _update_patch_debug(
        self,
        patcher_out: Dict[str, Any],
        encoder_out: Dict[str, torch.Tensor],
        attention_mask: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            patch_scores = patcher_out["scores"].detach()
            patch_lengths = encoder_out["patch_lengths"].detach()
            patch_padding_mask = encoder_out["patch_padding_mask"].detach()
            valid_patch_mask = ~patch_padding_mask
            valid_patch_lengths = patch_lengths[valid_patch_mask].float()
            num_patches = valid_patch_mask.sum(dim=1)

            avg_patch_len = valid_patch_lengths.mean().item() if valid_patch_lengths.numel() > 0 else 0.0
            max_patch_len = valid_patch_lengths.max().item() if valid_patch_lengths.numel() > 0 else 0.0
            avg_num_patches = num_patches.float().mean().item()
            avg_patch_score = (
                (patch_scores * attention_mask).sum(dim=1)
                / attention_mask.sum(dim=1).clamp_min(1.0)
            ).mean().item()

            self._last_patch_scores = patch_scores
            self._last_patch_lengths = patch_lengths
            self._last_num_patches = num_patches
            self._last_patch_stats = {
                "avg_num_patches": avg_num_patches,
                "avg_patch_len": avg_patch_len,
                "max_patch_len": max_patch_len,
                "avg_patch_score": avg_patch_score,
            }

    def get_timestep_embedding(self, t: torch.Tensor, dim: int) -> torch.Tensor:
        half_dim = dim // 2
        exponent = math.log(10000) / max(half_dim - 1, 1)
        frequencies = torch.exp(torch.arange(half_dim, device=t.device) * -exponent)
        arguments = t[:, None] * frequencies[None, :]
        embedding = torch.cat((arguments.sin(), arguments.cos()), dim=-1)
        if dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1, 0, 0))
        return embedding

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        observed_trajectory: torch.Tensor,
        intervals: torch.Tensor,
        attention_mask: torch.Tensor,
        observed_mask: torch.Tensor,
    ) -> torch.Tensor:
        _, seq_len, _ = x_t.shape
        if seq_len > self.max_len:
            raise ValueError(
                f"Input sequence length {seq_len} exceeds model max_len={self.max_len}"
            )

        cached_condition = self._load_condition_cache(
            observed_trajectory=observed_trajectory,
            intervals=intervals,
            attention_mask=attention_mask,
            observed_mask=observed_mask,
        )
        if cached_condition is not None:
            cond_tokens = cached_condition["cond_tokens"]
            pooled = cached_condition["pooled"]
        else:
            point_cond = self.cond_embedder(observed_trajectory)
            interval_tokens = self.interval_embedder(intervals.unsqueeze(-1))
            mask_features = torch.stack([observed_mask, attention_mask], dim=-1)
            mask_tokens = self.mask_embedder(mask_features)

            if self.use_adaptive_patch:
                patcher_out = self.adaptive_patcher(
                    trajectory=observed_trajectory,
                    attention_mask=attention_mask,
                    intervals=intervals,
                    observed_mask=observed_mask,
                )
                encoder_out = self.patch_encoder(
                    trajectory=observed_trajectory,
                    attention_mask=attention_mask,
                    patch2point_mask=patcher_out["patch2point_mask"],
                    intervals=intervals,
                    observed_mask=observed_mask,
                )
                patch_tokens = encoder_out["patch_tokens"]
                if hasattr(self, "patch_proj"):
                    patch_tokens = self.patch_proj(patch_tokens)

                if self.use_point_patch_fusion and self.point_patch_fusion is not None:
                    fused_point = self.point_patch_fusion(
                        point_q=point_cond,
                        patch_kv=patch_tokens,
                        patch_padding_mask=encoder_out["patch_padding_mask"],
                    )
                else:
                    fused_point = point_cond

                cond_tokens = fused_point + interval_tokens + mask_tokens
                self._update_patch_debug(
                    patcher_out=patcher_out,
                    encoder_out=encoder_out,
                    attention_mask=attention_mask,
                )
            else:
                self._clear_patch_debug()
                cond_tokens = point_cond + interval_tokens + mask_tokens

            pooled = cond_tokens * attention_mask.unsqueeze(-1)
            pooled = pooled.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            self._store_condition_cache(
                observed_trajectory=observed_trajectory,
                intervals=intervals,
                attention_mask=attention_mask,
                observed_mask=observed_mask,
                cond_tokens=cond_tokens,
                pooled=pooled,
            )

        x = self.x_embedder(x_t) + cond_tokens + self.pos_embed[:, :seq_len, :]

        t_embedding = self.get_timestep_embedding(t * 1000.0, self.hidden_size)
        t_embedding = self.t_embedder(t_embedding)

        condition = t_embedding + self.context_projector(pooled)

        for block in self.blocks:
            x = block(x, condition, attention_mask)

        output = self.final_layer(x, condition)
        return output * attention_mask.unsqueeze(-1)


def build_flow_source(
    observed_trajectory: torch.Tensor,
    observed_mask: torch.Tensor,
    attention_mask: torch.Tensor,
    noise: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if noise is None:
        noise = torch.randn_like(observed_trajectory)

    attention_mask = attention_mask.unsqueeze(-1)
    observed_mask = observed_mask.unsqueeze(-1)
    unknown_mask = (attention_mask - observed_mask).clamp_min(0.0)
    x_0 = observed_trajectory + noise * unknown_mask
    return x_0 * attention_mask, unknown_mask


def build_flow_path(
    x_0: torch.Tensor,
    x_1: torch.Tensor,
    t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    t = t.view(-1, 1, 1)
    x_t = (1.0 - t) * x_0 + t * x_1
    target_velocity = x_1 - x_0
    return x_t, target_velocity


def compute_spectral_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weight_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if weight_mask is not None:
        prediction = prediction * weight_mask
        target = target * weight_mask

    pred_fft = torch.fft.rfft(prediction.permute(0, 2, 1), dim=-1)
    target_fft = torch.fft.rfft(target.permute(0, 2, 1), dim=-1)

    loss_amp = torch.abs(pred_fft.abs() - target_fft.abs()).mean()
    loss_phase = torch.abs(pred_fft.angle() - target_fft.angle()).mean()
    return loss_amp + 0.1 * loss_phase


@torch.no_grad()
def sample_trajectory_flow(
    model: SingleStageTrajectoryFlow,
    observed_trajectory: torch.Tensor,
    intervals: torch.Tensor,
    attention_mask: torch.Tensor,
    observed_mask: torch.Tensor,
    steps: int = 16,
    noise: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if steps <= 0:
        raise ValueError("steps must be a positive integer")

    device = observed_trajectory.device
    attention = attention_mask.unsqueeze(-1)
    known_mask = observed_mask.unsqueeze(-1)
    if hasattr(model, "clear_condition_cache"):
        model.clear_condition_cache()
    x, unknown_mask = build_flow_source(
        observed_trajectory=observed_trajectory,
        observed_mask=observed_mask,
        attention_mask=attention_mask,
        noise=noise,
    )
    dt = 1.0 / steps

    for step in range(steps):
        t = torch.full((observed_trajectory.shape[0],), step / steps, device=device)
        velocity = model(
            x_t=x,
            t=t,
            observed_trajectory=observed_trajectory,
            intervals=intervals,
            attention_mask=attention_mask,
            observed_mask=observed_mask,
        )
        x = x + velocity * dt * unknown_mask
        x = x * unknown_mask + observed_trajectory * known_mask
        x = x * attention

    if hasattr(model, "clear_condition_cache"):
        model.clear_condition_cache()
    return x
