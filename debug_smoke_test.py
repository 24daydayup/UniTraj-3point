import torch

from utils.adaptive_patch_encoder import AdaptivePatchEncoder
from utils.adaptive_patcher import AdaptiveTrajectoryPatcher
from utils.flow_matching import SingleStageTrajectoryFlow, sample_trajectory_flow


def main():
    torch.manual_seed(7)

    batch_size, seq_len = 2, 16
    trajectory = torch.randn(batch_size, seq_len, 2)
    trajectory[0, 6:10] += 2.5
    trajectory[1, 10:] *= 0.25

    observed_mask = torch.ones(batch_size, seq_len)
    observed_mask[:, 8:] = 0.0
    attention_mask = torch.ones(batch_size, seq_len)
    attention_mask[1, 14:] = 0.0
    intervals = torch.ones(batch_size, seq_len)
    intervals[:, 0] = 0.0

    patcher = AdaptiveTrajectoryPatcher(
        coord_dim=2,
        hidden_dim=32,
        score_mode="rule",
        threshold_global=0.55,
        threshold_relative=0.10,
        min_patch_len=2,
        max_patch_len=8,
        smooth_kernel=3,
        learned_weight=0.5,
    )
    patcher_out = patcher(
        trajectory=trajectory * observed_mask.unsqueeze(-1),
        attention_mask=attention_mask,
        intervals=intervals,
        observed_mask=observed_mask,
    )
    patch2point_mask = patcher_out["patch2point_mask"]
    assert patch2point_mask.shape[0] == batch_size
    assert patch2point_mask.shape[-1] == seq_len

    encoder = AdaptivePatchEncoder(
        point_dim=2,
        model_dim=64,
        interval_dim=1,
        use_observed_flag=True,
        num_heads=4,
        dropout=0.1,
        num_refine_layers=2,
        max_patch_len_embed=16,
    )
    encoder_out = encoder(
        trajectory=trajectory * observed_mask.unsqueeze(-1),
        attention_mask=attention_mask,
        patch2point_mask=patch2point_mask,
        intervals=intervals,
        observed_mask=observed_mask,
    )
    assert encoder_out["patch_tokens"].shape[0] == batch_size
    assert encoder_out["patch_tokens"].shape[-1] == 64

    model = SingleStageTrajectoryFlow(
        input_dim=2,
        hidden_size=64,
        depth=2,
        num_heads=4,
        max_len=32,
        adaptive_patch_cfg={
            "enabled": True,
            "score_mode": "rule",
            "hidden_dim": 32,
            "threshold_global": 0.55,
            "threshold_relative": 0.10,
            "min_patch_len": 2,
            "max_patch_len": 8,
            "smooth_kernel": 3,
            "learned_weight": 0.5,
            "patch_encoder_dim": 64,
            "patch_encoder_heads": 4,
            "patch_encoder_layers": 2,
            "patch_dropout": 0.1,
            "use_patch_length_embedding": True,
            "use_point_patch_fusion": True,
            "fusion_heads": 4,
            "fusion_dropout": 0.1,
            "detach_patch_scores": False,
        },
    )

    x_t = torch.randn(batch_size, seq_len, 2)
    observed = trajectory * observed_mask.unsqueeze(-1)
    t = torch.rand(batch_size)

    output = model(
        x_t=x_t,
        t=t,
        observed_trajectory=observed,
        intervals=intervals,
        attention_mask=attention_mask,
        observed_mask=observed_mask,
    )
    assert output.shape == (batch_size, seq_len, 2)

    loss = output.pow(2).mean()
    loss.backward()

    model.eval()
    sampled = sample_trajectory_flow(
        model=model,
        observed_trajectory=observed,
        intervals=intervals,
        attention_mask=attention_mask,
        observed_mask=observed_mask,
        steps=4,
    )
    assert sampled.shape == (batch_size, seq_len, 2)
    print("smoke test passed")


if __name__ == "__main__":
    main()
