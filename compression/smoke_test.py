"""冒烟测试：验证 LlavaScissorCompressor 的基本功能正确性。

测试内容：
- 输出形状和设备正确
- 压缩比在合理范围内
- 所有统计字段正常填充
- 相同输入 + 相同种子 = 相同输出（确定性）
"""

import argparse

import numpy as np
import torch

from compression import compress_flat_video_features


def run_smoke_test(device: str, dtype: torch.dtype) -> None:
    """执行冒烟测试。"""
    torch.manual_seed(7)
    np.random.seed(7)

    frames = 4
    tokens_per_frame = 16
    hidden_dim = 32
    image_feat = torch.randn(
        frames * tokens_per_frame,
        hidden_dim,
        device=device,
        dtype=dtype,
    )

    compressed, stats = compress_flat_video_features(
        image_feat=image_feat,
        ori_token_num=tokens_per_frame,
        tau=0.80,
        epsilon=0.20,
    )

    # 输出形状检查
    assert compressed.ndim == 2, f"输出应为二维张量，实际 {compressed.ndim} 维"
    assert compressed.shape[1] == hidden_dim, (
        f"隐藏维度应保持不变: {hidden_dim} vs {compressed.shape[1]}"
    )
    assert compressed.device == image_feat.device, "设备应与输入一致"
    assert compressed.dtype == image_feat.dtype, "数据类型应与输入一致"

    # 压缩比检查
    assert 0 < compressed.shape[0] <= image_feat.shape[0], (
        f"压缩后的 token 数 ({compressed.shape[0]}) 应在 1 和输入 ({image_feat.shape[0]}) 之间"
    )

    # 统计字段检查
    assert stats.original_tokens == frames * tokens_per_frame
    assert stats.frames == frames
    assert stats.tokens_per_frame == tokens_per_frame
    assert stats.final_tokens == compressed.shape[0]
    assert stats.tau == 0.80
    assert stats.epsilon == 0.20
    assert stats.elapsed_ms > 0, "耗时应为正数"

    # 确定性检查：相同输入 + 重置种子 = 完全相同的输出
    np.random.seed(7)
    compressed_again, stats_again = compress_flat_video_features(
        image_feat=image_feat,
        ori_token_num=tokens_per_frame,
        tau=0.80,
        epsilon=0.20,
    )

    assert stats_again.final_tokens == stats.final_tokens, (
        f"确定性失败：token 数不一致 {stats.final_tokens} vs {stats_again.final_tokens}"
    )
    assert torch.allclose(compressed, compressed_again), "确定性失败：张量值不一致"

    print("smoke_test 通过")
    print(f"  输入形状: {tuple(image_feat.shape)}")
    print(f"  压缩后: {tuple(compressed.shape)}")
    print(stats)

    run_improvement_smoke_cases(image_feat, tokens_per_frame)


def run_improvement_smoke_cases(image_feat: torch.Tensor, tokens_per_frame: int) -> None:
    """检查新增改进开关的基本契约，不做固定数值金标准。"""
    baseline, baseline_stats = compress_flat_video_features(
        image_feat=image_feat,
        ori_token_num=tokens_per_frame,
        tau=0.80,
        epsilon=0.20,
        adaptive_tau=False,
        component_merge="mean",
        original_merge_strategy="hard",
        temporal_strategy="global",
    )

    adaptive, adaptive_stats = compress_flat_video_features(
        image_feat=image_feat,
        ori_token_num=tokens_per_frame,
        tau=0.80,
        epsilon=0.20,
        adaptive_tau=True,
        adaptive_tau_mode="redundancy",
        adaptive_tau_min=0.50,
        adaptive_tau_max=0.95,
    )
    assert adaptive.shape[1] == image_feat.shape[1]
    assert len(adaptive_stats.spatial_frame_taus) == adaptive_stats.frames
    assert all(0.50 <= tau <= 0.95 for tau in adaptive_stats.spatial_frame_taus)
    assert torch.isfinite(adaptive).all()

    weighted, weighted_stats = compress_flat_video_features(
        image_feat=image_feat,
        ori_token_num=tokens_per_frame,
        tau=0.80,
        epsilon=0.20,
        component_merge="centrality",
        component_merge_temperature=0.10,
        merge_original_tokens=False,
    )
    assert weighted.shape[0] == weighted_stats.after_temporal_tokens
    assert weighted_stats.component_merge == "centrality"
    assert torch.isfinite(weighted).all()

    soft_refill, soft_refill_stats = compress_flat_video_features(
        image_feat=image_feat,
        ori_token_num=tokens_per_frame,
        tau=0.80,
        epsilon=0.20,
        original_merge_strategy="soft",
        soft_merge_topk=2,
        soft_merge_temperature=0.10,
    )
    assert soft_refill.shape[0] == soft_refill_stats.final_tokens
    assert soft_refill_stats.original_merge_strategy == "soft"
    assert torch.isfinite(soft_refill).all()

    windowed, windowed_stats = compress_flat_video_features(
        image_feat=image_feat,
        ori_token_num=tokens_per_frame,
        tau=0.80,
        epsilon=0.20,
        temporal_strategy="windowed",
        temporal_window_size=2,
    )
    assert 0 < windowed.shape[0] <= baseline_stats.after_spatial_tokens
    assert windowed_stats.temporal_strategy == "windowed"
    assert torch.isfinite(windowed).all()

    np.random.seed(11)
    no_op_baseline, _ = compress_flat_video_features(
        image_feat=image_feat,
        ori_token_num=tokens_per_frame,
        tau=0.80,
        epsilon=0.20,
    )
    np.random.seed(11)
    no_op_windowed, _ = compress_flat_video_features(
        image_feat=image_feat,
        ori_token_num=tokens_per_frame,
        tau=0.80,
        epsilon=0.20,
        temporal_strategy="windowed",
        temporal_window_size=baseline_stats.frames,
    )
    assert torch.allclose(no_op_baseline, no_op_windowed), (
        "window size >= frame count should match global temporal compression"
    )

    combined, combined_stats = compress_flat_video_features(
        image_feat=image_feat,
        ori_token_num=tokens_per_frame,
        tau=0.80,
        epsilon=0.20,
        adaptive_tau=True,
        component_merge="centrality",
        original_merge_strategy="soft",
        temporal_strategy="windowed",
        temporal_window_size=2,
    )
    assert 0 < combined.shape[0] <= image_feat.shape[0]
    assert combined_stats.component_merge == "centrality"
    assert combined_stats.original_merge_strategy == "soft"
    assert torch.isfinite(combined).all()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLaVA-Scissor 冒烟测试")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="运行设备 (默认: cuda)")
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float32",
                        help="张量数据类型 (默认: float32)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    run_smoke_test(args.device, dtype)


if __name__ == "__main__":
    main()
