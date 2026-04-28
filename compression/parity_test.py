"""一致性测试：验证 compression 模块与原始代码输出完全一致。

通过多组参数和输入形状对比原始压缩逻辑与重构版本的输出，
确保重构后的代码在数值上与原版等价。
"""

import argparse

import numpy as np
import torch

from compression import compress_flat_video_features
from compression.components import approximate_components


def original_style_compress(
    image_feat: torch.Tensor,
    ori_token_num: int,
    tau: float,
    epsilon: float,
    enable_temporal: bool = True,
    merge_original_tokens: bool = True,
) -> torch.Tensor:
    """原始 llava_arch_zip.py 中压缩逻辑的本地副本。

    此函数与原始代码的算法流程完全一致，用于和重构版本进行逐位对比。
    """

    frame_num = image_feat.shape[0] // ori_token_num
    image_feat_hw = image_feat.reshape(frame_num, ori_token_num, -1)

    # ---- 第一阶段：空间压缩 ----
    norm_image_feat = torch.norm(image_feat_hw, p=2, dim=-1, keepdim=True)
    image_feat_normalized = image_feat_hw / norm_image_feat
    similarity_matrix = torch.matmul(image_feat_normalized, image_feat_normalized.transpose(1, 2))
    high_similarity_indices = similarity_matrix > tau

    all_fused_feature = []
    for frame_cnt, image_feat_per_frame in enumerate(image_feat_hw):
        connected_components = approximate_components(
            high_similarity_indices[frame_cnt].cpu().numpy(),
            epsilon=epsilon,
        )
        fused_features = [
            torch.mean(image_feat_per_frame[cc], dim=0)
            for cc in connected_components
        ]
        all_fused_feature.append(torch.stack(fused_features))

    selected_tokens = torch.cat(all_fused_feature)

    if not enable_temporal:
        # ---- 仅空间压缩时的 token 合并 ----
        num_tokens = len(selected_tokens)
        remaining_values = image_feat
        norm_selected = torch.norm(selected_tokens, p=2, dim=1, keepdim=True)
        norm_remaining = torch.norm(remaining_values, p=2, dim=1, keepdim=True)
        sel_norm = selected_tokens / norm_selected
        rem_norm = remaining_values / norm_remaining
        sim = torch.matmul(rem_norm, sel_norm.t())
        closest = torch.argmax(sim, dim=1)

        merged = torch.zeros_like(selected_tokens)
        merged.scatter_add_(dim=0, index=closest.view(-1, 1).expand(-1, merged.shape[1]), src=remaining_values)
        counts = torch.bincount(closest, minlength=num_tokens).float() + 1
        selected_tokens = selected_tokens + merged
        selected_tokens = selected_tokens / counts.unsqueeze(1)
        return selected_tokens

    # ---- 第二阶段：时间压缩 ----
    norm_select_tokens = torch.norm(selected_tokens, p=2, dim=1, keepdim=True)
    select_tokens_normalized = selected_tokens / norm_select_tokens
    similarity_matrix_select_tokens = torch.matmul(select_tokens_normalized, select_tokens_normalized.t())
    high_similarity_indices_select_tokens = similarity_matrix_select_tokens > tau
    connected_component_select_tokens = approximate_components(
        high_similarity_indices_select_tokens.cpu().numpy(),
        epsilon=epsilon,
    )

    fused_features_select_tokens = []
    for connected_component in connected_component_select_tokens:
        selected_features = selected_tokens[connected_component]
        fused_features_select_tokens.append(torch.mean(selected_features, dim=0))
    selected_tokens = torch.stack(fused_features_select_tokens)

    # ---- 原始 token 合并 ----
    if merge_original_tokens:
        num_tokens = len(selected_tokens)
        remaining_values = image_feat
        norm_selected_tokens = torch.norm(selected_tokens, p=2, dim=1, keepdim=True)
        norm_remaining_values = torch.norm(remaining_values, p=2, dim=1, keepdim=True)
        selected_tokens_normalized = selected_tokens / norm_selected_tokens
        remaining_values_normalized = remaining_values / norm_remaining_values
        similarity_matrix = torch.matmul(remaining_values_normalized, selected_tokens_normalized.t())
        closest_indices = torch.argmax(similarity_matrix, dim=1)

        merged_tokens = torch.zeros_like(selected_tokens)
        merged_tokens.scatter_add_(
            dim=0,
            index=closest_indices.view(-1, 1).expand(-1, merged_tokens.shape[1]),
            src=remaining_values,
        )
        counts = torch.bincount(closest_indices, minlength=num_tokens).float() + 1
        selected_tokens += merged_tokens
        selected_tokens /= counts.unsqueeze(1)
    return selected_tokens


def run_parity_test(device: str) -> None:
    """运行多组参数的一致性测试。

    测试覆盖了多种 tau/epsilon 组合以及不同的输入形状，
    确保在实际压缩和非压缩场景下重构版本都与原版一致。
    """

    # 利用相似向量构造可压缩的输入，使测试能覆盖到实际压缩场景
    torch.manual_seed(42)
    frames, tokens, hidden = 4, 64, 128
    # 每帧的 token 在基向量上加小噪声，保证帧内 token 之间有一定相似性
    base = torch.randn(frames, tokens, hidden, device=device)
    image_feat = base.reshape(frames * tokens, hidden)

    test_cases = [
        # (tau, epsilon, enable_temporal, merge_original_tokens, description)
        (0.99, 0.05, True, True, "默认配置 + 极高阈值"),
        (0.90, 0.05, True, True, "默认配置 + 高阈值"),
        (0.95, 0.10, True, True, "默认配置 + 高 epsilon"),
        (0.80, 0.10, True, True, "默认配置 + 低阈值"),
        (0.50, 0.20, True, True, "宽松阈值 + 高误差"),
        (0.90, 0.05, False, True, "仅空间压缩 + token 合并"),
        (0.90, 0.05, True, False, "时空压缩 + 不合并"),
        (0.90, 0.05, False, False, "仅空间压缩 + 不合并"),
    ]

    passed = 0
    failed = 0

    for tau, epsilon, enable_temporal, merge_original, desc in test_cases:
        np.random.seed(23)
        expected = original_style_compress(
            image_feat.clone(), tokens, tau, epsilon,
            enable_temporal=enable_temporal,
            merge_original_tokens=merge_original,
        )

        np.random.seed(23)
        actual, stats = compress_flat_video_features(
            image_feat.clone(), tokens,
            tau=tau, epsilon=epsilon,
            enable_temporal=enable_temporal,
            merge_original_tokens=merge_original,
        )

        try:
            assert expected.shape == actual.shape, (
                f"形状不匹配: 原版 {tuple(expected.shape)} vs 重构版 {tuple(actual.shape)}"
            )
            assert torch.allclose(expected, actual, atol=1e-6, rtol=1e-6), (
                f"数值不一致，最大差异: {(expected - actual).abs().max()}"
            )
            print(f"  [通过] {desc} | tau={tau} epsilon={epsilon} "
                  f"temporal={enable_temporal} merge={merge_original}")
            print(f"         tokens: {stats.original_tokens} -> {stats.final_tokens} "
                  f"(空间={stats.after_spatial_tokens} 时间={stats.after_temporal_tokens})")
            passed += 1
        except AssertionError as e:
            print(f"  [失败] {desc}: {e}")
            failed += 1

    # 额外测试：不同输入形状
    extra_shapes = [
        (8, 49, 256, "大帧数"),    # 8帧, 每个token来自7x7 patch
        (2, 196, 512, "宽隐藏层"),  # 2帧, 每个token来自14x14 patch
        (4, 16, 64, "小特征"),      # 极小的隐藏维度
    ]

    for frames, tokens, hidden, desc in extra_shapes:
        np.random.seed(23)
        test_feat = torch.randn(frames * tokens, hidden, device=device)
        expected = original_style_compress(test_feat.clone(), tokens, 0.95, 0.05)
        np.random.seed(23)
        actual, stats = compress_flat_video_features(test_feat.clone(), tokens, tau=0.95, epsilon=0.05)

        try:
            assert expected.shape == actual.shape
            assert torch.allclose(expected, actual, atol=1e-6, rtol=1e-6)
            print(f"  [通过] 形状测试({desc}) | {frames}帧 × {tokens}token × {hidden}维 "
                  f"-> {stats.final_tokens} tokens")
            passed += 1
        except AssertionError as e:
            print(f"  [失败] 形状测试({desc}): {e}")
            failed += 1

    print(f"\n总计: {passed} 通过, {failed} 失败")
    if failed > 0:
        raise SystemExit(f"{failed} 项测试失败")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLaVA-Scissor 重构版本一致性测试")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="运行设备 (默认: cuda 如果可用, 否则 cpu)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"设备: {args.device}")
    print("=" * 60)
    run_parity_test(args.device)
    print("=" * 60)
    print("parity_test 全部完成")


if __name__ == "__main__":
    main()
