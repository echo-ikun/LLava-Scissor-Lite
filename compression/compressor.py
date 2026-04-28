import time

import torch

from .components import approximate_components
from .config import ScissorConfig
from .stats import CompressionStats


class LlavaScissorCompressor:
    """LLaVA-Scissor 视频 token 压缩器的可读实现。

    两阶段压缩流程：
    1. 空间压缩：在每个帧内，合并语义相似的 patch token
    2. 时间压缩：跨帧合并语义相似的空间压缩 token
    3. （可选）将原始 token 合并回压缩后的 token 集合
    """

    def __init__(self, config: ScissorConfig | None = None):
        self.config = config or ScissorConfig()

    def compress_video_tokens(
        self,
        image_feat: torch.Tensor,
        tokens_per_frame: int,
    ) -> tuple[torch.Tensor, CompressionStats]:
        """压缩扁平化的视频特征。

        参数:
            image_feat: 形状为 [frames * tokens_per_frame, hidden_dim] 的张量。
            tokens_per_frame: 每帧经过空间池化后的视觉 token 数量。

        返回:
            (压缩后的token, 压缩统计信息)
        """

        self._validate_inputs(image_feat, tokens_per_frame)
        start = time.perf_counter()

        original_tokens = image_feat.shape[0]
        frame_num = original_tokens // tokens_per_frame
        # 将扁平 token 重排为 [frames, tokens, hidden]
        per_frame_tokens = image_feat.reshape(frame_num, tokens_per_frame, -1)

        # 第一阶段：帧内空间压缩
        selected_tokens = self._spatial_compress(per_frame_tokens)
        after_spatial_tokens = selected_tokens.shape[0]

        # 第二阶段：跨帧时间压缩
        if self.config.enable_temporal:
            selected_tokens = self._temporal_compress(selected_tokens)
        after_temporal_tokens = selected_tokens.shape[0]

        # 可选：将原始 token 合并回压缩 token
        if self.config.merge_original_tokens:
            selected_tokens = self._merge_original_tokens(image_feat, selected_tokens)

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        stats = CompressionStats(
            original_tokens=original_tokens,
            frames=frame_num,
            tokens_per_frame=tokens_per_frame,
            after_spatial_tokens=after_spatial_tokens,
            after_temporal_tokens=after_temporal_tokens,
            final_tokens=selected_tokens.shape[0],
            tau=self.config.tau,
            epsilon=self.config.epsilon,
            elapsed_ms=elapsed_ms,
        )
        return selected_tokens, stats

    def _spatial_compress(self, per_frame_tokens: torch.Tensor) -> torch.Tensor:
        """逐帧进行空间压缩：通过相似度 + SCC 聚类合并每个帧内的 token。"""
        fused_frames = []
        high_similarity = self._similarity_mask_per_frame(per_frame_tokens)

        for frame_idx, frame_tokens in enumerate(per_frame_tokens):
            components = approximate_components(
                high_similarity[frame_idx].cpu().numpy(),
                epsilon=self.config.epsilon,
            )
            fused_frames.append(self._mean_components(frame_tokens, components))

        return torch.cat(fused_frames, dim=0)

    def _temporal_compress(self, selected_tokens: torch.Tensor) -> torch.Tensor:
        """跨帧时间压缩：在所有空间压缩后的 token 上运行 SCC 聚类。"""
        high_similarity = self._similarity_mask(selected_tokens)
        components = approximate_components(
            high_similarity.cpu().numpy(),
            epsilon=self.config.epsilon,
        )
        return self._mean_components(selected_tokens, components)

    def _merge_original_tokens(
        self,
        original_tokens: torch.Tensor,
        selected_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """将每个原始 token 合并到余弦相似度最近的压缩 token 中。

        与原始代码保持相同的平均规则：压缩 token 本身被视为一份样本，
        原始 token 按其最近归属加回后再取平均。
        """
        selected_norm = self._safe_l2_normalize(selected_tokens, dim=1)
        original_norm = self._safe_l2_normalize(original_tokens, dim=1)
        similarity = torch.matmul(original_norm, selected_norm.t())
        closest_indices = torch.argmax(similarity, dim=1)

        merged_tokens = torch.zeros_like(selected_tokens)
        merged_tokens.scatter_add_(
            dim=0,
            index=closest_indices.view(-1, 1).expand(-1, selected_tokens.shape[1]),
            src=original_tokens,
        )

        # 计数 +1 是因为压缩 token 本身算一次
        counts = torch.bincount(closest_indices, minlength=selected_tokens.shape[0]).to(
            device=selected_tokens.device,
            dtype=selected_tokens.dtype,
        )
        counts = counts + 1
        return (selected_tokens + merged_tokens) / counts.unsqueeze(1)

    def _mean_components(
        self,
        tokens: torch.Tensor,
        components: list[list[int]],
    ) -> torch.Tensor:
        """对每个连通分量内的 token 做均值池化。"""
        fused = [tokens[component].mean(dim=0) for component in components]
        return torch.stack(fused, dim=0)

    def _similarity_mask_per_frame(self, per_frame_tokens: torch.Tensor) -> torch.Tensor:
        """计算每帧内部 token 的余弦相似度矩阵，返回超过阈值的布尔掩码。"""
        normalized = self._safe_l2_normalize(per_frame_tokens, dim=-1)
        similarity = torch.matmul(normalized, normalized.transpose(1, 2))
        return similarity > self.config.tau

    def _similarity_mask(self, tokens: torch.Tensor) -> torch.Tensor:
        """计算全局 token 的余弦相似度矩阵，返回超过阈值的布尔掩码。"""
        normalized = self._safe_l2_normalize(tokens, dim=1)
        similarity = torch.matmul(normalized, normalized.t())
        return similarity > self.config.tau

    def _safe_l2_normalize(self, tensor: torch.Tensor, dim: int) -> torch.Tensor:
        """安全的 L2 归一化，避免除零错误。"""
        norm = torch.norm(tensor, p=2, dim=dim, keepdim=True).clamp_min(1e-12)
        return tensor / norm

    def _validate_inputs(self, image_feat: torch.Tensor, tokens_per_frame: int) -> None:
        """验证输入张量的形状和参数合法性。"""
        if image_feat.ndim != 2:
            raise ValueError(f"image_feat 必须是二维张量，当前形状为 {tuple(image_feat.shape)}")
        if tokens_per_frame <= 0:
            raise ValueError("tokens_per_frame 必须为正整数")
        if image_feat.shape[0] % tokens_per_frame != 0:
            raise ValueError(
                "image_feat 的长度必须能被 tokens_per_frame 整除: "
                f"{image_feat.shape[0]} vs {tokens_per_frame}"
            )


def compress_flat_video_features(
    image_feat: torch.Tensor,
    ori_token_num: int,
    tau: float = 0.95,
    epsilon: float = 0.05,
    enable_temporal: bool = True,
    merge_original_tokens: bool = True,
) -> tuple[torch.Tensor, CompressionStats]:
    """为 llava_arch_zip.py 提供的兼容性封装函数。"""

    compressor = LlavaScissorCompressor(
        ScissorConfig(
            tau=tau,
            epsilon=epsilon,
            enable_temporal=enable_temporal,
            merge_original_tokens=merge_original_tokens,
        )
    )
    return compressor.compress_video_tokens(image_feat, tokens_per_frame=ori_token_num)
