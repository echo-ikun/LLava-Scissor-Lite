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
        selected_tokens, spatial_frame_tokens, spatial_frame_taus = self._spatial_compress(per_frame_tokens)
        after_spatial_tokens = selected_tokens.shape[0]

        # 第二阶段：跨帧时间压缩
        if self.config.enable_temporal:
            if self.config.temporal_strategy == "windowed":
                selected_tokens = self._windowed_temporal_compress(
                    selected_tokens,
                    spatial_frame_tokens,
                )
            else:
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
            spatial_frame_tokens=tuple(spatial_frame_tokens),
            spatial_frame_taus=tuple(spatial_frame_taus),
            temporal_strategy=self.config.temporal_strategy,
            component_merge=self.config.component_merge,
            original_merge_strategy=self.config.original_merge_strategy,
        )
        return selected_tokens, stats

    def _spatial_compress(self, per_frame_tokens: torch.Tensor) -> tuple[torch.Tensor, list[int], list[float]]:
        """逐帧进行空间压缩：通过相似度 + SCC 聚类合并每个帧内的 token。"""
        fused_frames = []
        frame_token_counts: list[int] = []
        high_similarity, frame_taus = self._similarity_mask_per_frame(per_frame_tokens)

        for frame_idx, frame_tokens in enumerate(per_frame_tokens):
            components = approximate_components(
                high_similarity[frame_idx].cpu().numpy(),
                epsilon=self.config.epsilon,
            )
            fused = self._merge_components(frame_tokens, components)
            fused_frames.append(fused)
            frame_token_counts.append(fused.shape[0])

        return torch.cat(fused_frames, dim=0), frame_token_counts, frame_taus

    def _temporal_compress(self, selected_tokens: torch.Tensor) -> torch.Tensor:
        """跨帧时间压缩：在所有空间压缩后的 token 上运行 SCC 聚类。"""
        high_similarity = self._similarity_mask(selected_tokens)
        components = approximate_components(
            high_similarity.cpu().numpy(),
            epsilon=self.config.epsilon,
        )
        return self._merge_components(selected_tokens, components)

    def _windowed_temporal_compress(
        self,
        selected_tokens: torch.Tensor,
        frame_token_counts: list[int],
    ) -> torch.Tensor:
        """窗口化时间压缩，降低长视频远距离误合并风险。"""
        window_size = max(1, self.config.temporal_window_size)
        fused_windows = []
        start = 0

        for frame_start in range(0, len(frame_token_counts), window_size):
            window_counts = frame_token_counts[frame_start:frame_start + window_size]
            window_token_count = sum(window_counts)
            window_tokens = selected_tokens[start:start + window_token_count]
            start += window_token_count

            if window_tokens.shape[0] == 0:
                continue
            fused_windows.append(self._temporal_compress(window_tokens))

        if not fused_windows:
            return selected_tokens

        fused_tokens = torch.cat(fused_windows, dim=0)
        if self.config.temporal_window_global_refine and fused_tokens.shape[0] > 1:
            fused_tokens = self._temporal_compress(fused_tokens)
        return fused_tokens

    def _merge_original_tokens(
        self,
        original_tokens: torch.Tensor,
        selected_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """将每个原始 token 合并回压缩 token。"""
        if self.config.original_merge_strategy == "soft":
            return self._soft_merge_original_tokens(original_tokens, selected_tokens)
        return self._hard_merge_original_tokens(original_tokens, selected_tokens)

    def _hard_merge_original_tokens(
        self,
        original_tokens: torch.Tensor,
        selected_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """原始 LLaVA-Scissor 最近邻平均回填。"""
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

    def _soft_merge_original_tokens(
        self,
        original_tokens: torch.Tensor,
        selected_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """将原始 token 按相似度软分配到 top-k 压缩 token。"""
        selected_norm = self._safe_l2_normalize(selected_tokens, dim=1)
        original_norm = self._safe_l2_normalize(original_tokens, dim=1)
        similarity = torch.matmul(original_norm, selected_norm.t())

        topk = max(1, min(self.config.soft_merge_topk, selected_tokens.shape[0]))
        top_values, top_indices = torch.topk(similarity, k=topk, dim=1)
        temperature = max(self.config.soft_merge_temperature, 1e-6)
        weights = torch.softmax((top_values / temperature).float(), dim=1).to(original_tokens.dtype)

        expanded_indices = top_indices.reshape(-1, 1).expand(-1, selected_tokens.shape[1])
        weighted_tokens = (
            original_tokens.unsqueeze(1) * weights.unsqueeze(2)
        ).reshape(-1, selected_tokens.shape[1])

        merged_tokens = torch.zeros_like(selected_tokens)
        merged_tokens.scatter_add_(dim=0, index=expanded_indices, src=weighted_tokens)

        counts = torch.zeros(
            selected_tokens.shape[0],
            device=selected_tokens.device,
            dtype=selected_tokens.dtype,
        )
        counts.scatter_add_(dim=0, index=top_indices.reshape(-1), src=weights.reshape(-1))
        return (selected_tokens + merged_tokens) / (counts + 1).unsqueeze(1)

    def _merge_components(
        self,
        tokens: torch.Tensor,
        components: list[list[int]],
    ) -> torch.Tensor:
        """合并连通分量，默认保持原论文均值池化。"""
        if self.config.component_merge == "centrality":
            fused = [self._centrality_merge(tokens[component]) for component in components]
        else:
            fused = [tokens[component].mean(dim=0) for component in components]
        return torch.stack(fused, dim=0)

    def _centrality_merge(self, component_tokens: torch.Tensor) -> torch.Tensor:
        """按 token 到 component prototype 的中心性加权合并。"""
        if component_tokens.shape[0] == 1:
            return component_tokens[0]

        prototype = component_tokens.mean(dim=0, keepdim=True)
        token_norm = self._safe_l2_normalize(component_tokens, dim=1)
        prototype_norm = self._safe_l2_normalize(prototype, dim=1)
        centrality = torch.matmul(token_norm, prototype_norm.t()).squeeze(1)
        temperature = max(self.config.component_merge_temperature, 1e-6)
        weights = torch.softmax((centrality / temperature).float(), dim=0).to(component_tokens.dtype)
        return torch.sum(component_tokens * weights.unsqueeze(1), dim=0)

    def _similarity_mask_per_frame(self, per_frame_tokens: torch.Tensor) -> tuple[torch.Tensor, list[float]]:
        """计算每帧内部 token 相似度掩码，并返回实际使用的 tau。"""
        normalized = self._safe_l2_normalize(per_frame_tokens, dim=-1)
        similarity = torch.matmul(normalized, normalized.transpose(1, 2))
        frame_taus = self._frame_taus(similarity)
        tau_tensor = torch.tensor(
            frame_taus,
            device=similarity.device,
            dtype=similarity.dtype,
        ).view(-1, 1, 1)
        return similarity > tau_tensor, frame_taus

    def _similarity_mask(self, tokens: torch.Tensor) -> torch.Tensor:
        """计算全局 token 的余弦相似度矩阵，返回超过阈值的布尔掩码。"""
        normalized = self._safe_l2_normalize(tokens, dim=1)
        similarity = torch.matmul(normalized, normalized.t())
        return similarity > self.config.tau

    def _frame_taus(self, similarity: torch.Tensor) -> list[float]:
        """根据帧内相似度分布得到每帧 SCC 阈值。"""
        frame_count, token_count, _ = similarity.shape
        if not self.config.adaptive_tau:
            return [float(self.config.tau)] * frame_count

        if token_count <= 1:
            return [float(self.config.tau)] * frame_count

        off_diagonal = ~torch.eye(token_count, dtype=torch.bool, device=similarity.device)
        frame_taus: list[float] = []

        for frame_idx in range(frame_count):
            values = similarity[frame_idx][off_diagonal].float()
            if self.config.adaptive_tau_mode == "quantile":
                tau = torch.quantile(values, self.config.adaptive_tau_quantile).item()
            else:
                mean_similarity = values.mean().item()
                tau = self.config.tau + self.config.adaptive_tau_strength * (
                    self.config.tau - mean_similarity
                )
            tau = min(max(tau, self.config.adaptive_tau_min), self.config.adaptive_tau_max)
            frame_taus.append(float(tau))

        return frame_taus

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
    adaptive_tau: bool = False,
    adaptive_tau_mode: str = "redundancy",
    adaptive_tau_strength: float = 0.25,
    adaptive_tau_quantile: float = 0.85,
    adaptive_tau_min: float = 0.50,
    adaptive_tau_max: float = 0.995,
    component_merge: str = "mean",
    component_merge_temperature: float = 0.07,
    original_merge_strategy: str = "hard",
    soft_merge_topk: int = 4,
    soft_merge_temperature: float = 0.07,
    temporal_strategy: str = "global",
    temporal_window_size: int = 4,
    temporal_window_global_refine: bool = False,
) -> tuple[torch.Tensor, CompressionStats]:
    """为 llava_arch_zip.py 提供的兼容性封装函数。"""

    compressor = LlavaScissorCompressor(
        ScissorConfig(
            tau=tau,
            epsilon=epsilon,
            enable_temporal=enable_temporal,
            merge_original_tokens=merge_original_tokens,
            adaptive_tau=adaptive_tau,
            adaptive_tau_mode=adaptive_tau_mode,
            adaptive_tau_strength=adaptive_tau_strength,
            adaptive_tau_quantile=adaptive_tau_quantile,
            adaptive_tau_min=adaptive_tau_min,
            adaptive_tau_max=adaptive_tau_max,
            component_merge=component_merge,
            component_merge_temperature=component_merge_temperature,
            original_merge_strategy=original_merge_strategy,
            soft_merge_topk=soft_merge_topk,
            soft_merge_temperature=soft_merge_temperature,
            temporal_strategy=temporal_strategy,
            temporal_window_size=temporal_window_size,
            temporal_window_global_refine=temporal_window_global_refine,
        )
    )
    return compressor.compress_video_tokens(image_feat, tokens_per_frame=ori_token_num)
