from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ScissorConfig:
    """LLaVA-Scissor token 压缩器的配置。

    参数:
        tau: 余弦相似度阈值，相似度 > tau 的 token 会被归入同一连通分量。
             值越高，合并条件越严格，压缩率越低。
        epsilon: 近似 SCC 算法的误差容忍度，控制采样数量。
                值越小，近似精度越高，但速度越慢。
        enable_temporal: 是否启用跨帧时间压缩（第二阶段）。
        merge_original_tokens: 是否在压缩后将原始 token 合并回压缩 token。
        adaptive_tau: 是否根据每帧 token 冗余度自适应调整空间 SCC 阈值。
        adaptive_tau_mode: `redundancy` 根据平均相似度调节；`quantile` 使用相似度分位数。
        adaptive_tau_strength: 冗余度模式下的调节强度，值越大每帧 tau 差异越明显。
        adaptive_tau_quantile: 分位数模式下使用的相似度分位数。
        adaptive_tau_min/max: 自适应 tau 的裁剪范围。
        component_merge: 连通分量内部合并方式。`mean` 保持原论文；`centrality` 使用中心性加权。
        component_merge_temperature: 中心性 softmax 温度。
        original_merge_strategy: 原始 token 回填方式。`hard` 保持原论文；`soft` 分配到 top-k 代表 token。
        soft_merge_topk: soft 回填时每个原始 token 分配到的代表 token 数。
        soft_merge_temperature: soft 回填的相似度 softmax 温度。
        temporal_strategy: `global` 保持原论文全局 temporal SCC；`windowed` 使用窗口化 temporal SCC。
        temporal_window_size: windowed temporal SCC 的窗口帧数。
        temporal_window_global_refine: 是否在窗口 SCC 后再做一次全局 SCC。
    """

    tau: float = 0.95
    epsilon: float = 0.05
    enable_temporal: bool = True
    merge_original_tokens: bool = True
    adaptive_tau: bool = False
    adaptive_tau_mode: Literal["redundancy", "quantile"] = "redundancy"
    adaptive_tau_strength: float = 0.25
    adaptive_tau_quantile: float = 0.85
    adaptive_tau_min: float = 0.50
    adaptive_tau_max: float = 0.995
    component_merge: Literal["mean", "centrality"] = "mean"
    component_merge_temperature: float = 0.07
    original_merge_strategy: Literal["hard", "soft"] = "hard"
    soft_merge_topk: int = 4
    soft_merge_temperature: float = 0.07
    temporal_strategy: Literal["global", "windowed"] = "global"
    temporal_window_size: int = 4
    temporal_window_global_refine: bool = False
