from dataclasses import dataclass


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
    """

    tau: float = 0.95
    epsilon: float = 0.05
    enable_temporal: bool = True
    merge_original_tokens: bool = True
