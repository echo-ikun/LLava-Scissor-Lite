from dataclasses import dataclass
from typing import Tuple


@dataclass
class CompressionStats:
    """压缩过程的运行时统计信息，用于调试和 A/B 对照。"""

    original_tokens: int
    frames: int
    tokens_per_frame: int
    after_spatial_tokens: int
    after_temporal_tokens: int
    final_tokens: int
    tau: float
    epsilon: float
    elapsed_ms: float
    spatial_frame_tokens: Tuple[int, ...] = ()
    spatial_frame_taus: Tuple[float, ...] = ()
    temporal_strategy: str = "global"
    component_merge: str = "mean"
    original_merge_strategy: str = "hard"
