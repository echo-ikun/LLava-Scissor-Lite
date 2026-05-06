"""Compatibility import path for official backend integrations."""

from compression import (  # noqa: F401
    CompressionStats,
    LlavaScissorCompressor,
    ScissorConfig,
    compress_flat_video_features,
)

__all__ = [
    "CompressionStats",
    "LlavaScissorCompressor",
    "ScissorConfig",
    "compress_flat_video_features",
]
