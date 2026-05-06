from dataclasses import dataclass


DEFAULT_CHECKPOINT = (
    "/home/zyk/.cache/huggingface/hub/"
    "models--BBBBCHAN--LLaVA-Scissor-baseline-0.5B/"
    "snapshots/2f2df55a6a20a712707aa62a1a309e0ffee3b912"
)


@dataclass
class ReproduceScissorConfig:
    """Runtime config for the slim LLaVA-Scissor reproduction runner."""

    backend: str = "official"
    checkpoint_path: str = DEFAULT_CHECKPOINT
    model_name: str = "llava_qwen_zip"
    conv_template: str = "qwen_2"
    device: str = "cuda"
    device_map: str = "auto"
    attn_implementation: str = "sdpa"
    max_frames: int = 4
    max_new_tokens: int = 16
    temperature: float = 0.0
    do_sample: bool = False
    scissor_adaptive_tau: bool = False
    scissor_adaptive_tau_mode: str = "redundancy"
    scissor_adaptive_tau_strength: float = 0.25
    scissor_adaptive_tau_quantile: float = 0.85
    scissor_adaptive_tau_min: float = 0.50
    scissor_adaptive_tau_max: float = 0.995
    scissor_component_merge: str = "mean"
    scissor_component_merge_temperature: float = 0.07
    scissor_original_merge_strategy: str = "hard"
    scissor_soft_merge_topk: int = 4
    scissor_soft_merge_temperature: float = 0.07
    scissor_temporal_strategy: str = "global"
    scissor_temporal_window_size: int = 4
    scissor_temporal_window_global_refine: bool = False
    offline: bool = True
    suppress_warnings: bool = True
    seed: int | None = None
