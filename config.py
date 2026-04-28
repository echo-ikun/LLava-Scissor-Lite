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
    offline: bool = True
    suppress_warnings: bool = True
    seed: int | None = None
