import argparse
import copy
import os
import sys
import time
import warnings
from dataclasses import dataclass

warnings.filterwarnings("ignore")

import numpy as np
import torch

from config import DEFAULT_CHECKPOINT, ReproduceScissorConfig


@dataclass
class GenerationResult:
    """Output returned by a ScissorLite backend."""

    text: str
    elapsed_s: float
    frame_shape: tuple[int, ...]
    prompt: str
    backend: str


def load_video_frames(video_path: str, max_frames: int) -> np.ndarray:
    """Uniformly sample video frames as `[frames, height, width, channels]`."""

    if max_frames <= 0:
        raise ValueError("max_frames must be positive")

    from decord import VideoReader, cpu

    vr = VideoReader(video_path, ctx=cpu(0))
    if len(vr) == 0:
        raise ValueError(f"video has no frames: {video_path}")

    indices = np.linspace(0, len(vr) - 1, max_frames, dtype=int).tolist()
    return vr.get_batch(indices).asnumpy()


class OfficialScissorRunner:
    """Faithful runner that reuses the original LLaVA-Scissor stack."""

    def __init__(self, config: ReproduceScissorConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.image_processor = None

        if config.offline:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    def load(self) -> None:
        if self.model is not None:
            return

        from llava.model.builder import load_pretrained_model

        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            self.config.checkpoint_path,
            None,
            self.config.model_name,
            device_map=self.config.device_map,
            attn_implementation=self.config.attn_implementation,
        )
        self._apply_scissor_config()
        self.model.eval()

    @torch.no_grad()
    def generate(self, video_path: str, question: str) -> GenerationResult:
        self.load()
        self._set_seed()
        start = time.perf_counter()

        frames = load_video_frames(video_path, self.config.max_frames)
        image_tensor = self.image_processor.preprocess(frames, return_tensors="pt")["pixel_values"]
        image_tensor = image_tensor.half().to(self.config.device)

        prompt = self._build_prompt(question)
        input_ids = self._image_prompt_ids(prompt)
        output_ids = self.model.generate(
            input_ids,
            images=[image_tensor],
            image_sizes=[frame.size for frame in frames],
            do_sample=self.config.do_sample,
            temperature=self.config.temperature,
            max_new_tokens=self.config.max_new_tokens,
            modalities=["video"],
        )
        text = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        return GenerationResult(text, time.perf_counter() - start, tuple(frames.shape), prompt, "official")

    def _build_prompt(self, question: str) -> str:
        from llava.constants import DEFAULT_IMAGE_TOKEN
        from llava.conversation import conv_templates

        conv = copy.deepcopy(conv_templates[self.config.conv_template])
        conv.append_message(conv.roles[0], f"{DEFAULT_IMAGE_TOKEN}\n{question}")
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()

    def _image_prompt_ids(self, prompt: str) -> torch.Tensor:
        from llava.constants import IMAGE_TOKEN_INDEX
        from llava.mm_utils import tokenizer_image_token

        return tokenizer_image_token(
            prompt,
            self.tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        ).unsqueeze(0).to(self.config.device)

    def _set_seed(self) -> None:
        if self.config.seed is None:
            return
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)

    def _apply_scissor_config(self) -> None:
        """Expose Lite improvement switches to the official LLaVA-Scissor model config."""
        if self.model is None:
            return
        model_config = self.model.config
        model_config.mm_zip_adaptive_tau = self.config.scissor_adaptive_tau
        model_config.mm_zip_adaptive_tau_mode = self.config.scissor_adaptive_tau_mode
        model_config.mm_zip_adaptive_tau_strength = self.config.scissor_adaptive_tau_strength
        model_config.mm_zip_adaptive_tau_quantile = self.config.scissor_adaptive_tau_quantile
        model_config.mm_zip_adaptive_tau_min = self.config.scissor_adaptive_tau_min
        model_config.mm_zip_adaptive_tau_max = self.config.scissor_adaptive_tau_max
        model_config.mm_zip_component_merge = self.config.scissor_component_merge
        model_config.mm_zip_component_merge_temperature = self.config.scissor_component_merge_temperature
        model_config.mm_zip_original_merge_strategy = self.config.scissor_original_merge_strategy
        model_config.mm_zip_soft_merge_topk = self.config.scissor_soft_merge_topk
        model_config.mm_zip_soft_merge_temperature = self.config.scissor_soft_merge_temperature
        model_config.mm_zip_temporal_strategy = self.config.scissor_temporal_strategy
        model_config.mm_zip_temporal_window_size = self.config.scissor_temporal_window_size
        model_config.mm_zip_temporal_window_global_refine = self.config.scissor_temporal_window_global_refine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Faithful slim LLaVA-Scissor inference")
    parser.add_argument("--backend", choices=["official", "nano"], default="official")
    parser.add_argument("--video", required=True, help="Path to the input video")
    parser.add_argument("--question", default="Describe this video in detail.")
    parser.add_argument("--checkpoint", default=os.environ.get("REPRODUCE_SCISSOR_CHECKPOINT", DEFAULT_CHECKPOINT))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-frames", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--online", action="store_true", help="Allow Hugging Face online lookup/downloads")
    parser.add_argument("--scissor-adaptive-tau", action="store_true")
    parser.add_argument("--scissor-adaptive-tau-mode", choices=["redundancy", "quantile"], default="redundancy")
    parser.add_argument("--scissor-adaptive-tau-strength", type=float, default=0.25)
    parser.add_argument("--scissor-adaptive-tau-quantile", type=float, default=0.85)
    parser.add_argument("--scissor-adaptive-tau-min", type=float, default=0.50)
    parser.add_argument("--scissor-adaptive-tau-max", type=float, default=0.995)
    parser.add_argument("--scissor-component-merge", choices=["mean", "centrality"], default="mean")
    parser.add_argument("--scissor-component-merge-temperature", type=float, default=0.07)
    parser.add_argument("--scissor-original-merge-strategy", choices=["hard", "soft"], default="hard")
    parser.add_argument("--scissor-soft-merge-topk", type=int, default=4)
    parser.add_argument("--scissor-soft-merge-temperature", type=float, default=0.07)
    parser.add_argument("--scissor-temporal-strategy", choices=["global", "windowed"], default="global")
    parser.add_argument("--scissor-temporal-window-size", type=int, default=4)
    parser.add_argument("--scissor-temporal-window-global-refine", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not os.path.exists(args.video):
        sys.exit(f"video not found: {args.video}")
    if not os.path.isdir(args.checkpoint):
        sys.exit(f"checkpoint not found: {args.checkpoint}")

    config = ReproduceScissorConfig(
        backend=args.backend,
        checkpoint_path=args.checkpoint,
        device=args.device,
        max_frames=args.max_frames,
        max_new_tokens=args.max_new_tokens,
        scissor_adaptive_tau=args.scissor_adaptive_tau,
        scissor_adaptive_tau_mode=args.scissor_adaptive_tau_mode,
        scissor_adaptive_tau_strength=args.scissor_adaptive_tau_strength,
        scissor_adaptive_tau_quantile=args.scissor_adaptive_tau_quantile,
        scissor_adaptive_tau_min=args.scissor_adaptive_tau_min,
        scissor_adaptive_tau_max=args.scissor_adaptive_tau_max,
        scissor_component_merge=args.scissor_component_merge,
        scissor_component_merge_temperature=args.scissor_component_merge_temperature,
        scissor_original_merge_strategy=args.scissor_original_merge_strategy,
        scissor_soft_merge_topk=args.scissor_soft_merge_topk,
        scissor_soft_merge_temperature=args.scissor_soft_merge_temperature,
        scissor_temporal_strategy=args.scissor_temporal_strategy,
        scissor_temporal_window_size=args.scissor_temporal_window_size,
        scissor_temporal_window_global_refine=args.scissor_temporal_window_global_refine,
        offline=not args.online,
        seed=args.seed,
    )
    if args.backend == "official":
        runner_cls = OfficialScissorRunner
    else:
        from backends.nano import NanoScissorRunner

        runner_cls = NanoScissorRunner
    runner = runner_cls(config)
    result = runner.generate(args.video, args.question)

    print(f"backend: {result.backend}")
    print(f"frames: {result.frame_shape}")
    print(f"elapsed_s: {result.elapsed_s:.2f}")
    print(result.text)


if __name__ == "__main__":
    main()
