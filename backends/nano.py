import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

from compression import LlavaScissorCompressor, ScissorConfig

from config import ReproduceScissorConfig
from inference import GenerationResult, load_video_frames


_IMAGE_TOKEN = "<image>"
_CHAT_TEMPLATE = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n{image}\n{question}<|im_end|>\n"
    "<|im_start|>assistant\n"
)


class NanoScissorRunner:
    """Small standalone backend for understanding the full inference path.

    This backend is intentionally more self-contained than the official runner,
    but it is still an approximation. Use `backend=official` for faithful
    experiments and `backend=nano` for readable end-to-end code study.
    """

    def __init__(self, config: ReproduceScissorConfig | None = None):
        self.config = config or ReproduceScissorConfig()
        self.checkpoint_path = self.config.checkpoint_path
        self.device = self.config.device
        self.dtype = torch.float16

        self.cfg = None
        self.vision_tower = None
        self.projector = None
        self.llm = None
        self.tokenizer = None
        self.compressor = LlavaScissorCompressor(self._scissor_config())

        if self.config.offline:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    def _scissor_config(self) -> ScissorConfig:
        return ScissorConfig(
            adaptive_tau=self.config.scissor_adaptive_tau,
            adaptive_tau_mode=self.config.scissor_adaptive_tau_mode,
            adaptive_tau_strength=self.config.scissor_adaptive_tau_strength,
            adaptive_tau_quantile=self.config.scissor_adaptive_tau_quantile,
            adaptive_tau_min=self.config.scissor_adaptive_tau_min,
            adaptive_tau_max=self.config.scissor_adaptive_tau_max,
            component_merge=self.config.scissor_component_merge,
            component_merge_temperature=self.config.scissor_component_merge_temperature,
            original_merge_strategy=self.config.scissor_original_merge_strategy,
            soft_merge_topk=self.config.scissor_soft_merge_topk,
            soft_merge_temperature=self.config.scissor_soft_merge_temperature,
            temporal_strategy=self.config.scissor_temporal_strategy,
            temporal_window_size=self.config.scissor_temporal_window_size,
            temporal_window_global_refine=self.config.scissor_temporal_window_global_refine,
        )

    def load(self) -> None:
        if self.llm is not None:
            return

        with open(os.path.join(self.checkpoint_path, "config.json")) as f:
            self.cfg = json.load(f)

        state = load_file(os.path.join(self.checkpoint_path, "model.safetensors"))
        self.vision_tower = self._build_vision_tower(state)
        self.projector = self._build_projector(state)
        self.llm, self.tokenizer = self._build_llm(state)

    @torch.no_grad()
    def generate(
        self,
        video_path: str,
        question: str,
        max_frames: int | None = None,
        max_new_tokens: int | None = None,
    ) -> GenerationResult:
        self.load()
        start = time.perf_counter()
        frames = load_video_frames(video_path, max_frames or self.config.max_frames)
        visual_tokens = self.encode_vision(frames)
        prompt = _CHAT_TEMPLATE.format(image=_IMAGE_TOKEN, question=question)
        input_embeds, attention_mask = self._build_input_embeds(prompt, visual_tokens)

        output_ids = self.llm.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens or self.config.max_new_tokens,
            do_sample=self.config.do_sample,
            temperature=self.config.temperature if self.config.temperature > 0 else 1.0,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        output_ids = output_ids[0]
        if output_ids.shape[0] >= input_embeds.shape[1]:
            output_ids = output_ids[input_embeds.shape[1] :]
        text = self.tokenizer.decode(output_ids, skip_special_tokens=True)

        return GenerationResult(
            text=text.strip(),
            elapsed_s=time.perf_counter() - start,
            frame_shape=tuple(frames.shape),
            prompt=prompt,
            backend="nano",
        )

    def encode_vision(self, frames) -> torch.Tensor:
        pixel_values = self._preprocess_frames(frames)
        out = self.vision_tower(pixel_values, output_hidden_states=True)
        features = out.hidden_states[-1]

        # Match the official path more closely: projector first, then 2D pool.
        projected = self.projector(features)
        pooled = self._bilinear_pool(projected, stride=2)
        flat = pooled.reshape(-1, pooled.shape[-1])
        compressed, _ = self.compressor.compress_video_tokens(flat, pooled.shape[1])
        return compressed

    def _build_vision_tower(self, state):
        from transformers import SiglipVisionModel

        siglip_path = self.cfg["mm_vision_tower"]
        model = SiglipVisionModel.from_pretrained(siglip_path)

        vision_state = {}
        prefix = "model.vision_tower.vision_tower."
        for key, value in state.items():
            if key.startswith(prefix):
                vision_state[key.removeprefix(prefix)] = value
        if vision_state:
            model.load_state_dict(vision_state, strict=False)

        del model.vision_model.encoder.layers[-1:]
        model.vision_model.head = nn.Identity()
        model.requires_grad_(False).eval()
        return model.to(self.device, dtype=self.dtype)

    def _build_projector(self, state):
        in_dim = self.cfg["mm_hidden_size"]
        out_dim = self.cfg["hidden_size"]
        projector = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )
        projector[0].weight.data = state["model.mm_projector.0.weight"].to(self.dtype)
        projector[0].bias.data = state["model.mm_projector.0.bias"].to(self.dtype)
        projector[2].weight.data = state["model.mm_projector.2.weight"].to(self.dtype)
        projector[2].bias.data = state["model.mm_projector.2.bias"].to(self.dtype)
        return projector.to(self.device, dtype=self.dtype).eval()

    def _build_llm(self, state):
        from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

        cfg = self.cfg
        qwen_cfg = Qwen2Config(
            vocab_size=cfg["vocab_size"],
            hidden_size=cfg["hidden_size"],
            intermediate_size=cfg["intermediate_size"],
            num_hidden_layers=cfg["num_hidden_layers"],
            num_attention_heads=cfg["num_attention_heads"],
            num_key_value_heads=cfg["num_key_value_heads"],
            max_position_embeddings=cfg["max_position_embeddings"],
            rms_norm_eps=cfg["rms_norm_eps"],
            tie_word_embeddings=cfg.get("tie_word_embeddings", True),
            rope_theta=cfg["rope_theta"],
            bos_token_id=cfg["bos_token_id"],
            eos_token_id=cfg["eos_token_id"],
            use_sliding_window=cfg.get("use_sliding_window", False),
        )
        llm = Qwen2ForCausalLM(qwen_cfg)
        llm_state = {
            key: value.to(self.dtype)
            for key, value in state.items()
            if not key.startswith("model.mm_projector.") and not key.startswith("model.vision_tower.")
        }
        missing, unexpected = llm.load_state_dict(llm_state, strict=False)
        important_missing = [key for key in missing if not key.startswith("lm_head.")]
        if important_missing or unexpected:
            print(f"[nano] load_state_dict missing={important_missing[:5]} unexpected={unexpected[:5]}")
        llm.eval()
        tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_path)
        return llm.to(self.device, dtype=self.dtype), tokenizer

    def _preprocess_frames(self, frames) -> torch.Tensor:
        tensor = torch.from_numpy(frames).float().permute(0, 3, 1, 2)
        tensor = tensor / 255.0
        tensor = F.interpolate(tensor, size=(384, 384), mode="bicubic", align_corners=False)
        tensor = (tensor - 0.5) / 0.5
        return tensor.to(self.device, dtype=self.dtype)

    def _bilinear_pool(self, features: torch.Tensor, stride: int) -> torch.Tensor:
        frames, tokens, hidden = features.shape
        side = int(tokens**0.5)
        features = features.view(frames, side, side, hidden).permute(0, 3, 1, 2).contiguous()
        pooled_side = (side + stride - 1) // stride
        features = F.interpolate(features, size=(pooled_side, pooled_side), mode="bilinear")
        return features.permute(0, 2, 3, 1).reshape(frames, pooled_side * pooled_side, hidden)

    def _build_input_embeds(self, prompt: str, visual_tokens: torch.Tensor):
        parts = prompt.split(_IMAGE_TOKEN)
        embeds = []
        for idx, part in enumerate(parts):
            if part:
                tokenized = self.tokenizer(part, return_tensors="pt", add_special_tokens=False)
                ids = tokenized.input_ids.to(self.device)
                embeds.append(self.llm.model.embed_tokens(ids).to(visual_tokens.dtype))
            if idx < len(parts) - 1:
                embeds.append(visual_tokens.unsqueeze(0))

        input_embeds = torch.cat(embeds, dim=1)
        attention_mask = torch.ones(1, input_embeds.shape[1], device=self.device, dtype=torch.long)
        return input_embeds, attention_mask
