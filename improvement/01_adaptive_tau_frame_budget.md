# Improvement 01: Frame-Adaptive SCC Threshold

## Motivation

LLaVA-Scissor uses a fixed cosine threshold `tau` to build the SCC graph for every frame. This is faithful to the original method, but it assumes that all frames should be compressed with the same semantic granularity. In video, this is rarely true: static/background-heavy frames can be compressed aggressively, while motion-heavy or visually dense frames should keep more components.

This improvement keeps the original SCC idea but replaces the single spatial threshold with per-frame thresholds `tau_f`.

## References

- LLaVA-Scissor: SCC uses token similarity and `mm_zip_tau` to control connected components and retention.  
  https://arxiv.org/abs/2506.21862  
  https://github.com/HumanMLLM/LLaVA-Scissor
- DyToK: motivates dynamic per-frame token retention using keyframe priors.  
  https://arxiv.org/abs/2512.06866
- KiToke: motivates content-adaptive token selection from global redundancy/diversity.  
  https://arxiv.org/abs/2604.03414
- Unified STC: motivates treating video compression as a global allocation problem under low token budgets.  
  https://arxiv.org/abs/2603.21957

## Implementation

Files changed:

- `compression/config.py`
- `compression/compressor.py`
- `compression/stats.py`
- `compression/smoke_test.py`

Switches:

```python
adaptive_tau: bool = False
adaptive_tau_mode: Literal["redundancy", "quantile"] = "redundancy"
adaptive_tau_strength: float = 0.25
adaptive_tau_quantile: float = 0.85
adaptive_tau_min: float = 0.50
adaptive_tau_max: float = 0.995
```

Default behavior is unchanged: `adaptive_tau=False` makes every frame use `tau`.

## Method

For each frame, compute the off-diagonal cosine similarity distribution.

`redundancy` mode:

```text
tau_f = tau + strength * (tau - mean_similarity_f)
```

The intuition is:

- if a frame is highly redundant, its mean similarity is high and the threshold becomes lower or near the base threshold;
- if a frame is visually diverse, its mean similarity is low and the threshold rises, preserving more components.

`quantile` mode:

```text
tau_f = quantile(off_diagonal_similarity_f, q)
```

This directly derives the graph threshold from the frame's similarity distribution.

Both modes clamp `tau_f` into `[adaptive_tau_min, adaptive_tau_max]`.

## Ablation

Suggested experiments:

| Setting | adaptive_tau | mode | strength / q |
|---|---|---|---|
| Baseline Scissor | False | - | - |
| Ours-A1 | True | redundancy | 0.15 |
| Ours-A2 | True | redundancy | 0.25 |
| Ours-A3 | True | quantile | 0.85 |
| Ours-A4 | True | quantile | 0.90 |

Report per-frame tokens and per-frame `tau_f` in addition to accuracy.

## Expected Benefit

- Better content adaptation without changing the LLM.
- Less over-compression on complex frames.
- Less wasted budget on static frames.
- Maintains the original LLaVA-Scissor SCC interpretation.

## Risks

- If `tau_f` becomes too high on complex frames, compression may become weak and latency gains shrink.
- If `tau_f` becomes too low on static frames, small but important details can be merged away.
- Query-specific tasks may still need query-aware budget allocation; this method is content-aware, not query-aware.
