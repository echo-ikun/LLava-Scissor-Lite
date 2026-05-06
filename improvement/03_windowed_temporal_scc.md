# Improvement 03: Windowed Temporal SCC

## Motivation

Original LLaVA-Scissor performs temporal SCC globally after spatial compression. This removes cross-frame redundancy, but it can also merge visually similar tokens that belong to different events in a long video.

For example, the same object can appear before and after a state change. Global temporal SCC may merge them because they look similar, even though the temporal distinction is needed for the answer.

This improvement makes temporal SCC windowed.

## References

- LLaVA-Scissor: two-stage spatial and temporal SCC.  
  https://arxiv.org/abs/2506.21862
- AOT: uses temporal clips/keyframe anchors and preserves distinct tokens for temporal dynamics.  
  https://arxiv.org/abs/2603.01400
- KiToke: uses interval-aware merging to preserve temporal coherence.  
  https://arxiv.org/abs/2604.03414
- ForestPrune: adds semantic, spatial, and temporal constraints through forest modeling.  
  https://arxiv.org/abs/2603.22911
- Unified STC: highlights that low-budget compression can lose QA evidence if allocation is not safe.  
  https://arxiv.org/abs/2603.21957

## Implementation

Files changed:

- `compression/config.py`
- `compression/compressor.py`
- `compression/stats.py`
- `compression/smoke_test.py`

Switches:

```python
temporal_strategy: Literal["global", "windowed"] = "global"
temporal_window_size: int = 4
temporal_window_global_refine: bool = False
```

Default behavior is unchanged: `temporal_strategy="global"` uses the original temporal SCC over all spatially compressed tokens.

## Method

The spatial stage now records how many compressed tokens remain per frame:

```text
spatial_frame_tokens = [n_1, n_2, ..., n_F]
```

Windowed temporal SCC groups adjacent frames:

```text
frames 0..W-1 -> temporal SCC
frames W..2W-1 -> temporal SCC
...
```

Optionally, a second global refinement can run on the concatenated window outputs:

```python
temporal_window_global_refine=True
```

This retains the original Scissor behavior as a special case:

```text
temporal_window_size >= number_of_frames
```

## Ablation

Suggested experiments:

| Setting | temporal_strategy | window_size | global_refine |
|---|---|---:|---|
| Baseline Scissor | global | - | - |
| Ours-T2 | windowed | 2 | False |
| Ours-T4 | windowed | 4 | False |
| Ours-T8 | windowed | 8 | False |
| Ours-T4G | windowed | 4 | True |

Long-video benchmarks should show the clearest difference:

- EgoSchema
- LongVideoBench
- MLVU
- Video-MME medium/long split

## Expected Benefit

- Reduces long-range false merges.
- Keeps event boundaries safer.
- Makes temporal SCC more scalable for long videos.
- Provides a clean path toward future content-adaptive intervals.

## Risks

- Too small a window may miss true long-range redundancy.
- Too large a window degenerates to original global temporal SCC.
- `global_refine=True` can reintroduce long-range false merges if not combined with a stricter threshold.
