# ScissorLite Improvement Switches

All improvements are training-free and disabled by default. The default configuration remains numerically equivalent to the original LLaVA-Scissor path.

## Switch Summary

| Improvement | Switches | Default |
|---|---|---|
| Frame-adaptive SCC threshold | `adaptive_tau`, `adaptive_tau_mode`, `adaptive_tau_strength`, `adaptive_tau_quantile`, `adaptive_tau_min`, `adaptive_tau_max` | off |
| Centrality-weighted component merge | `component_merge="centrality"` | `mean` |
| Soft original-token refill | `original_merge_strategy="soft"`, `soft_merge_topk`, `soft_merge_temperature` | `hard` |
| Windowed temporal SCC | `temporal_strategy="windowed"`, `temporal_window_size`, `temporal_window_global_refine` | `global` |

## Direct Python Use

```python
from compression import compress_flat_video_features

compressed, stats = compress_flat_video_features(
    image_feat,
    ori_token_num=196,
    tau=0.95,
    epsilon=0.05,
    adaptive_tau=True,
    component_merge="centrality",
    original_merge_strategy="soft",
    temporal_strategy="windowed",
    temporal_window_size=4,
)
```

## CLI Use

The Lite `inference.py` exposes the same switches and writes them into the official model config before generation.

```bash
python inference.py \
  --backend official \
  --video /path/to/video.mp4 \
  --question "What happens in the video?" \
  --scissor-adaptive-tau \
  --scissor-component-merge centrality \
  --scissor-original-merge-strategy soft \
  --scissor-temporal-strategy windowed \
  --scissor-temporal-window-size 4
```

The official LLaVA backend reads the matching model config fields:

```text
mm_zip_adaptive_tau
mm_zip_adaptive_tau_mode
mm_zip_adaptive_tau_strength
mm_zip_adaptive_tau_quantile
mm_zip_adaptive_tau_min
mm_zip_adaptive_tau_max
mm_zip_component_merge
mm_zip_component_merge_temperature
mm_zip_original_merge_strategy
mm_zip_soft_merge_topk
mm_zip_soft_merge_temperature
mm_zip_temporal_strategy
mm_zip_temporal_window_size
mm_zip_temporal_window_global_refine
```

For compatibility with the official code path, the repo now provides
`reproduce_scissor.compression`, which re-exports the ScissorLite compression
implementation.

## Suggested Ablations

| Name | adaptive_tau | component_merge | original_merge_strategy | temporal_strategy |
|---|---|---|---|---|
| Scissor | False | mean | hard | global |
| A | True | mean | hard | global |
| W | False | centrality | hard | global |
| S | False | mean | soft | global |
| T | False | mean | hard | windowed |
| AWS | True | centrality | soft | global |
| AWST | True | centrality | soft | windowed |

## Verification

Use the project environment with PyTorch installed:

```bash
/home/zyk/miniconda3/envs/tinyllava_video/bin/python -m compression.parity_test --device cpu
/home/zyk/miniconda3/envs/tinyllava_video/bin/python -m compression.smoke_test --device cpu
```
