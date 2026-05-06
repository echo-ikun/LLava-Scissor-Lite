# Improvement 02: Centrality-Weighted Merge and Soft Original-Token Refill

## Motivation

Original LLaVA-Scissor merges each SCC component with a simple mean, then assigns each original token to its nearest compressed token for another average merge. This is clean and faithful, but two information-loss points remain:

1. Tokens inside a component are not equally representative.
2. Hard nearest-neighbor refill forces each original token to contribute to only one compressed token.

This improvement keeps SCC components but changes the aggregation rule.

## References

- LLaVA-Scissor: SCC aims to preserve semantic coverage rather than only salient tokens.  
  https://arxiv.org/abs/2506.21862
- AOT: uses optimal transport to aggregate information from pruned tokens back into anchors.  
  https://arxiv.org/abs/2603.01400
- Unified STC: merges unselected tokens and refills the retained token pool for information integrity.  
  https://arxiv.org/abs/2603.21957
- ForestPrune: motivates treating tokens differently according to their structural role.  
  https://arxiv.org/abs/2603.22911

## Implementation

Files changed:

- `compression/config.py`
- `compression/compressor.py`
- `compression/stats.py`
- `compression/smoke_test.py`

Switches:

```python
component_merge: Literal["mean", "centrality"] = "mean"
component_merge_temperature: float = 0.07

original_merge_strategy: Literal["hard", "soft"] = "hard"
soft_merge_topk: int = 4
soft_merge_temperature: float = 0.07
```

Default behavior is unchanged:

- `component_merge="mean"` keeps original SCC mean pooling.
- `original_merge_strategy="hard"` keeps original nearest-token refill.

## Method

### Centrality-Weighted Component Merge

For one SCC component `C`, compute a prototype:

```text
prototype = mean(C)
```

Then score each token by cosine similarity to this prototype:

```text
score_i = cosine(token_i, prototype)
weight_i = softmax(score_i / temperature)
fused = sum_i weight_i * token_i
```

This makes the most representative token contribute more while still preserving boundary tokens softly.

### Soft Original-Token Refill

Original Scissor uses:

```text
original token -> nearest compressed token
```

Soft refill uses:

```text
original token -> top-k compressed tokens
weights = softmax(similarity / temperature)
```

Then each original token contributes fractionally to several compressed tokens. This is a lightweight approximation of AOT-style soft information transport without implementing full Sinkhorn OT.

## Ablation

Suggested experiments:

| Setting | component_merge | original_merge_strategy |
|---|---|---|
| Baseline Scissor | mean | hard |
| Ours-W | centrality | hard |
| Ours-S | mean | soft |
| Ours-WS | centrality | soft |

Also sweep:

- `component_merge_temperature`: `0.05, 0.07, 0.10`
- `soft_merge_topk`: `2, 4, 8`
- `soft_merge_temperature`: `0.05, 0.07, 0.10`

## Expected Benefit

- Less semantic smoothing than plain mean pooling.
- Better preservation of component centers.
- Better recovery of subtle context from original tokens.
- Minimal engineering risk because token count and output shape stay unchanged.

## Risks

- Low temperature can over-focus on one token and reduce component coverage.
- High temperature degenerates toward mean pooling.
- Soft refill can blur distinct compressed tokens if top-k is too large.
