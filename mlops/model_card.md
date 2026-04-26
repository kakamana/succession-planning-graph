# Model Card — Succession-Planning Graph Recommender

## Intended use
Decision-aid for HRBPs and senior leadership: rank internal succession candidates for any `manager_id`, with three sub-scores (`skill_match`, `structural_proximity`, `performance`) and a blended `readiness_score`. Always paired with a current development plan and (for Director+) a D&I review. Never an automated promotion trigger.

## Method
- 40-dim per-employee skills vectors + manager-hierarchy adjacency.
- Truncated SVD (k=16) on `[skills | adjacency]` as a spectral-embedding stand-in for a full GNN.
- Sub-scores: cosine skill-match (blended with embedding cosine 0.7/0.3), inverse shortest-path distance, normalised performance rating.
- Blend: `0.6 · skill + 0.3 · structural + 0.1 · performance`.
- Optional slate-diversity gate: max single-subgroup share ≤ 80%.

## Training data
Synthetic 2,000-employee org tree + skills profile (see `data/data_card.md`).

## Metrics
| Metric | Target |
|--------|--------|
| Recall@5 (held-out promotions) | ≥ 45% |
| nDCG@10 | ≥ 0.55 |
| Median percentile of held-out promotions | ≥ 75 |
| Latency P95 | < 2 s |

## Limitations
- The TruncatedSVD stand-in is not a true GNN — message-passing depth is one hop on the symmetrised adjacency.
- Skills profiles in the synthetic frame are static; real skills drift over time and require quarterly refresh.
- Slate-diversity gate is a post-hoc swap, not a constrained-optimisation step.

## Ethical considerations
- Lookalike-bias risk → slate-diversity gate.
- Decision-aid framing in every API response.
- D&I signoff required for Director+ slates.

## Retraining
- Nightly batch on the org snapshot.
- Trigger: ≥ 5% net headcount change.

## Ownership
- On-call DS: Asad
- Runbook: `mlops/runbook.md`
