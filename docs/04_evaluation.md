# Evaluation Plan — Succession-Planning Graph Recommender

## 1. Held-out promotion set
We synthesise a labelled set of "promotions" by sampling, for each `manager_id`, the actual successor among the manager's direct reports (highest performance × tenure-weighted skill match in our ground-truth generator). We hide the promotion link, run the recommender against the rest of the org, and measure whether the held-out successor is in the top-k.

## 2. Primary scorecard

| Recommender | Recall@1 | Recall@5 | Recall@10 | nDCG@10 | Median pct |
|-------------|----------|----------|-----------|---------|------------|
| Random | – | – | – | – | – |
| Skill-only (cosine) | – | – | – | – | – |
| Structural-only (1/spd) | – | – | – | – | – |
| Spectral-only (SVD) | – | – | – | – | – |
| **Blended (0.6/0.3/0.1)** | – | – | – | – | – |

## 3. Weight sensitivity
Sweep `(α, β, γ) ∈ ` simplex with step 0.1; report Recall@5 surface. Confirm `(0.6, 0.3, 0.1)` is at or near the optimum.

## 4. Slate-diversity gate
- For roles ≥ Director, report the share of slates that hit the gate at default ≤ 80%.
- Quantify the Recall@5 cost of the gate.

## 5. Latency
- P50 / P95 of `/succession` for k = 5 and k = 20 over 1,000 random `manager_id`s.

## 6. Robustness
- 10% of `manager_id` mappings randomly broken → confirm fallback to skill-only ranking.
- 25% of skill-vector entries set to NaN → confirm the system still returns a non-empty ranked list.
- Embeddings re-fit on a 30-day-shifted org snapshot → measure rank-correlation to verify stability.

## 7. Business impact
- Avg time-to-shortlist (current vs with recommender) on the held-out promotion set.
- HR-BP NPS on a sample of returned shortlists (qualitative).

## 8. Deployment readiness checklist
- [ ] Held-out scorecard populated
- [ ] Slate-diversity gate enabled by default for Director+
- [ ] `/succession` returns the four-field response in < 2 seconds
- [ ] `mlops/model_card.md` includes lookalike-bias caveat
- [ ] UI shows three sub-scores + one-line "why" per candidate
