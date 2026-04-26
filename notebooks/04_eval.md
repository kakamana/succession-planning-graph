# Notebook 04 — Evaluation

## 1. Held-out promotions
>>> Build the held-out set from the synthetic generator's ground-truth promotions.

## 2. Recall / nDCG
>>> Loop the four recommenders over the held-out set; populate `docs/04_evaluation.md` §2.

## 3. Weight sensitivity
>>> Sweep simplex of `(α, β, γ)` and plot Recall@5 surface.

## 4. Slate-diversity cost
>>> Toggle the gate on/off; report Recall@5 delta.

## 5. Latency
>>> P50 / P95 over 1,000 random `manager_id`s.

## 6. Conclusions
>>> Draft the Medium-article "Results" section.
