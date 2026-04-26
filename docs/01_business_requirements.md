# Business Requirements — Succession-Planning Graph Recommender

## 1. Problem Statement
A typical succession plan is a hand-curated PowerPoint that becomes stale within a quarter. Leaders want a **reactive recommender**: pick a manager position (by `manager_id`), get the top-k internal succession candidates ranked by a **readiness score** that combines *skill match*, *structural proximity in the org*, and *performance*. Target user: HR Business Partners + senior leadership during reorgs and unexpected exits.

## 2. Stakeholders
| Role | Interest | Success criterion |
|------|----------|-------------------|
| HR Business Partner | Faster, defensible succession shortlists | Top-5 candidates with three sub-scores in < 2 seconds |
| Senior leadership | Continuity for critical roles | Recall@5 of held-out internal promotions ≥ 45% |
| L&D | Identify development gaps | Skill-gap delta vs incumbent reported per candidate |
| D&I | Diverse candidate slates | Slate auditable for over-concentration of any subgroup |

## 3. Business Objectives
1. Surface **k = 5** ranked successors for any `manager_id` in < 2 seconds.
2. Each candidate carries **three sub-scores** (skill_match, structural_proximity, performance) plus a blended `readiness_score` and a one-line "why".
3. Held-out evaluation: rank actual internal promotions in the top-5 ≥ 45% of the time.
4. Optional fairness gate: no shortlist > 80% of any one subgroup.

## 4. KPIs
| KPI | Definition | Target | Baseline |
|-----|-----------|--------|----------|
| Recall@5 | held-out promotions appearing in top-5 | ≥ 45% | 32% (skill-only) |
| nDCG@10 | rank quality of held-out promotions | ≥ 0.55 | 0.41 |
| Median percentile of held-out promotions | – | ≥ 75 | – |
| Latency (P95) | API response time | < 2 s | – |
| Slate-diversity | max subgroup share in top-5 | ≤ 80% | – |

## 5. Scope
**In scope:** internal succession only; manager hierarchy as a tree; 40-dim skills profile per employee.
**Out of scope:** external hiring; promotion-readiness time-to-readiness modelling (separate project); multi-incumbent role pooling.

## 6. Constraints & Assumptions
- **PII:** `emp_id` is anonymized; no names in the model layer.
- **Compute:** CPU only; embeddings recomputed nightly.
- **Explainability:** every candidate carries a one-line "why" and three sub-scores.

## 7. Risks
| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Stale skills profile → poor matches | Medium | High | Quarterly profile-refresh requirement; surface profile age in UI |
| "Lookalike" recommendations entrenching bias | Medium | High | Slate-diversity gate; D&I signoff for shortlists ≥ Director |
| Org-graph error (wrong manager_id) → wrong proximity | Low | Medium | Validate hierarchy is a tree at load time; log violations |
| Over-reliance on the recommender | Medium | Medium | UI banner: "decision aid, not auto-action"; signoff workflow |

## 8. Timeline
- **Week 1** — Synthetic org + skills profile generator
- **Week 2** — Spectral embeddings + structural-distance scoring
- **Week 3** — Held-out promotion eval + slate-diversity gate
- **Week 4** — FastAPI + Next.js with three-sub-score cards
- **Week 5** — Replace stub with small GraphSAGE (stretch); content; ship
