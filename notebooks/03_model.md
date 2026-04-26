# Notebook 03 — Modeling: Spectral Embeddings + Scoring

## 1. Fit embeddings
>>> `from succession_graph.models import fit_embeddings`
>>> `Z = fit_embeddings(df, G, k=16)`

## 2. Score one manager
>>> `from succession_graph.models import succession_for`
>>> `succession_for(df, G, Z, manager_id="E-0001", k=5)`

## 3. Slate-diversity gate
>>> `from succession_graph.models import enforce_slate_diversity`
>>> `enforce_slate_diversity(top_k, attr_series, max_share=0.8)`

## 4. Persist
>>> `models.save({"Z": Z, "emp_index": emp_index}, "embeddings.pkl")`
