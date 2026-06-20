# P1 Test Report — Criteria Map & JD-Intent Embedding

**Phase:** P1  
**Date:** 2026-06-19 (updated 2026-06-20)  
**Test file:** `tests/test_p1.py`  
**Result:** ✅ **24/24 passed** (~27s — dominated by model inference in semantic tests)  
**Environment:** Python **3.11.9** venv

> **Update log (2026-06-20):** original report was 20/20 with a **single** frozen JD-intent vector.
> The plan was later revised (EXECUTION_PLAN §2.5.a) to use a **multi-query JD-intent set** for the
> blended role signal (`s_dense = max cosine over a small set of intent vectors`). That made the single
> vector a *blocker* for the revised design, so we generated `config/jd_intent_embeddings.npy` (shape
> `(4, 384)`) alongside the legacy single vector (kept for back-compat) and added a new test group
> (`TestJdIntentSet`, +4 tests → 24 total). Both artifacts are frozen.

---

## What P1 Was About

P1 has two deliverables that must both be locked before any scoring code is written.

The first is `docs/project_docs/criteria_map.md` — a human-reviewed document that maps every JD requirement, every behavioral signal, and every candidate schema field to a decision: score it, proxy it, or drop it. Without this map, scoring decisions are made ad-hoc in code and become impossible to audit or defend. The map is the single source of truth for *why* each feature exists in the ranker.

The second is the frozen JD-intent embedding — a 384-dimensional vector that captures what a good candidate's career history *looks like*, not what keywords they list. This embedding is generated once at dev-time using `all-MiniLM-L6-v2` and saved to `config/jd_intent_embedding.npy`. At ranking runtime, the ranker loads this file directly — no model, no network, no inference cost. The embedding is the core of `s_role_fit`, the dominant scoring component that defeats the keyword-stuffing trap.

**Design revision — multi-query intent set.** The single vector is preserved, but the role signal is now
**blended and multi-query** (EXECUTION_PLAN §2.5.a): `config/jd_intent_embeddings.npy` holds a `(4, 384)`
set of frozen intent vectors ("production retrieval/ranking", "recsys/search at a product company", "eval
frameworks NDCG/MRR/MAP", "embeddings + vector DB in prod"), and runtime computes `s_dense` as the **max
cosine over that set**. This is sharper than a single centroid for the top band, where 50% of the score
(NDCG@10) is decided. The query strings live in `config/scoring_config.yaml → role_fit.intent_queries`, so
regenerating the set is a one-command, config-driven step.

---

## How the Tests Were Designed

The 24 tests are split across five groups.

**Group 1 — Criteria Map Integrity (5 tests)**  
These verify that `criteria_map.md` exists, contains all seven required sections (A through G), explicitly mentions each of the five hard disqualifier categories (pure research, consulting-only, LangChain-only, honeypot, CV/speech/robotics domain mismatch), documents sentinel value handling, and is substantive enough in length to not be a stub. These are structural checks — they ensure the document is complete enough to be a real design artifact, not a placeholder.

**Group 2 — Embedding File & Metadata (4 tests)**  
These verify that both output files from the embedding generation step exist on disk (`jd_intent_embedding.npy` and `jd_embedding_meta.yaml`), that the metadata YAML contains the required fields (model name, dimension, generation date, normalization flag), and that the normalization flag is explicitly `true`. The normalization check matters because the ranker uses dot product as a proxy for cosine similarity — this only works correctly if both the JD embedding and candidate embeddings are L2-normalized.

**Group 3 — Embedding Numerical Properties (5 tests)**  
These load the actual saved embedding and verify its mathematical properties: it is a 1D float32 numpy array, its dimension matches what the metadata claims, its L2 norm is 1.0 (within floating-point tolerance of 1e-5), and it contains no NaN or Inf values. These are regression guards — if the embedding generation step silently produces a corrupt file, these tests catch it before it propagates into the ranker.

**Group 4 — Semantic Sanity (3 tests) + Batch Similarity (3 tests)**  
The semantic tests are the most important in P1. They embed three pairs of contrasting career descriptions and verify that the JD embedding ranks them in the correct order:

- An ML engineer who built a production retrieval system should score higher than a marketing manager who ran campaigns.
- A recommendation systems engineer should score higher than a civil engineer.
- A candidate whose career history describes actually building ML systems should score higher than a candidate who just lists AI keywords in a skills dump.

The third test is the direct anti-keyword-stuffing check — it is the empirical proof that the embedding is pointing at the right semantic space. The batch similarity tests verify the vectorized dot-product function works correctly: correct output shape, self-similarity of 1.0, and near-zero similarity for an orthogonal vector.

**Group 5 — Multi-Query JD-Intent Set (4 tests, `TestJdIntentSet`)**  
These cover the revised blended role signal. They verify: (1) `jd_intent_embeddings.npy` and its metadata exist on disk; (2) the array is a 2-D `(Q, 384)` float32 matrix with **L2-normalized rows**, and **Q matches the number of `intent_queries` in `scoring_config.yaml`** (catches a stale artifact if the query list changes without regeneration); (3) `max_query_similarity` returns the right shape and each intent self-matches at ~1.0; (4) the multi-query `s_dense` still ranks a genuine production-ML description above a marketing description — the anti-keyword guard carried into the new signal.

---

## Passing Criteria

- `criteria_map.md` exists, has all 7 sections, mentions all 5 disqualifier categories, documents sentinels, and is >2000 characters
- `jd_intent_embedding.npy` and `jd_embedding_meta.yaml` both exist on disk
- Metadata contains `model`, `embedding_dim`, `generated_date`, `normalized: true`
- Embedding is a 1D float32 array with L2 norm = 1.0 ± 1e-5, no NaN/Inf
- Embedding dimension matches the value recorded in metadata
- ML engineer career text scores higher than marketing manager text (cosine similarity comparison)
- RecSys engineer career text scores higher than civil engineer text
- Genuine ML career description scores higher than a keyword-stuffed skills list
- Batch similarity returns shape `(N,)`, self-similarity = 1.0 ± 1e-5, orthogonal vector ≈ 0
- Multi-query set is `(Q, 384)`, rows L2-normalized, `Q` matches `role_fit.intent_queries`; `s_dense` ranks ML over marketing

---

## How We Know It Passed

On a clean Python 3.11 environment, pytest reported `24 passed in ~27s` with exit code `0`. The runtime is dominated by the semantic tests, which load the sentence-transformers model and run inference on short text pairs — this is expected and only happens at test time, not at ranking runtime. All 24 tests showed `PASSED` individually, with only harmless `huggingface_hub` `resume_download` deprecation warnings.

One test initially failed (`test_criteria_map_mentions_key_disqualifiers`) because it checked for the string `research_only` (underscore form) while the criteria map uses natural language ("pure research", "research only"). The test was corrected to check for the actual phrasing used in the document — the document itself was not changed.

**Artifact integrity note.** The legacy single vector was independently verified (parsing the `.npy` header directly): 384-dim, float32, L2 norm = 1.00000005, no NaN/Inf. The new multi-query set is `(4, 384)` with unit-norm rows.

---

## What This Unlocks

With P1 green, the project has a frozen, tested semantic anchor for role-fit scoring and a fully documented, human-reviewed decision record for every feature in the ranker. P2 (honeypot and disqualifier detection) can now be built with direct reference to the disqualifier gates defined in `criteria_map.md` Section F, and P3 (feature extractors) can be built with direct reference to Sections A, B, C, and E.
