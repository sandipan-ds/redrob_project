# Project Understanding

This is a hackathon challenge ("INDIA.RUNS" by Redrob) to build a **Senior AI Engineer candidate ranker** for a pool of 100,000 candidates, outputting a top-100 CSV.

**The hard problem:** The dataset is adversarial on purpose:
- ~80 fabricated **honeypots** in the pool
- ~1,249/3,000 measured cases of `career_history[].title` being **scrambled relative to `description`** (so naive title-matching punishes you)
- The JD is explicit: "A Marketing Manager with all AI keywords is NOT a fit; a Tier-5 Backend Engineer who built a recsys IS" — i.e. the scoring must read careers, not keywords

**Scoring (hidden ground truth):** `0.50·NDCG@10 + 0.30·NDCG@50 + 0.15·MAP + 0.05·P@10`

**Hard constraints:** CPU-only, ≤5 min ranking step, ≤16 GB RAM, no network, no hosted LLM at ranking time. Precompute (embeddings) is allowed to exceed 5 min, the ranking step is not.

**Output contract (from `validate_submission.py`):** CSV with header `candidate_id,rank,score,reasoning`; exactly 100 data rows; ranks 1–100 unique; score non-increasing; tie-break by `candidate_id` asc; `CAND_XXXXXXX` format; 1–2 sentence reasoning.

**Current state:** P0 + P1 done (config loader, data loader, criteria map, frozen JD-intent embedding via `all-MiniLM-L6-v2`, 384-dim). P2–P8 still to do.

**Architecture the plan proposes:** Offline precompute embeds all `career_history[].description` with MiniLM → cached vectors. Runtime does a matrix-vector cosine → top-1-2K shortlist → scalar feature rerank + multiplicative gates + behavior multiplier → top-100 → CSV. Reasoning is a deterministic 1–2 sentence template, no LLM.

---

# What I Agree With (most of it — this is a strong plan)

1. **Two-stage offline/runtime split.** Embedding 100K live would blow the 5-min budget. This is the only viable shape given the constraints.
2. **Max-pool over career descriptions for `s_role_fit`.** Correctly defeats "one ML description averaged with nine marketing descriptions."
3. **Sentinels (`-1`, `{}`) = unknown, never bad.** Critical; treating them as "low score" would systematically punish candidates who never linked GitHub.
4. **Multiplicative gates for disqualifiers.** A honeypot can out-earn a subtractive penalty; ×0.01 cannot be out-earned. This is the right math.
5. **Demoting the `title` feature** after the 1,249/3,000 mismatch finding. The plan correctly traded a "clean" feature for a "noisy but more predictive" one.
6. **Honeypot detection as safety net, not strategy.** A correct ranker avoids them emergently; the detector is belt-and-suspenders. Avoids overfitting to sample honeypots.
7. **Calibrate ≤4 macro knobs, freeze the rest.** With ~50 hand labels, fitting 40+ weights = guaranteed overfit. The plan is intellectually honest about this.
8. **Top-20 manual audit before each submission.** NDCG@10 is 50% of the score; 20 profiles × 5 min = cheap insurance on half the points.
9. **`M_behavior` band [0.5, 1.1] with `neutral_base=0.85`.** Correctly stops availability from vaulting weak fits above strong ones.
10. **Reasoning 1–2 sentences, not 100 words.** Matches `submission_spec.docx` §2 and the worked example format in `sample_submission.csv` ("HR Manager with 6.1 yrs; 9 AI core skills; response rate 0.76.").
11. **Vendor MiniLM weights, Python 3.11 pin, `--network none` Docker.** Reproducibility done right.

---

# What I Disagree With / Would Push Back On

## High-priority concerns

1. **Sole reliance on a single 384-dim MiniLM centroid for role-fit is the plan's biggest single bet.** A small general-purpose encoder is fine for keyword-ish similarity but will likely blur nuanced distinctions (production-vs-research, recsys-vs-vision, applied-vs-theoretical). I'd add at least one second retrieval signal — a **BM25 / TF-IDF lexical match** against the JD's named concepts (NDCG, MRR, MAP, embeddings, vector DB, retrieval) — and merge the two scores before shortlisting. A candidate with strong lexical but weak semantic match (or vice versa) is a different signal than one who matches both.

2. **Max-pool over descriptions has a known failure mode the plan doesn't address:** a single description that *superficially* mentions "production ML" can pull a candidate up even if 9 of 10 entries are unrelated. Suggest a **top-K mean over the K-best descriptions** (e.g., K=2 or 3) rather than pure max — or weight each description's contribution by `duration_months` so a 6-month consulting stint can't outvote a 4-year ML stint.

3. **Penalty stackability is too aggressive as stated.** Consulting-only × research-only = 0.03 is a death sentence for a candidate who is *either* of those, even if the other label is borderline. The plan should either (a) define a *primary* gate (use the worst) and *secondary* gates as additional demotion, or (b) use `max(gates) × ∏(secondary)^0.5` to soften stacking.

4. **`github_activity_score` is sentinel `-1` for most candidates** (the plan admits low coverage). If 60-70% of the pool has github=-1, then github becomes a discriminator *only* for the 30% who have it — a self-selected group. I'd explicitly drop github from `M_behavior` and lean harder on `last_active_date` and `interview_completion_rate` (universally populated).

5. **No "career recency" weighting on `s_role_fit`.** The JD specifically says "no production code in last 18 months" is a gate — that's a *career-history recency* concept, not a platform-activity concept. The plan has this as a binary gate but doesn't say: even when not gated, a candidate's role-fit should be weighted by how recent their ML-relevant descriptions are. A 2018 ML stint + 2024 marketing career ≠ a 2024 ML stint.

## Medium-priority concerns

6. **The `title` demotion is correct in principle, but the plan doesn't say what to do with a candidate whose description is empty or thin.** With `s_role_fit` near zero for them, they get 0 on the dominant feature regardless of a legitimate "AI Engineer" title. Need an explicit "title prior as fallback when description is empty" path (not zero-weight).

7. **The retrieval shortlist cutoff `k` is treated as a single knob.** I'd argue for a **two-band retrieval**: top-K by cosine for the semantic-first pass, plus a small "second-chance" set of candidates with very high skill/experience scores but moderate cosine. Without the second band, you risk dropping a candidate whose description is technical-jargon-heavy but who is a real fit.

8. **The 4-macro-knob calibration may be too few.** The weight split, behavior band, penalty floor, and `k` are interdependent (changing `w_role` changes the *effective* range of `M_behavior`). I'd also let `recency_thresholds` for `last_active_date` and the per-signal weights inside `M_behavior` be tunable — they have direct operational meaning, not just curve-fitting risk.

9. **No way to verify the JD-intent embedding aligns with the hidden ground truth.** The embedding is built from one document; the ground truth was generated by humans with a specific interpretation of "Senior AI Engineer." If those diverge, you're scoring against the wrong centroid. Suggest: also embed the top-5 sample candidates (peeked from `sample_candidates.json`) and verify the JD-intent vector points at them — sanity check before locking it.

10. **Reasoning hallucination test needs a stronger form than "no skill/employer not in profile."** Pre-extract a whitelist of allowed entities (skills, employers, years) from each candidate, then assert every emitted token is in that whitelist. The current plan's substring check can miss, e.g., a hallucinated year or company variant.

## Lower-priority concerns

11. **`profile.summary` is "kept as proxy" in the criteria map but never used explicitly in the plan.** Either commit to using it (with low weight, behind career descriptions) or drop it. The current "kept as proxy" middle is what causes drift later.

12. **The plan is conservative about what to score (good) but under-defines the `skill` feature.** `s_skill` is mentioned as "weighted endorsement curve + duration as trust + prefer platform-verified" but the *exact* mapping from `skills[].name` to JD-core skills (e.g., does "Vector DB" match `Milvus`? does "Pinecone" match?) needs a synonym-collapse table. Without it, two equivalent skills score differently.

13. **No multilingual handling.** Indian names, Hindi skill strings, regional location terms. The embedding model handles this; skill name matching may not. Minor.

14. **"Open-source contributions" is proxied by `github_activity_score` only.** Many senior candidates don't GitHub. Better proxy: any mention of public artifacts, OSS projects, papers, or talks in `career_history[].description`.

15. **"Shipped to real users at meaningful scale" is proxied as `company_size ≥ 201-500` + product industry.** This is fine, but excludes candidates from large-but-IT-services companies (TCS, Infosys) who shipped real products for clients. The JD penalizes *consulting-only* careers, not all IT-services. Be careful with the company-size/industry gate.

---

# Summary

The plan is **~85% right** — architecture, math, constraints, and Phase-4/7 hygiene are all sound. The 15% I'd change is around (a) defending the single-MiniLM bet with a second retrieval signal, (b) fixing max-pool and penalty-stacking failure modes, (c) adding career-recency weighting, and (d) tightening the reasoning hallucination test. None of these are show-stoppers; they're the difference between "strong submission" and "winning submission" on NDCG@10.
