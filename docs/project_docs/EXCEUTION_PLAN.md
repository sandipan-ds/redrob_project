# Redrob Challenge — Execution Plan (v2, Production-Grade)

> **Challenge:** Intelligent Candidate Discovery & Ranking. Rank the **top 100** candidates
> from a **100,000**-candidate pool against a Senior AI Engineer JD.
> **Hard reality from the spec:** CPU-only, **no network / no hosted-LLM calls during ranking**,
> **≤ 5 min wall-clock**, ≤ 16 GB RAM. Scored on a **hidden tier-based ground truth** with
> `0.50·NDCG@10 + 0.30·NDCG@50 + 0.15·MAP + 0.05·P@10`. **3 submissions max, no leaderboard.**

---

## 0. North Star

We are NOT building "the model that counts the most AI keywords." The JD explicitly states that
is a **trap baked into the dataset**. We are building a system that reasons about the **gap between
what a profile *says* and what a career *shows***, applies the JD's explicit disqualifiers, modulates
by **behavioral availability signals**, and avoids **honeypots** — all within a strict CPU/latency budget.

**Win conditions (in priority order):**
1. Get the **top-10 ruthlessly right** (50% of score is NDCG@10).
2. **Never rank a honeypot in the top-100** (>10% honeypot rate = Stage-3 disqualification).
3. **Reproducible** inside a 5-min / 16 GB / CPU-only / offline Docker sandbox.
4. **Explainable** — fact-grounded reasoning + a design we can defend in a live interview.

---

## 1. Ground-Truth Mapping: JD ↔ Signals ↔ Candidate Schema

Before any scoring, we map the three sources and separate **signal from noise**. This is done
**once, offline** (LLM allowed at dev-time only) and **frozen into `config/scoring_config.yaml`**.

Reference inputs:
- `docs/reference_docs/job_description.docx` — **primary** fit reference.
- `docs/reference_docs/redrob_signals_doc.docx` — **secondary** signal provider (behavioral multiplier).
- `data/samples/sample_candidates.json` + `data/samples/candidate_schema.json` — field inventory.

### 1.1 What the JD *means* (decoded from `job_description.docx`)

| JD statement | Operational rule | Strength |
|---|---|---|
| "Right answer is NOT most AI keywords" | Down-weight raw skill-count; keyword stuffing must not win | **decisive** |
| "Marketing Manager with all AI keywords = NOT a fit" | **Title/career-vs-claims gap** is the dominant feature | **decisive** |
| "Tier-5 who built a recsys at a product company IS a fit" | Read `career_history.description` semantically, not `current_title` | **decisive** |
| Pure-research-only (no production) | Hard disqualifier → multiplicative penalty | **gate** |
| <12-mo LangChain-only "AI experience" | Disqualifier unless pre-LLM ML production shown | **gate** |
| No production code in last 18 mo | Disqualifier | **gate** |
| Consulting-only career (TCS/Infosys/Wipro/Accenture/Cognizant/Capgemini) | Penalty unless prior product-company experience | **gate** |
| CV / speech / robotics without NLP/IR | Domain-mismatch penalty | **strong** |
| 5–9 yrs ("range, not requirement"); ideal 6–8, 4–5 in applied ML at product cos | Soft experience **band**, not a cliff | **strong** |
| Embeddings retrieval, vector DB, eval frameworks (NDCG/MRR/MAP) in production | Core positive skill signals | **strong** |
| Noida/Pune preferred; sub-30-day notice loved | Soft tie-breakers only | **weak** |
| "Weigh behavioral signals — inactive + low response = not available" | Availability **multiplier** on fit score | **strong** |

### 1.2 What's NOISE in the JD (do not encode)
Company-stage storytelling, culture/"vibe check", async-writing preference, comp logistics — these
are not derivable from the candidate JSON and must **not** become scoring features.

### 1.3 Signals doc → architecture decision
The signals doc explicitly says signals are a **"multiplier or modifier on top of skill-match
scoring."** We honor that literally: `final = fit_score × behavior_multiplier × penalty`.
Sentinels are **"unknown," never "bad"**: `github_activity_score = -1`, `offer_acceptance_rate = -1`,
`skill_assessment_scores = {}` → treated as neutral.

### 1.4 Common vs additional criteria (the mapping deliverable)
Output `docs/project_docs/criteria_map.md` listing: (a) criteria common to JD + signals + schema,
(b) JD-only criteria with no schema field (drop or proxy), (c) schema fields not in JD (e.g. `summary`,
`connection_count` — keep only if predictive). This is produced **once** and reviewed by a human.

---

## 2. Scoring Model (transparent, defensible)

```
fit_score = w_role   · s_role_fit      # semantic career↔JD match (HIGHEST weight)
          + w_skill  · s_skill         # JD-relevant skills, endorsement/duration-weighted
          + w_exp    · s_exp_band      # soft band peaking 6–8 yrs
          + w_edu    · s_education      # tier + CGPA (non-linear, see §3)
          + w_loc    · s_location      # Noida/Pune/relocation, soft

final     = fit_score · M_behavior · P_penalty
```

- **`s_role_fit` (dominant):** cosine similarity of `career_history.description` text to the JD-intent
  embedding (retrieval/ranking/recsys/search at *product* companies). Defeats the keyword trap because
  it reads what they *did*, not what they *titled* themselves.
- **`M_behavior` ∈ [~0.5, 1.1]:** from `recruiter_response_rate`, `last_active_date` recency,
  `open_to_work_flag`, `interview_completion_rate`. Neutral for sentinels.
- **`P_penalty` (multiplicative kill-switches):** honeypot → ~0; consulting-only/research-only/
  CV-speech-robotics → 0.1–0.3.

### 2.1 Why not LLM-per-candidate?
The spec forbids it: network off + 5 min for 100K candidates. **All LLM use is offline/dev-time**
(JD-criteria extraction, reasoning-template design, code review). Runtime is a lightweight feature
reranker over precomputed embeddings — exactly what the spec recommends.

---

## 3. Recruiter Weights → *Calibrated* Config

The raw idea was to ask the recruiter for per-criterion weights (Python 8/10, CGPA baseline, etc.).
**Problem:** with no ground truth, hand-set weights overfit to keywords. **Fix:** keep the weights as
a **config layer** with the non-linear curves below, then **calibrate** them by maximizing NDCG@10 on
a **local proxy eval set** (§5).

Non-linear scales to keep:
- **CGPA:** below `min_cgpa` (e.g. 7.0) → score = 1 on a 0–7 ramp; from 7→10 maps to recruiter rating;
  not linear from 0.
- **Endorsements:** linear 0 → `endorse_floor` (e.g. 30), capped at the recruiter rating.
- **Role match:** Data Scientist ≠ Software Engineer ≠ Business Analyst — a **role-affinity matrix**,
  not a binary match.

All of this lives in `config/scoring_config.yaml` (human-readable, version-controlled — the documented
"scoring config", **not** to be confused with the hidden ground truth).

---

## 4. Honeypot & Disqualifier Detection (run BEFORE ranking)

Honeypots (~80) are forced to tier 0; >10% in top-100 = disqualification. Detector rules:
- `years_of_experience` vs Σ`career_history.duration_months` mismatch beyond tolerance.
- `proficiency ∈ {advanced, expert}` with `duration_months == 0` (e.g. "expert in 10 skills, 0 yrs used").
- Tenure starting before plausible company existence; education `end_year < start_year`.
- Internally contradictory `summary` vs `current_title` vs all `career_history.description`.

Disqualifier gates (from §1.1): consulting-only, research-only, CV/speech/robotics-only, no-code-18mo.
**We do not special-case honeypots into the top** — a correct reader avoids them naturally; the detector
is a safety net.

---

## 5. Local Proxy Evaluation Harness (because there is NO leaderboard)

With only **3 blind submissions**, we cannot probe the server. We build our own:
- Hand-label ~50 sample candidates (+ a few synthesized honeypots) into **relevance tiers 0–5**.
- Implement **NDCG@10, NDCG@50, MAP, P@10** and the exact composite formula.
- Use it to (a) calibrate weights, (b) regression-test every change, (c) decide what to submit.

---

## 6. Reasoning Generation (no LLM at runtime)

A **deterministic generator** fills real feature values into varied, honest sentences (≈1–2 sentences,
the spec's recommendation). Must pass the Stage-4 checks: specific facts, JD connection, honest concerns,
**no hallucination**, variation, rank-tone consistency. Example:
> "Analytics/Backend Engineer, 6.9 yrs; built Spark/Airflow pipelines feeding ML at a product co.,
> but no production retrieval/ranking; AI skills endorsement-light. Response rate 0.34 — moderate
> availability."

---

## 7. Compute & Latency Architecture (hard gate)

```
Offline (no time limit):  parse → honeypot scan → embed JD + career descriptions → cache features/index
Runtime (≤ 5 min, CPU):   load cache → hybrid retrieve top ~1–2K → feature rerank → top-100 → CSV
```
- Embeddings precomputed offline; runtime is vectorized NumPy/sklearn, no GPU, no network.
- Verify on a 16 GB CPU-only machine before each submission.

---

## 8. Submission & Reproducibility Deliverables

- `rank.py --candidates ./candidates.jsonl --out ./submission.csv` (single command, ≤5 min).
- `requirements.txt` (pinned), `Dockerfile` that builds & runs **offline** unmodified.
- Public **sandbox link** (HF Spaces / Streamlit / Colab) running a ≤100-candidate sample.
- `submission_metadata.yaml` at repo root (`has_network_during_ranking: false`,
  `uses_gpu_for_inference: false`, `honeypot_check_done: true`).
- **Real, iterative git history** (flat single-dump history is penalized at Stage 4).
- Validate every CSV with `docs/reference_docs/validate_submission.py` before uploading.

### 8.1 Output format law (from `validate_submission.py`)
Header exactly `candidate_id,rank,score,reasoning`; exactly 100 data rows; ranks 1–100 unique;
**score non-increasing**; ties broken by **candidate_id ascending**; IDs match `^CAND_[0-9]{7}$`; UTF-8.

---

## 9. Phased Build Plan

| Phase | Deliverable | Exit criterion |
|---|---|---|
| P0 | Repo scaffold, `config/scoring_config.yaml`, data loaders, schema validation | Loads 100K JSONL, validates schema |
| P1 | `criteria_map.md` + frozen JD-intent embedding (offline LLM at dev-time) | Human-reviewed mapping signed off |
| P2 | Honeypot + disqualifier detectors | Catches all sample honeypots; 0 false-kills on clear fits |
| P3 | Feature extractors (role-gap, exp-band, skill, edu, behavior) | Unit-tested on 50 samples |
| P4 | Hybrid retrieve → rerank pipeline | Top-100 produced under 5 min on CPU |
| P5 | Local eval harness + weight calibration | NDCG@10 maximized on proxy set |
| P6 | Deterministic reasoning generator | Passes all 6 Stage-4 reasoning checks |
| P7 | Docker + sandbox + metadata + git hygiene | Reproduces offline within limits |
| P8 | Final validation + submit | `validate_submission.py` passes |

---

## 10. Risk Register

| Risk | Mitigation |
|---|---|
| Accidentally rewards keyword stuffing | Role-gap feature dominates; honeypot/keyword regression tests |
| LLM-at-runtime creeps back in | Hard CI check: ranking step asserts no network import/calls |
| Overfit to 50 samples | Keep config simple; prefer monotonic, defensible rules |
| Misses 5-min budget on 100K | Precompute offline; profile runtime each phase |
| Reasoning hallucinates | Generator only emits values present in the profile |
| Burn submissions blindly | Decide via local proxy eval, not by submitting |

---

## Appendix A — Original v1 Notes (preserved for traceability)

> The following are the original planning notes this plan was derived from. Retained verbatim so
> the design rationale and recruiter-weighting intuition are not lost.

1. The reference for shortlisting the resumes is based on the two docs —
   a. `docs/reference_docs/job_description.docx`
   b. `docs/reference_docs/redrob_signals_doc.docx` (secondary signal provider for ranking).
   We consider both, objectively.

2. Rather than assuming the criteria and their weightage, take input from the recruiter on how much
   each criterion is valued. Before that, map the signals doc ↔ JD ↔ sample candidate fields: which
   criteria are common across the three sources? Which are additional (e.g. the `summary` field)?
   How much of the JD is relevant for selecting JSON-based candidate profiles, and how much is
   unnecessary? (Probable solution: a good prompt to force the LLM to extract the *signal* criteria
   from the JD + signals doc — done offline at dev-time.)

3. Take a weightage score from the recruiter, e.g. Python 8/10, SQL 6/10, LangChain 8/10,
   Eye-to-detail 8/10, Education/Graduation 8/10 (by college tier, Tier 1 best), Experience 8/10,
   Role=Data Scientist 8/10 (catch: software engineer / data analyst / business analyst / data engineer
   should NOT score the same — role-affinity matrix), Endorsements (linear 0→30 floor, capped at rating),
   CGPA (min 7 baseline; 7→1, 10→rating, non-linear below baseline).

4. The weighted document then serves as the **scoring config** (NOT the hidden ground truth).

5. Base the reasoning on the scoring config + candidate JSON, ~100 words, detailed but concise —
   generated **deterministically at runtime** (no hosted LLM call).

6. Latency is a criterion — see `docs/reference_docs/submission_spec.docx`.

7. Honeypots: if total experience doesn't match the sum of individual-organisation experience it may
   be a honeypot; more examples in `submission_spec.docx`. Honeypots are a rejection criterion.

