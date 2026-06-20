# P3 Test Report — Feature Extractors

**Phase:** P3  
**Date:** 2026-06-20  
**Test file:** `tests/test_p3.py`  
**Result:** ✅ **39/39 passed** (~21s — dominated by sentence-transformers embedding in role_fit tests)  
**Environment:** Python **3.11.9** venv

---

## What P3 Was About

P3 builds the **six pure feature extractors** that assemble `fit_score`
and modulate the final score (EXCEUTION_PLAN §2, PHASED_BUILD_PLAN §P3):

```
fit_score = w_role·s_role_fit + w_skill·s_skill + w_exp·s_exp_band
         + w_edu·s_education + w_loc·s_location
final     = fit_score · m_behavior · p_penalty
```

All six are **pure functions**: they read only the candidate dict
(plus a pre-computed embeddings array for `s_role_fit`) and the config,
and return a normalized score. No I/O, no network, no model loading.
The runtime (P4 `rank.py`) calls them in the rerank step; the offline
pipeline (P4 `precompute.py`) handles the embedding work.

The five fit-component extractors implement the §2.5 refinements
verbatim: blended multi-query role signal with top-K-mean pooling and
single combined per-description weight (§2.5.a/b/f), thin-desc fallback
to title prior (§2.5.h), synonym collapse for skills (GLM-v2 #A4),
`github_activity_score` dropped from behavior (§2.5.g), sentinels
(`-1`, `{}`) treated as neutral.

---

## How the Tests Were Designed

The 39 tests are split across seven groups.

**Group 1 — `s_role_fit` (5 tests, `TestRoleFit`)**  
The dominant feature. The tests load the frozen multi-query intent
set (`config/jd_intent_embeddings.npy`, shape `(4, 384)`) and embed
synthetic descriptions in-process with the sentence-transformers model
(dev-time, allowed) so the role_fit function operates on real
embeddings. Coverage:
- Returns a float in [0, 1] for a realistic product-co ML engineer.
- An ML-engineering description scores higher than a marketing one
  (the anti-keyword-stuffing guarantee).
- A candidate with total description chars below `min_desc_chars`
  (default 40) falls back to the role-affinity title prior (e.g.
  `current_title="ML Engineer"` → 0.95).
- A recent (2024) ML stint outweighs an old (2016) one — but **only
  when the top-K-mean pool actually contains multiple descriptions
  with different recencies**. The test gives each candidate two
  descriptions (one relevant ML stint + one older marketing stint) so
  the K=2 pool exercises the recency weighting; single-description
  candidates fall back to pure max and would mask the effect.
- All 50 sample candidates return a score in [0, 1] with no exceptions.

**Group 2 — `s_skill` (5 tests, `TestSkill`)**  
The synonym-collapse property is the most important test in P3.
- A canonical skill ("RAG") matches `jd_core_skills` and scores positively.
- A candidate listing both "RAG" and "Retrieval-Augmented Generation"
  scores **identically** to a candidate listing just one of them — the
  synonym map collapses both to the canonical "RAG" and they are
  counted once, not twice. This is the GLM-v2 #A4 double-count fix.
- Noise skills ("Photoshop", etc.) are excluded.
- A Redrob platform-verified `skill_assessment_scores` entry (0–100)
  overrides the self-reported `proficiency`/`endorsements` for that
  skill (the §5.1 trust platform-verified rule).
- The endorsement curve caps at `endorse_floor` (30): 100 endorsements
  produce the same score as 30.

**Group 3 — `s_exp_band` (7 tests, `TestExperience`)**  
The soft band. The ideal range [6, 8] returns 1.0; the acceptable
taper [4, 6] and [8, 12] is monotonic in each direction; below
`hard_min_yrs` (2) the score is small but non-zero (the "sub-30-day
notice loved" anti-cliff); over-qualified candidates (14, 20+ years)
are penalised mildly but stay above 0.4. The `in_range` test sweeps
yoe across {0.5, 2, 4, 6, 7, 8, 12, 15, 25} and asserts every result is
in [0, 1].

**Group 4 — `s_education` (7 tests, `TestEducation`)**  
Tier score combined with non-linear CGPA ramp and field-relevance
multiplier. Coverage:
- Tier 1 + CGPA 9.0 + CS field → high score.
- Higher CGPA → higher score (monotonicity).
- CS field scores higher than Fine Arts (relevant_field_bonus).
- Unknown tier (0.30) is below tier_3 (0.50) — explicit neutral,
  not a bonus.
- Empty education list → 0.0.
- All scores in [0, 1] across tier × grade × field combinations.
- CGPA parser handles `"8.24 CGPA"`, `"74%"`, `"3.8 GPA"`, `"First Class"`,
  `""`, `None`. The first three yield a positive score; the last three
  fall back to the neutral 0.5 (missing CGPA is NOT punished).

**Group 5 — `m_behavior` (7 tests, `TestBehavior`)**  
The multiplier in [0.5, 1.1]. Coverage:
- `None` and `{}` signals → exact `neutral_base` (0.85).
- All-sentinel signals contribute nothing → exact `neutral_base`.
  (The test uses `last_active_date = "2026-02-20"` — ~4 months ago, in
  the moderate recency tier which contributes 0.0 — so the only
  non-sentinel channels don't move the multiplier off neutral.)
- The most-negative possible signal combo never drops below
  `min_multiplier` (0.50).
- The most-positive signal combo never exceeds `max_multiplier` (1.10).
- `open_to_work_flag = True` strictly increases the multiplier
  (verified with low-signal values so the off/on versions both stay
  below the cap).
- A recent `last_active_date` strictly beats a stale one.
- **`github_activity_score` does not affect the multiplier** (§2.5.g).
  Setting it to -1 vs 100 produces identical results.

**Group 6 — `s_location` (7 tests, `TestLocation`)**  
Substring match on "City, Region" data.
- "Noida, Uttar Pradesh" → `preferred_score` (1.0).
- "Pune, Maharashtra" → `preferred_score` (substring hits "Pune").
- "Hyderabad, Telangana" → `also_welcome_score` (0.85).
- "Jaipur, Rajasthan" + `willing_to_relocate=True` →
  `willing_to_relocate_score` (0.7).
- "Toronto, Canada" (no willing_to_relocate) → `outside_india_score` (0.2).
- Empty location + no willing_to_relocate → `outside_india_score`.
- All scores in [0, 1] across the location sweep.

**Group 7 — 50-sample smoke (1 test, `TestFiftySampleSmoke`)**  
The single most valuable test in P3: it iterates all 50 real sample
candidates, calls every extractor on each, and asserts every result
is in its declared range with no exceptions. This is the "works on
real data" regression guard — none of the synthetic tests exercise
the actual data shape, edge cases, or distribution.

---

## Passing Criteria

- Every extractor returns a float in its declared range.
- `m_behavior` returns a float in `[min_multiplier, max_multiplier]`.
- ML-engineering career descriptions score higher than marketing ones
  on `s_role_fit`.
- A candidate listing both "RAG" and "Retrieval-Augmented Generation"
  scores the same as a candidate listing just one (no double-count).
- Higher CGPA → higher `s_education`; 7 yrs → higher `s_exp_band` than
  1 yr; a top-K-mean ML description pool → higher `s_role_fit` than
  a pure-marketing pool.
- All-sentinel `m_behavior` inputs → exact `neutral_base`; never
  drops below `min_multiplier`; never exceeds `max_multiplier`.
- `github_activity_score` does not affect `m_behavior` (§2.5.g).
- All 50 real sample candidates return valid scores for every extractor.

---

## How We Know It Passed

On a clean Python 3.11 environment, pytest reported `39 passed in 21.12s`
with exit code `0`. The runtime is dominated by the `s_role_fit` tests,
which load `sentence-transformers` (`all-MiniLM-L6-v2`) and embed
synthetic + real descriptions to produce deterministic embeddings for
the role-fit function. The model is loaded once and cached across
tests. All 39 tests showed `PASSED` individually, with only harmless
`huggingface_hub` `resume_download` deprecation warnings.

The full suite is now **114/114 green** (P0: 20, P1: 24, P2: 18, P3: 39,
P4: 13 — the P4 full-100K latency test is opt-in and skipped by default).

### Development notes (what had to be fixed during P3)

Three test bugs surfaced and were corrected in `tests/test_p3.py`
itself, not in the feature code. Each reflects a real property of the
features that the test had to express more precisely.

1. **Recency weighting is invisible with one description.** The
   `test_recent_outweighs_old` test originally gave each candidate a
   single description. With K=1 the pool falls back to pure max, and
   the per-description weight has nothing to reweight. The test was
   fixed to give each candidate TWO descriptions (an ML stint + a
   marketing stint with different end dates) so the K=2 pool actually
   exercises the duration × recency composition. With one relevant
   + one irrelevant description and a duration_norm of 1.0 for both,
   the recent candidate's relevant stint gets a higher weight and the
   weighted mean differs from the old candidate's.

2. **Sentinel test contaminated by last_active_date.** The
   `test_sentinels_yield_neutral_base` test originally used
   `last_active_date = "2025-12-01"` (~6.5 months ago, in the stale
   tier) which contributes -0.05, giving 0.85 - 0.05 = 0.80 instead
   of the expected 0.85. The test was fixed to use a date in the
   moderate tier (`"2026-02-20"`, ~4 months ago) which contributes 0.0
   by design, so the sentinel-bearing channels are the only ones left
   to evaluate.

3. **Open-to-work test maxed out the cap.** The
   `test_open_to_work_increases` test originally used moderate
   response/interview rates that, combined with the OOTW bonus, pushed
   both the off and on versions above `max_multiplier` (1.1), so the
   clamp zeroed the difference. The test was fixed to use zero
   response/interview rates so the off version stays below the cap and
   the OOTW bonus moves the on version up by exactly 0.10.

None of these are feature bugs — they're properties of the design
that the test code needed to express more carefully. The feature code
behaves exactly as the §2.5 spec describes.

---

## What This Unlocks

With P3 green, every piece of `fit_score` has a tested, config-driven
extractor. P4 (`scoring.py` + `retrieve.py` + `precompute.py` +
`rank.py`) can now assemble the full pipeline: offline embed all career
descriptions, load the cached vectors at runtime, compute the six
features per shortlisted candidate, apply `m_behavior` × `p_penalty`,
sort, write the top-100 CSV. The P3 features are the spine that P4
connects and that P5 will calibrate against the proxy eval set.
