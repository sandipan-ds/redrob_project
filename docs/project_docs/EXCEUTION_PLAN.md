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
| Noida/Pune preferred (Hyderabad/Mumbai/Delhi NCR also welcome); sub-30-day notice loved | Soft tie-breakers only; match city by **substring** (data is `"City, Region"`) | **weak** |
| "Weigh behavioral signals — inactive + low response = not available" | Availability **multiplier** on fit score | **strong** |

#### 1.1.a What the Strength labels mean (and can we put numbers on them?)

The `Strength` column is a **priority/severity tag**, not itself a score. It answers the question
*"how much should this rule be allowed to move a candidate's final rank, and in which direction?"*
It maps to **where in the formula the rule acts** (a weight, a soft band, or a multiplicative gate),
which is exactly why the four labels behave so differently in numbers.

| Label | Meaning (plain English) | Where it acts in the formula | Approx. quantified effect |
|---|---|---|---|
| **decisive** | This *is* the thesis of the JD. Getting it wrong = wrong answer. It dominates ordering. | Drives the **largest fit weight** (`w_role = 0.45`) and the semantic role-fit feature | Can swing fit by the full top weight: **≈ 0–45 of the 100 "fit points."** A profile that fails the career-vs-claims read loses most of its score. |
| **gate** | A hard disqualifier. Not "scored down a bit" — **multiplicatively killed.** | A `P_penalty` factor multiplied onto the final score | Multiplies final by **0.01–0.40** (i.e. removes **60–99%** of the score). Honeypot ≈ ×0.01, langchain-only-junior ≈ ×0.40. |
| **strong** | A real, material positive/negative feature — matters a lot but is *tradeable* against other strong features. | Mid-weight fit features (`w_skill = 0.25`, `w_exp = 0.15`) or the behavior multiplier band | Contributes on the order of **±15–25 fit points**, or ×0.5–1.1 via `M_behavior`. |
| **weak** | A tie-breaker only. Should never change who is in vs out of the top-100, only order among near-ties. | Smallest fit weight (`w_loc = 0.05`) | **≤ 5 fit points.** Only separates candidates who are otherwise ~equal. |

**Can we express them out of 100?** Yes, as a *budget*, with the important caveat that fit and the
multipliers live on different axes:

- **Additive axis (fit_score, normalized to ~100):** decisive ≈ **45/100**, strong ≈ **15–25/100 each**,
  weak ≈ **≤ 5/100**. These are literally the `weights:` block in `config/scoring_config.yaml`
  (`role_fit 0.45`, `skill 0.25`, `experience 0.15`, `education 0.10`, `location 0.05`).
- **Multiplicative axis (gates, expressed as a retained-fraction %):** gate = "keep **1–40%** of the
  score." A 90/100 fit profile that trips a research-only gate (×0.20) ends up at an effective **18/100**,
  i.e. out of contention — which is the whole point of a gate vs. a weight.

So the honest one-line answer: **`decisive`/`strong`/`weak` are positions on a 0–45 / 15–25 / 0–5
additive-point bracket; `gate` is not on that bracket at all — it is a 1–40% multiplier that can erase
any number of points.** This separation is deliberate: it lets a single disqualifier override an
otherwise-excellent keyword-rich profile, which is precisely the trap the JD is testing for.

### 1.2 What's NOISE in the JD (do not encode)
Company-stage storytelling, culture/"vibe check", async-writing preference, comp logistics — these
are not derivable from the candidate JSON and must **not** become scoring features.

#### 1.2.a Concrete examples of each noise category (grounded in the sample data)

"Noise" here means *language in the JD that a human recruiter cares about but that has **no reliable
field** in the candidate JSON*, so trying to score it would just inject bias or hallucination. Examples,
using profiles from `data/samples/sample_candidates.json` (and the same patterns appear in
`data/originals/candidates.jsonl`):

- **Company-stage storytelling** ("we're a Series-B rocketship, want people who thrive in 0→1 chaos").
  The schema gives us `current_company_size` (e.g. `"10001+"`, `"201-500"`) and `current_industry`,
  but **nothing about funding stage, growth phase, or "0→1 vs scale" narrative.** Example: `CAND_0000001`
  (Ira Vora) is at `Mindtree`, size `10001+`, industry `IT Services` — we can read *size* and *industry*
  (and we do, as a product-company proxy), but we **cannot** read "early-stage builder energy." So we
  encode size/industry and drop the stage story.

- **Culture / "vibe check"** ("must be a low-ego, high-trust teammate"). Nothing in the JSON measures
  ego, collaboration style, or values. The closest fields — `profile.summary` and
  `career_history[].description` — are *self-authored marketing copy*. Example: `CAND_0000001`'s summary
  ("I'm a backend/data hybrid … building competence on the ML side") tells us about *claimed focus*, not
  about culture fit. We use that text **only** for semantic role-fit, never as a "vibe" feature.

- **Async-writing preference** ("writes clearly, async-first"). The JD values written communication, but
  the only writing sample we have is the `summary`/`description` text — and its quality is confounded by
  whoever wrote the profile, not how the person communicates at work. Example: `CAND_0000002`
  (Saanvi Sethi) has a fluent, well-written summary, yet her career is Operations/Marketing — good prose
  must **not** earn AI-fit points. So writing quality is explicitly **not scored** (it stays a Stage-4
  human-interview criterion in `criteria_map.md` §B).

- **Comp logistics** (budget, equity split, salary banding). `expected_salary_range_inr_lpa` exists
  (e.g. `CAND_0000001`: 18.7–36.1 LPA) but per the JD it is **not a fit criterion**, so `criteria_map.md`
  §C marks it **Drop**. Encoding it would rank people by cheapness, not fit.

The rule of thumb: **if a JD phrase can only be answered by a field we don't have, or by reading
self-written copy as if it were ground truth, it is noise — keep it out of the math and leave it for
the human interview stage.**

### 1.3 Signals doc → architecture decision
The signals doc explicitly says signals are a **"multiplier or modifier on top of skill-match
scoring."** We honor that literally: `final = fit_score × behavior_multiplier × penalty`.
Sentinels are **"unknown," never "bad"**: `github_activity_score = -1`, `offer_acceptance_rate = -1`,
`skill_assessment_scores = {}` → treated as neutral.

> **What "sentinels are unknown, never bad" means.** A *sentinel* is a placeholder value the dataset
> uses to say *"we have no data here,"* **not** *"the candidate scored zero here."* In this dataset the
> sentinels are `-1` for numeric fields that are normally `≥ 0` (e.g. `github_activity_score`,
> `offer_acceptance_rate`) and `{}` (empty dict) for `skill_assessment_scores`. The trap is that `-1` is
> numerically *lower* than any real score, so if you fed it straight into a formula it would **punish**
> a candidate for missing data — e.g. someone who simply never linked GitHub (`github_activity_score = -1`)
> would look *worse* than someone with a genuinely terrible score of `0.1`. That is wrong: absence of a
> signal is not evidence of a weakness.
>
> So we **map every sentinel to neutral before scoring** — it contributes nothing (neither bonus nor
> penalty) to `M_behavior`, and the multiplier falls back to its `neutral_base` for that signal. Concrete
> examples from `data/samples/sample_candidates.json`: `CAND_0000002` has
> `github_activity_score = -1`, `offer_acceptance_rate = -1`, and `skill_assessment_scores = {}` — all
> three are read as *"no info,"* so her availability multiplier is driven only by the signals she *does*
> have (`last_active_date`, `recruiter_response_rate`, `interview_completion_rate`), not dragged down by
> the three blanks. By contrast `CAND_0000001` has a *real* `github_activity_score = 9.2` and a populated
> `skill_assessment_scores`, so those are scored on their merits. In short: **missing ≠ low.**

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

#### 2.0.a Where do the weights come from? (the honest answer)

**No document hands us these numbers.** We checked: `job_description.docx` describes *what* a good
candidate looks like in prose, and `redrob_signals_doc.docx` says only that behavioral signals should
act as *"a multiplier or modifier on top of skill-match scoring"* — it gives **ranges and types for the
23 signals but zero weights, zero multiplier values, and no formula.** So every number in §2 and in
`config/scoring_config.yaml` is **our engineering choice**, derived from the JD's *emphasis*, not copied
from a spec. That is exactly why §3 and §5 exist: the weights start as **defensible priors** and are
then **calibrated** by maximizing NDCG@10 on the local proxy eval set. They are starting points, not
sacred constants.

The priors are set by translating the JD's own language into relative importance:

- `w_role = 0.45` — the JD's thesis ("read what they *did*, not what they *titled* themselves") is
  **decisive**, so role-fit gets the single largest slice. It alone is ~45% of fit because the whole
  challenge is built to punish keyword-matching over career-reading.
- `w_skill = 0.25` — skills matter but are gameable (the keyword trap), so they are second, not first.
- `w_exp = 0.15` — the JD calls 5–9 yrs a *"range, not a requirement,"* so experience is a **soft band**,
  not a top driver.
- `w_edu = 0.10`, `w_loc = 0.05` — education is a minor positive; location is explicitly a **soft
  tie-breaker only**. They sum with the rest to **1.0** (enforced in config).

#### 2.0.b Why `M_behavior ∈ [~0.5, 1.1]` — what's the reasoning behind 0.5 and 1.1?

`M_behavior` is a **multiplier on an already-computed fit score**, so its job is to *nudge*, not to
*decide*. The bounds encode a deliberate asymmetry that mirrors the signals doc's logic ("a
perfect-on-paper candidate who hasn't logged in for 6 months … is, for hiring purposes, not actually
available"):

- **Upper bound ≈ 1.1 (a small +10% reward).** Being highly available is *nice-to-have*, not a reason to
  vault a weaker candidate over a stronger one. If we let availability multiply by, say, ×1.5, a mediocre
  but eager candidate could outrank a clearly better fit — which corrupts NDCG@10. Capping the *bonus* at
  ~+10% keeps availability as a **tie-breaker among comparable fits**, never a fit substitute.
- **Lower bound ≈ 0.5 (at most −50%).** Genuinely unavailable candidates (stale + non-responsive) should
  be pushed down hard, but **not to zero** — that role belongs to the `P_penalty` gates. A great engineer
  who is merely slow to respond is still a great engineer; halving is a strong demotion without being a
  death sentence.
- **`neutral_base = 0.85`** (in config) is the value used when signals are missing/sentinel: it sits
  *below* 1.0 on purpose, so that candidates with **proven** availability (real, strong signals) can
  climb toward 1.1, while "unknown" candidates neither get the reward nor get punished as if "bad."

These exact numbers (`min_multiplier 0.50`, `max_multiplier 1.10`, `neutral_base 0.85`, and the per-signal
weights like `response_rate_weight 0.3`) are **not from any doc** — they are tunable knobs in
`config/scoring_config.yaml`, chosen so behavior modulates rank by roughly **±10–15% in practice** and
then refined during calibration (§5). If calibration shows availability should matter more or less, these
are the first numbers we move.

#### 2.0.c How was `P_penalty` decided?

`P_penalty` is **multiplicative and < 1** because the JD frames these as **disqualifiers**, not
deductions — and the only honest way to express "this should be disqualified, regardless of how good the
rest of the profile looks" is to multiply the whole score down. (Subtracting a fixed number wouldn't
work: a keyword-stuffed honeypot could rack up enough additive points to survive a subtraction. A
multiplier of ×0.01 cannot be out-earned.) The *magnitude* of each penalty is graded by **how absolute
the JD is about it:**

| Gate | Config score | Why this severity |
|---|---|---|
| Honeypot | `×0.01` | Fabricated profile — must be effectively *removed* (>10% in top-100 = Stage-3 disqualification). Near-zero, not exactly zero, so it still sorts deterministically. |
| Consulting-only | `×0.15` | JD penalizes *unless* prior product-company experience — strong but recoverable, so harsh-but-not-fatal. |
| Research-only | `×0.20` | JD wants *production*; pure research is a near-miss, not a fraud. |
| No recent code (18mo) | `×0.25` | Currency matters but a recent gap is less damning than never having shipped. |
| Domain mismatch (CV/speech/robotics, no NLP/IR) | `×0.30` | Wrong specialization, but adjacent ML skill is transferable, so the mildest *gate*. |
| LangChain-only junior (<12mo, no pre-LLM ML) | `×0.40` | The softest gate — junior, not disqualified, just heavily down-weighted. |

Again, **no document specifies 0.01 / 0.15 / 0.20 / …** — these are our calibrated priors in
`config/scoring_config.yaml`. The ordering (honeypot ≪ consulting < research < no-code < domain <
langchain) is the part that is *principled*; the precise decimals are tuned so that no gated candidate
can climb back into the top-100 while a borderline-but-legitimate candidate isn't annihilated.

### 2.5 Post-review refinements (folded in from GLM/MIMO/MINIMAX critiques)

Three independent reviews converged on the same handful of weak points. These are **not**
re-architecture — the spine (offline precompute + NumPy rerank, distrust titles, multiplicative gates,
sentinels-neutral, few-knob calibration) is unchanged. They are sharpenings, recorded here and wired into
the build in `PHASED_BUILD_PLAN.md`.

#### 2.5.a 🔴 A second, orthogonal role signal (the unanimous concern)

**Problem:** ~45% of fit, and effectively all of NDCG@10, rides on **one** signal — cosine of career
descriptions to a **single** 1,200-char JD-intent vector via `all-MiniLM-L6-v2` (a short-text similarity
model, not a fine-grained "did-they-ship-it" discriminator). Genuine fits will cluster at near-identical
cosine and then be ordered by `candidate_id` — precisely where half the score is won or lost.

**Fix (commit to this, do not leave it "optional"):** at the **rerank** stage, combine the dense cosine
with a second, orthogonal role signal and take a calibrated blend:

```
s_role_fit = w_dense · s_dense  +  w_lex · s_lex
```

- **`s_dense` — multi-query dense, max-pooled.** Replace the single JD-intent vector with a small set of
  **frozen intent vectors** (e.g. "production retrieval/ranking", "recsys/search at a product company",
  "eval frameworks: NDCG/MRR/MAP", "embeddings + vector DB in prod"). Score = **max** cosine over the
  query set, then **max over the candidate's descriptions** (see §2.5.b for the pooling refinement).
- **`s_lex` — production-evidence lexical match (BM25/TF-IDF).** A small, *generalized* lexicon of
  production-shipping verbs **and synonyms** ("shipped, deployed, launched, rolled out, served, in
  production, A/B, inference service, online") **+** retrieval/ranking/recsys/search terms, matched
  against the descriptions. This catches real engineers whose phrasing differs from our guessed words —
  the failure mode §2.5.c warns about.
- `w_dense` / `w_lex` are part of the calibrated macro-knob budget (§5.1), default ~0.7 / 0.3.

**Pre-freeze sanity check (from MINIMAX #9):** before locking the JD-intent vectors, embed the few
clearly-good sample candidates and confirm the intent vectors actually point at them (high cosine). If the
hand-distilled intent diverges from what a real fit looks like, we are scoring against the wrong centroid.

#### 2.5.b 🟠 Pooling refinement: top-K mean, not pure max

Pure max-pool has a known failure: a *single* description that superficially name-drops "production ML"
can vault a candidate whose other entries are unrelated. **Refinement:** use **top-K mean** (K≈2) over a
candidate's per-description cosines, optionally **weighting each description by `duration_months`** so a
6-month consulting stint cannot out-vote a 4-year ML stint. `pool ∈ {max, topk_mean}` and `K` live in
`cfg.role_fit`; default `topk_mean`, K=2. (Max remains a valid fallback when a candidate has one
description.)

#### 2.5.c 🟠 `research_only` must be conjunctive, not a keyword-*absence* rule

A gate that fires on "none of {production, deployed, shipped, users, scale} appear" is **the keyword trap
inverted** — it punishes a real production engineer who happened to write "rolled out / served / A/B
tested" instead of our exact words. **Fix:** `research_only` fires **only if all three** hold:
(1) no production-lexicon match (using the *broad* synonym set from §2.5.a), **and**
(2) no product-company tenure ever, **and**
(3) explicit research framing in descriptions/titles ("research scientist", "academic", "lab", "PhD
thesis work"). Otherwise it degrades to a mild soft feature, not a ×0.20 gate. This directly honors the
plan's own "avoid false-kills" rule (§4.1).

#### 2.5.d 🟠 Penalty stacking: primary gate + softened secondaries

`consulting_only × research_only = 0.03` is a death sentence even when one label is borderline. **Fix:**
apply the **worst (smallest) gate at full strength**, and **soften the rest**:

```
P_penalty = min(gates) · Π(other gates)^0.5      # geometric softening of secondary gates
```

So a candidate who is squarely consulting-only (×0.15) but only *borderline* research-only doesn't get
crushed to 0.03; the secondary gate is dampened. Honeypot remains an exception — it always applies at
full ×0.01.

#### 2.5.e 🟠 Gate magnitudes are calibratable, not frozen by assertion

The six gate scores (0.01 … 0.40) are hand-set numbers that the plan previously *excluded* from its
"≤4 calibrated knobs" budget — i.e. ~6 hidden knobs. **Fix:** introduce a **single global penalty scale**
`p_scale ∈ cfg.penalties` that multiplies all *non-honeypot* gate severities, and make `p_scale` **one of
the calibrated macro knobs** (§5.1). This preserves the principled *ordering* while letting the proxy data
set overall severity, and it keeps the honest knob-count accurate.

#### 2.5.f 🟠 Career-recency weighting on role-fit (distinct from the 18-mo gate)

The JD's "no production code in last 18 months" is a *career-recency* concept, currently only a binary
gate. **Even when not gated**, a candidate's role-fit should be **recency-weighted**: a 2024 ML stint
should outweigh a 2018 ML stint followed by a marketing career. Weight each description's role-fit
contribution by a recency factor derived from its `start_date`/`end_date` (recent → 1.0, old → decayed).
Decay half-life is a `cfg.role_fit` parameter.

**Composition rule (pinned — duration × recency are ONE weight, not two):** to avoid calibrating two
entangled per-description reweightings (the "interdependent knobs" failure mode), each description *d* gets
a **single** weight `w_d = duration_norm(d) × recency_decay(d)`, where
`duration_norm = min(duration_months/24, 1.0)` and
`recency_decay = 0.5 ** (months_since_end / recency_half_life_months)`. The top-K mean (§2.5.b) is the
`w_d`-weighted mean of the K highest per-description cosines. See `SYSTEM_DESIGN.md` §4.1.1.

#### 2.5.g 🟢 Drop `github_activity_score` from `M_behavior`

It is sentinel `-1` for a large fraction of the pool, so it only discriminates a **self-selected** subset.
**Fix:** remove it from the behavior multiplier and lean on the **universally-populated** signals
(`last_active_date`, `recruiter_response_rate`, `interview_completion_rate`). (`criteria_map.md` §E should
move it from "✅ (weak)" to "❌ dropped — low coverage".)

#### 2.5.h 🟢 Thin/empty-description fallback (the only place `title` is allowed back)

If a candidate has **no usable description text**, `s_dense`/`s_lex` are ~0 and a legitimate "AI Engineer"
would score zero on the dominant feature. **Fix:** when (and only when) the concatenated description text
is empty/below a min length, fall back to the **role-affinity title prior** (§3.1) for the role component.
This is the single justified use of the otherwise-demoted title lookup — it is a *fallback*, never a
*bonus*, and it resolves the apparent "drop the role-affinity table entirely" suggestion: we keep it
precisely for this case.

#### 2.5.i 🟢 Bottom-of-top-100 / "fewer than 100 real fits"

The pool may contain far fewer than 100 genuine fits. The bottom of our list will necessarily be
"best of the weak." **Design stance:** (1) the score is honestly low there and reasoning says so
("adjacent skills only — filler below the likely cutoff", matching the spec's own rank-100 example);
(2) ordering among weak candidates falls back to the *stable* secondary features (exp band, then
`candidate_id` asc) so it stays deterministic and validator-clean; (3) we do **not** fabricate confidence
we don't have — NDCG@50/MAP reward getting the *relative* order right even among mediocre candidates.

#### 2.5.j 🟢 Three P2-implementation decisions carried from GLM-v2 (§A2/A3/A5)

To keep these from being forgotten, they are recorded here and implemented in P2/P3 (not now):

- **`consulting_only` — generalize beyond the 10-name list.** A hard-coded company list misses hidden-pool
  variants (Genpact, LTIMindtree, IBM India…). The detector should *also* use
  `current_industry == "IT Services"` + `company_size` bands as a generalized consulting proxy, and the
  "prior product-company experience" exemption should be a real check (`industry != "IT Services"` stint
  ever). Keep the name list as a high-precision booster, not the sole signal.
- **`langchain_only_junior` — reconsider as a gate.** It is detection-hard and can false-fire on a senior
  who recently *added* LangChain. A LangChain-only junior's descriptions won't embed near "built
  retrieval/ranking in production," so role-fit + the exp band already demote them. **Decision:** demote it
  from a hard ×0.40 gate to a mild soft feature unless the conjunctive conditions (junior exp **and**
  no pre-2022 ML **and** LangChain is the only AI signal) all hold.
- **`profile.summary` — commit, don't leave "as proxy".** Use it as a **low-weight supplementary input to
  the role-fit text** (behind `career_history[].description`), and specifically as part of the thin-desc
  fallback (§2.5.h) when descriptions are empty. `criteria_map.md` §C is updated to reflect this commitment
  rather than the ambiguous "keep as proxy".

### 2.1 Why not LLM-per-candidate?
The spec forbids it: network off + 5 min for 100K candidates. **All LLM use is offline/dev-time**
(JD-criteria extraction, reasoning-template design, code review). Runtime is a lightweight feature
reranker over precomputed embeddings — exactly what the spec recommends.

#### 2.1.a "5 minutes for 100K or for 100?" — settling the ambiguity (from `submission_spec.docx` §10.5)

There are **two different runs** in the spec, and they have **two different scales**. This is the source
of the confusion, so to be explicit:

1. **The real grading run — full 100K pool, ≤ 5 min.** Stage-3 reproduction runs *our* `rank.py` on the
   **full 100,000-candidate** file inside the organizers' own sandbox. §10.3 is explicit: *"the ranking
   step that produces the CSV must complete within \[the 5-minute window]."* Pre-computation
   (embeddings, indexes) is allowed to run *before* and *may exceed* 5 minutes, but the **ranking step
   over all 100K must finish in ≤ 5 min on CPU.** **This is the budget we design the architecture around
   (§7).**

2. **The mandatory public sandbox — ≤ 100 candidates, ≤ 5 min.** §10.5 says the hosted sandbox link only
   needs to *"Accept a small candidate sample (**≤ 100 candidates**) … Run end-to-end … Complete within
   the compute budget (≤ 5 min on CPU)."* It then states plainly: *"It does **not** need to handle the
   full 100K pool — small-sample reproducibility is what we're checking. The full reproduction at Stage 3
   happens in our own sandbox."*

**So:** we are *evaluated* on **100K in 5 min** (Stage 3, the run that matters), and we merely *demo*
**≤ 100 in 5 min** in the public sandbox (a cheap "does it run at all?" sanity check, Stage 1). The
sandbox being only 100 candidates is **not** a relaxation of our performance target — it's just a
lower-stakes smoke test. Our latency engineering in §7 (offline embedding + vectorized NumPy rerank over
a hybrid-retrieved shortlist) is sized for the **100K** case; the 100-candidate sandbox then runs the
identical code path trivially within budget.

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

### 3.1 The role-affinity matrix — what it is and how its scores are set

**Is it a similarity matrix?** Conceptually yes, but it is **not learned** and it is **not the main
role signal.** Two clarifications:

1. **It's a one-dimensional affinity lookup, not a full N×N matrix.** As implemented in
   `config/scoring_config.yaml → role_affinity:`, it maps **`current_title` (one role) → an affinity
   score in [0, 1] for the *one* target role (Senior AI Engineer)**. We don't need a square matrix of
   "every role vs every role" because we only ever rank against a **single** JD. So it is the *target
   column* of a similarity matrix, not the whole matrix:

   ```
   "AI Engineer"        : 0.95     "Data Scientist"   : 0.80
   "ML Engineer"        : 0.95     "Backend Engineer" : 0.50   # check ML-adjacency in history
   "Research Scientist" : 0.70     "Business Analyst" : 0.10
   "NLP Engineer"       : 0.90     "Marketing Manager": 0.05
   "Search/Ranking Eng" : 0.90     "default"          : 0.20   # unknown title
   ```

2. **How are the scores determined?** They are **hand-set ordinal priors derived from the JD's intent**,
   not measured from data — and crucially they are a **prior/tie-breaker, not the dominant feature.**
   The dominant role signal is `s_role_fit`, the *semantic cosine similarity* between the
   `career_history[].description` text and the frozen JD-intent embedding (§2). The affinity lookup
   exists to handle the **title** quickly and to encode the JD's explicit "these titles are not equal"
   rule. The scoring procedure for the title score is:

   - **Anchor the endpoints.** Exact target role = `1.0`; clearly unrelated business roles
     (Marketing/Operations Manager) ≈ `0.05`; unknown title falls back to `default = 0.20`.
   - **Rank everything in between by *distance to "builds production AI/ML systems."*** Direct ML
     engineering titles cluster at `0.90–0.95`; analytical-but-not-engineering (Data Scientist `0.80`,
     Applied Scientist `0.85`) sit just below; **ambiguous adjacent titles** (Backend/Data/Analytics/
     Software Engineer `0.40–0.50`) are deliberately *mid-band with a "check career history" caveat*,
     because the JD's whole point is that a Tier-5 *Backend Engineer who actually built a recsys* is a
     fit. The low title score is **overridden** by a high `s_role_fit` when the description proves real
     ML work.
   - **Pure-research risk is encoded in the title too:** `Research Scientist = 0.70` (not higher),
     flagged to cross-check against the research-only gate (§4).

   So the numbers are **principled in their *ordering*** (which the JD dictates) but **arbitrary in their
   exact decimals** (our priors). Like all of §3, they are **calibrated in P5** by maximizing NDCG@10 on
   the proxy set — if the data says Data Scientist should be `0.85` not `0.80`, we move it.

   **Why hand-set and not learned?** There is no ground-truth label set to learn a matrix from (no
   leaderboard, only 3 blind submissions). A learned matrix on 50 hand-labeled samples would overfit; a
   small, monotonic, human-defensible lookup is more robust and is **explainable in the live interview**,
   which is a Stage-4 scoring criterion. The semantic `s_role_fit` carries the real discriminative load;
   the affinity matrix is the cheap, transparent title prior layered on top.

#### 3.1.a ⚠️ Data reality: `title` is *decoupled* from `description` — demote the title feature

**Measured fact (not assumed).** Profiling `data/originals/candidates.jsonl` shows that
`career_history[].title` is **systematically inconsistent** with its own `career_history[].description`.
In a scan of the first 3,000 candidates, **1,249 career entries** had a title that did not match the work
their description actually describes. Concrete example — `CAND_0000004`:

| Listed title | Actual description (what the work was) |
|---|---|
| "Marketing Manager" | *"Mechanical engineering design role… CAD (SolidWorks, Creo), FEA (ANSYS)…"* |
| "Operations Manager" | *"Content writing and SEO strategy for a tech-focused publication…"* |
| "Business Analyst" | *"Operations management role at a logistics company… fulfillment across 3 warehouses…"* |

This is **by design** — it is the dataset's instantiation of the JD's own warning: *"A candidate who has
all the AI keywords… but whose title is 'Marketing Manager' is not a fit… [but] a Tier-5 candidate… if
their career history shows they built a recommendation system… is a fit."* **The title is the trap; the
description is the signal.**

**Consequences for this design (corrections to the §3.1 plan above):**

1. **The `role_affinity[current_title]` lookup is built on an unreliable field and is therefore
   *demoted*** — from a scored sub-feature to, at most, a **very weak prior / cross-check**, not a
   meaningful contributor to `s_role_fit`. We must **not** reward a candidate for a good-looking title,
   nor punish a good description for a bad title. Where title and description disagree, **the description
   wins, unconditionally.**
2. **The real role label should be derived from the *descriptions themselves*, not the title.** Instead
   of "what does `current_title` say," we ask "**does any `career_history[].description` actually
   describe production retrieval / ranking / recsys / search / applied ML?**" — answered by the semantic
   `s_role_fit` cosine to the frozen JD-intent embedding. This is the only career signal the data lets us
   trust.
3. **Honeypot/decoy resistance.** Because titles are scrambled, any scheme that leans on titles is
   exploitable by profiles whose titles *look* adjacent. Leaning on descriptions (and the honeypot
   integrity checks in §4) is what actually keeps decoys out of the top-100.

> **Net change:** keep the affinity table in `config/scoring_config.yaml` for transparency and as a
> last-resort fallback when a profile has *no usable description text*, but its weight in the role score
> drops to near-zero. `s_role_fit` over descriptions becomes effectively the **sole** role signal. This
> strengthens — not weakens — the plan's North Star (read what they *did*, not what they *titled*).

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

#### 4.1 What "we do not special-case honeypots into the top — the detector is a safety net" means

This sentence is making a **design-philosophy claim**, and it's worth unpacking because it's easy to
misread as "we ignore honeypots."

A naïve solution to *"don't rank honeypots in the top-100"* would be to **build the system around hunting
honeypots** — write lots of brittle rules to *detect and forcibly eject* them. We are saying the
opposite: **we don't architect the ranker around honeypots at all.** Here's the reasoning:

- **A honeypot is, by construction, a profile that *looks* great on keywords but falls apart when you
  read the career.** That is the *exact same failure mode* the JD's keyword-trap is testing. So a ranker
  that correctly reads **career-vs-claims** (our dominant `s_role_fit` feature, §2) will **naturally**
  score honeypots low — their fabricated/contradictory descriptions simply don't embed near
  "built production retrieval/ranking at a product company." We don't need a special "is-honeypot?"
  branch to keep them out of the top; **doing the main job well already keeps them out.** In other words,
  *avoiding honeypots is a side-effect of being a correct reader, not a separate feature.*

- **The explicit honeypot detector (§4 rules) is therefore a *safety net*, not the primary mechanism.**
  Its job is to catch the *structural* impossibilities a semantic reader might miss — e.g. `years_of_
  experience` not matching `Σ duration_months`, "expert in 10 skills with `duration_months == 0`",
  `education end_year < start_year`, tenure before a company existed. These are *integrity checks on the
  data*, layered **before** ranking, so that even if a fabricated profile somehow scored well, the
  ×0.01 honeypot penalty (§2.0.c) removes it. Belt **and** suspenders.

- **Why phrase it this way at all?** Because the failure mode it guards against is over-engineering: if
  you *special-case* honeypots into/out of the ranking (e.g. hard-coding "any profile mentioning these 80
  patterns → tier 0"), you risk (a) **false-kills** on legitimate candidates who happen to look similar,
  and (b) a fragile system that only works because it memorized the *sample* honeypots and breaks on the
  hidden pool's variants. The robust posture is: **rank correctly so honeypots lose on merit; run the
  detector as an independent integrity gate so none can slip through.** This also reads well at Stage-4
  review — it shows we understood the trap conceptually, not just pattern-matched the examples.

In one line: ***honeypot avoidance is an emergent property of reading careers honestly; the detector is
the redundant guardrail, not the strategy.***

---

## 5. Local Proxy Evaluation Harness (because there is NO leaderboard)

With only **3 blind submissions**, we cannot probe the server. We build our own:
- Hand-label ~50 sample candidates (+ a few synthesized honeypots) into **relevance tiers 0–5**.
- Implement **NDCG@10, NDCG@50, MAP, P@10** and the exact composite formula.
- Use it to (a) calibrate weights, (b) regression-test every change, (c) decide what to submit.

### 5.1 ⚠️ Calibrate few knobs, not many — the label budget is tiny
**Honest constraint:** ~50 hand-labels is **far too little to fit** the ~40 individually-weighted skills
+ 6 top-level weights + role-affinity table + 6 penalty constants + the behavior model that currently
live in `config/scoring_config.yaml`. Fitting all of them on 50 points = guaranteed overfit, and it is
**indefensible at the Stage-5 interview**. So we deliberately **freeze most parameters by principle and
calibrate only a handful of macro knobs:**

- **Calibrate (the macro knobs):** the top-level weight split (`role/skill/exp/edu/loc` — including
  `w_edu`, which the JD's anti-credentialist stance suggests may want to drop toward ~0.05), the behavior
  band width (`min/max/neutral_base`), the **global penalty scale `p_scale`** (§2.5.e, replaces the 6
  hidden gate constants), the retrieval cutoff `k`, and the role-fit blend `w_dense`/`w_lex` (§2.5.a).
- **Freeze by principle (do not fit):** the per-skill weights — now **synonym-collapsed via
  `config.skills.skill_synonyms`** so e.g. `RAG` ≡ `Retrieval-Augmented Generation` scores once, not twice
  (§GLM-v2 #A4) — the role-affinity decimals (already demoted in §3.1.a), and the gate ordering.
- **Trust platform-verified signals over self-reported ones:** `skill_assessment_scores` (e.g.
  `CAND_0000001`: NLP 38.8) is harder to game than `proficiency`/`endorsements` (the same profile claims
  *advanced, 52 endorsements* in **Speech Recognition** — a domain the JD penalizes). Endorsements/
  proficiency must stay **subordinate** to description-derived evidence.

### 5.2 ⚠️ Top-10 hardening (NDCG@10 = 50% of the score)
The pool is mostly noise — measured title distribution is dominated by **Business Analyst, HR Manager,
Mechanical Engineer, Accountant, Project Manager…**; `Software Engineer` is only ~3,450 of 100,000 and
real AI/ML titles are rare. The JD agrees: *"we'd rather see 10 great matches than 1000 maybes."* With so
few genuine fits, **NDCG@10 (half the score) is decided by a handful of profiles.** Therefore, before
*every* submission we run an explicit **manual audit of the final top ~20**: read each career history,
confirm real production retrieval/ranking/recsys evidence, confirm no honeypot/decoy slipped in, and
confirm the reasoning is honest and rank-consistent. This human gate is cheap (≤20 profiles) and directly
protects 50% of the score — the pipeline is not trusted blindly for the top band.

### 5.3 ⚠️ Anti-keyword regression test (encode the JD's central warning as a test)

`docs/reference_docs/sample_submission.csv` is the **canonical bad output** — it ranks HR Managers,
Content Writers, Mechanical Engineers, Accountants and Marketing Managers in the top rows, carried purely
by AI-keyword count. It is the exact trap the JD describes. We turn that into a guard: compute the
**pure AI-keyword-count ordering** on the sample pool and **assert our top-10 diverges from it** (e.g.
rank-correlation near zero / low overlap). If our ranker ever starts resembling the keyword baseline, this
test fails loudly. Add as `tests/test_anti_keyword.py` (wired in P5).

---

## 6. Reasoning Generation (no LLM at runtime)

A **deterministic generator** fills real feature values into varied, honest sentences. The length target
is **exactly what `submission_spec.docx` §2 states: "a 1-2 sentence justification"** — *not* the ~100
words from the v1 notes (Appendix A item 5 is struck-through and superseded for this reason). Must pass
all six Stage-4 checks: **specific facts, JD connection, honest concerns, no hallucination, variation,
rank-tone consistency.** Example (1 sentence):
> "Analytics/Backend Engineer, 6.9 yrs; built Spark/Airflow pipelines feeding ML at a product co.,
> but no production retrieval/ranking; AI skills endorsement-light; response rate 0.34 — moderate
> availability."

**Anti-hallucination guarantee:** the generator may only emit values that are *literally present* in the
candidate JSON (years, titles, named skills, signal values). Because `title` is unreliable (see §3.1.a),
reasoning should describe the **work from `career_history[].description`**, not the job title, to avoid
contradicting the candidate's own record. Variation is achieved by sentence-template rotation keyed on
rank band + dominant feature, so the 10 sampled rows at Stage-4 read as genuinely different.

**Hallucination test = entity whitelist, not substring (from MINIMAX #10).** A substring check can miss a
hallucinated *year* or a company-name variant. Instead, **pre-extract a whitelist** of allowed entities
per candidate (skill names, employers, numeric years/values, signal numbers) and assert **every
content-bearing token the generator emits is in that whitelist**. This is a hard P6 exit test.

---

## 7. Compute & Latency Architecture (hard gate)

```
Offline (no time limit):  parse → honeypot scan → embed JD + career descriptions → cache features/index
Runtime (≤ 5 min, CPU):   load cache → hybrid retrieve top ~1–2K → feature rerank → top-100 → CSV
```
- Embeddings precomputed offline; runtime is vectorized NumPy/sklearn, no GPU, no network.
- Verify on a 16 GB CPU-only machine before each submission.

### 7.1 Measured budget (grounded in the real `data/originals/candidates.jsonl`)

The "≤5 min / ≤16 GB" gate must be **sized, not asserted.** Real measured dataset facts:

| Quantity | Measured value |
|---|---|
| Candidates | **100,000** |
| File size | **487 MB** JSONL |
| `career_history` descriptions | **~300,171** (avg ~3 per candidate), **avg ~396 chars** |
| Embedding model (`config/jd_embedding_meta.yaml`) | `all-MiniLM-L6-v2`, **384-dim**, normalized |

**What this implies for the budget split (spec §3 / §10.3 — precompute may exceed 5 min; only the
ranking step that writes the CSV must fit):**

| Stage | Work | Where | Budget |
|---|---|---|---|
| **Offline (uncapped)** | Embed ~300K descriptions with MiniLM on CPU; build index; cache vectors + parsed features | dev-time / Docker build | minutes → low tens of min; **no 5-min cap** |
| **Runtime (≤5 min, hard)** | `np.load` cached vectors → cosine vs JD-intent → feature math → sort → write 100 rows | `rank.py` | **must be ≤5 min on 100K, CPU-only** |

**Memory sizing (must stay ≪ 16 GB):**

- Candidate-level pooled vectors: `100,000 × 384 × 4 B ≈ 154 MB`.
- Per-description vectors (kept un-pooled for top-K-mean + recency at runtime): `~300,171 × 384 × 4 B ≈ 461 MB`.
- Raw JSONL is 487 MB but is **streamed/parsed offline**; runtime loads only the cached arrays + a
  compact feature table. Comfortably under 16 GB.

**Disk sizing (spec §3 hard limit: ≤ 5 GB intermediate state):**

| Artifact | Size |
|---|---|
| Vendored `all-MiniLM-L6-v2` weights | ~90 MB |
| `career_embeddings.npy` (per-description, un-pooled) | ~461 MB |
| `candidate_index.parquet` (scalar features + dates) | ~50–150 MB |
| Raw `candidates.jsonl` (input, not "intermediate") | 487 MB |
| **Total intermediate state** | **≈ 0.6–0.7 GB ≪ 5 GB** ✅ |

**Hard fail-tests (CI, run before every submission):**

1. **Latency gate:** `rank.py` on the full 100K must complete in **< 5 min wall** on a 16 GB CPU-only
   box; the job *fails the build* if it doesn't. Profile each phase (load / score / sort / write).
2. **No-network assertion:** the ranking step asserts zero outbound calls / no hosted-LLM imports.
3. **Cache-presence assertion:** runtime must find precomputed embeddings; it must **never** embed at
   runtime (embedding 300K texts live would blow the budget).

> **Why this matters:** the sandbox (§10.5) only exercises ≤100 candidates, so it will *always* pass
> trivially and gives **false confidence**. The number that actually gates Stage-3 reproduction is the
> **full 100K runtime**, which is why it is the explicit fail-test here.

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
| **Trusting `current_title`** (titles are scrambled vs descriptions — §3.1.a, 1249/3000 measured) | Demote title-affinity to near-zero; derive role from `career_history[].description` only; description wins on conflict |
| LLM-at-runtime creeps back in | Hard CI check: ranking step asserts no network import/calls |
| **Overfit to ~50 labels** (config has ~40 skill weights + many constants) | Calibrate ≤4 macro knobs only (§5.1); freeze the rest by principle; collapse skill synonyms |
| Misses 5-min budget on 100K | Precompute offline; **measured budget table + hard latency fail-test on full 100K (§7.1)**; sandbox's 100-row pass is *not* sufficient evidence |
| **False confidence from sandbox (≤100 rows)** | Gate on full-100K runtime, not the sandbox sample (§7.1) |
| Reasoning hallucinates / wrong length | Generator emits only profile-present values; **1–2 sentences per spec §2** (not ~100 words); describe work, not title |
| **Location field is `"City, Region"`; JD welcomes Hyderabad/Mumbai/Delhi NCR too** | Substring-match cities; widen preferred list beyond Noida/Pune (soft `w_loc=0.05` only) |
| `no_recent_code` gate is low-coverage (`github_activity_score=-1` is a sentinel for many) | Lean on recency of ML-relevant `career_history` descriptions, not GitHub score |
| Burn submissions blindly | Decide via local proxy eval + top-20 manual audit (§5.2), not by submitting |

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

5. Base the reasoning on the scoring config + candidate JSON, ~~~100 words~~, detailed but concise —
   generated **deterministically at runtime** (no hosted LLM call).
   > **Superseded — see §6.** This v1 note said "~100 words," but `submission_spec.docx` §2 specifies
   > **"a 1-2 sentence justification"** (its worked examples in §6 are single sentences). The authoritative
   > target is **1–2 sentences**, not ~100 words. A 100-word block across 100 rows also increases exposure
   > to the Stage-4 "variation / templated reasoning" penalty. The v1 wording is retained struck-through
   > for traceability only.

6. Latency is a criterion — see `docs/reference_docs/submission_spec.docx`.

7. Honeypots: if total experience doesn't match the sum of individual-organisation experience it may
   be a honeypot; more examples in `submission_spec.docx`. Honeypots are a rejection criterion.

