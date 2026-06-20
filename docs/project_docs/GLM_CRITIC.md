# GLM Critic — Review of the Redrob Execution Plan

> **Reviewer:** GLM-5.2 (opencode-go/glm-5.2)
> **Reviewed artifact:** `docs/project_docs/EXCEUTION_PLAN.md` (v2, Production-Grade)
> **Cross-referenced primary sources:** `job_description.docx`, `submission_spec.docx`,
> `redrob_signals_doc.docx` (extracted and read in full), `data/samples/candidate_schema.json`,
> `data/samples/sample_candidates.json`, `docs/reference_docs/sample_submission.csv`,
> `docs/reference_docs/validate_submission.py`, `docs/reference_docs/submission_metadata_template.yaml`.
> **Cross-referenced project docs:** `criteria_map.md`, `SYSTEM_DESIGN.md`, `PHASED_BUILD_PLAN.md`,
> `P0_TEST_REPORT.md`, `P1_TEST_REPORT.md`, `config/scoring_config.yaml`, `config/jd_embedding_meta.yaml`.
> **Repo state at review:** P0 (17/17) and P1 (20/20) reported green; 9 git commits (see concern #4).

---

## 1. What the project is

**Redrob Hackathon — Intelligent Candidate Discovery & Ranking.** Rank the **top 100** of a
**100,000**-candidate pool against a *Senior AI Engineer — Founding Team* JD. Scored on a hidden
tier ground truth: `0.50·NDCG@10 + 0.30·NDCG@50 + 0.15·MAP + 0.05·P@10`. Hard gates: the ranking
step must run **≤5 min wall-clock, ≤16 GB RAM, CPU-only, network OFF, ≤5 GB intermediate disk**.
**3 blind submissions, no leaderboard.** Five evaluation stages: format validation → scoring →
**code reproduction + honeypot-rate check (>10% honeypots in top-100 = Stage-3 DQ)** → manual
review (6 reasoning checks + **git-history authenticity** + code quality) → defend-your-work
interview.

**The dataset is adversarial by design:**
- Pool is dominated by irrelevant roles (HR Manager, Mechanical Engineer, Accountant, Project
  Manager…); real ML titles are rare (~3,450 Software Engineers of 100,000).
- ~80 **honeypots** (impossible profiles, forced to tier 0).
- **Titles are deliberately scrambled vs. career descriptions** — the JD explicitly states the
  "right answer" is *not* "find candidates whose skills section contains the most AI keywords"
  (that's a built-in trap); the real signal is the gap between what a profile *says* and what a
  career *shows*. The provided `sample_submission.csv` demonstrates the trap in action: HR Manager /
  Content Writer / Marketing Manager occupy the top rows, carried by "8–9 AI core skills."

**The plan's architecture (P0/P1 reported green, 17/17 and 20/20 tests):**
- Two-stage split: **offline precompute** (uncapped) + **runtime rerank** (≤5 min, NumPy-only).
  Offline embeds the JD-intent + ~300K career descriptions with `all-MiniLM-L6-v2` (384-d) and
  caches vectors; runtime is a dot-product retrieve → feature rerank → sort → CSV.
- `fit_score = 0.45·role + 0.25·skill + 0.15·exp + 0.10·edu + 0.05·loc`;
  `final = fit_score · M_behavior · P_penalty`.
- **Dominant signal:** `s_role_fit` = max-pooled cosine of `career_history[].description` to a
  frozen JD-intent vector; **titles demoted to near-zero** (measured 1,249/3,000
  title↔description mismatches).
- `M_behavior ∈ [0.5, 1.1]` (sentinels `-1`/`{}` = neutral, never bad); `P_penalty` = multiplicative
  gates (honeypot 0.01, consulting 0.15, research 0.20, no-recent-code 0.25, domain-mismatch 0.30,
  langchain-junior 0.40).
- Local proxy eval (~50 hand-labels), calibrate **≤4 macro knobs** (freeze the rest to avoid
  overfitting 50 labels). 1–2 sentence deterministic reasoning (emit only profile-present values,
  describe work not title).

---

## 2. How much I agree

**~85–90%.** The core thesis is correct and well-grounded in the primary sources (not just the
plan's own summary of them). Specifically I agree with:

- **North Star (read careers, not keywords)** — matches the JD's explicit warning verbatim; the
  sample CSV is live proof of the trap. This is the right thing to optimize.
- **Offline/online split + NumPy-only runtime** — necessary and sufficient; ~154 MB pooled vectors
  make the 5-min/CPU budget trivial. The spec (§10.3) explicitly allows precompute to exceed 5 min.
- **Multiplicative gates, not additive penalties** — a keyword-stuffed honeypot can out-earn a
  subtraction but cannot out-earn ×0.01. This directly protects NDCG@10 *and* the Stage-3 honeypot
  filter.
- **Max-pool over descriptions** — mean would dilute one real ML stint under marketing/ops noise;
  "is there *any* evidence they built retrieval/ranking in production?" is exactly the question the
  JD asks.
- **Distrusting titles** — the single most important data-reality correction in the plan, and
  empirically measured (1,249/3,000), not assumed.
- **Sentinels = neutral; behavior as a narrow-band modulator (≤+10 / −50%)** — availability must
  modulate, not replace, fit, or an eager mediocre candidate corrupts NDCG@10.
- **≤4 calibrated knobs, freeze the rest** — fitting ~40 skill weights + 6 gate constants on 50
  labels would be indefensible at the Stage-5 interview.
- **1–2 sentence reasoning, describe work not title, no hallucination** — maps 1:1 to the Stage-4
  checks; the plan caught and superseded the v1 "~100 words" error.
- **Honeypot avoidance as emergent + detector as safety net** — special-casing the 80 visible
  honeypots overfits and risks false-kills on the hidden pool.
- **Full-100K latency gate as the real CI test, not the ≤100 sandbox** — the sandbox gives false
  confidence and the plan correctly says so.
- **Intellectual honesty throughout** — the plan repeatedly states "no document gives us these
  numbers; they're our priors." That is exactly the right posture for a defend-your-work interview.

---

## 3. What I don't agree with / concerns

In rough priority order. The first two I would push back on hard; the rest are tuning/process.

### Concern 1 — A single 384-d embedding cosine is too blunt to decide 50% of the score at the top

45% of fit — and effectively all of NDCG@10 — rides on one signal: cosine of career descriptions
to *one* hand-distilled 1,200-char JD-intent vector via `all-MiniLM-L6-v2` (a short-sentence
similarity model, not a fine-grained "did-they-ship-it" discriminator). Genuine fits will cluster
at near-identical cosine and then be ordered by `candidate_id` — exactly where NDCG@10 is won and
lost.

**Fix:** add an orthogonal role signal on the shortlist rerank — either a multi-query JD-intent
(separate frozen vectors for "retrieval/ranking," "production ML at product companies," "eval
frameworks," take the max cosine) and/or a small hand-built production-evidence lexicon
("shipped / deployed / rolled out / served / A-B / inference service" + recsys / retrieval /
search terms) combined with the cosine. The plan mentions multi-intent as an option but commits to
a single frozen vector — that is the wrong call when half the score lives in the top 10.

### Concern 2 — `research_only` as a keyword-*absence* rule is the keyword trap in mirror form

The detector (PHASED_BUILD_PLAN §P2) flags "no production/deployed/shipped/users/scale across
descriptions." This is the keyword trap inverted: it rewards the *presence* of the exact keywords
we guessed and penalizes their absence. A real production ML engineer whose descriptions say
"launched / rolled out / served / A-B tested / inference service" — but not the specific words on
the list — gets falsely gated ×0.20. That is a disqualifier triggered by *missing the exact
keywords we guessed* — the same failure mode the North Star rejects.

It also risks the false-kills the plan itself warns against (§4.1: "false-kills on legitimate
candidates who happen to look similar").

**Fix:** soften `research_only` from a hard gate to a soft feature unless *all three* hold:
(a) no production-lexicon match, (b) no product-company tenure, (c) explicit research-only
signals (academic / lab / "research scientist" framing in descriptions). And make the **gate
scores** one of the calibrated macro knobs — a single global scale preserves the ordering while
letting data set severity. Right now six independent multipliers are frozen by assertion, which is
~6 hidden knobs the plan does not count toward its "≤4 calibrated" budget.

### Concern 3 — Proxy calibration on 50 self-labels is circular

The proxy labels are hand-set by the same person who derived the weights, against the same JD.
"Maximize NDCG@10 on the proxy" will almost by construction re-confirm the priors — the labeler's
mental model *is* the formula. The plan is honest that 50 labels cannot fit 40 knobs but misses
that 50 self-labels also cannot *independently validate* the 4 macro knobs.

**Fix:** have a second person label a held-out ~20 blind and check inter-rater agreement, or
synthesize adversarial near-miss decoys (profiles that look like fits by the formula but should
not be) and confirm they rank below genuine fits. Either gives the proxy set some independence
from the weight-setter.

### Concern 4 — The git history is already off-track for Stage-4

Stage-4 penalizes "flat history with no iteration." The actual log (9 commits) has two
`Initial commit`s, a `first commit`, a merge, and only two phase-labeled commits; **P0 has no
labeled commit** — it is bundled into `first commit`. The plan promises "small per-phase commits"
(PHASED_BUILD_PLAN §0) but execution already contradicts it. Cheap to fix now; expensive once
P2–P8 land in big dumps.

**Fix:** going forward, one labeled commit per phase with the documented prefix
(`P2: honeypot + disqualifier gates`, etc.); optionally a small retroactive cleanup commit that
documents the P0/P1 split in `CHANGELOG.md` without rewriting history.

### Concern 5 — No explicit anti-keyword regression test

The risk register lists "honeypot/keyword regression tests" but never specifies the most
JD-relevant one: assert the ranker's top-10 has *low correlation with a pure AI-keyword-count
baseline* on the sample. The provided `sample_submission.csv` is the canonical bad output — a
test that our top-10 does not resemble it (e.g., overlap < some threshold, or rank-correlation
near zero) is a cheap, powerful guard that directly encodes the JD's central warning.

**Fix:** add to `tests/test_p5.py` (or a new `test_anti_keyword.py`) a regression that computes
the AI-keyword-count ordering on the sample and asserts our top-10 diverges from it.

### Concern 6 — Education at 0.10 is too high for this JD

The JD is explicitly anti-credentialist ("5–9 is a range, not a requirement," title-chaser
warning, focus on shipped systems) and never mentions CGPA. tier_3 / CGPA is noisy
(`CAND_0000001` is LPU / tier_3 yet a reasonable data-eng-adjacent fit). I would cut `w_edu` to
~0.05 and move the freed weight to role-fit (→ 0.50) or a production-evidence feature. At minimum
make it a calibrated knob instead of frozen.

### Concern 7 — Minor gaps

- **(a) The ≤5 GB disk budget is never sized in §7.** Vendored MiniLM (~90 MB) + 461 MB un-pooled
  vectors + parquet + 487 MB raw JSONL is comfortably fine, but the measured-budget table should
  say so — disk is a hard Stage-3 constraint alongside time and RAM.
- **(b) `consulting_only`'s 10-company list vs. the JD's "etc."** Hidden-pool variants
  (Genpact, LTIMindtree, Mphasis is listed but others may not be) escape; the "prior
  product-company experience" exemption needs a real product-co detector, currently unspecified.
  Leaning on `current_industry != "IT Services"` + `company_size` bands would generalize better
  than a name list.
- **(c) `langchain_only_junior` (×0.40) is the softest gate but detection-hard.** A senior ML
  engineer who recently *added* LangChain could false-fire. Consider dropping it as a gate and
  letting role-fit + the exp band handle juniors naturally — a LangChain-only junior's
  descriptions will not embed near "built retrieval/ranking in production" anyway, so the dominant
  feature already demotes them without a brittle rule.

---

## 4. Net assessment

The strategy and architecture are sound and I would build largely as written. The two changes I
would insist on **before P3** are:

1. A second, orthogonal role signal for the top-band rerank (concern 1).
2. Replacing the keyword-absence `research_only` gate with a high-confidence conjunctive detector
   and folding the gate magnitudes into the calibrated-knob budget (concern 2).

Everything else is calibration or process hygiene and can be addressed in P5 / P7 without
re-architecting. The plan's strongest property is its intellectual honesty about which numbers are
principled vs. guessed — preserving that honesty into the actual code (no magic numbers, everything
in `scoring_config.yaml`, everything defensible at the Stage-5 interview) is more important than
any individual weight value.
