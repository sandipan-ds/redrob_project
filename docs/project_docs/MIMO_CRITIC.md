# MIMO CRITIC — Execution Plan Review

---

## What I Understood

This is the **Redrob Challenge** — rank the **top 100 candidates** from **100K** against a **Senior AI Engineer** JD. Hard constraints: **CPU-only, no network during ranking, ≤5 min, ≤16 GB RAM, 3 submissions max, no leaderboard.**

Scoring formula: `0.50·NDCG@10 + 0.30·NDCG@50 + 0.15·MAP + 0.05·P@10`

**The central trap:** The JD explicitly says "the right answer is NOT most AI keywords." The dataset is designed so keyword-stuffing profiles (Marketing Managers with AI skills listed) look great on paper but have career descriptions that prove zero ML/retrieval/ranking work. Meanwhile, a Backend Engineer who actually built a recsys is a fit even without AI in their title.

**Data reality from samples:**
- Titles are deliberately **decoupled from descriptions** (e.g., "Marketing Manager" → career description is mechanical engineering design)
- Many summaries are boilerplate ("Lately I've been curious about AI...")
- `redrob_signals` has sentinel values (`-1`, `{}`) meaning "unknown, not bad"
- Honeypots (~80) are structurally impossible profiles (>10% in top-100 = disqualification)

---

## What I Agree With (Strongly)

1. **"Read what they *did*, not what they *titled* themselves"** as the dominant feature (`s_role_fit` = 0.45 weight). This is exactly right. The samples prove it — `s_role_fit` over `career_history[].description` is the only trustworthy signal.

2. **Demoting `current_title` (§3.1.a).** The measured 1,249/3,000 title-description mismatches make this undeniable. Title affinity should be near-zero fallback only.

3. **Multiplicative penalty gates (§2.0.c, §4).** Honeypots at ×0.01, consulting-only at ×0.15, etc. This is the correct structure — "disqualifiers" must erase score, not just subtract. A keyword-stuffed profile can survive subtraction but not a ×0.01 multiplier.

4. **Sentinels as "unknown, never bad" (§1.3).** Mapping `-1` → neutral before scoring is essential. CAND_0000002 having `github_activity_score = -1` should not penalize her.

5. **"Belt AND suspenders" on honeypots (§4.1).** The philosophy that correct career-reading naturally avoids honeypots, with the detector as a safety net, is the right architecture. Building *around* honeypots would be brittle.

6. **Calibrating ≤4 macro knobs, not 40 (§5.1).** With ~50 hand-labels, fitting everything would guarantee overfit. Freezing per-skill weights and role-affinity decimals by principle is honest and defensible.

7. **§7.1 measured budget.** Actually computing memory (154 MB for 100K × 384-dim × 4B) instead of asserting it. The sandbox's 100-row test is correctly identified as false confidence — the real gate is the full 100K runtime.

8. **§5.2 top-10 manual audit.** With 50% of the score decided by NDCG@10 and very few genuine fits in the pool, a human gate on ~20 profiles is cheap insurance.

9. **1–2 sentence reasoning (§6), not ~100 words.** Correctly updated from v1 after reading `submission_spec.docx` §2.

---

## What I Disagree / Have Concerns About

### 1. `s_role_fit` as cosine similarity — what are we embedding, exactly?

The plan says cosine similarity of `career_history.description` to a "JD-intent embedding." But there's ambiguity:
- Is it cosine between each description and a single JD-intent vector? Or across all descriptions for a candidate?
- If pooled (mean-pooling ~3 descriptions per candidate), noise descriptions will dilute signal. CAND_0000001 has 2 descriptions, both data engineering — easy. But a candidate with 3 descriptions where only 1 is ML-relevant gets averaged down.
- The embedding model (`all-MiniLM-L6-v2`) is good but small — will it reliably distinguish "built retrieval/ranking" from "managed support tickets"?

**Concern:** The plan doesn't specify the exact pooling strategy or how multi-career-description candidates are handled. This matters because the *fraction of career spent on relevant work* is arguably more important than the average.

### 2. The skill-scoring approach feels under-specified

§2 mentions `w_skill = 0.25` but doesn't detail how `s_skill` is computed. The plan says "JD-relevant skills, endorsement/duration-weighted" but:
- Which skills are JD-relevant? The plan says "collapse synonyms" but doesn't list them.
- Endorsements are self-reported and gameable (CAND_0000001 has 52 endorsements in Speech Recognition — a domain the JD penalizes).
- `skill_assessment_scores` (platform-verified) are much more trustworthy but only some candidates have them.

**Concern:** With 25% of fit score riding on skills, the feature engineering here needs more clarity before implementation.

### 3. The role-affinity table is mostly dead weight

After §3.1.a correctly demotes `current_title`, the role-affinity matrix (`config/scoring_config.yaml → role_affinity`) is described as "near-zero fallback." If it's truly near-zero, why maintain it at all? It adds config complexity for a feature that's been identified as unreliable. A simpler design: just use `s_role_fit` and drop the title lookup entirely.

### 4. The behavioral multiplier range might be too conservative

`M_behavior ∈ [0.5, 1.1]` means availability can at most change rank by ~±10-15%. But the signals doc says "a perfect-on-paper candidate who hasn't logged in for 6 months is not actually available." If that's the case, shouldn't a truly stale candidate (last_active 6+ months ago, response rate 0.1) get penalized more than ×0.85? The current range might not push unavailable candidates down hard enough.

### 5. The sample_submission.csv is a trap — and the plan doesn't mention it

The provided `sample_submission.csv` ranks HR Managers, Content Writers, Mechanical Engineers, Accountants, and Marketing Managers in the top 10. This is clearly a **naïve baseline** (probably keyword-count based) that demonstrates exactly what the JD says *not* to do. The plan doesn't explicitly call this out as a cautionary artifact, which could trip up a new reader.

### 6. Location matching needs more precision

§1.1 says "match city by substring (data is `City, Region`)" and `w_loc = 0.05`. But the JD lists "Noida/Pune preferred (Hyderabad/Mumbai/Delhi NCR also welcome)." The location field format varies — some are just city names (`Toronto`, `Austin`), some are `City, Region` (`Chennai, Tamil Nadu`). Simple substring matching should work, but the plan doesn't address candidates who are "willing to relocate" vs. not — is that factored into location scoring?

### 7. Missing: what happens if you run out of genuinely good candidates?

The pool is dominated by Business Analysts, HR Managers, Mechanical Engineers, Accountants. There may be very few genuine Senior AI Engineer fits (maybe 50-200?). The plan assumes you can find 100 good candidates, but if the true positive pool is smaller than 100, your top-100 will necessarily include "best of the bad" candidates. The plan should address how to handle this gracefully — particularly how to rank the bottom half of the 100 where there's no good signal to differentiate.

---

## Overall Verdict

**~85-90% agreement.** This is an unusually well-thought-out plan. The analysis of the JD traps, the title-description mismatch measurement, the sentinel handling, and the honest acknowledgment that weights are engineering choices (not spec-provided) are all excellent. The biggest gaps are in the details of `s_role_fit` implementation, skill scoring specifics, and edge cases at the bottom of the top-100. None of these are fatal — they're implementation questions that need answering during P3-P4.
