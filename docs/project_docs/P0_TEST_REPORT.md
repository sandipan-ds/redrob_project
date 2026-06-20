# P0 Test Report — Repo Scaffold & Data Loading

**Phase:** P0  
**Date:** 2026-06-19 (updated 2026-06-20)  
**Test file:** `tests/test_p0.py`  
**Result:** ✅ **20/20 passed** (~5s; ~0.3s without the 100K smoke test)  
**Environment:** Python **3.11.9** venv (see "Environment note" below)

> **Update log (2026-06-20):** original report was 17/17. Added a **100K loader smoke test**
> (`TestFull100KLoad`, +3 tests → 20 total) that exercises the real P0 exit criterion against the
> production `data/originals/candidates.jsonl`, and rebuilt the environment on Python 3.11 after a
> dependency-compatibility issue. Both are documented below.

---

## What P0 Was About

P0 is the foundation phase. Before any scoring or ranking logic can be built, two things must work reliably:

1. **The scoring config loads correctly** — `config/scoring_config.yaml` is the single source of truth for all weights, thresholds, and rules. If it loads broken or with bad values, every downstream phase is wrong.
2. **Candidate data loads and validates correctly** — the ranker needs to ingest up to 100,000 candidate records from a JSONL file without crashing, silently corrupting data, or choking on known edge cases like sentinel values (`-1` for missing GitHub scores, `{}` for empty skill assessments).

The exit criterion stated in the execution plan was: *"Loads 100K JSONL, validates schema."* The tests operationalize exactly that.

---

## How the Tests Were Designed

The 20 tests are split across four groups, each targeting a distinct failure mode.

**Group 1 — Config Loader (4 tests)**  
These verify that `scoring_config.yaml` is structurally sound before any scoring runs. The key check is that the five component weights (role fit, skill, experience, education, location) sum to 1.0 — if they don't, the scoring formula silently produces wrong scores. A test also confirms that pointing the loader at a non-existent file raises a clean error rather than a silent failure. (The config has since grown a `role_fit:` block and `penalties.p_scale`, plus a `skills.skill_synonyms` collapse map — all of which load under the same validator without breaking the weights-sum invariant.)

**Group 2 — Sample JSON Loading (6 tests)**  
These run against the real `data/samples/sample_candidates.json` file — the actual data provided by the challenge. They check that every candidate has the required top-level sections, that `candidate_id` matches the `CAND_XXXXXXX` format the submission validator enforces, that `career_history` is never empty (a schema requirement), and that sentinel values like `github_activity_score = -1` and `offer_acceptance_rate = -1` are accepted as valid rather than rejected as errors. Sentinels are explicitly called out in the execution plan as "unknown, never bad."

**Group 3 — JSONL Loading & Schema Validator (7 tests)**  
These use synthetic candidate records written to temporary files to test the JSONL streaming path — the format the actual 100K dataset will arrive in. They cover the happy path (valid record loads correctly), the default lenient mode (bad records are skipped with a warning, not a crash), the strict mode (bad records raise a `SchemaValidationError`), and batch loading of multiple records. Three unit tests also hit the schema validator directly to confirm it catches a malformed `candidate_id` and missing top-level keys.

**Group 4 — Full 100K Load Smoke Test (3 tests, `TestFull100KLoad`)**  
These run against the **real production `data/originals/candidates.jsonl` (~487 MB, ~100K rows)** — the exact P0 exit criterion ("loads 100K JSONL, validates schema"), which Groups 2–3 only approximated with samples and synthetic records. They assert: (1) the loader returns a **generator, not a list** (so the 100K pool streams and never materializes — protecting the 16 GB budget); (2) the first 1,000 real records validate in strict mode and look well-formed; (3) the **entire file** streams once with ≥90K records loaded, **zero malformed IDs**, **zero duplicate IDs**, negligible skip-rate, and the whole pass completes well under a sane time ceiling (a loader-regression guard, distinct from the `rank.py` ≤5-min ranking budget). The group is **skipped gracefully** (not failed) when the production file is absent, so fresh clones / CI without the dataset still go green.

---

## Passing Criteria

A test passes when the assertion it makes holds true. The criteria across the suite were:

- Config loads as a dict with all 9 required sections present and weights summing to 1.0 ± 0.01
- Sample candidates load as a non-empty list with no missing required keys
- Every `candidate_id` matches `^CAND_[0-9]{7}$`
- Every candidate has at least one `career_history` entry
- Sentinel values (`-1`, `{}`) are accepted without validation errors
- A valid JSONL record round-trips correctly (written to temp file, read back, ID matches)
- An invalid JSONL record is skipped in lenient mode and raises in strict mode
- The schema validator returns an empty error list for a well-formed record and a non-empty list for a bad one
- The full 100K file streams as a generator with ≥90K valid records, unique `CAND_` IDs, and a streaming-load time under the regression ceiling

---

## How We Know It Passed

The suite was re-run on a clean Python 3.11 environment. The final line of output was:

```
20 passed in 5.33s     # ~5s is dominated by the full 100K streaming smoke test
```

Every test showed `PASSED` with no errors. The exit code was `0`, which is pytest's signal for a fully green run. No tests were marked as expected failures. (The 100K smoke test will report as **skipped** rather than passed on machines that don't have `data/originals/candidates.jsonl`.)

---

## Environment note (Python 3.11 — important for reproduction)

The pinned dependencies (`numpy==1.26.4`, `scikit-learn==1.4.2`, `torch==2.3.1`,
`sentence-transformers==3.0.1`) have **no prebuilt wheels for Python 3.12+/3.14** — on those interpreters
`pip install -r requirements.txt` falls back to a source build and fails. The `.venv` must be created with
**Python 3.11** (verified on 3.11.9):

```
py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

The Stage-3 Docker image must likewise use `python:3.11-slim`. This is documented in the `requirements.txt`
header and is a hard reproducibility requirement.

---

## What This Unlocks

With P0 green, the project has a stable foundation: the config is readable and sane, and candidate data — in both the sample JSON format and the **real production JSONL** (now exercised end-to-end at 100K scale) — loads cleanly with proper validation. P1 (criteria mapping and JD-intent embedding) and P2 (honeypot and disqualifier detection) can now be built on top of these loaders with confidence.
