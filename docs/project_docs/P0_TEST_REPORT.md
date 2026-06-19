# P0 Test Report — Repo Scaffold & Data Loading

**Phase:** P0  
**Date:** 2026-06-19  
**Test file:** `tests/test_p0.py`  
**Result:** ✅ 17/17 passed (0.33s)

---

## What P0 Was About

P0 is the foundation phase. Before any scoring or ranking logic can be built, two things must work reliably:

1. **The scoring config loads correctly** — `config/scoring_config.yaml` is the single source of truth for all weights, thresholds, and rules. If it loads broken or with bad values, every downstream phase is wrong.
2. **Candidate data loads and validates correctly** — the ranker needs to ingest up to 100,000 candidate records from a JSONL file without crashing, silently corrupting data, or choking on known edge cases like sentinel values (`-1` for missing GitHub scores, `{}` for empty skill assessments).

The exit criterion stated in the execution plan was: *"Loads 100K JSONL, validates schema."* The tests operationalize exactly that.

---

## How the Tests Were Designed

The 17 tests are split across three groups, each targeting a distinct failure mode.

**Group 1 — Config Loader (4 tests)**  
These verify that `scoring_config.yaml` is structurally sound before any scoring runs. The key check is that the five component weights (role fit, skill, experience, education, location) sum to 1.0 — if they don't, the scoring formula silently produces wrong scores. A test also confirms that pointing the loader at a non-existent file raises a clean error rather than a silent failure.

**Group 2 — Sample JSON Loading (6 tests)**  
These run against the real `data/samples/sample_candidates.json` file — the actual data provided by the challenge. They check that every candidate has the required top-level sections, that `candidate_id` matches the `CAND_XXXXXXX` format the submission validator enforces, that `career_history` is never empty (a schema requirement), and that sentinel values like `github_activity_score = -1` and `offer_acceptance_rate = -1` are accepted as valid rather than rejected as errors. Sentinels are explicitly called out in the execution plan as "unknown, never bad."

**Group 3 — JSONL Loading & Schema Validator (7 tests)**  
These use synthetic candidate records written to temporary files to test the JSONL streaming path — the format the actual 100K dataset will arrive in. They cover the happy path (valid record loads correctly), the default lenient mode (bad records are skipped with a warning, not a crash), the strict mode (bad records raise a `SchemaValidationError`), and batch loading of multiple records. Three unit tests also hit the schema validator directly to confirm it catches a malformed `candidate_id` and missing top-level keys.

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

---

## How We Know It Passed

pytest was run with the `-v` flag, which prints each test name and its result individually. The final line of output was:

```
17 passed in 0.33s
```

Every test showed `PASSED` with no warnings or errors. The exit code was `0`, which is pytest's signal for a fully green run. No tests were skipped or marked as expected failures.

---

## What This Unlocks

With P0 green, the project has a stable foundation: the config is readable and sane, and candidate data — in both the sample JSON format and the production JSONL format — loads cleanly with proper validation. P1 (criteria mapping and JD-intent embedding) and P2 (honeypot and disqualifier detection) can now be built on top of these loaders with confidence.
