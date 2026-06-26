# redrob_project — Redrob Candidate Ranker

Rank the top 100 of a 100,000-candidate pool against a Senior AI Engineer JD.
Submission is a single CSV. CPU-only, ≤ 5 min wall-clock at ranking runtime,
≤ 16 GB RAM, no network during ranking.

Built per `docs/project_docs/EXCEUTION_PLAN.md` (canonical spec) and
`docs/project_docs/PHASED_BUILD_PLAN.md` (build sequence). Test reports per
phase in `docs/project_docs/P[N]_TEST_REPORT.md`.

## Quick start

### Local dev (Python 3.11)

```bash
py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt

# 50-sample end-to-end (offline, no network):
.\.venv\Scripts\python -m src.precompute --candidates data/samples/sample_candidates.json --output artifacts/sample
.\.venv\Scripts\python -m src.rank       --candidates data/samples/sample_candidates.json --cache artifacts/sample --out outputs/sample_submission.csv --top-n 100
```

### Full 100K pool (offline precompute, then run)

```bash
# Precompute (uncapped per submission_spec §10.3; ~70 min on a 100K pool):
python -m src.precompute --candidates data/originals/candidates.jsonl --output artifacts/full

# Rank the top 100 (≤ 5 min on CPU; measured 20.3s on full 100K):
python -m src.rank --candidates data/originals/candidates.jsonl --cache artifacts/full --out outputs/submission.csv --top-n 100
```

> **Note:** `artifacts/full/` (the production precompute, ~440 MB) is
> **gitignored**. It exceeds GitHub's 100 MB file size limit. The
> precompute is reproducible from `data/originals/candidates.jsonl` in
> ~70 min on 8-core CPU. The vendored `artifacts/sample/` (50-sample
> precompute, ~225 KB) is sufficient for dev, tests, and the HF Space
> sandbox.

### Docker (offline, sandbox-compatible)

```bash
docker build -t redrob-ranker .
docker run --rm --network none \
    -v ${PWD}/data:/work/data \
    -v ${PWD}/outputs:/work/outputs \
    redrob-ranker \
    python -m src.rank --candidates /work/data/candidates.jsonl \
                           --cache /work/artifacts \
                           --out /work/outputs/submission.csv
```

The `Dockerfile` vendors the `all-MiniLM-L6-v2` model into
`models/all-MiniLM-L6-v2/` so the image runs offline. The
`src/rank.py` no-network guard (overrides `socket.socket` at import time)
fails loudly if anything tries to make a network call during ranking.

## Hard constraints (submission_spec §3)

- **CPU-only** at ranking runtime. No GPU.
- **≤ 5 min** wall-clock for the full-100K ranking step. Precompute is uncapped.
- **≤ 16 GB RAM.**
- **No network** at ranking runtime. No hosted LLM, no API calls.
- **No LLM at runtime.** All ML/embedding work is offline, dev-time only.

## Scoring

The hidden ground truth is a tier-based score:
`composite = 0.50·NDCG@10 + 0.30·NDCG@50 + 0.15·MAP + 0.05·P@10`.
NDCG@10 is **half the score** — the top-10 is the single highest-leverage
band to get right.

## Architecture

```
fit_score = 0.45·s_role_fit + 0.25·s_skill + 0.15·s_exp
          + 0.10·s_edu + 0.05·s_loc
final     = fit_score × M_behavior × P_penalty
```

- `s_role_fit` (DOMINANT): multi-query dense cosine + lexical match, top-K-mean
  pooled, recency-weighted. Reads `career_history[].description`, NOT
  `current_title` (titles are scrambled vs descriptions in the data —
  measured 1,249/3,000 mismatches).
- `M_behavior ∈ [0.5, 1.1]`: narrow-band multiplier on the universally-
  populated signals (last_active_date, recruiter_response_rate,
  interview_completion_rate).
- `P_penalty`: honeypot (×0.01) + 5 non-honeypot gates (consulting_only,
  research_only, no_recent_code, domain_mismatch, langchain_only_junior)
  combined as `min(non_hp) × Π(other non_hp)^0.5` with a single
  calibratable `p_scale` knob.

Two-stage split (allowed by spec §10.3 — only the *ranking step* must
fit in ≤ 5 min):

- **Offline (uncapped)**: embed JD-intent + candidate career descriptions
  with `all-MiniLM-L6-v2`; cache vectors.
- **Runtime (≤ 5 min hard)**: `np.load` cached vectors → cosine → feature
  math → sort → write 100 rows.

## Project layout

```
config/                  scoring_config.yaml (single source of truth) + frozen JD-intent embeddings
data/samples/            50-sample dev set (sample_candidates.json)
data/originals/          full 100K pool (candidates.jsonl) — gitignored
docs/                    reference_docs (spec/JD/signals), project_docs (plan/test reports)
models/all-MiniLM-L6-v2/ vendored model weights (P7; ~91 MB)
src/                     production code: config_loader, data_loader, jd_embedding, features/*,
                         honeypot, disqualifiers, scoring, precompute, retrieve, rank, reasoning
scripts/calibrate.py     P5 calibration driver
tests/                   pytest suite: test_p0..p7 + test_anti_keyword
artifacts/               precomputed embeddings (gitignored except small sample/)
outputs/                 submission CSVs (gitignored)
submission_metadata.yaml spec-required metadata
Dockerfile               P7 offline-reproducible image
```

## Tests

```bash
pytest tests/ -q           # 158 tests, ~75s on 8-core
```

The suite covers P0–P7 and the anti-keyword regression guard.
The full-100K latency test is opt-in (`RUN_P4_FULL=1`) because the
precompute pass takes minutes on the production file.
