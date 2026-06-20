"""
test_p4.py — P4 exit criterion tests.

P4 exit criterion (PHASED_BUILD_PLAN §P4):
  - Running the pipeline on the ≤100-row sample produces a CSV that
    passes validate_submission.py (adapted for <100 rows in dev).
  - On the full 100K (skip if absent): produces exactly 100 rows and
    the runtime ranking step completes in < 5 min on CPU.
  - No honeypot from the sample appears in the produced top-N.
  - No network calls at runtime (submission_spec §3).
"""

from __future__ import annotations

import csv
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from src.config_loader import load_config

CFG = load_config()

REPO = Path(__file__).parent.parent
SAMPLE = REPO / "data" / "samples" / "sample_candidates.json"
SAMPLE_CACHE = REPO / "artifacts" / "sample"
SAMPLE_OUT = REPO / "outputs" / "sample_submission.csv"
FULL_POOL = REPO / "data" / "originals" / "candidates.jsonl"

CAND_ID_RE = re.compile(r"^CAND_[0-9]{7}$")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def precomputed_sample() -> dict:
    """Run precompute on the 50-sample and return the meta dict."""
    if not SAMPLE.exists():
        pytest.skip("50-sample file missing")
    from src.precompute import run_precompute
    meta = run_precompute(SAMPLE, SAMPLE_CACHE, force=True)
    return meta


@pytest.fixture(scope="module")
def ranked_sample(precomputed_sample) -> Path:
    """Run rank.py on the 50-sample and return the output CSV path."""
    out = REPO / "outputs" / "sample_submission.csv"
    if out.exists():
        out.unlink()
    rc = subprocess.run(
        [
            sys.executable, "-m", "src.rank",
            "--candidates", str(SAMPLE),
            "--cache", str(SAMPLE_CACHE),
            "--out", str(out),
            "--top-n", "50",
        ],
        cwd=REPO, capture_output=True, text=True, timeout=120,
    )
    assert rc.returncode == 0, f"rank.py failed:\nSTDOUT:\n{rc.stdout}\nSTDERR:\n{rc.stderr}"
    return out


# ---------------------------------------------------------------------------
# Precompute
# ---------------------------------------------------------------------------

class TestPrecompute:
    def test_produces_artifacts(self, precomputed_sample):
        assert (SAMPLE_CACHE / "career_embeddings.npy").exists()
        assert (SAMPLE_CACHE / "candidate_offsets.npz").exists()
        assert (SAMPLE_CACHE / "precompute_meta.yaml").exists()

    def test_embeddings_shape_and_unit_norm(self, precomputed_sample):
        from src.precompute import load_precomputed
        emb, off, meta = load_precomputed(SAMPLE_CACHE)
        assert emb.ndim == 2
        assert emb.shape[1] == 384  # all-MiniLM-L6-v2 dim
        norms = np.linalg.norm(emb, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_offsets_match_candidate_count(self, precomputed_sample):
        from src.precompute import load_precomputed
        _emb, off, meta = load_precomputed(SAMPLE_CACHE)
        assert len(off["candidate_ids"]) == 50  # 50-sample
        assert len(off["offsets"]) == 51  # M+1
        assert int(off["offsets"][-1]) == meta["num_descriptions"]


# ---------------------------------------------------------------------------
# Rank — end-to-end on the 50-sample
# ---------------------------------------------------------------------------

class TestRankSample:
    def test_produces_csv(self, ranked_sample):
        assert ranked_sample.exists()
        assert ranked_sample.stat().st_size > 0

    def test_csv_format(self, ranked_sample):
        rows = list(csv.reader(ranked_sample.open(encoding="utf-8")))
        assert rows[0] == ["candidate_id", "rank", "score", "reasoning"]
        # 50-sample → 50 data rows.
        assert len(rows) - 1 == 50, f"Expected 50 rows, got {len(rows) - 1}"

    def test_ranks_unique_and_1_to_n(self, ranked_sample):
        rows = list(csv.reader(ranked_sample.open(encoding="utf-8")))[1:]
        ranks = [int(r[1]) for r in rows]
        assert ranks == list(range(1, len(ranks) + 1)), "Ranks not 1..N in order"
        assert len(set(ranks)) == len(ranks), "Duplicate ranks"

    def test_score_non_increasing(self, ranked_sample):
        rows = list(csv.reader(ranked_sample.open(encoding="utf-8")))[1:]
        scores = [float(r[2]) for r in rows]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], (
                f"Score not non-increasing at row {i}: {scores[i]} < {scores[i+1]}"
            )

    def test_tie_break_by_candidate_id_ascending(self, ranked_sample):
        rows = list(csv.reader(ranked_sample.open(encoding="utf-8")))[1:]
        for i in range(len(rows) - 1):
            s1, c1 = float(rows[i][2]), rows[i][0]
            s2, c2 = float(rows[i + 1][2]), rows[i + 1][0]
            if s1 == s2:
                assert c1 < c2, f"Tie at row {i} not broken by candidate_id asc: {c1} vs {c2}"

    def test_ids_match_pattern(self, ranked_sample):
        rows = list(csv.reader(ranked_sample.open(encoding="utf-8")))[1:]
        for r in rows:
            assert CAND_ID_RE.match(r[0]), f"Bad id format: {r[0]}"

    def test_reasoning_non_empty(self, ranked_sample):
        rows = list(csv.reader(ranked_sample.open(encoding="utf-8")))[1:]
        for r in rows:
            assert r[3].strip(), f"Empty reasoning for {r[0]}"

    def test_no_honeypots_in_top_n(self, ranked_sample):
        """The 50-sample has no structural impossibilities (verified in P2).
        This is a regression guard: the runtime must not introduce any."""
        from src.honeypot import detect_honeypot
        from src.data_loader import load_candidates_json
        candidates = {c["candidate_id"]: c for c in load_candidates_json(SAMPLE)}
        rows = list(csv.reader(ranked_sample.open(encoding="utf-8")))[1:]
        for r in rows:
            cid = r[0]
            cand = candidates[cid]
            is_hp, reasons = detect_honeypot(cand, CFG)
            assert not is_hp, f"Honeypot in top-N: {cid} ({reasons})"


# ---------------------------------------------------------------------------
# Runtime safety
# ---------------------------------------------------------------------------

class TestRuntimeSafety:
    def test_no_network_guard(self):
        """rank.py overrides socket.socket to raise on any call. Verify
        the guard is in effect after import."""
        # Importing rank.py applies the guard.
        import src.rank  # noqa: F401
        with pytest.raises(RuntimeError, match="Network access blocked"):
            socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def test_no_model_loaded_at_runtime(self):
        """rank.py must NOT import sentence_transformers. If a future
        edit accidentally adds the import, this test catches it."""
        import src.rank as r
        src_text = Path(r.__file__).read_text(encoding="utf-8")
        assert "sentence_transformers" not in src_text, (
            "rank.py must not import sentence_transformers at runtime — "
            "all embedding is offline (precompute.py)."
        )


# ---------------------------------------------------------------------------
# Full 100K latency gate (opt-in: set RUN_P4_FULL=1 to enable)
# This test does real precompute on the production 100K pool, which takes
# minutes to tens of minutes (uncapped per spec §10.3). The 5-min gate
# only applies to the RANKING step, not the precompute. To keep the
# default test run fast, this is opt-in via the RUN_P4_FULL env var.
# ---------------------------------------------------------------------------

class TestFullPoolLatency:
    def test_full_pool_runs_under_5_minutes(self):
        if not FULL_POOL.exists():
            pytest.skip(f"Full 100K pool not present at {FULL_POOL}")
        if not os.environ.get("RUN_P4_FULL"):
            pytest.skip(
                "Full 100K latency test is opt-in. Set RUN_P4_FULL=1 to enable "
                "(will take minutes for precompute + ≤5 min for ranking)."
            )
        cache_dir = REPO / "artifacts" / "full"
        out = REPO / "outputs" / "submission.csv"
        # Precompute (uncapped; may take minutes to tens of minutes).
        from src.precompute import run_precompute
        run_precompute(FULL_POOL, cache_dir, force=True)
        # Ranking step — the ≤5 min hard gate.
        t0 = time.perf_counter()
        rc = subprocess.run(
            [
                sys.executable, "-m", "src.rank",
                "--candidates", str(FULL_POOL),
                "--cache", str(cache_dir),
                "--out", str(out),
            ],
            cwd=REPO, capture_output=True, text=True, timeout=300,
        )
        elapsed = time.perf_counter() - t0
        assert rc.returncode == 0, f"rank.py failed:\n{rc.stderr}"
        assert elapsed < 300, f"Ranking step took {elapsed:.1f}s (>5 min)"
        # Output must have exactly 100 rows.
        rows = list(csv.reader(out.open(encoding="utf-8")))
        assert len(rows) - 1 == 100
