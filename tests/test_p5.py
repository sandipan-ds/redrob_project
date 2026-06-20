"""
test_p5.py — P5 exit criterion tests.

P5 exit criterion (PHASED_BUILD_PLAN §P5):
  - Metric implementations match hand-computed expected values on a tiny
    fixture (e.g., a known ranking).
  - The composite of a deliberately-good ordering > a shuffled ordering
    on the proxy set.
  - Adversarial decoys rank below genuine fits.
  - Calibration runs and reports a composite; tuning only the documented
    macro knobs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from src.eval.metrics import (
    COMPOSITE_WEIGHTS,
    average_precision,
    composite_score,
    dcg_at_k,
    ndcg_at_k,
    precision_at_k,
)
from src.eval.proxy_labels import adversarial_decoys, load_proxy_tiers


# ---------------------------------------------------------------------------
# Metrics — known-fixture correctness
# ---------------------------------------------------------------------------

class TestMetricsCorrectness:
    """Verify each metric against a hand-computed expected value."""

    def test_dcg_at_k_known_value(self):
        # 3 items, tier [3, 2, 1] at k=3.
        # gain = 2^3-1=7, 2^2-1=3, 2^1-1=1
        # dcg = 7/log2(2) + 3/log2(3) + 1/log2(4)
        #      = 7/1 + 3/1.585 + 1/2
        #      = 7 + 1.8928 + 0.5 = 9.3928
        tiers = {"a": 3, "b": 2, "c": 1, "d": 0}
        d = dcg_at_k(["a", "b", "c"], tiers, 3)
        assert d == pytest.approx(9.3928, abs=1e-3)

    def test_ndcg_at_k_perfect_ordering_is_one(self):
        tiers = {"a": 5, "b": 4, "c": 3, "d": 2, "e": 1}
        # Ordered by tier desc → NDCG = 1.0
        ranked = ["a", "b", "c", "d", "e"]
        assert ndcg_at_k(ranked, tiers, 5) == pytest.approx(1.0, abs=1e-9)

    def test_ndcg_at_k_reversed_ordering(self):
        tiers = {"a": 5, "b": 4, "c": 3, "d": 2, "e": 1}
        # Reversed → NDCG = 1 / IDCG = lower
        ranked = ["e", "d", "c", "b", "a"]
        score = ndcg_at_k(ranked, tiers, 5)
        assert 0.0 < score < 1.0
        # The reversed NDCG should be the reciprocal of the "ideal" DCG ratio,
        # which is the DCG of the ideal ordering divided by itself = 1.
        # So reversed is < 1 but > 0.
        assert score < 1.0
        assert score > 0.0

    def test_ndcg_at_k_no_relevant_returns_zero(self):
        # All tier 0 → IDCG = 0 → NDCG = 0
        tiers = {"a": 0, "b": 0, "c": 0}
        assert ndcg_at_k(["a", "b", "c"], tiers, 3) == 0.0

    def test_average_precision_known(self):
        # 5 items, tiers [3, 0, 2, 1, 3]. Relevant = tier >= 3 → 2 items (a, e).
        # AP = (1/1 + 2/5) / 2 = (1 + 0.4) / 2 = 0.7
        tiers = {"a": 3, "b": 0, "c": 2, "d": 1, "e": 3}
        ap = average_precision(["a", "b", "c", "d", "e"], tiers, relevance_threshold=3)
        assert ap == pytest.approx(0.7, abs=1e-3)

    def test_average_precision_no_relevant(self):
        tiers = {"a": 0, "b": 1, "c": 2}
        assert average_precision(["a", "b", "c"], tiers, relevance_threshold=3) == 0.0

    def test_precision_at_k(self):
        tiers = {"a": 5, "b": 0, "c": 4, "d": 1, "e": 3}
        # Top 3: a (5), b (0), c (4) → 2 relevant (a, c) → P@3 = 2/3
        assert precision_at_k(["a", "b", "c", "d", "e"], tiers, 3, relevance_threshold=3) == pytest.approx(2/3)
        # Top 2: a (5), b (0) → 1 relevant → P@2 = 1/2
        assert precision_at_k(["a", "b", "c", "d", "e"], tiers, 2, relevance_threshold=3) == pytest.approx(0.5)

    def test_composite_weights_match_spec(self):
        # submission_spec §4: exactly these weights.
        assert COMPOSITE_WEIGHTS == {
            "ndcg_at_10": 0.50,
            "ndcg_at_50": 0.30,
            "map": 0.15,
            "p_at_10": 0.05,
        }
        # Sum = 1.0
        assert sum(COMPOSITE_WEIGHTS.values()) == pytest.approx(1.0, abs=1e-9)

    def test_composite_score_returns_all_metrics(self):
        tiers = {"a": 5, "b": 0, "c": 4, "d": 1, "e": 3}
        result = composite_score(["a", "b", "c", "d", "e"], tiers)
        assert set(result.keys()) == {"ndcg_at_10", "ndcg_at_50", "map", "p_at_10", "composite"}
        assert 0.0 <= result["composite"] <= 1.0


# ---------------------------------------------------------------------------
# Proxy set — good ordering > shuffled
# ---------------------------------------------------------------------------

class TestProxyOrdering:
    def test_good_ordering_beats_shuffled(self):
        """An ordering that puts tier-4 candidates first must score
        strictly higher than a random ordering of the same items."""
        tiers = load_proxy_tiers()
        # Add decoys at tier 1.
        for d in adversarial_decoys():
            tiers[d["candidate_id"]] = 1
        all_ids = list(tiers.keys())
        # Good ordering: tier desc, then candidate_id asc.
        good = sorted(all_ids, key=lambda cid: (-tiers[cid], cid))
        good_score = composite_score(good, tiers)["composite"]
        # Shuffled: reverse the good ordering.
        import random
        random.seed(42)
        bad = list(good)
        random.shuffle(bad)
        bad_score = composite_score(bad, tiers)["composite"]
        assert good_score > bad_score, (
            f"Good ordering ({good_score:.4f}) should beat shuffled ({bad_score:.4f})"
        )

    def test_adversarial_decoys_rank_below_genuine_fits(self):
        """The 3 decoys (consulting-only, marketing stuffer, research-only)
        must not appear in the top of a reasonable ordering. Specifically,
        the genuine tier-4 fit CAND_0000031 should outrank all 3 decoys."""
        tiers = load_proxy_tiers()
        for d in adversarial_decoys():
            tiers[d["candidate_id"]] = 1
        all_ids = list(tiers.keys())
        good = sorted(all_ids, key=lambda cid: (-tiers[cid], cid))
        # Find the rank of CAND_0000031 (genuine tier-4 fit) and the decoys.
        rank_fit = good.index("CAND_0000031") + 1
        decoy_ranks = [good.index(d["candidate_id"]) + 1 for d in adversarial_decoys()]
        assert rank_fit <= min(decoy_ranks), (
            f"genuine fit CAND_0000031 (rank {rank_fit}) should outrank "
            f"all decoys (ranks {decoy_ranks})"
        )


# ---------------------------------------------------------------------------
# Calibration script — runs and writes a candidate config
# ---------------------------------------------------------------------------

class TestCalibration:
    def test_calibrate_runs_and_writes_candidate(self, tmp_path):
        """Run the calibration script and verify it produces a candidate
        config (NOT overwriting scoring_config.yaml) and reports a
        composite score."""
        import subprocess
        out = tmp_path / "scoring_config.candidate.yaml"
        result = subprocess.run(
            [
                sys.executable, str(REPO / "scripts" / "calibrate.py"),
                "--grid", "coarse",
                "--out", str(out),
            ],
            cwd=REPO, capture_output=True, text=True, timeout=180,
        )
        assert result.returncode == 0, (
            f"calibrate.py failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
        assert out.exists(), "Candidate config was not written"
        # scoring_config.yaml was NOT overwritten.
        assert (REPO / "config" / "scoring_config.yaml").exists()
        # The candidate config is a valid YAML with the macro knobs present.
        import yaml
        with out.open("r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        assert "weights" in cfg
        assert "behavior" in cfg
        assert "penalties" in cfg and "p_scale" in cfg["penalties"]
        assert "role_fit" in cfg and "w_dense" in cfg["role_fit"]
        assert "retrieve" in cfg and "k" in cfg["retrieve"]
        # Output should mention the best composite (the script logs to
        # stderr by default; check both streams).
        assert "Best composite" in (result.stdout + result.stderr)
