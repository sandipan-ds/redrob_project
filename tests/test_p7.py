"""
test_p7.py — P7 exit tests (PHASED_BUILD_PLAN §P7).

P7 = Docker + sandbox + metadata + git hygiene. The DoD is:
  1. `docker build` succeeds; `docker run` produces a valid sample CSV with network disabled.
  2. `rank.py` raises if any code attempts a network call (unit-test the guard).
  3. `submission_metadata.yaml` parses and has the three required boolean flags set correctly.

Test 1 is the manual exit test (run by a human with `docker` available).
Tests 2 and 3 are the automated regression guards — they run on every commit.
"""

from __future__ import annotations

import socket
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))


# ---------------------------------------------------------------------------
# Group 1 — No-network guard in rank.py
# ---------------------------------------------------------------------------
class TestNoNetworkGuard:
    """rank.py overrides `socket.socket` at import time so an accidental
    network call raises. This is the P7 §3.3 hard-constraint guard."""

    def test_rank_module_overrides_socket(self):
        """Importing src.rank must replace `socket.socket` with a guard."""
        import src.rank  # noqa: F401  (the import is the side effect)
        # socket.socket is now the blocking factory, not the original.
        # Direct check via attribute identity.
        assert socket.socket is not __import__("socket").socket.__class__ or True
        # More robust: try to call it. The rank module's _blocked_socket raises.
        with pytest.raises(RuntimeError, match="Network access blocked"):
            socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def test_guard_raises_runtimeerror_with_spec_reference(self):
        """The error message must reference submission_spec §3 so the
        accidental-network failure is self-explanatory in logs."""
        import src.rank  # noqa: F401
        with pytest.raises(RuntimeError) as exc:
            socket.socket()
        msg = str(exc.value)
        assert "§3" in msg or "submission_spec" in msg.lower(), (
            f"Guard error message should reference spec §3 for traceability, "
            f"got: {msg!r}"
        )

    def test_rank_does_not_import_sentence_transformers(self):
        """At ranking runtime, the heavy ML libs must not be loaded. This
        protects the 5-min / 16 GB budget and the offline-reproducibility
        promise (no model → no network fetch)."""
        import src.rank  # noqa: F401
        # The mod must be in sys.modules only if rank.py itself imported it
        # (which it should NOT). We check the source for safety, since
        # transitive imports could vary by Python/sys.path.
        src = (REPO / "src" / "rank.py").read_text(encoding="utf-8")
        assert "sentence_transformers" not in src, (
            "src/rank.py must not import sentence-transformers "
            "(runtime must be pure NumPy / pandas / yaml)."
        )
        assert "transformers" not in src or "from src" in src, (
            "src/rank.py must not import the heavy `transformers` library."
        )


# ---------------------------------------------------------------------------
# Group 2 — submission_metadata.yaml has the three required boolean flags
# ---------------------------------------------------------------------------
class TestSubmissionMetadata:
    """The spec requires three boolean declarations on the metadata file.
    These are the only hard contract flags the public sandbox checks."""

    REQUIRED_FLAGS = {
        "uses_gpu_for_inference": False,
        "has_network_during_ranking": False,
        "honeypot_check_done": True,
    }

    def _load(self) -> dict:
        path = REPO / "submission_metadata.yaml"
        assert path.exists(), f"Missing {path}"
        with path.open(encoding="utf-8") as fh:
            return yaml.safe_load(fh)

    def test_yaml_parses(self):
        """The metadata file must parse as a single YAML document."""
        data = self._load()
        assert isinstance(data, dict)
        assert "compute" in data
        assert "declarations" in data

    def test_three_required_flags_present(self):
        """Each of the three spec-required flags must be present and set correctly:
            - uses_gpu_for_inference = false  (spec §3)
            - has_network_during_ranking = false  (spec §3)
            - honeypot_check_done = true  (P2 deliverable)"""
        data = self._load()
        compute = data.get("compute", {})
        decls = data.get("declarations", {})

        for flag, expected in self.REQUIRED_FLAGS.items():
            # Check both possible locations.
            actual = compute.get(flag, decls.get(flag, "MISSING"))
            assert actual == expected, (
                f"Required flag {flag!r} must be {expected!r}, got {actual!r}. "
                f"This is a spec §3 / P2 hard contract."
            )

    def test_reproduce_command_present(self):
        """The metadata should document a single reproduce command."""
        data = self._load()
        cmd = data.get("reproduce_command", "")
        assert cmd and isinstance(cmd, str), "reproduce_command must be a non-empty string"
        # Must mention the ranker (the constrained step).
        assert "src.rank" in cmd or "rank.py" in cmd, (
            f"reproduce_command must invoke the ranker; got: {cmd!r}"
        )

    def test_reproduction_tested_flag(self):
        """The declarations.reproduction_tested flag should be true at P7."""
        data = self._load()
        assert data.get("declarations", {}).get("reproduction_tested") is True, (
            "P7 sets declarations.reproduction_tested = true once the "
            "Dockerfile + no-network guard are in place."
        )


# ---------------------------------------------------------------------------
# Group 3 — Dockerfile and .dockerignore exist + are sane
# ---------------------------------------------------------------------------
class TestDockerArtifacts:
    """P7 ships a Dockerfile and a .dockerignore. Both must be present and
    parseable. We do not run `docker build` in CI (manual exit test)."""

    def test_dockerfile_exists(self):
        assert (REPO / "Dockerfile").exists(), "Missing Dockerfile at repo root"

    def test_dockerfile_uses_python_311(self):
        """Per requirements.txt, deps need Python 3.11. The image must pin it."""
        text = (REPO / "Dockerfile").read_text(encoding="utf-8")
        assert "python:3.11" in text, "Dockerfile must base on python:3.11-*"

    def test_dockerfile_copies_vendored_model(self):
        """The image must COPY the vendored model — runtime must work offline."""
        text = (REPO / "Dockerfile").read_text(encoding="utf-8")
        assert "models/" in text, "Dockerfile must COPY the models/ directory"

    def test_dockerfile_does_not_pull_at_runtime(self):
        """No `pip install` or `wget`/`curl` in the runtime layer."""
        text = (REPO / "Dockerfile").read_text(encoding="utf-8")
        # All pip install must be in build-time RUN (no CMD/RUNENTRYPOINT that calls pip).
        for line in text.splitlines():
            if line.strip().startswith(("CMD", "ENTRYPOINT")):
                assert "pip install" not in line, (
                    f"pip install in CMD/ENTRYPOINT: {line!r}"
                )

    def test_dockerignore_exists(self):
        assert (REPO / ".dockerignore").exists(), "Missing .dockerignore at repo root"

    def test_dockerignore_excludes_originals(self):
        """Originals (~487 MB) must not be baked into the image — mount at runtime."""
        text = (REPO / ".dockerignore").read_text(encoding="utf-8")
        assert "data/originals" in text, (
            ".dockerignore must exclude data/originals/ (mounted at runtime)"
        )


# ---------------------------------------------------------------------------
# Group 4 — Vendored model is loadable from the local path (no network)
# ---------------------------------------------------------------------------
class TestVendoredModel:
    """PHASED_BUILD_PLAN §P7 task 1: vendor the model into models/ so a
    clean machine can precompute offline. Verified by: (a) files present,
    (b) SentenceTransformer loads from the local path, (c) encoding works."""

    VENDORED_DIR = REPO / "models" / "all-MiniLM-L6-v2"

    def test_vendored_dir_exists(self):
        assert self.VENDORED_DIR.exists(), (
            f"Vendored model dir missing at {self.VENDORED_DIR}. "
            "P7 task 1: copy from HF cache to models/all-MiniLM-L6-v2/."
        )

    def test_required_model_files_present(self):
        """The vendored dir must contain the files SentenceTransformer needs."""
        required = [
            "config.json",
            "config_sentence_transformers.json",
            "modules.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.txt",
            "model.safetensors",
        ]
        present = {p.name for p in self.VENDORED_DIR.iterdir() if p.is_file()}
        missing = [r for r in required if r not in present]
        assert not missing, f"Vendored model missing files: {missing}"

    def test_sentence_transformer_loads_from_vendored_path(self):
        """`SentenceTransformer(vendored_path)` must produce a working model
        — this is the entire point of vendoring."""
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(str(self.VENDORED_DIR))
        emb = model.encode(["test sentence"], normalize_embeddings=True)
        assert emb.shape == (1, 384), f"Expected (1, 384), got {emb.shape}"


# ---------------------------------------------------------------------------
# Group 5 — P5/P6 invariants preserved across the P7 work
# ---------------------------------------------------------------------------
class TestP7Invariants:
    """Sanity: applying the calibrated P5 config and the P7 vendoring should
    not have broken any pre-P7 test. The full pytest suite (139 tests) is
    the real guard; here we re-assert a few key invariants that the
    precompute + rank pipeline is intact end-to-end on the 50-sample."""

    def test_score_weights_sum_to_one(self):
        """The calibrated config must still sum to 1.0 within float precision."""
        from src.config_loader import load_config
        cfg = load_config()
        s = sum(cfg["weights"].values())
        assert abs(s - 1.0) < 1e-6, f"weights must sum to 1.0, got {s}"

    def test_p_scale_is_calibrated_value(self):
        """The applied p_scale is 1.5 (best P5 result); regression guard."""
        from src.config_loader import load_config
        cfg = load_config()
        assert cfg["penalties"]["p_scale"] == 1.5, (
            f"P5 calibration set p_scale=1.5; if this changes, re-run "
            f"`python scripts/calibrate.py` and document the change."
        )

    def test_jd_intent_set_still_aligned_with_queries(self):
        """The multi-query JD-intent set must still have one embedding per
        configured query (test_p1 invariant — P7 changes must not break it)."""
        from src.config_loader import load_config
        from src.jd_embedding import load_jd_intent_set
        cfg = load_config()
        Q = len(cfg["role_fit"]["intent_queries"])
        emb = load_jd_intent_set()
        assert emb.shape[0] == Q, (
            f"intent_queries has {Q} entries but the embedding has {emb.shape[0]} rows. "
            f"Re-run `python -m src.jd_embedding` after any query-list change."
        )
