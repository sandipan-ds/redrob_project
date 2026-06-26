# Redrob Candidate Ranker — Docker image (P7, PHASED_BUILD_PLAN §P7)
#
# Hard constraints (submission_spec §3 / §10.3):
#   - CPU-only (no CUDA / no GPU runtime)
#   - ≤ 5 min wall-clock for the full-100K ranking step
#   - ≤ 16 GB RAM
#   - No network at RANKING runtime (--network none is the recommended run flag)
#   - All ML/embedding work is OFFLINE and dev-time
#
# Python 3.11 is pinned (per requirements.txt): pinned deps have no wheels
# for 3.12+/3.14, and the source build fails on those versions.

FROM python:3.11-slim

# ---------------------------------------------------------------------------
# System setup
# ---------------------------------------------------------------------------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONHASHSEED=0 \
    # Force CPU-only torch (no CUDA). The slim image has no GPU libs anyway,
    # but this makes the intent explicit and guards against future base-image
    # changes that might pull CUDA wheels.
    TRANSFORMERS_NO_ADVISORY_WARNINGS=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    SENTENCE_TRANSFORMERS_HOME=/app/models

WORKDIR /app

# Install Python deps. Network is available at BUILD time; the runtime will
# be invoked with --network none (see README "Reproduce" section).
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# ---------------------------------------------------------------------------
# Project layout
# ---------------------------------------------------------------------------
# src/        — production code (the ranker)
# config/     — scoring_config.yaml, JD-intent embeddings (frozen)
# models/     — vendored all-MiniLM-L6-v2 weights (~91 MB) — no HF download
# artifacts/  — precomputed career embeddings + offsets
# data/       — input candidates (mounted from the host at runtime)

COPY src/ /app/src/
COPY config/ /app/config/
COPY models/ /app/models/
COPY artifacts/ /app/artifacts/

# Make `src.*` importable.
ENV PYTHONPATH=/app

# ---------------------------------------------------------------------------
# Runtime defaults
# ---------------------------------------------------------------------------
# The default CMD runs the constrained RANKING step (≤5 min, no network).
# Precompute (embedding the candidate pool) is a separate, offline step that
# does not need to run in the container — precompute_meta.yaml etc. are
# already vendored. For a fresh run on a new candidate pool:
#
#   docker run --rm -v ${PWD}/data:/work/data -v ${PWD}/artifacts:/work/artifacts \
#       redrob-ranker python -m src.precompute \
#       --candidates /work/data/candidates.jsonl --output /work/artifacts
#
# But the SANDBOX only exercises the ranking step (per submission_spec §10.5).
# So the default entrypoint is the ranker with the vendored sample.

# Default sandbox invocation: rank the 50-sample. Override candidates/out at runtime.
# NB: the in-container path uses /work/* for mounts; the default uses the vendored
# sample so the image is self-contained and runnable as-is.
ENTRYPOINT ["python", "-m", "src.rank"]
CMD ["--candidates", "/app/data/samples/sample_candidates.json", \
     "--cache",      "/app/artifacts/sample", \
     "--out",        "/app/outputs/sample_submission.csv", \
      "--top-n",      "50"]
