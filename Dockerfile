# Redrob Candidate Ranker — Docker image (lean, multi-stage).
#
# Hard constraints (submission_spec §3 / §10.3):
#   - CPU-only (no CUDA / no GPU runtime)
#   - ≤ 5 min wall-clock for the full-100K ranking step
#   - ≤ 16 GB RAM
#   - No network at RANKING runtime (--network none is the recommended run flag)
#   - All ML/embedding work is OFFLINE and dev-time
#
# Python 3.11 is pinned (per requirements-runtime.txt): pinned deps have no
# wheels for 3.12+/3.14, and the source build fails on those versions.
#
# Image size target: <1.5 GB (was 10.1 GB before the multi-stage rewrite).

# =============================================================================
# Stage 1: builder — install Python deps into a throwaway layer
# =============================================================================
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Install deps into a prefix we can copy cleanly.
COPY requirements-runtime.txt /build/requirements-runtime.txt
RUN pip install --prefix=/install --no-cache-dir -r /build/requirements-runtime.txt

# =============================================================================
# Stage 2: runtime — minimal image, deps + app code only
# =============================================================================
FROM python:3.11-slim

# No-bytecode + unbuffered for clean logs. HF_HUB_OFFLINE + the no-network
# guard in src/rank.py enforce the spec §3 "no network at ranking" rule.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    TRANSFORMERS_NO_ADVISORY_WARNINGS=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    SENTENCE_TRANSFORMERS_HOME=/app/models

# Copy only the installed deps from the builder.
COPY --from=builder /install /usr/local

# Strip transitive dependencies that sentence-transformers pulls in but
# our actual code path doesn't use. Verified that src/*.py and app.py
# don't import these at runtime:
#   - pyarrow  (149 MB, pulled by streamlit/pandas) — not used
#   - scipy    (83 MB, pulled by sklearn)           — not used
#   - sklearn  (32 MB, pulled by sentence-transformers) — not used
#   - sympy    (30 MB, pulled by torch)              — not used
#   - transformers (45 MB, pulled by sentence-transformers) — not used
# This drops ~340 MB from the image without breaking any functionality.
RUN pip uninstall -y --no-input \
        pyarrow scipy scikit-learn sympy transformers 2>&1 | tail -3

WORKDIR /app

# Project layout (the .dockerignore excludes everything not listed here).
# src/        — production code
# config/     — scoring_config.yaml + frozen JD-intent embeddings
# models/     — vendored all-MiniLM-L6-v2 weights (~87 MB, no HF download)
# artifacts/  — precomputed career embeddings (sample only; full/ is excluded)
# data/       — vendored 50-sample (for the default UI behavior)
# app.py      — Streamlit web UI
COPY src/ /app/src/
COPY config/ /app/config/
COPY models/ /app/models/
COPY artifacts/ /app/artifacts/
COPY data/ /app/data/
COPY app.py /app/app.py
COPY .streamlit/ /app/.streamlit/

# Make src.* importable.
ENV PYTHONPATH=/app

# Strip any .pyc / .pyo that the pip install might have left behind.
# (PYTHONDONTWRITEBYTECODE helps, but belt-and-suspenders.)
RUN find /usr/local -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete && \
    find /usr/local -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true

# Expose Streamlit's default port (HF Spaces standard).
EXPOSE 7860

# Health check: HF Spaces uses this to determine if the Space is up.
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/_stcore/health').read()" \
    || exit 1

# Default: run the Streamlit web UI (HF Spaces standard).
# For a CLI-only sandbox run (spec §10.5 small-sample check), override:
#
#   docker run --rm --network none redrob-ranker \
#     python -m src.rank --candidates /app/data/samples/sample_candidates.json \
#                        --cache /app/artifacts/sample \
#                        --out /app/outputs/sample_submission.csv \
#                        --top-n 50
#
# For a full 100K run with mounted volumes:
#
#   docker run --rm --network none \
#     -v ${PWD}/data:/work/data \
#     -v ${PWD}/artifacts:/work/artifacts \
#     -v ${PWD}/outputs:/work/outputs \
#     redrob-ranker python -m src.rank \
#       --candidates /work/data/originals/candidates.jsonl \
#       --cache /work/artifacts/full \
#       --out /work/outputs/submission.csv \
#       --top-n 100

ENTRYPOINT ["streamlit", "run", "app.py", \
            "--server.port=7860", \
            "--server.address=0.0.0.0", \
            "--server.headless=true", \
            "--server.maxUploadSize=1024", \
            "--browser.gatherUsageStats=false"]
CMD []
