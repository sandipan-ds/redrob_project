#!/usr/bin/env bash
# push_to_hf.sh — Push the redrob-ranker repo to a Hugging Face Space.
#
# This is Path B: push the code to HF, HF builds the Docker image
# from the Dockerfile.
#
# Prerequisites:
#   1. Git installed (always true)
#   2. Python with pip (for the huggingface_hub CLI)
#   3. A Hugging Face account (https://huggingface.co/join)
#   4. A HF access token with write scope (https://huggingface.co/settings/tokens)
#
# Usage:
#   1. Create a .env file in the project root with your token:
#        echo 'HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxx' > .env
#      OR set HF_TOKEN as an env var before running this script
#   2. Run: bash push_to_hf.sh
#   3. After the Space is running, update submission_metadata.yaml
#      with the sandbox_link
#
# Sandbox link format: https://huggingface.co/spaces/<username>/<space-name>

set -e

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HF_USERNAME="${HF_USERNAME:-sandipanarnab}"          # your HF username
SPACE_NAME="${SPACE_NAME:-redrob-ranker}"             # the Space repo name

# ---------------------------------------------------------------------------
# Load HF_TOKEN from .env (if it exists) or expect it in the environment
# ---------------------------------------------------------------------------
# We use Python's python-dotenv to load .env, then export HF_TOKEN to bash.
# This way the rest of the script (and the hf CLI, and the git push URL)
# all see the same token.

if [ -z "${HF_TOKEN:-}" ] && [ -f .env ]; then
    echo "Loading HF_TOKEN from .env (via python-dotenv)..."
    # Try a few Python interpreters in order. Suppress errors with
    # "|| true" so a missing path doesn't exit (we have set -e).
    HF_TOKEN=$(
        {
            .venv/Scripts/python.exe load_hf_token.py 2>/dev/null ||
            .venv/Scripts/python    load_hf_token.py 2>/dev/null ||
            venv/bin/python          load_hf_token.py 2>/dev/null ||
            python3                  load_hf_token.py 2>/dev/null ||
            python                   load_hf_token.py 2>/dev/null
        } | tail -n 1
    )
    if [ -n "$HF_TOKEN" ]; then
        export HF_TOKEN
        echo "  Loaded (length: ${#HF_TOKEN})"
    else
        echo "  .env exists but HF_TOKEN not set in it"
    fi
fi

if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN is not set."
    echo "Either:"
    echo "  1. Create a .env file: echo 'HF_TOKEN=hf_xxx' > .env"
    echo "  2. Or set the env var: export HF_TOKEN=hf_xxx"
    echo "  3. Or pass inline: HF_TOKEN=hf_xxx bash push_to_hf.sh"
    exit 1
fi

# ---------------------------------------------------------------------------
# Step 1: Install / verify huggingface_hub CLI
# ---------------------------------------------------------------------------
if ! command -v hf &> /dev/null && ! command -v huggingface-cli &> /dev/null; then
    echo "Installing huggingface_hub CLI..."
    pip install "huggingface_hub[cli]" >/dev/null
fi

# ---------------------------------------------------------------------------
# Step 2: Log in to Hugging Face
# ---------------------------------------------------------------------------
echo "=== Step 1/5: Log in to Hugging Face ==="
hf auth login --token "$HF_TOKEN" --add-to-git-credential 2>&1 | tail -3
echo ""

# ---------------------------------------------------------------------------
# Step 3: Create the Space repo
# ---------------------------------------------------------------------------
echo "=== Step 2/5: Create the Space repo ==="
echo "  $ hf repo create $SPACE_NAME --repo-type space --space-sdk docker --exist-ok"
hf repo create "$SPACE_NAME" --repo-type space --space-sdk docker --exist-ok 2>&1 | tail -3
echo ""

# ---------------------------------------------------------------------------
# Step 4: Add the HF Space as a git remote
# ---------------------------------------------------------------------------
echo "=== Step 3/5: Add the HF Space as a git remote ==="
HF_REPO="https://oauth2:${HF_TOKEN}@huggingface.co/spaces/${HF_USERNAME}/${SPACE_NAME}"
if ! git remote get-url hf &>/dev/null; then
    git remote add hf "$HF_REPO"
    echo "  (remote 'hf' added)"
else
    git remote set-url hf "$HF_REPO"
    echo "  (remote 'hf' URL updated)"
fi
echo ""

# ---------------------------------------------------------------------------
# Step 5: Push the code (HF builds the Docker image from Dockerfile)
# ---------------------------------------------------------------------------
echo "=== Step 4/5: Push the code (HF will build the Docker image) ==="
echo "  $ git push hf main:main"
git push hf main:main 2>&1
echo ""

# ---------------------------------------------------------------------------
# Step 6: Wait for HF to build, then prompt to update submission_metadata.yaml
# ---------------------------------------------------------------------------
SANDBOX_LINK="https://huggingface.co/spaces/${HF_USERNAME}/${SPACE_NAME}"
echo "=== Step 5/5: Wait for HF to build (~10-15 min for first build) ==="
echo "  Monitor at: ${SANDBOX_LINK}"
echo ""
echo "Once the Space shows 'Running', update submission_metadata.yaml:"
echo "  sandbox_link: \"${SANDBOX_LINK}\""
echo ""
echo "Quick test (paste this in your terminal after the build finishes):"
echo "  curl -s ${SANDBOX_LINK}/_stcore/health"
echo "  (should return: ok)"
