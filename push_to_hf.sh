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
#   1. Set HF_TOKEN env var (or run 'hf auth login' first)
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
HF_TOKEN="${HF_TOKEN:-}"                              # set via env or run `hf auth login`

# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
if [ "$HF_USERNAME" = "YOUR_HF_USERNAME" ]; then
    echo "ERROR: set HF_USERNAME to your HF username."
    echo "Edit push_to_hf.sh or pass as env var:"
    echo "  HF_USERNAME=sandipanarnab bash push_to_hf.sh"
    exit 1
fi

if [ -z "$HF_TOKEN" ] && ! command -v huggingface-cli &> /dev/null; then
    echo "ERROR: set HF_TOKEN env var or install huggingface_hub CLI first."
    echo "  pip install 'huggingface_hub[cli]'"
    echo "  export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxx"
    exit 1
fi

# ---------------------------------------------------------------------------
# Step 1: Install / verify huggingface_hub CLI
# ---------------------------------------------------------------------------
if ! command -v huggingface-cli &> /dev/null; then
    echo "Installing huggingface_hub CLI..."
    pip install "huggingface_hub[cli]" >/dev/null
fi

# ---------------------------------------------------------------------------
# Step 2: Log in to Hugging Face
# ---------------------------------------------------------------------------
echo "=== Step 1/5: Log in to Hugging Face ==="
if [ -n "$HF_TOKEN" ]; then
    # Non-interactive: use the token directly
    hf auth login --token "$HF_TOKEN" --add-to-git-credential
else
    # Interactive: prompt for the token
    hf auth login
fi
echo ""

# ---------------------------------------------------------------------------
# Step 3: Create the Space repo
# ---------------------------------------------------------------------------
echo "=== Step 2/5: Create the Space repo ==="
echo "  $ hf repo create $SPACE_NAME --type space --space-sdk docker --exist-ok"
hf repo create "$SPACE_NAME" --type space --space-sdk docker --exist-ok
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
