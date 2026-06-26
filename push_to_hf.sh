#!/usr/bin/env bash
# push_to_hf.sh — Push the redrob-ranker image to a Hugging Face Space.
#
# Prerequisites:
#   1. Docker installed and running (verified: image built locally at redrob-ranker:latest)
#   2. Hugging Face account (https://huggingface.co/join)
#   3. Either:
#      a) Docker Hub account (Path A below) — image gets pushed to Docker Hub,
#         HF Space pulls from there
#      b) huggingface_hub CLI installed (`pip install "huggingface_hub[cli]"`)
#         (Path B below) — code is pushed to HF, image is built from the Dockerfile
#
# Usage:
#   1. Edit the variables below (HF_USERNAME, DOCKERHUB_USERNAME)
#   2. Run: bash push_to_hf.sh
#   3. Choose Path A or Path B when prompted
#
# After the Space is created, update submission_metadata.yaml:
#   sandbox_link: "https://huggingface.co/spaces/sandipanarnab/redrob-ranker"

set -e

# ---------------------------------------------------------------------------
# Configuration — edit these before running
# ---------------------------------------------------------------------------
HF_USERNAME="${HF_USERNAME:-sandipanarnab}"           # e.g., "sandipan"
DOCKERHUB_USERNAME="${DOCKERHUB_USERNAME:-sandipanarnab}"  # e.g., "sandipan"
SPACE_NAME="${SPACE_NAME:-redrob-ranker}"
IMAGE_NAME="redrob-ranker:latest"
TAGGED_IMAGE="${DOCKERHUB_USERNAME}/${IMAGE_NAME}"

# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------
if [ "$HF_USERNAME" = "YOUR_HF_USERNAME" ]; then
    echo "ERROR: set HF_USERNAME and DOCKERHUB_USERNAME before running."
    echo "Edit push_to_hf.sh or pass as env vars:"
    echo "  HF_USERNAME=sandipan DOCKERHUB_USERNAME=sandipan bash push_to_hf.sh"
    exit 1
fi

if ! command -v docker &> /dev/null; then
    echo "ERROR: docker not found. Install Docker Desktop first."
    exit 1
fi

if ! docker image inspect "$IMAGE_NAME" &> /dev/null; then
    echo "ERROR: image $IMAGE_NAME not found locally. Build it first:"
    echo "  docker build -t $IMAGE_NAME ."
    exit 1
fi

echo "=== Pre-flight OK ==="
echo "  Image: $IMAGE_NAME (built locally)"
echo "  HF username: $HF_USERNAME"
echo "  Docker Hub username: $DOCKERHUB_USERNAME"
echo "  Space name: $SPACE_NAME"
echo ""

# ---------------------------------------------------------------------------
# Path A: Docker Hub + Docker SDK Space
# ---------------------------------------------------------------------------
path_a() {
    echo "=== Path A: Docker Hub + Docker SDK Space ==="
    echo ""
    echo "Step 1/4: Log in to Docker Hub"
    echo "  $ docker login -u $DOCKERHUB_USERNAME"
    docker login -u "$DOCKERHUB_USERNAME"
    echo ""
    echo "Step 2/4: Tag the image"
    echo "  $ docker tag $IMAGE_NAME $TAGGED_IMAGE"
    docker tag "$IMAGE_NAME" "$TAGGED_IMAGE"
    echo ""
    echo "Step 3/4: Push to Docker Hub (this will take a few minutes for ~3.4 GB)"
    echo "  $ docker push $TAGGED_IMAGE"
    docker push "$TAGGED_IMAGE"
    echo ""
    echo "Step 4/4: Create the HF Space (do this in the browser)"
    echo "  1. Go to https://huggingface.co/new-space"
    echo "  2. Name: $SPACE_NAME"
    echo "  3. SDK: Docker"
    echo "  4. Docker image: $TAGGED_IMAGE"
    echo "  5. Hardware: CPU basic (free tier, 2 vCPU / 16 GB)"
    echo "  6. Click 'Create Space' — HF will pull the image and start it"
    echo ""
    echo "Once the Space is running, update submission_metadata.yaml:"
    echo "  sandbox_link: \"https://huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME\""
}

# ---------------------------------------------------------------------------
# Path B: Push the repo to HF (HF builds the image from the Dockerfile)
# ---------------------------------------------------------------------------
path_b() {
    echo "=== Path B: Push repo to HF (HF builds the image) ==="
    echo ""
    if ! command -v huggingface-cli &> /dev/null; then
        echo "Installing huggingface_hub CLI..."
        pip install "huggingface_hub[cli]"
    fi
    echo ""
    echo "Step 1/5: Log in to Hugging Face"
    echo "  $ huggingface-cli login"
    huggingface-cli login
    echo ""
    echo "Step 2/5: Create the Space repo"
    echo "  $ huggingface-cli repo create $SPACE_NAME --type space --space-sdk docker"
    huggingface-cli repo create "$SPACE_NAME" --type space --space-sdk docker || true
    echo ""
    echo "Step 3/5: Add the HF Space as a git remote"
    HF_REPO="https://oauth2:$(huggingface-cli whoami | grep -oP 'token: \K\S+')@huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME"
    echo "  $ git remote add hf $HF_REPO"
    git remote add hf "$HF_REPO" 2>/dev/null || echo "  (remote 'hf' already exists)"
    echo ""
    echo "Step 4/5: Push the code (HF will build the Docker image from Dockerfile)"
    echo "  $ git push hf main:main"
    git push hf main:main
    echo ""
    echo "Step 5/5: Wait for HF to build the image (~10-15 min for first build)"
    echo "  Monitor at: https://huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME"
    echo ""
    echo "Once the Space is running, update submission_metadata.yaml:"
    echo "  sandbox_link: \"https://huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME\""
}

# ---------------------------------------------------------------------------
# Main: ask which path
# ---------------------------------------------------------------------------
echo "Which deployment path?"
echo "  A) Docker Hub + Docker SDK Space (simpler, image already built)"
echo "  B) Push repo to HF (HF builds the image from Dockerfile)"
echo ""
read -p "Path (A/B): " path
echo ""

case "$path" in
    [Aa]) path_a ;;
    [Bb]) path_b ;;
    *) echo "Invalid path. Use A or B."; exit 1 ;;
esac
