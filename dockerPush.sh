#!/bin/bash

# Input parameters
REPOSITORY="coinstacteam"            # Default repository
COMPUTATION_HANDLE="nfc-single-round-closedform-regression-vbm" # Default computation handle
PLATFORM="${PLATFORM:-linux/amd64}"

# Get the current commit hash if not provided
COMMIT_HASH=${1:-$(git rev-parse HEAD)}

# Full tags for the image
LATEST_IMAGE="${REPOSITORY}/${COMPUTATION_HANDLE}:latest"
COMMIT_IMAGE="${REPOSITORY}/${COMPUTATION_HANDLE}:${COMMIT_HASH}"

# Build the image using Dockerfile-prod
echo "Building the image with Dockerfile-prod for ${PLATFORM}..."
docker build --platform "${PLATFORM}" -f Dockerfile-prod -t "${LATEST_IMAGE}" -t "${COMMIT_IMAGE}" .

echo "Pushing ${LATEST_IMAGE} to Docker Hub..."
docker push "${LATEST_IMAGE}"

# Push the image to the repository
echo "Pushing ${COMMIT_IMAGE} to Docker Hub..."
docker push "${COMMIT_IMAGE}"

echo "Push complete. Images available at ${LATEST_IMAGE} and ${COMMIT_IMAGE}"
