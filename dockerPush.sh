#!/bin/bash

# Input parameters
REPOSITORY="dylanrmartin/computations"            # Default repository
COMPUTATION_HANDLE="single_round_ridge_regression" # Default computation handle

# Get the current commit hash if not provided
COMMIT_HASH=${1:-$(git rev-parse HEAD)}

# Full tag for the image
IMAGE_TAG="${REPOSITORY}:${COMPUTATION_HANDLE}-${COMMIT_HASH}"

# Build the image using Dockerfile-prod
echo "Building the image with Dockerfile-prod..."
docker build -f Dockerfile-prod -t "${COMPUTATION_HANDLE}" .

# Tag the image for Docker Hub
echo "Tagging the image as ${IMAGE_TAG}..."
docker tag "${COMPUTATION_HANDLE}" "${IMAGE_TAG}"

# Push the image to the repository
echo "Pushing ${IMAGE_TAG} to Docker Hub..."
docker push "${IMAGE_TAG}"

echo "Push complete. Image available at ${IMAGE_TAG}"