#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_WORKSPACE="$SCRIPT_DIR"
REMOTE_WORKSPACE="/workspace"
IMAGE_NAME="neuroflame-vbm-dev"
CONTAINER_NAME="${IMAGE_NAME}-sim-$$"
SIMULATOR_WORKSPACE="simulator_workspace"
RESULT_ROOT="$LOCAL_WORKSPACE/test_output/simulate_job"

if [[ $# -lt 1 ]]; then
    echo "Usage: ./run_local_simulation.sh site1,site2[,site3...] [--no-build]" >&2
    exit 1
fi

SITES_ARG=""
BUILD_IMAGE=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-build) BUILD_IMAGE=0 ;;
        *)
            if [[ -n "$SITES_ARG" ]]; then
                echo "Unexpected argument: $1" >&2
                exit 1
            fi
            SITES_ARG="$1"
            ;;
    esac
    shift
done

IFS=',' read -r -a SITES <<< "$SITES_ARG"
if [[ ${#SITES[@]} -lt 2 ]]; then
    echo "VBM simulation requires at least two participants" >&2
    exit 1
fi

if [[ $BUILD_IMAGE -eq 1 ]]; then
    docker build --platform linux/amd64 -t "$IMAGE_NAME" -f Dockerfile-dev "$LOCAL_WORKSPACE"
fi

docker run --rm \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    --name "$CONTAINER_NAME" \
    -v "$LOCAL_WORKSPACE:$REMOTE_WORKSPACE" \
    -w "$REMOTE_WORKSPACE" \
    "$IMAGE_NAME:latest" \
    /bin/bash -lc "set -euo pipefail; rm -rf ./job ./$SIMULATOR_WORKSPACE; python makeJob.py '$SITES_ARG'; python debugger.py ./job -w ./$SIMULATOR_WORKSPACE -n ${#SITES[@]} -t ${#SITES[@]} -c '$SITES_ARG'"

for site in "${SITES[@]}"; do
    test -f "$RESULT_ROOT/$site/index.html"
done
