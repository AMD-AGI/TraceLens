#!/usr/bin/env bash
set -e

usage() {
    echo "Usage: $0 <tracelens_path>"
    echo "  tracelens_path: Absolute path to the TraceLens-internal repository"
    exit 1
}

if [ -z "$1" ]; then
    usage
fi

TRACELENS_PATH="$(realpath "$1")"
if [ ! -d "${TRACELENS_PATH}" ]; then
    echo "Error: TraceLens path does not exist: ${TRACELENS_PATH}"
    exit 1
fi

CONTAINER_NAME="sglang-deepseek-mi355x"
TRACELENS_MOUNT="/workspace/TraceLens-internal"

echo "Using TraceLens path: ${TRACELENS_PATH}"
echo "Starting Docker container..."
docker run -d \
    --user root \
    --name ${CONTAINER_NAME} \
    --group-add video \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    -w /app/ \
    --ipc=host \
    --network=host \
    --shm-size 64G \
    --device=/dev/kfd \
    --device=/dev/dri \
    -e SGLANG_USE_AITER=1 \
    -v "${TRACELENS_PATH}:${TRACELENS_MOUNT}" \
    lmsysorg/sglang:v0.5.9-rocm700-mi35x \
    tail -f /dev/null

echo "Waiting for container to start..."
sleep 2

echo "Applying patches to sglang..."
docker exec ${CONTAINER_NAME} bash -c "
    set -e
    SGLANG_DIR=\$(python3 -c 'import sglang; import os; print(os.path.dirname(os.path.dirname(sglang.__file__)))')
    echo \"sglang directory: \${SGLANG_DIR}\"
    cd \"\${SGLANG_DIR}\"
    echo \"Applying patches...\"
    for patch in ${TRACELENS_MOUNT}/examples/custom_workflows/inference_analysis/sglang_roofline_patches/*.patch; do
        if [ -f \"\$patch\" ]; then
            echo \"Applying: \$(basename \$patch)\"
            git apply \"\$patch\" || exit 1
        fi
    done
    echo \"Installing TraceLens...\"
    pip install --no-deps ${TRACELENS_MOUNT}
"

echo "Done! Entering container..."
docker exec -it ${CONTAINER_NAME} bash