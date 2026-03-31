#!/usr/bin/env bash
set -e

usage() {
    echo "Usage: $0 <tracelens_path> [gpu_type]"
    echo "  tracelens_path: Absolute path to the TraceLens-internal repository"
    echo "  gpu_type:       'mi300' or 'mi355' (default: mi355)"
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

GPU_TYPE="${2:-mi355}"
case "${GPU_TYPE}" in
    mi300)
        DOCKER_IMAGE="rocm/atom:rocm7.1.1-ubuntu24.04-pytorch2.9-atom0.1.1-MI300x"
        CONTAINER_NAME="atom-deepseek-mi300x"
        ;;
    mi355)
        DOCKER_IMAGE="rocm/atom:rocm7.1.1-ubuntu24.04-pytorch2.9-atom0.1.1-MI355x"
        CONTAINER_NAME="atom-deepseek-mi355x"
        ;;
    *)
        echo "Error: Invalid gpu_type '${GPU_TYPE}'. Must be 'mi300' or 'mi355'."
        usage
        ;;
esac

TRACELENS_MOUNT="/workspace/TraceLens-internal"

echo "Using TraceLens path: ${TRACELENS_PATH}"
echo "Using GPU type: ${GPU_TYPE} (image: ${DOCKER_IMAGE})"
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
    --mount type=bind,src=/data,dst=/data \
    --device=/dev/kfd \
    --device=/dev/dri \
    -e SGLANG_USE_AITER=1 \
    -v /home:/home \
    -v "${TRACELENS_PATH}:${TRACELENS_MOUNT}" \
    ${DOCKER_IMAGE} \
    tail -f /dev/null

echo "Waiting for container to start..."
sleep 2

echo "Applying patches to ATOM..."
docker exec ${CONTAINER_NAME} bash -c "
    set -e
    ATOM_DIR=\$(python3 -c 'import atom; import os; print(os.path.dirname(os.path.dirname(atom.__file__)))')
    echo \"ATOM directory: \${ATOM_DIR}\"
    cd \"\${ATOM_DIR}\"
    echo \"Applying patches...\"
    for patch in ${TRACELENS_MOUNT}/examples/custom_workflows/inference_analysis/atom_roofline_patches/*.patch; do
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
