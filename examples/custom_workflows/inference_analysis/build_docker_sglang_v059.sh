#!/usr/bin/env bash
set -e

usage() {
    echo "Usage: $0 <tracelens_path> [gpu_type] [docker build args...]"
    echo "  tracelens_path: Path to the TraceLens-internal repository"
    echo "  gpu_type:       'mi300' or 'mi350/mi355' (default: mi350)"
    exit 1
}

if [ -z "$1" ]; then
    usage
fi

TRACELENS_PATH="$(realpath "$1")"
shift

if [ ! -d "${TRACELENS_PATH}" ]; then
    echo "Error: TraceLens path does not exist: ${TRACELENS_PATH}"
    exit 1
fi

GPU_TYPE="${1:-mi350}"
case "${GPU_TYPE}" in
    mi300)
        BASE_IMAGE="lmsysorg/sglang:v0.5.9-rocm700-mi30x"
        ;;
    mi350|mi355)
        BASE_IMAGE="lmsysorg/sglang:v0.5.9-rocm700-mi35x"
        ;;
    *)
        echo "Error: Invalid gpu_type '${GPU_TYPE}'. Must be 'mi300' or 'mi350/mi355'."
        usage
        ;;
esac
shift 2>/dev/null || true

echo "Building SGLang docker image"
echo "  Base image : ${BASE_IMAGE}"
echo "  GPU type   : ${GPU_TYPE}"
echo "  TraceLens  : ${TRACELENS_PATH}"

docker build "$@" -f - "${TRACELENS_PATH}" <<DOCKERFILE
FROM ${BASE_IMAGE}

COPY . /tmp/TraceLens-internal

RUN SGLANG_DIR=\$(pip show sglang | grep "Editable project location" | cut -d' ' -f4 | xargs dirname) && \\
    cd "\${SGLANG_DIR}" && \\
    for patch in /tmp/TraceLens-internal/examples/custom_workflows/inference_analysis/sglang_roofline_patches/*.patch; do \\
        [ -f "\$patch" ] && git apply "\$patch"; \\
    done && \\
    pip install --no-deps /tmp/TraceLens-internal && \\
    rm -rf /tmp/TraceLens-internal

WORKDIR /workspace
DOCKERFILE
