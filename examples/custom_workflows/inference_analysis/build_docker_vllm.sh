#!/usr/bin/env bash
set -e

usage() {
    echo "Usage: $0 <vllm-version> <path-to-TraceLens> [--base-image <image>] [docker build args...]"
    echo ""
    echo "  vllm-version    One of: v18, v19 (shorthand for v0.18.0, v0.19.0)"
    echo "  --base-image    Override the default base Docker image for the selected vllm version"
    echo ""
    echo "Examples:"
    echo "  $0 v14 /home/user/TraceLens -t tracelens-vllm"
    echo "  $0 v16 . -t tracelens-vllm:v16 --no-cache"
    echo "  $0 v18 . --base-image my-custom/vllm:latest -t tracelens-vllm:custom"
    exit 1
}

if [ -z "$1" ] || [ -z "$2" ]; then
    usage
fi

VLLM_VERSION="$1"
shift

case "${VLLM_VERSION}" in
    v18)
        BASE_IMAGE="vllm/vllm-openai-rocm:v0.18.0"
        PATCH_FILE="config_vllm_v0.18.0.patch"
        ;;
    v19)
        BASE_IMAGE="vllm/vllm-openai-rocm:v0.19.0"
        PATCH_FILE="config_vllm_v0.19.0.patch"
        ;;
    *)
        echo "Error: unsupported vllm version '${VLLM_VERSION}'"
        echo "Supported versions: v18"
        exit 1
        ;;
esac

TRACELENS_REPO="$(cd "$1" && pwd)"
shift

CUSTOM_BASE_IMAGE=""
REMAINING_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --base-image)
            CUSTOM_BASE_IMAGE="$2"
            shift 2
            ;;
        *)
            REMAINING_ARGS+=("$1")
            shift
            ;;
    esac
done

if [ -n "${CUSTOM_BASE_IMAGE}" ]; then
    BASE_IMAGE="${CUSTOM_BASE_IMAGE}"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_PATH="examples/custom_workflows/inference_analysis/${PATCH_FILE}"

if [ ! -f "${TRACELENS_REPO}/${PATCH_PATH}" ]; then
    echo "Error: patch file not found: ${TRACELENS_REPO}/${PATCH_PATH}"
    exit 1
fi

echo "Building TraceLens vLLM docker image"
echo "  Base image : ${BASE_IMAGE}"
echo "  Patch file : ${PATCH_FILE}"
echo "  TraceLens  : ${TRACELENS_REPO}"

docker build "${REMAINING_ARGS[@]}" -f - "${TRACELENS_REPO}" <<DOCKERFILE
FROM ${BASE_IMAGE}

COPY . /tmp/TraceLens

RUN VLLM_DIR=\$(python -c "import vllm, os; print(os.path.join(os.path.dirname(vllm.__file__), '..'))") && \\
    cd "\${VLLM_DIR}" && \\
    (git apply /tmp/TraceLens/${PATCH_PATH} || patch -p1 --fuzz=10 < /tmp/TraceLens/${PATCH_PATH}) && \\
    pip install --no-deps /tmp/TraceLens && \\
    rm -rf /tmp/TraceLens

WORKDIR /workspace
DOCKERFILE
