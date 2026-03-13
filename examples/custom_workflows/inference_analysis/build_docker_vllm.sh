#!/usr/bin/env bash
set -e

usage() {
    echo "Usage: $0 <vllm-version> <path-to-TraceLens-internal> [docker build args...]"
    echo ""
    echo "  vllm-version    One of: v14, v15, v16, v17 (shorthand for v0.14.0, v0.15.0, v0.16.0, v0.17.0)"
    echo ""
    echo "Examples:"
    echo "  $0 v14 /home/user/TraceLens-internal -t tracelens-vllm"
    echo "  $0 v16 . -t tracelens-vllm:v16 --no-cache"
    exit 1
}

if [ -z "$1" ] || [ -z "$2" ]; then
    usage
fi

VLLM_VERSION="$1"
shift

case "${VLLM_VERSION}" in
    v14)
        BASE_IMAGE="rocm/vllm-dev:preview_releases_rocm_v0.14.0_20260120"
        PATCH_FILE="vllm_v0.14.0.patch"
        ;;
    v15)
        BASE_IMAGE="rocm/vllm-dev:preview_releases_rocm_v0.15.0_20260130"
        PATCH_FILE="vllm_v0.15.0.patch"
        ;;
    v16)
        BASE_IMAGE="rocm/vllm-dev:preview_rocm70_releases_rocm_v0.16.0_20260223"
        PATCH_FILE="vllm_v0.16.0.patch"
        ;;
    v17)
        BASE_IMAGE="vllm/vllm-openai-rocm:v0.17.0"
        PATCH_FILE="vllm_v0.17.0.patch"
        ;;
    *)
        echo "Error: unsupported vllm version '${VLLM_VERSION}'"
        echo "Supported versions: v14, v15, v16, v17"
        exit 1
        ;;
esac

TRACELENS_REPO="$(cd "$1" && pwd)"
shift

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

docker build "$@" -f - "${TRACELENS_REPO}" <<DOCKERFILE
FROM ${BASE_IMAGE}

COPY . /tmp/TraceLens-internal

RUN VLLM_DIR=\$(python -c "import vllm, os; print(os.path.join(os.path.dirname(vllm.__file__), '..'))") && \\
    cd "\${VLLM_DIR}" && \\
    (git apply /tmp/TraceLens-internal/${PATCH_PATH} || patch -p1 --fuzz=10 < /tmp/TraceLens-internal/${PATCH_PATH}) && \\
    pip install --no-deps /tmp/TraceLens-internal && \\
    rm -rf /tmp/TraceLens-internal

WORKDIR /workspace
DOCKERFILE
