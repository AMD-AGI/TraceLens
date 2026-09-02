#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

set -e

usage() {
    echo "Usage: $0 <vllm-version> <path-to-TraceLens> [--base-image <image>] [docker build args...]"
    echo ""
    echo "  vllm-version    One of: v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28 (shorthand for v0.14.0 ... v0.28.0)"
    echo "  --base-image    Override the default base Docker image for the selected vllm version"
    echo ""
    echo "  Each version applies the matching vllm_patches/config_vllm_*.patch."
    echo ""
    echo "  v14-v25 patches add the profiler_config.capture_torch_profiler and"
    echo "  profiler_config.detailed_trace_annotation options, plus graph-capture tracing"
    echo "  for the V1 model runner."
    echo ""
    echo "  v26+ ship both config options upstream, so those patches instead add the"
    echo "  graph-capture tracing that upstream still lacks: the V2 model runner (decoder"
    echo "  and speculator, plus the encoder from v0.28.0) and the V1 encoder. Traces are"
    echo "  written per subsystem as graph_capture_rank_0[_encoder|_speculator].*."
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
    v14)
        BASE_IMAGE="rocm/vllm-dev:preview_releases_rocm_v0.14.0_20260120"
        PATCH_FILE="config_vllm_v0.14.0.patch"
        ;;
    v15)
        BASE_IMAGE="rocm/vllm-dev:preview_releases_rocm_v0.15.0_20260130"
        PATCH_FILE="config_vllm_v0.15.0.patch"
        ;;
    v16)
        BASE_IMAGE="rocm/vllm-dev:preview_rocm70_releases_rocm_v0.16.0_20260223"
        PATCH_FILE="config_vllm_v0.16.0.patch"
        ;;
    v17)
        BASE_IMAGE="vllm/vllm-openai-rocm:v0.17.0"
        PATCH_FILE="config_vllm_v0.17.0.patch"
        ;;
    v18)
        BASE_IMAGE="vllm/vllm-openai-rocm:v0.18.0"
        PATCH_FILE="config_vllm_v0.18.0.patch"
        ;;
    v19)
        BASE_IMAGE="vllm/vllm-openai-rocm:v0.19.0"
        PATCH_FILE="config_vllm_v0.19.0.patch"
        ;;
    v20)
        BASE_IMAGE="rocm/vllm-dev:preview_v0.20.0_20260429"
        PATCH_FILE="config_vllm_v0.20.0.patch"
        ;;
    v21)
        BASE_IMAGE="vllm/vllm-openai-rocm:v0.21.0"
        PATCH_FILE="config_vllm_v0.21.0.patch"
        ;;
    v22)
        BASE_IMAGE="vllm/vllm-openai-rocm:v0.22.0"
        PATCH_FILE="config_vllm_v0.22.0.patch"
        ;;
    v23)
        BASE_IMAGE="vllm/vllm-openai-rocm:v0.23.0"
        PATCH_FILE="config_vllm_v0.23.0.patch"
        ;;
    v24)
        BASE_IMAGE="vllm/vllm-openai-rocm:v0.24.0"
        PATCH_FILE="config_vllm_v0.24.0.patch"
        ;;
    v25)
        BASE_IMAGE="vllm/vllm-openai-rocm:v0.25.0"
        PATCH_FILE="config_vllm_v0.25.0.patch"
        ;;
    v26)
        BASE_IMAGE="vllm/vllm-openai-rocm:v0.26.0"
        PATCH_FILE="config_vllm_v0.26.0.patch"
        ;;
    v27)
        BASE_IMAGE="vllm/vllm-openai-rocm:v0.27.0"
        PATCH_FILE="config_vllm_v0.27.0.patch"
        ;;
    v28)
        BASE_IMAGE="vllm/vllm-openai-rocm:v0.28.0"
        PATCH_FILE="config_vllm_v0.28.0.patch"
        ;;
    *)
        echo "Error: unsupported vllm version '${VLLM_VERSION}'"
        echo "Supported versions: v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28"
        echo ""
        echo "vLLM v0.26.0 and later ship capture_torch_profiler and detailed_trace_annotation"
        echo "upstream, so the v26+ patches only add graph-capture tracing to the paths"
        echo "upstream still leaves untraced: the V2 model runner and the V1 encoder. A stock"
        echo "image is enough if you only need V1 decoder capture traces."
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

PATCH_PATH="examples/custom_workflows/inference_analysis/vllm_patches/${PATCH_FILE}"
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
    pip install --upgrade /tmp/TraceLens && \\
    rm -rf /tmp/TraceLens

WORKDIR /workspace
DOCKERFILE
