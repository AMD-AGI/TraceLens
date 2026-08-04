#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

set -e

usage() {
    echo "Usage: $0 <xdit-version> <path-to-TraceLens> [--base-image <image>] [docker build args...]"
    echo ""
    echo "  xdit-version    One of: v26.6"
    echo "  --base-image    Override the default base Docker image for the selected xDiT version"
    echo ""
    echo "Examples:"
    echo "  $0 v26.6 /home/user/TraceLens -t tracelens-xdit:v26.6"
    echo "  $0 v26.6 . --base-image my-custom/xdit:latest -t tracelens-xdit:custom"
    exit 1
}

if [ -z "$1" ] || [ -z "$2" ]; then
    usage
fi

XDIT_VERSION="$1"
shift

case "${XDIT_VERSION}" in
    v26.6)
        BASE_IMAGE="rocm/pytorch-xdit:v26.6"
        PATCH_FILE="config_xdit_v26.6.patch"
        XDIT_COMMIT="2b8b5b709e3c63bcbf0f0640e11e916a15a85b46"
        ;;
    *)
        echo "Error: unsupported xDiT version '${XDIT_VERSION}'"
        echo "Supported versions: v26.6"
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
PATCH_PATH="examples/custom_workflows/inference_analysis/xdit_patches/${PATCH_FILE}"

if [ ! -f "${TRACELENS_REPO}/${PATCH_PATH}" ]; then
    echo "Error: patch file not found: ${TRACELENS_REPO}/${PATCH_PATH}"
    exit 1
fi

echo "Building TraceLens xDiT docker image"
echo "  Base image : ${BASE_IMAGE}"
echo "  Patch file : ${PATCH_FILE}"
echo "  TraceLens  : ${TRACELENS_REPO}"

docker build "${REMAINING_ARGS[@]}" -f - "${TRACELENS_REPO}" <<DOCKERFILE
FROM ${BASE_IMAGE}

COPY . /tmp/TraceLens

RUN XDIT_DIR=\$(python -c "import xfuser, os; print(os.path.join(os.path.dirname(xfuser.__file__), '..'))") && \
    cd "\${XDIT_DIR}" && \
    git checkout -- . && \
    git fetch origin && \
    git checkout ${XDIT_COMMIT} && \
    (git apply /tmp/TraceLens/${PATCH_PATH} || patch -p1 --fuzz=10 < /tmp/TraceLens/${PATCH_PATH}) && \
    pip install --upgrade /tmp/TraceLens && \
    rm -rf /tmp/TraceLens

WORKDIR /workspace
DOCKERFILE
