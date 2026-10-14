#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

set -e

usage() {
    echo "Usage: $0 <atom-version> <path-to-TraceLens> [--base-image <image>] [--atom-dir <dir>] [docker build args...]"
    echo ""
    echo "  atom-version    One of: v13, v14, v15, v16rc"
    echo "                  (shorthand for ATOM 0.1.3, 0.1.4, 0.1.5, 0.1.6rc)"
    echo "  --base-image    Override the default base Docker image for the selected atom version"
    echo "  --atom-dir      Override the directory containing the atom package"
    echo "                  (default: derived from 'import atom')"
    echo ""
    echo "  Each version applies the matching atom_roofline_patches/atom_0_1_*/ patches, which add"
    echo "  the ATOM_ENABLE_DETAILED_ANNOTATION environment variable, the detailed prefill/decode"
    echo "  annotations, and the graph-capture profiler. ATOM merged these upstream on 2026-07-21"
    echo "  (ROCm/ATOM#477), so nightly builds from 2026-07-22 onward need no image from this"
    echo "  script - run a stock rocm/atom-dev:nightly_* image and analyse the traces on the host."
    echo ""
    echo "Examples:"
    echo "  $0 v15 /home/user/TraceLens -t tracelens-atom"
    echo "  $0 v16rc . -t tracelens-atom:v16rc --no-cache"
    echo "  $0 v13 . --base-image my-custom/atom:latest -t tracelens-atom:custom"
    exit 1
}

if [ -z "$1" ] || [ -z "$2" ]; then
    usage
fi

ATOM_VERSION="$1"
shift

case "${ATOM_VERSION}" in
    v13)
        BASE_IMAGE="rocm/atom-dev:nightly_202605301523"
        PATCH_DIR="atom_0_1_3"
        ;;
    v14)
        BASE_IMAGE="rocm/atom-dev:atom0.1.4-aiter0.1.15"
        PATCH_DIR="atom_0_1_4"
        ;;
    v15)
        BASE_IMAGE="rocm/atom-dev:atom0.1.5-aiter0.1.16"
        PATCH_DIR="atom_0_1_5"
        ;;
    v16rc)
        # No image is published for release/v0.1.6, so this uses the closest
        # nightly built from main after that branch point.
        BASE_IMAGE="rocm/atom-dev:nightly_202607091539"
        PATCH_DIR="atom_0_1_6rc"
        ;;
    *)
        echo "Error: unsupported atom version '${ATOM_VERSION}'"
        echo "Supported versions: v13, v14, v15, v16rc"
        echo ""
        echo "ATOM merged the detailed annotations and the graph-capture profiler upstream on"
        echo "2026-07-21 (ROCm/ATOM#477), after release/v0.1.6 was branched. Nightly builds from"
        echo "2026-07-22 onward therefore need no patched image: run a stock"
        echo "rocm/atom-dev:nightly_* image instead."
        exit 1
        ;;
esac

TRACELENS_REPO="$(cd "$1" && pwd)"
shift

CUSTOM_BASE_IMAGE=""
ATOM_DIR_OVERRIDE=""
REMAINING_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --base-image)
            CUSTOM_BASE_IMAGE="$2"
            shift 2
            ;;
        --atom-dir)
            ATOM_DIR_OVERRIDE="$2"
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

PATCH_DIR_PATH="examples/custom_workflows/inference_analysis/atom_roofline_patches/${PATCH_DIR}"

if [ ! -d "${TRACELENS_REPO}/${PATCH_DIR_PATH}" ]; then
    echo "Error: patch directory not found: ${TRACELENS_REPO}/${PATCH_DIR_PATH}"
    exit 1
fi

# Resolve the directory holding the atom package
if [ -n "${ATOM_DIR_OVERRIDE}" ]; then
    ATOM_DIR_CMD="echo ${ATOM_DIR_OVERRIDE}"
else
    ATOM_DIR_CMD="python3 -c \"import importlib.util, os; print(os.path.dirname(os.path.dirname(importlib.util.find_spec('atom').origin)))\""
fi

echo "Building TraceLens Atom docker image"
echo "  Base image      : ${BASE_IMAGE}"
echo "  Atom version    : ${ATOM_VERSION}"
echo "  TraceLens       : ${TRACELENS_REPO}"
echo "  Patch directory : ${PATCH_DIR_PATH}"

# Only the patches are copied in: the image is used to collect traces 
PATCH_DIR_CONTAINER="/tmp/atom_roofline_patches"
docker build "${REMAINING_ARGS[@]}" -f - "${TRACELENS_REPO}" <<DOCKERFILE
FROM ${BASE_IMAGE}

COPY ${PATCH_DIR_PATH} ${PATCH_DIR_CONTAINER}

RUN ATOM_DIR=\$(${ATOM_DIR_CMD}) && \\
    cd "\${ATOM_DIR}" && \\
    for patch in ${PATCH_DIR_CONTAINER}/*.patch; do \\
        if [ -f "\$patch" ]; then \\
            echo "Applying \$(basename "\$patch")..." && \\
            (git apply "\$patch" || patch -p1 --fuzz=10 < "\$patch") || \\
            { echo "Failed to apply \$patch"; exit 1; }; \\
        fi \\
    done && \\
    rm -rf ${PATCH_DIR_CONTAINER}

WORKDIR /workspace
DOCKERFILE
