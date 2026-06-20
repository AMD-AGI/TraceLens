#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCHES_ROOT="${SCRIPT_DIR}/sglang_roofline_patches"

usage() {
    cat <<EOF
Usage: $0 <tracelens_path> [options] [docker build args...]

Build a TraceLens + SGLang inference-analysis image on AMD ROCm base images.

Positional:
  tracelens_path   Path to the TraceLens repository

Options:
  --sglang-version <ver>   SGLang version to patch (default: 0.5.9)
                           - 0.5.9  / v059   : sglang_roofline_patches/sglang_0_5_9/
                           - 0.5.11 / v0511 : sglang_roofline_patches/sglang_0_5_11/
                           - 0.5.12 / v0512 : sglang_roofline_patches/sglang_0_5_12/
  --gpu-type <type>        mi300 | mi350 | mi355 (default: mi350)
  --base-image <image>     Override the default base image
  -h, --help               Show this help

Legacy positional (still supported):
  $0 <tracelens_path> [mi300|mi350|mi355] [docker build args...]

Base images:
  0.5.9  MI300X : lmsysorg/sglang:v0.5.9-rocm700-mi30x
  0.5.9  MI355X : lmsysorg/sglang:v0.5.9-rocm700-mi35x
  0.5.11 MI300X : lmsysorg/sglang:v0.5.11-rocm720-mi30x
  0.5.11 MI355X : lmsysorg/sglang:v0.5.11-rocm720-mi35x
  0.5.12 MI300X : lmsysorg/sglang:v0.5.12-rocm720-mi30x
  0.5.12 MI355X : lmsysorg/sglang:v0.5.12-rocm720-mi35x

Examples:
  $0 /path/to/TraceLens --sglang-version 0.5.12 --gpu-type mi300 -t tracelens-sglang:0.5.12-mi300
  $0 /path/to/TraceLens --sglang-version 0.5.11 --gpu-type mi300 -t tracelens-sglang:0.5.11-mi300
  $0 /path/to/TraceLens mi350 -t tracelens-sglang:0.5.9-mi350
EOF
    exit 1
}

SGLANG_VERSION="0.5.9"
GPU_TYPE="mi350"
CUSTOM_BASE_IMAGE=""
TRACELENS_PATH=""
DOCKER_ARGS=()

if [ -z "$1" ]; then
    usage
fi

# Legacy: second positional arg may be gpu type before flags/docker args.
if [[ $# -ge 2 && "$2" != --* ]]; then
    case "$2" in
        mi300|mi350|mi355)
            GPU_TYPE="$2"
            shift
            ;;
    esac
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --sglang-version)
            SGLANG_VERSION="$2"
            shift 2
            ;;
        --gpu-type|--gpu)
            GPU_TYPE="$2"
            shift 2
            ;;
        --base-image)
            CUSTOM_BASE_IMAGE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        --)
            shift
            DOCKER_ARGS+=("$@")
            break
            ;;
        *)
            if [ -z "${TRACELENS_PATH}" ]; then
                TRACELENS_PATH="$(realpath "$1")"
            else
                DOCKER_ARGS+=("$1")
            fi
            shift
            ;;
    esac
done

if [ -z "${TRACELENS_PATH}" ]; then
    echo "Error: tracelens_path is required."
    usage
fi

if [ ! -d "${TRACELENS_PATH}" ]; then
    echo "Error: TraceLens path does not exist: ${TRACELENS_PATH}"
    exit 1
fi

normalize_version() {
    case "$1" in
        0.5.9|v059|059|5.9)
            echo "0.5.9"
            ;;
        0.5.11|v0511|0511|5.11)
            echo "0.5.11"
            ;;
        0.5.12|v0512|0512|5.12)
            echo "0.5.12"
            ;;
        *)
            echo ""
            ;;
    esac
}

SGLANG_VERSION="$(normalize_version "${SGLANG_VERSION}")"
if [ -z "${SGLANG_VERSION}" ]; then
    echo "Error: unsupported --sglang-version. Use 0.5.9, 0.5.11, or 0.5.12."
    exit 1
fi

resolve_base_image() {
    local version="$1"
    local gpu="$2"
    case "${version}:${gpu}" in
        0.5.9:mi300)
            echo "lmsysorg/sglang:v0.5.9-rocm700-mi30x"
            ;;
        0.5.9:mi350|0.5.9:mi355)
            echo "lmsysorg/sglang:v0.5.9-rocm700-mi35x"
            ;;
        0.5.11:mi300)
            echo "lmsysorg/sglang:v0.5.11-rocm720-mi30x"
            ;;
        0.5.11:mi350|0.5.11:mi355)
            echo "lmsysorg/sglang:v0.5.11-rocm720-mi35x"
            ;;
        0.5.12:mi300)
            echo "lmsysorg/sglang:v0.5.12-rocm720-mi30x"
            ;;
        0.5.12:mi350|0.5.12:mi355)
            echo "lmsysorg/sglang:v0.5.12-rocm720-mi35x"
            ;;
        *)
            echo ""
            ;;
    esac
}

case "${GPU_TYPE}" in
    mi300|mi350|mi355) ;;
    *)
        echo "Error: invalid --gpu-type '${GPU_TYPE}'. Must be mi300, mi350, or mi355."
        exit 1
        ;;
esac

BASE_IMAGE="$(resolve_base_image "${SGLANG_VERSION}" "${GPU_TYPE}")"
if [ -z "${BASE_IMAGE}" ]; then
    echo "Error: no base image mapping for version=${SGLANG_VERSION} gpu=${GPU_TYPE}"
    exit 1
fi

if [ -n "${CUSTOM_BASE_IMAGE}" ]; then
    BASE_IMAGE="${CUSTOM_BASE_IMAGE}"
fi

PATCH_DIR="sglang_$(echo "${SGLANG_VERSION}" | tr '.' '_')"
PATCH_DIR_PATH="${PATCHES_ROOT}/${PATCH_DIR}"
if [ ! -d "${PATCH_DIR_PATH}" ]; then
    echo "Error: patch directory not found: ${PATCH_DIR_PATH}"
    exit 1
fi

echo "Building SGLang docker image"
echo "  Base image      : ${BASE_IMAGE}"
echo "  SGLang version  : ${SGLANG_VERSION}"
echo "  GPU type        : ${GPU_TYPE}"
echo "  TraceLens       : ${TRACELENS_PATH}"
echo "  Patch directory : ${PATCH_DIR_PATH}"

PATCH_DIR_CONTAINER="/tmp/TraceLens/examples/custom_workflows/inference_analysis/sglang_roofline_patches/${PATCH_DIR}"
docker build "${DOCKER_ARGS[@]}" -f - "${TRACELENS_PATH}" <<DOCKERFILE
FROM ${BASE_IMAGE}

COPY . /tmp/TraceLens

RUN SGLANG_DIR=\$(pip show sglang | grep "Editable project location" | cut -d' ' -f4 | xargs dirname) && \\
    cd "\${SGLANG_DIR}" && \\
    for patch in ${PATCH_DIR_CONTAINER}/*.patch; do \\
        if [ -f "\$patch" ]; then \\
            echo "Applying \$(basename "\$patch")..." && \\
            (git apply "\$patch" || patch -p1 --fuzz=10 < "\$patch") || \\
            { echo "Failed to apply \$patch"; exit 1; }; \\
        fi \\
    done && \\
    pip install --no-deps /tmp/TraceLens && \\
    rm -rf /tmp/TraceLens

WORKDIR /workspace
DOCKERFILE
