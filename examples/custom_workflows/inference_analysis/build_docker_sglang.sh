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
                           - 0.5.9  / v059  : sglang_roofline_patches/sglang_0_5_9/
                           - 0.5.11 / v0511 : sglang_roofline_patches/sglang_0_5_11/
                           - 0.5.12 / v0512 : sglang_roofline_patches/sglang_0_5_12/
                           - 0.5.13 / v0513 : sglang_roofline_patches/sglang_0_5_13/
                           - 0.5.14 / v0514 : sglang_roofline_patches/sglang_0_5_14/
                           - 0.5.15 / v0515 : sglang_roofline_patches/sglang_0_5_15/
                           - 0.5.16 / v0516 : sglang_roofline_patches/sglang_0_5_16/
                           - 0.5.17 / v0517 : sglang_roofline_patches/sglang_0_5_17/
                           - 0.5.18 / v0518 : sglang_roofline_patches/sglang_0_5_18/
  --gpu-type <type>        mi300 | mi350 | mi355 (default: mi350)
  --base-image <image>     Override the default base image
  --patch-dir <name>       Patch directory under sglang_roofline_patches/
                           instead of the one derived from --sglang-version.
                           Use for bases that have drifted from a release tag,
                           e.g. --patch-dir sglang_0_5_17_sgldev for the
                           rocm/sgl-dev nightly.
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
  0.5.13 MI300X : lmsysorg/sglang:v0.5.13-rocm720-mi30x
  0.5.13 MI355X : lmsysorg/sglang:v0.5.13-rocm720-mi35x
  0.5.14 MI300X : lmsysorg/sglang:v0.5.14-rocm720-mi30x
  0.5.14 MI355X : lmsysorg/sglang:v0.5.14-rocm720-mi35x
  0.5.15 MI300X : lmsysorg/sglang:v0.5.15-rocm720-mi30x
  0.5.15 MI355X : lmsysorg/sglang:v0.5.15-rocm720-mi35x
  0.5.16 MI300X : lmsysorg/sglang:v0.5.16-rocm720-mi30x
  0.5.16 MI355X : lmsysorg/sglang:v0.5.16-rocm720-mi35x
  0.5.17 MI300X : lmsysorg/sglang:v0.5.17-rocm720-mi30x
  0.5.17 MI355X : lmsysorg/sglang:v0.5.17-rocm720-mi35x
  0.5.18 MI300X : lmsysorg/sglang-rocm:v0.5.18-rocm724-mi30x-20260824
  0.5.18 MI355X : lmsysorg/sglang-rocm:v0.5.18-rocm724-mi35x-20260824

Note:
  On a ROCm 7.2.4 base the build also overwrites the HIP and roctracer copies
  that the torch wheel vendors under torch/lib with the image's own 7.2.4 build.
  Those vendored copies are 7.2.0, libtorch_hip.so carries DT_RPATH \$ORIGIN so
  LD_LIBRARY_PATH cannot redirect them, and on 7.2.0 every kernel launched inside
  a HIP graph replay is missing from the trace -- decode steps record only the
  hipGraphLaunch. This is skipped on 7.2.0 and 7.0 bases.

Note:
  SGLang 0.5.18 carries the detailed-annotation and per-batch-size graph-capture
  work upstream, so its patch set only adds the kernel shape profiler and the
  shape_discovery plumbing that reaches it.

Note:
  On SGLang 0.5.13 and 0.5.14 the kernel-shape wrapping is incompatible with the
  EAGLE/MTP speculative *overlap* decode, so the speculative patches disable
  capture profiling on the speculative graph runners (and, on 0.5.14, the
  target-verify graph) to keep MTP fault-free. Non-MTP shape profiling is
  unaffected. Full MTP shape profiling (capture + execution) works on 0.5.15+.

Examples:
  $0 /path/to/TraceLens --sglang-version 0.5.11 --gpu-type mi300 -t tracelens-sglang:0.5.11-mi300
  $0 /path/to/TraceLens --sglang-version 0.5.16 --gpu-type mi355 -t tracelens-sglang:0.5.16-mi355
  $0 /path/to/TraceLens mi350 -t tracelens-sglang:0.5.9-mi350
EOF
    exit 1
}

SGLANG_VERSION="0.5.9"
GPU_TYPE="mi350"
CUSTOM_BASE_IMAGE=""
CUSTOM_PATCH_DIR=""
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
        --patch-dir)
            CUSTOM_PATCH_DIR="$2"
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
        0.5.13|v0513|0513|5.13)
            echo "0.5.13"
            ;;
        0.5.14|v0514|0514|5.14)
            echo "0.5.14"
            ;;
        0.5.15|v0515|0515|5.15)
            echo "0.5.15"
            ;;
        0.5.16|v0516|0516|5.16)
            echo "0.5.16"
            ;;
        0.5.17|v0517|0517|5.17)
            echo "0.5.17"
            ;;
        0.5.18|v0518|0518|5.18)
            echo "0.5.18"
            ;;
        *)
            echo ""
            ;;
    esac
}

SGLANG_VERSION="$(normalize_version "${SGLANG_VERSION}")"
if [ -z "${SGLANG_VERSION}" ]; then
    echo "Error: unsupported --sglang-version. Use 0.5.9, 0.5.11, 0.5.12, 0.5.13, 0.5.14, 0.5.15, 0.5.16, 0.5.17, or 0.5.18."
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
        0.5.13:mi300)
            echo "lmsysorg/sglang:v0.5.13-rocm720-mi30x"
            ;;
        0.5.13:mi350|0.5.13:mi355)
            echo "lmsysorg/sglang:v0.5.13-rocm720-mi35x"
            ;;
        0.5.14:mi300)
            echo "lmsysorg/sglang:v0.5.14-rocm720-mi30x"
            ;;
        0.5.14:mi350|0.5.14:mi355)
            echo "lmsysorg/sglang:v0.5.14-rocm720-mi35x"
            ;;
        0.5.15:mi300)
            echo "lmsysorg/sglang:v0.5.15-rocm720-mi30x"
            ;;
        0.5.15:mi350|0.5.15:mi355)
            echo "lmsysorg/sglang:v0.5.15-rocm720-mi35x"
            ;;
        0.5.16:mi300)
            echo "lmsysorg/sglang:v0.5.16-rocm720-mi30x"
            ;;
        0.5.16:mi350|0.5.16:mi355)
            echo "lmsysorg/sglang:v0.5.16-rocm720-mi35x"
            ;;
        0.5.17:mi300)
            echo "lmsysorg/sglang:v0.5.17-rocm720-mi30x"
            ;;
        0.5.17:mi350|0.5.17:mi355)
            echo "lmsysorg/sglang:v0.5.17-rocm720-mi35x"
            ;;
        0.5.18:mi300)
            echo "lmsysorg/sglang-rocm:v0.5.18-rocm724-mi30x-20260824"
            ;;
        0.5.18:mi350|0.5.18:mi355)
            echo "lmsysorg/sglang-rocm:v0.5.18-rocm724-mi35x-20260824"
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

PATCH_DIR="${CUSTOM_PATCH_DIR:-sglang_$(echo "${SGLANG_VERSION}" | tr '.' '_')}"
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
    pip install --upgrade /tmp/TraceLens && \\
    rm -rf /tmp/TraceLens

# On a ROCm 7.2.4 base, make torch load the image's own HIP and roctracer instead
# of the 7.2.0 copies its wheel vendors. libtorch_hip.so carries DT_RPATH \$ORIGIN,
# so overwriting these two files is the only way to change which runtime profiles;
# left as shipped, kernels inside a HIP graph replay never reach the trace.
# Keyed on the ROCm version in the image rather than the tag so --base-image works.
RUN set -eu; \\
    ROCM_VERSION="\$(cat /opt/rocm/.info/version 2>/dev/null || echo unknown)"; \\
    case "\${ROCM_VERSION}" in \\
        7.2.4*) \\
            TORCH_LIB="\$(python3 -c 'import torch, pathlib; print(pathlib.Path(torch.__file__).resolve().parent / "lib")')"; \\
            HIP_SRC="\$(find /opt/rocm/lib -maxdepth 1 -type f -name 'libamdhip64.so.*.*' | head -1)"; \\
            RT_SRC="\$(find /opt/rocm/lib -maxdepth 1 -type f -name 'libroctracer64.so.*.*' | head -1)"; \\
            if [ -z "\${HIP_SRC}" ] || [ -z "\${RT_SRC}" ]; then \\
                echo "ROCm \${ROCM_VERSION} base but no HIP/roctracer found under /opt/rocm/lib"; \\
                exit 1; \\
            fi; \\
            cp -a "\${HIP_SRC}" "\${TORCH_LIB}/libamdhip64.so"; \\
            cp -a "\${RT_SRC}" "\${TORCH_LIB}/libroctracer64.so"; \\
            echo "torch/lib now carries \$(basename "\${HIP_SRC}") and \$(basename "\${RT_SRC}")"; \\
            ;; \\
        *) \\
            echo "Base ROCm \${ROCM_VERSION} is not 7.2.4; leaving torch/lib as shipped"; \\
            ;; \\
    esac

WORKDIR /workspace
DOCKERFILE
