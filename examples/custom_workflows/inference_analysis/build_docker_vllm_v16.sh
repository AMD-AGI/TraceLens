#!/usr/bin/env bash
set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <path-to-TraceLens-internal> [docker build args...]"
    echo "Example: $0 /home/user/TraceLens-internal -t tracelens-vllm"
    exit 1
fi

TRACELENS_REPO="$(cd "$1" && pwd)"
shift
docker build "$@" -f - "${TRACELENS_REPO}" <<'DOCKERFILE'
FROM rocm/vllm-dev:preview_rocm70_releases_rocm_v0.16.0_20260223

COPY . /tmp/TraceLens-internal

RUN VLLM_DIR=$(python -c "import vllm, os; print(os.path.join(os.path.dirname(vllm.__file__), '..'))") && \
    cd "${VLLM_DIR}" && \
    git apply /tmp/TraceLens-internal/examples/custom_workflows/inference_analysis/vllm_v0.16.0.patch && \
    pip install --no-deps /tmp/TraceLens-internal && \
    rm -rf /tmp/TraceLens-internal

WORKDIR /workspace
DOCKERFILE
