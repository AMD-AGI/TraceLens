#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

set -euo pipefail

# ---------------------------------------------------------------------------
# Start a vLLM Docker container, install pi + TraceLens inside it, configure pi
# from the running vLLM /v1/models endpoint, and run Analysis evals with --pi.
#
# Usage:
#   bash run_pi_analysis_in_docker.sh <tracelens_root> [standalone|comparative] [options] -- <vllm-args...>
#
# <vllm-args> are passed to vllm (prepend "vllm serve" automatically when omitted).
#
# Examples:
#   bash run_pi_analysis_in_docker.sh /data/tracelens_local_testing/tracelens -- \
#     MiniMaxAI/MiniMax-M3 --trust-remote-code --port 30000 --tensor-parallel-size 4
#
#   TEST_IDS="gemm_01" NUM_REPEATS=1 \
#     bash run_pi_analysis_in_docker.sh /data/tracelens_local_testing/tracelens standalone -- \
#     vllm serve MiniMaxAI/MiniMax-M3 --port 30000
#
# Environment:
#   DOCKER_IMAGE        Inference server Docker image (default: vllm/vllm-openai-rocm:nightly)
#   CONTAINER_NAME      Container name (default: tracelens_pi_evals)
#   WORK_DIR            Host dir mounted at /workspace (default: parent of tracelens_root)
#   VLLM_PORT           API port if not given in vllm args (default: 30000)
#   VLLM_READY_TIMEOUT  Seconds to wait for /v1/models (default: 1800)
#   DOCKER_RUN_ARGS     Extra whitespace-separated docker run arguments
#   SKIP_EVAL           Set to 1 to configure only (no eval harness)
#   Harness env vars (TEST_IDS, NUM_REPEATS, etc.) pass through to run_repeatability_parallel.sh
# ---------------------------------------------------------------------------

DOCKER_IMAGE="${DOCKER_IMAGE:-vllm/vllm-openai-rocm:nightly}"
CONTAINER_NAME="${CONTAINER_NAME:-tracelens_pi_evals}"
VLLM_PORT="${VLLM_PORT:-30000}"
VLLM_READY_TIMEOUT="${VLLM_READY_TIMEOUT:-1800}"
SKIP_EVAL="${SKIP_EVAL:-0}"

DEFAULT_DOCKER_RUN_ARGS=(
    --device /dev/dri
    --device /dev/kfd
    --device /dev/infiniband
    --network host
    --ipc host
    --group-add video
    --cap-add SYS_PTRACE
    --security-opt seccomp=unconfined
    --privileged
    --shm-size 128G
)

usage() {
    cat <<'EOF'
Usage: bash run_pi_analysis_in_docker.sh <tracelens_root> [standalone|comparative] [options] -- <vllm-args...>

  tracelens_root         TraceLens repo on the host (correct branch already checked out)
  standalone|comparative Harness comparison scope (default: standalone)

Options:
  --docker-image IMAGE   Inference server Docker image (default: vllm/vllm-openai-rocm:nightly)
  --container-name NAME    Docker container name (default: tracelens_pi_evals)
  --work-dir DIR           Host directory mounted at /workspace (default: <parent>)
  --vllm-port PORT         Default API port when not set in vllm args (default: 30000)
  -h, --help               Show this help

  Arguments after -- are passed to vllm. If they do not start with "vllm" or "serve",
  "vllm serve" is prepended automatically.

Environment:
  DOCKER_RUN_ARGS          Extra arguments appended to docker run
  VLLM_READY_TIMEOUT       Seconds to wait for http://localhost:<port>/v1/models
  SKIP_EVAL=1              Set up vLLM + pi only; skip the eval harness

Example:
  bash run_pi_analysis_in_docker.sh /data/tracelens_local_testing/tracelens -- \
    MiniMaxAI/MiniMax-M3 --trust-remote-code --block-size 128 --port 30000
EOF
}

die() {
    echo "ERROR: $*" >&2
    exit 1
}

TRACELENS_ROOT=""
COMPARISON_SCOPE="${COMPARISON_SCOPE:-standalone}"
VLLM_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --docker-image)
            [[ $# -ge 2 ]] || die "--docker-image requires a value"
            DOCKER_IMAGE="$2"
            shift 2
            ;;
        --container-name)
            [[ $# -ge 2 ]] || die "--container-name requires a value"
            CONTAINER_NAME="$2"
            shift 2
            ;;
        --work-dir)
            [[ $# -ge 2 ]] || die "--work-dir requires a value"
            WORK_DIR="$2"
            shift 2
            ;;
        --vllm-port)
            [[ $# -ge 2 ]] || die "--vllm-port requires a value"
            VLLM_PORT="$2"
            shift 2
            ;;
        standalone|comparative)
            COMPARISON_SCOPE="$1"
            shift
            ;;
        --)
            shift
            VLLM_ARGS=("$@")
            break
            ;;
        -*)
            die "Unknown option: $1"
            ;;
        *)
            if [[ -z "$TRACELENS_ROOT" ]]; then
                TRACELENS_ROOT="$1"
                shift
            else
                die "Unexpected argument: $1 (use -- before vllm arguments)"
            fi
            ;;
    esac
done

if [[ -z "$TRACELENS_ROOT" ]]; then
    usage >&2
    die "tracelens_root is required"
fi

if [[ ${#VLLM_ARGS[@]} -eq 0 ]]; then
    die "vllm arguments are required after --"
fi

if [[ ! -d "$TRACELENS_ROOT" ]]; then
    die "TraceLens root not found: $TRACELENS_ROOT"
fi

TRACELENS_ROOT="$(cd "$TRACELENS_ROOT" && pwd)"
REPO_BASENAME="$(basename "$TRACELENS_ROOT")"
WORK_DIR="${WORK_DIR:-$(dirname "$TRACELENS_ROOT")}"
WORK_DIR="$(cd "$WORK_DIR" && pwd)"

if [[ "$TRACELENS_ROOT" != "$WORK_DIR/"* && "$TRACELENS_ROOT" != "$WORK_DIR" ]]; then
    die "tracelens_root must be inside WORK_DIR ($WORK_DIR)"
fi

CONTAINER_REPO="/workspace/$REPO_BASENAME"
HARNESS="$TRACELENS_ROOT/agent_evals/Analysis/eval_scripts/run_repeatability_parallel.sh"
if [[ ! -f "$HARNESS" ]]; then
    die "Eval harness not found: $HARNESS"
fi

# Normalize vllm command line.
if [[ "${VLLM_ARGS[0]}" == vllm ]]; then
    :
elif [[ "${VLLM_ARGS[0]}" == serve ]]; then
    VLLM_ARGS=(vllm "${VLLM_ARGS[@]}")
else
    VLLM_ARGS=(vllm serve "${VLLM_ARGS[@]}")
fi

for ((i = 0; i < ${#VLLM_ARGS[@]}; i++)); do
    if [[ "${VLLM_ARGS[i]}" == --port && $((i + 1)) -lt ${#VLLM_ARGS[@]} ]]; then
        VLLM_PORT="${VLLM_ARGS[i + 1]}"
    elif [[ "${VLLM_ARGS[i]}" == --port=* ]]; then
        VLLM_PORT="${VLLM_ARGS[i]#--port=}"
    fi
done

VLLM_CMD_QUOTED=""
for arg in "${VLLM_ARGS[@]}"; do
    VLLM_CMD_QUOTED+="$(printf '%q ' "$arg")"
done

mkdir -p "$WORK_DIR/.pi/agent" "$WORK_DIR/venv_tracelens"

DOCKER_ARGS=("${DEFAULT_DOCKER_RUN_ARGS[@]}")
if [[ -n "${DOCKER_RUN_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    EXTRA=($DOCKER_RUN_ARGS)
    DOCKER_ARGS+=("${EXTRA[@]}")
fi

cleanup() {
    if [[ -n "${CONTAINER_NAME:-}" ]]; then
        docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    echo "Removing existing container: $CONTAINER_NAME"
    docker rm -f "$CONTAINER_NAME" >/dev/null
fi

echo "========================================="
echo "  pi + vLLM Analysis Eval Launcher"
echo "  TraceLens root:  $TRACELENS_ROOT"
echo "  Work dir:        $WORK_DIR -> /workspace"
echo "  Container repo:  $CONTAINER_REPO"
echo "  Docker image:    $DOCKER_IMAGE"
echo "  Container:       $CONTAINER_NAME"
echo "  vLLM port:       $VLLM_PORT"
echo "  vLLM command:    ${VLLM_ARGS[*]}"
echo "  Comparison:      $COMPARISON_SCOPE"
echo "========================================="
echo ""

echo "Starting Docker container..."
# shellcheck disable=SC2086
docker run -d --name "$CONTAINER_NAME" \
    "${DOCKER_ARGS[@]}" \
    -v "$WORK_DIR:/workspace:rw" \
    --entrypoint bash \
    "$DOCKER_IMAGE" \
    -c 'sleep infinity' >/dev/null

echo "Container started. Running setup inside container..."

EXEC_ENV=(
    -e "VLLM_PORT=$VLLM_PORT"
    -e "VLLM_READY_TIMEOUT=$VLLM_READY_TIMEOUT"
    -e "CONTAINER_REPO=$CONTAINER_REPO"
    -e "COMPARISON_SCOPE=$COMPARISON_SCOPE"
    -e "SKIP_EVAL=$SKIP_EVAL"
    -e "VLLM_CMD=$VLLM_CMD_QUOTED"
)
for var in TEST_IDS NUM_REPEATS MAX_PARALLEL SLEEP_BETWEEN TEST_TRACES_CSV RESULTS_ROOT REPORT_DIR SUITE_NAME SKIP_POST_PROCESSING; do
    if [[ -n "${!var:-}" ]]; then
        EXEC_ENV+=(-e "$var=${!var}")
    fi
done

docker exec "${EXEC_ENV[@]}" -i "$CONTAINER_NAME" bash -s <<'INNER'
set -euo pipefail

VLLM_PORT="${VLLM_PORT:-30000}"
VLLM_READY_TIMEOUT="${VLLM_READY_TIMEOUT:-1800}"
CONTAINER_REPO="${CONTAINER_REPO:?}"
COMPARISON_SCOPE="${COMPARISON_SCOPE:-standalone}"
SKIP_EVAL="${SKIP_EVAL:-0}"
PI_AGENT_DIR="/workspace/.pi/agent"
VENV_DIR="/workspace/venv_tracelens"
VLLM_LOG="/workspace/vllm.log"

die() {
    echo "ERROR: $*" >&2
    exit 1
}

echo "==> Starting vLLM: ${VLLM_CMD}"
# shellcheck disable=SC2086
eval "${VLLM_CMD}" >"$VLLM_LOG" 2>&1 &
VLLM_PID=$!

cleanup_inner() {
    if kill -0 "$VLLM_PID" 2>/dev/null; then
        kill "$VLLM_PID" 2>/dev/null || true
    fi
}
trap cleanup_inner EXIT

echo "==> Waiting for vLLM at http://localhost:${VLLM_PORT}/v1/models (timeout ${VLLM_READY_TIMEOUT}s)..."
models_json=""
deadline=$((SECONDS + VLLM_READY_TIMEOUT))
while (( SECONDS < deadline )); do
    if models_json="$(curl -sf "http://localhost:${VLLM_PORT}/v1/models" 2>/dev/null)"; then
        break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "vLLM exited early. Last log lines:" >&2
        tail -n 40 "$VLLM_LOG" >&2 || true
        die "vLLM process died before /v1/models became available"
    fi
    sleep 5
done

if [[ -z "$models_json" ]]; then
    echo "vLLM log tail:" >&2
    tail -n 40 "$VLLM_LOG" >&2 || true
    die "Timed out waiting for http://localhost:${VLLM_PORT}/v1/models"
fi

echo "==> Discovering model id from /v1/models..."
mkdir -p "$PI_AGENT_DIR"
python3 - "$PI_AGENT_DIR" "$VLLM_PORT" <<'PY'
import json
import pathlib
import sys
import urllib.request

agent_dir = pathlib.Path(sys.argv[1])
port = sys.argv[2]
url = f"http://localhost:{port}/v1/models"

with urllib.request.urlopen(url, timeout=30) as resp:
    payload = json.load(resp)

models = payload.get("data") or []
if not models:
    raise SystemExit("No models returned from /v1/models")

entries = []
for item in models:
    model_id = item.get("id")
    if not model_id:
        continue
    context = (
        item.get("max_model_len")
        or item.get("context_length")
        or item.get("contextWindow")
        or 1_000_000
    )
    entries.append({"id": model_id, "contextWindow": int(context)})

if not entries:
    raise SystemExit("Could not parse model id from /v1/models response")

config = {
    "providers": {
        "local": {
            "baseUrl": f"http://localhost:{port}/v1",
            "api": "openai-completions",
            "apiKey": "none",
            "models": entries,
        }
    }
}

out = agent_dir / "models.json"
out.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
print(f"Wrote {out} with model id(s): {', '.join(m['id'] for m in entries)}")
PY

export PI_CODING_AGENT_DIR="$PI_AGENT_DIR"
echo "PI_CODING_AGENT_DIR=$PI_CODING_AGENT_DIR"

if ! command -v pi >/dev/null 2>&1; then
    echo "==> Installing pi..."
    curl -fsSL https://pi.dev/install.sh | sh
fi

export PATH="${HOME}/.local/bin:${PATH}"
command -v pi >/dev/null 2>&1 || die "pi not found on PATH after install"

if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
    echo "==> Creating Python venv at $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
fi

# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

echo "==> Installing TraceLens editable from $CONTAINER_REPO..."
pip install -q --upgrade pip
pip install -q -e "$CONTAINER_REPO"

if [[ "$SKIP_EVAL" == "1" ]]; then
    echo "SKIP_EVAL=1 — setup complete, skipping eval harness."
    trap - EXIT
    exit 0
fi

HARNESS="$CONTAINER_REPO/agent_evals/Analysis/eval_scripts/run_repeatability_parallel.sh"
[[ -f "$HARNESS" ]] || die "Harness not found in container: $HARNESS"

export REPO_ROOT="$CONTAINER_REPO"

echo "==> Running eval harness: $HARNESS --pi $COMPARISON_SCOPE"
exec bash "$HARNESS" --pi "$COMPARISON_SCOPE"
INNER

echo ""
echo "Eval run finished. Results are under:"
echo "  $WORK_DIR/$REPO_BASENAME/agent_evals/Analysis/"
echo ""
echo "vLLM log inside container: /workspace/vllm.log"
echo "pi config: $WORK_DIR/.pi/agent/models.json"
