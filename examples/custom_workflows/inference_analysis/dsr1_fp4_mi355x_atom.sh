#!/usr/bin/env bash

source "$(dirname "$0")/benchmark_lib.sh"
export MODEL="deepseek-ai/DeepSeek-R1-0528"

export PORT=8000
export TP=8
export CONC=16
export ISL=2048
export OSL=16
export RANDOM_RANGE_RATIO=0.8
export RESULT_FILENAME="results.json"
export DP_ATTENTION="false"
export EP_SIZE=1

check_env_vars \
    MODEL \
    TP \
    CONC \
    ISL \
    OSL \
    RANDOM_RANGE_RATIO \
    RESULT_FILENAME \
    EP_SIZE \
    DP_ATTENTION

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

echo "TP: $TP, CONC: $CONC, ISL: $ISL, OSL: $OSL, EP_SIZE: $EP_SIZE, DP_ATTENTION: $DP_ATTENTION"

mkdir /home/profile
mkdir /workspace/
SERVER_LOG=/workspace/server.log
PORT=${PORT:-8888}
export ATOM_PROFILER_MORE=1
export OMP_NUM_THREADS=1
export ATOM_ENABLE_ROOFLINE_ANNOTATION=1


# Calculate max-model-len based on ISL and OSL
if [ "$ISL" = "1024" ] && [ "$OSL" = "1024" ]; then
    CALCULATED_MAX_MODEL_LEN=""
else
    CALCULATED_MAX_MODEL_LEN=" --max-model-len 10240 "
fi

if [ "$EP_SIZE" -gt 1 ]; then
  EP=" --enable-expert-parallel"
else
  EP=" "
fi

set -x

BLOCK_SIZE=${BLOCK_SIZE:-16}
python3 -m atom.entrypoints.openai_server \
    --model $MODEL \
    --server-port $PORT \
    -tp $TP \
    --kv_cache_dtype fp8 $CALCULATED_MAX_MODEL_LEN $EP \
    --torch-profiler-dir /home/profile \
    --enable-capture-profiling \
    --block-size $BLOCK_SIZE > $SERVER_LOG 2>&1 &

SERVER_PID=$!

#--enforce-eager \ <----for eager mode

# Wait for server to be ready
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

run_benchmark_serving \
    --model "$MODEL" \
    --port "$PORT" \
    --backend vllm \
    --input-len "$ISL" \
    --output-len "$OSL" \
    --random-range-ratio "$RANDOM_RANGE_RATIO" \
    --num-prompts "$((CONC * 2))" \
    --max-concurrency "$CONC" \
    --result-filename "$RESULT_FILENAME" \
    --result-dir /workspace/




# After throughput, run evaluation only if RUN_EVAL is true
if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT" --concurrent-requests $CONC
    append_lm_eval_summary
fi

# Shutdown server
echo "Shutting down Atom server (PID: $SERVER_PID)..."
kill $SERVER_PID
wait $SERVER_PID 2>/dev/null
echo "Server shutdown complete."

set +x

