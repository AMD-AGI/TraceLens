#!/usr/bin/env bash

source "$(dirname "$0")/benchmark_lib.sh"
export MODEL="deepseek-ai/DeepSeek-R1-0528"
export PORT=8000
export TP=8
export CONC=32
export ISL=2048
export OSL=32
export RANDOM_RANGE_RATIO=0.8
export RESULT_FILENAME="results.json"
export PROFILE_DIR="/home/profile"

check_env_vars \
    MODEL \
    TP \
    CONC \
    ISL \
    OSL \
    RANDOM_RANGE_RATIO \
    RESULT_FILENAME

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

hf download "$MODEL"
mkdir /workspace
# Reference
# https://rocm.docs.amd.com/en/docs-7.0-docker/benchmark-docker/inference-sglang-deepseek-r1-fp8.html

export SGLANG_USE_AITER=1

export RCCL_MSCCL_ENABLE=0
export ROCM_QUICK_REDUCE_QUANTIZATION=INT4

mkdir "$PROFILE_DIR"
export SGLANG_TORCH_PROFILER_DIR="$PROFILE_DIR"
export VLLM_TORCH_PROFILER_DIR="$PROFILE_DIR"
#export SGLANG_ENABLE_PROFILE_CUDA_GRAPH=1
export SGLANG_PROFILE_WITH_STACK=True
export SGLANG_PROFILE_RECORD_SHAPE=True
export SGLANG_ENABLE_PROFILER_METADATA=1
export SGLANG_WARMUP_TIMEOUT=10000


SERVER_LOG=/workspace/server.log
PORT=${PORT:-8888}

python3 -m sglang.launch_server \
    --attention-backend aiter \
    --model-path $MODEL \
    --host=0.0.0.0 \
    --port=$PORT \
    --tensor-parallel-size=$TP \
    --trust-remote-code \
    --watchdog-timeout=10000 \
    --chunked-prefill-size=196608 \
    --mem-fraction-static=0.8 --disable-radix-cache \
    --num-continuous-decode-steps=4 \
    --max-prefill-tokens=196608 \
    --disable-cuda-graph > $SERVER_LOG 2>&1 &
    
    #--cuda-graph-max-bs $CONC

SERVER_PID=$!

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

# Shutdownserver
echo "Shutting down vLLM server (PID: $SERVER_PID)..."
kill $SERVER_PID
wait $SERVER_PID 2>/dev/null
echo "Server shutdown complete."

# After throughput, run evaluation only if RUN_EVAL is true
if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT" --concurrent-requests $CONC
    append_lm_eval_summary
fi
set +x