#!/usr/bin/env bash
# Sweep --max-concurrency levels against each deployed config (fp16, AWQ,
# GPTQ), one at a time — a 24GB card can't hold all three models at once.
# For each config: start the vLLM server, wait for /health, run
# benchmark_serving.py at each concurrency level (capturing GPU stats
# alongside), then stop the server before moving to the next config.
#
# Usage: ./benchmarks/run_matrix.sh
# Env overrides: CONCURRENCY_LEVELS, NUM_PROMPTS, PORT, DATASET, RESULTS_DIR
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="${DATASET:-$ROOT/datasets/ShareGPT_V3_unfiltered_cleaned_split.json}"
RESULTS_DIR="${RESULTS_DIR:-$ROOT/results}"
PORT="${PORT:-8000}"
read -ra CONCURRENCY_LEVELS <<< "${CONCURRENCY_LEVELS:-1 4 8 16 32}"
NUM_PROMPTS="${NUM_PROMPTS:-100}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-600}"

# name:model:quantization
# AWQ uses awq_marlin (not awq) — this AWQ checkpoint is Marlin-compatible,
# and vLLM 0.8.5 only auto-upgrades to the fast Marlin kernel when
# --quantization is left unset entirely; passing awq explicitly forces the
# slow unoptimized path (see context.md AWQ investigation for the ~3.5x
# throughput difference this makes).
CONFIGS=(
  "fp16:meta-llama/Llama-3.2-3B-Instruct:"
  "awq:casperhansen/llama-3.2-3b-instruct-awq:awq_marlin"
  "gptq:ModelCloud/Llama-3.2-3B-Instruct-gptqmodel-4bit-vortex-v3:gptq"
)

mkdir -p "$RESULTS_DIR"

if [[ ! -f "$DATASET" ]]; then
  echo "Dataset not found at $DATASET — run scripts/download_dataset.sh first" >&2
  exit 1
fi

wait_for_health() {
  local deadline=$((SECONDS + HEALTH_TIMEOUT))
  until curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; do
    if (( SECONDS > deadline )); then
      echo "vLLM server did not become healthy within ${HEALTH_TIMEOUT}s" >&2
      return 1
    fi
    sleep 5
  done
}

wait_for_gpu_free() {
  local deadline=$((SECONDS + 60))
  until [[ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 1000 ]]; do
    if (( SECONDS > deadline )); then
      echo "GPU memory did not clear within 60s, continuing anyway" >&2
      return 0
    fi
    sleep 2
  done
}

stop_server() {
  kill "$1" 2>/dev/null || true
  pkill -f "vllm serve" 2>/dev/null || true
  wait_for_gpu_free
}

for entry in "${CONFIGS[@]}"; do
  IFS=':' read -r name model quant <<< "$entry"
  echo "=== Starting $name ($model) ==="
  "$ROOT/serving/launch_vllm.sh" "$model" "$quant" "$PORT" > "$RESULTS_DIR/${name}.server.log" 2>&1 &
  SERVER_PID=$!

  if ! wait_for_health; then
    echo "Skipping $name — server failed to start (see ${name}.server.log)" >&2
    stop_server "$SERVER_PID"
    continue
  fi

  for c in "${CONCURRENCY_LEVELS[@]}"; do
    echo "--- $name concurrency=$c ---"
    "$ROOT/benchmarks/capture_gpu_stats.sh" "$RESULTS_DIR/${name}-c${c}.gpu.csv" &
    GPU_PID=$!

    python3 "$ROOT/benchmarks/vendor/benchmark_serving.py" \
      --backend vllm \
      --model "$model" \
      --port "$PORT" \
      --dataset-name sharegpt \
      --dataset-path "$DATASET" \
      --max-concurrency "$c" \
      --num-prompts "$NUM_PROMPTS" \
      --save-result \
      --result-dir "$RESULTS_DIR" \
      --result-filename "${name}-c${c}.json"

    kill "$GPU_PID" 2>/dev/null || true
    wait "$GPU_PID" 2>/dev/null || true
  done

  echo "=== Stopping $name ==="
  stop_server "$SERVER_PID"
done

echo "Matrix sweep complete. Results in $RESULTS_DIR"
