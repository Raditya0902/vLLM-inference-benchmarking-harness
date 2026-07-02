#!/usr/bin/env bash
# Sweep --max-concurrency levels against the naive HF baseline server
# (baseline/hf_inference_server.py), mirroring benchmarks/run_matrix.sh's
# structure for the vLLM configs — but calling baseline/launch_baseline.sh
# instead, no shared code with the vLLM path. Starts the baseline server,
# waits for /health, runs benchmark_serving.py at each concurrency level
# (capturing GPU stats alongside), then stops the server.
#
# Usage: ./benchmarks/run_baseline_matrix.sh <model-repo>
# Env overrides: CONCURRENCY_LEVELS, NUM_PROMPTS, PORT, DATASET, RESULTS_DIR
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${1:?Usage: run_baseline_matrix.sh <model-repo>}"
DATASET="${DATASET:-$ROOT/datasets/ShareGPT_V3_unfiltered_cleaned_split.json}"
RESULTS_DIR="${RESULTS_DIR:-$ROOT/results}"
PORT="${PORT:-8000}"
read -ra CONCURRENCY_LEVELS <<< "${CONCURRENCY_LEVELS:-1 4 8 16 32}"
NUM_PROMPTS="${NUM_PROMPTS:-100}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-600}"

mkdir -p "$RESULTS_DIR"

if [[ ! -f "$DATASET" ]]; then
  echo "Dataset not found at $DATASET — run scripts/download_dataset.sh first" >&2
  exit 1
fi

wait_for_health() {
  local deadline=$((SECONDS + HEALTH_TIMEOUT))
  until curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; do
    if (( SECONDS > deadline )); then
      echo "Baseline server did not become healthy within ${HEALTH_TIMEOUT}s" >&2
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
  pkill -f "hf_inference_server.py" 2>/dev/null || true
  wait_for_gpu_free
}

echo "=== Starting baseline ($MODEL) ==="
"$ROOT/baseline/launch_baseline.sh" "$MODEL" "$PORT" > "$RESULTS_DIR/baseline.server.log" 2>&1 &
SERVER_PID=$!

if ! wait_for_health; then
  echo "Skipping baseline — server failed to start (see baseline.server.log)" >&2
  stop_server "$SERVER_PID"
  exit 1
fi

for c in "${CONCURRENCY_LEVELS[@]}"; do
  echo "--- baseline concurrency=$c ---"
  "$ROOT/benchmarks/capture_gpu_stats.sh" "$RESULTS_DIR/baseline-c${c}.gpu.csv" &
  GPU_PID=$!

  python3 "$ROOT/benchmarks/vendor/benchmark_serving.py" \
    --backend openai \
    --model "$MODEL" \
    --port "$PORT" \
    --dataset-name sharegpt \
    --dataset-path "$DATASET" \
    --max-concurrency "$c" \
    --num-prompts "$NUM_PROMPTS" \
    --save-result \
    --result-dir "$RESULTS_DIR" \
    --result-filename "baseline-c${c}.json"

  kill "$GPU_PID" 2>/dev/null || true
  wait "$GPU_PID" 2>/dev/null || true
done

echo "=== Stopping baseline ==="
stop_server "$SERVER_PID"

echo "Baseline sweep complete. Results in $RESULTS_DIR"
