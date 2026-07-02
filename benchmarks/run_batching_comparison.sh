#!/usr/bin/env bash
# Phase 1 addendum: continuous vs. static batching comparison, AWQ config
# only, concurrency 8/16/32 (see dev/active/vllm-benchmarking/context.md,
# 2026-07-01 addendum entry, for why AWQ/these levels and what "static"
# means here).
#
# vLLM 0.8.5 has no flag to disable continuous batching outright (its
# scheduler admits waiting requests into a freed slot every decode step,
# by design — see context.md). The "static" runs below use the closest
# configurable approximation: --max-num-seqs <concurrency> (caps the
# running batch to exactly the concurrency level) plus
# --enable-chunked-prefill false (removed if vLLM's V1 engine turns out to
# force chunked prefill back on — checked via server log after the first
# run, see check_chunked_prefill() below).
#
# "continuous" mode starts the server once (default config, unchanged from
# Phase 1) and sweeps all concurrency levels. "static" mode restarts the
# server per concurrency level, since --max-num-seqs must match the level
# under test.
#
# Usage: ./benchmarks/run_batching_comparison.sh
# Env overrides: CONCURRENCY_LEVELS, NUM_PROMPTS, PORT, DATASET, RESULTS_DIR
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="${DATASET:-$ROOT/datasets/ShareGPT_V3_unfiltered_cleaned_split.json}"
RESULTS_DIR="${RESULTS_DIR:-$ROOT/results}"
PORT="${PORT:-8000}"
read -ra CONCURRENCY_LEVELS <<< "${CONCURRENCY_LEVELS:-8 16 32}"
NUM_PROMPTS="${NUM_PROMPTS:-100}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-600}"

MODEL="casperhansen/llama-3.2-3b-instruct-awq"
QUANT="awq_marlin"

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

# Confirms from the server's own startup log whether chunked prefill was
# actually disabled, or force-re-enabled by vLLM's V1 engine (see
# context.md caveat) — printed, not asserted, so the sweep doesn't fail
# over documentation-only mismatch.
check_chunked_prefill() {
  local log="$1"
  # chunked_prefill_enabled= is the *resolved* SchedulerConfig field logged
  # at startup; enable_chunked_prefill= (no "_enabled") is just the raw CLI
  # arg echo pre-resolution and doesn't reflect what the V1 engine actually
  # did with it (confirmed empirically: V1 forces chunked_prefill_enabled
  # =True regardless of the requested value — see context.md).
  local match
  match="$(grep -io "chunked_prefill_enabled=[A-Za-z]*" "$log" 2>/dev/null | head -1)"
  if [[ -z "$match" ]]; then
    echo "NOTE: could not find chunked_prefill_enabled in server log ($log) — check manually." >&2
  else
    echo "Server log reports: $match (requested: false)" >&2
  fi
}

run_benchmark() {
  local tag="$1" c="$2"
  echo "--- $tag concurrency=$c ---"
  "$ROOT/benchmarks/capture_gpu_stats.sh" "$RESULTS_DIR/${tag}-c${c}.gpu.csv" &
  local gpu_pid=$!

  python3 "$ROOT/benchmarks/vendor/benchmark_serving.py" \
    --backend vllm \
    --model "$MODEL" \
    --port "$PORT" \
    --dataset-name sharegpt \
    --dataset-path "$DATASET" \
    --max-concurrency "$c" \
    --num-prompts "$NUM_PROMPTS" \
    --save-result \
    --result-dir "$RESULTS_DIR" \
    --result-filename "${tag}-c${c}.json"

  kill "$gpu_pid" 2>/dev/null || true
  wait "$gpu_pid" 2>/dev/null || true
}

echo "=== Starting AWQ (continuous batching, default) ==="
"$ROOT/serving/launch_vllm.sh" "$MODEL" "$QUANT" "$PORT" \
  > "$RESULTS_DIR/awq-continuous.server.log" 2>&1 &
SERVER_PID=$!

if ! wait_for_health; then
  echo "Continuous-batching server failed to start (see awq-continuous.server.log)" >&2
  stop_server "$SERVER_PID"
  exit 1
fi

for c in "${CONCURRENCY_LEVELS[@]}"; do
  run_benchmark "awq-continuous" "$c"
done

echo "=== Stopping continuous-batching server ==="
stop_server "$SERVER_PID"

for c in "${CONCURRENCY_LEVELS[@]}"; do
  echo "=== Starting AWQ (static-like: max-num-seqs=$c, chunked-prefill off) ==="
  EXTRA_ARGS="--max-num-seqs $c --no-enable-chunked-prefill" \
    "$ROOT/serving/launch_vllm.sh" "$MODEL" "$QUANT" "$PORT" \
    > "$RESULTS_DIR/awq-static-c${c}.server.log" 2>&1 &
  SERVER_PID=$!

  if ! wait_for_health; then
    echo "Skipping static c=$c — server failed to start (see awq-static-c${c}.server.log)" >&2
    stop_server "$SERVER_PID"
    continue
  fi

  check_chunked_prefill "$RESULTS_DIR/awq-static-c${c}.server.log"
  run_benchmark "awq-static" "$c"

  echo "=== Stopping static-like server (c=$c) ==="
  stop_server "$SERVER_PID"
done

echo "Batching comparison sweep complete. Results in $RESULTS_DIR"
