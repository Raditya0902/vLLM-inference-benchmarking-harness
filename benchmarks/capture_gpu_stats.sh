#!/usr/bin/env bash
# Poll nvidia-smi at a fixed interval and append CSV rows to a file, until
# killed. Meant to be started in the background around a benchmark run:
#
#   ./benchmarks/capture_gpu_stats.sh results/fp16-c8.gpu.csv &
#   GPU_CAPTURE_PID=$!
#   python benchmarks/vendor/benchmark_serving.py ...
#   kill "$GPU_CAPTURE_PID"
set -euo pipefail

OUT_FILE="${1:?Usage: capture_gpu_stats.sh <output-csv> [interval-seconds]}"
INTERVAL="${2:-1}"

# exec replaces this script's process with nvidia-smi so `kill "$!"` in the
# caller (capturing this script's own PID when backgrounded) actually signals
# nvidia-smi directly, instead of orphaning it as an unkillable child.
exec nvidia-smi \
  --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
  --format=csv \
  -l "$INTERVAL" \
  > "$OUT_FILE"
