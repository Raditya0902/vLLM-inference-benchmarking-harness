#!/usr/bin/env bash
# Standing pre-flight check (see tasks.md Phase 1): run a couple of
# requests through benchmark_serving.py against an already-running vLLM
# server before kicking off the full concurrency matrix. Run this after any
# change to benchmarks/ code, not just once.
#
# Usage: ./benchmarks/smoke_test.sh <model-repo> [port]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${1:?Usage: smoke_test.sh <model-repo> [port]}"
PORT="${2:-8000}"
DATASET="${DATASET:-$ROOT/datasets/ShareGPT_V3_unfiltered_cleaned_split.json}"

if [[ ! -f "$DATASET" ]]; then
  echo "Dataset not found at $DATASET — run scripts/download_dataset.sh first" >&2
  exit 1
fi

python3 "$ROOT/benchmarks/vendor/benchmark_serving.py" \
  --backend vllm \
  --model "$MODEL" \
  --port "$PORT" \
  --dataset-name sharegpt \
  --dataset-path "$DATASET" \
  --max-concurrency 1 \
  --num-prompts 2
