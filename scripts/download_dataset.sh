#!/usr/bin/env bash
# Download the ShareGPT conversation dataset used as the realistic workload
# for benchmarks/vendor/benchmark_serving.py (--dataset-name sharegpt).
set -euo pipefail

DATASET_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/datasets"
DATASET_FILE="$DATASET_DIR/ShareGPT_V3_unfiltered_cleaned_split.json"
DATASET_URL="https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"

mkdir -p "$DATASET_DIR"

if [[ -f "$DATASET_FILE" ]]; then
  echo "Already downloaded: $DATASET_FILE"
  exit 0
fi

wget -O "$DATASET_FILE" "$DATASET_URL"
echo "Downloaded to $DATASET_FILE"
