#!/usr/bin/env bash
# Launch a vLLM OpenAI-compatible server for one model configuration.
#
# Usage:
#   ./serving/launch_vllm.sh <model-repo> [quantization] [port]
#
# Examples:
#   ./serving/launch_vllm.sh meta-llama/Llama-3.2-3B-Instruct
#   ./serving/launch_vllm.sh casperhansen/llama-3.2-3b-instruct-awq awq
#   ./serving/launch_vllm.sh ModelCloud/Llama-3.2-3B-Instruct-gptqmodel-4bit-vortex-v3 gptq
set -euo pipefail

MODEL="${1:?Usage: launch_vllm.sh <model-repo> [quantization] [port]}"
QUANTIZATION="${2:-}"
PORT="${3:-8000}"

ARGS=(--port "$PORT")
if [[ -n "$QUANTIZATION" ]]; then
  ARGS+=(--quantization "$QUANTIZATION")
fi

vllm serve "$MODEL" "${ARGS[@]}"
