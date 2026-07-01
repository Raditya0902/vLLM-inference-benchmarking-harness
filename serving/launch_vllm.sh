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
# Llama-3.2-3B-Instruct's default max seq len (131072) needs more KV-cache
# memory than a 24GB GPU has free after loading weights. Our benchmark
# workload is realistic (ShareGPT-scale) prompts, not 131K-token ones, and
# capping this the same way across fp16/AWQ/GPTQ keeps the comparison fair.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
# The GPTQ kernel in this vllm version only supports float16 (not the
# bfloat16 vLLM auto-detects from the base model config), so force it
# unless the caller overrides DTYPE explicitly.
if [[ "$QUANTIZATION" == "gptq" ]]; then
  DTYPE="${DTYPE:-float16}"
else
  DTYPE="${DTYPE:-}"
fi

ARGS=(--port "$PORT" --max-model-len "$MAX_MODEL_LEN")
if [[ -n "$QUANTIZATION" ]]; then
  ARGS+=(--quantization "$QUANTIZATION")
fi
if [[ -n "$DTYPE" ]]; then
  ARGS+=(--dtype "$DTYPE")
fi

vllm serve "$MODEL" "${ARGS[@]}"
