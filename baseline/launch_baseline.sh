#!/usr/bin/env bash
# Launch the naive HF transformers.generate() baseline server.
#
# Usage:
#   ./baseline/launch_baseline.sh <model-repo> [port]
#
# Example:
#   ./baseline/launch_baseline.sh meta-llama/Llama-3.2-3B-Instruct
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${1:?Usage: launch_baseline.sh <model-repo> [port]}"
PORT="${2:-8000}"

# exec replaces this script's process with the python server so callers
# that background this script and capture `$!` can kill the actual server
# process — same reasoning as serving/launch_vllm.sh, but no code shared
# with it (see baseline/hf_inference_server.py docstring).
exec python3 "$ROOT/baseline/hf_inference_server.py" "$MODEL" "$PORT"
