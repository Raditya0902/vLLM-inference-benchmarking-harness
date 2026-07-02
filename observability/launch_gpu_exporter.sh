#!/usr/bin/env bash
# Launch nvidia_gpu_exporter (wraps nvidia-smi, exposes Prometheus metrics
# on :9835 by default) — chosen over DCGM for this project since it's a
# single static binary with no separate host-engine daemon to install.
#
# Usage: ./observability/launch_gpu_exporter.sh [port]
set -euo pipefail

INSTALL_DIR="${INSTALL_DIR:-$HOME/observability-tools}"
PORT="${1:-9835}"

exec "$INSTALL_DIR/nvidia_gpu_exporter/nvidia_gpu_exporter" \
  --web.listen-address=":${PORT}"
