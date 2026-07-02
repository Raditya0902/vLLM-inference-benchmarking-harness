#!/usr/bin/env bash
# Launch Prometheus scraping vLLM's /metrics + the GPU exporter, using the
# committed config at observability/prometheus.yml. Assumes
# install_observability_stack.sh has already installed the binary to
# ~/observability-tools/prometheus.
#
# Usage: ./observability/launch_prometheus.sh [port]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INSTALL_DIR="${INSTALL_DIR:-$HOME/observability-tools}"
DATA_DIR="${PROMETHEUS_DATA_DIR:-$HOME/observability-tools/prometheus-data}"
PORT="${1:-9090}"

mkdir -p "$DATA_DIR"

# exec so callers backgrounding this script can kill the real process via
# `$!` — same pattern as serving/launch_vllm.sh.
exec "$INSTALL_DIR/prometheus/prometheus" \
  --config.file="$ROOT/observability/prometheus.yml" \
  --storage.tsdb.path="$DATA_DIR" \
  --web.listen-address="0.0.0.0:${PORT}"
