#!/usr/bin/env bash
# Launch Grafana OSS, provisioning the Prometheus datasource and the
# vLLM-benchmark dashboard from the committed configs under
# observability/grafana/. Runtime state (provisioning copy with an
# absolute path baked in, data, logs) lives outside the repo since it's
# host-specific / not something to version control.
#
# If accessing Grafana through RunPod's HTTP proxy (https://<pod-id>-<port>
# .proxy.runpod.net) rather than a direct/local connection, set
# PUBLIC_HOSTNAME to that host (no protocol) — Grafana's CSRF origin check
# otherwise rejects the browser's requests with "origin not allowed" since
# it doesn't recognize the proxy's Origin header. Not needed for plain SSH
# port-forwarding / localhost access.
#
# Usage: PUBLIC_HOSTNAME=<pod-id>-3000.proxy.runpod.net ./observability/launch_grafana.sh [port]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GRAFANA_HOME="${GRAFANA_HOME:-$HOME/observability-tools/grafana}"
RUNTIME_DIR="${GRAFANA_RUNTIME_DIR:-$HOME/observability-tools/grafana-runtime}"
PORT="${1:-3000}"

mkdir -p "$RUNTIME_DIR/provisioning/datasources" "$RUNTIME_DIR/provisioning/dashboards" \
  "$RUNTIME_DIR/data" "$RUNTIME_DIR/logs"

cp "$ROOT/observability/grafana/provisioning/datasources/prometheus.yml" \
  "$RUNTIME_DIR/provisioning/datasources/prometheus.yml"

# Dashboard provider file needs an absolute path to the dashboard JSON
# directory, which is host-specific — generated here rather than committed.
cat > "$RUNTIME_DIR/provisioning/dashboards/dashboard.yml" <<EOF
apiVersion: 1
providers:
  - name: vllm-benchmark
    folder: ""
    type: file
    options:
      path: $ROOT/observability/grafana/dashboards
EOF

export GF_PATHS_PROVISIONING="$RUNTIME_DIR/provisioning"
export GF_PATHS_DATA="$RUNTIME_DIR/data"
export GF_PATHS_LOGS="$RUNTIME_DIR/logs"
export GF_SERVER_HTTP_PORT="$PORT"

if [[ -n "${PUBLIC_HOSTNAME:-}" ]]; then
  export GF_SERVER_ROOT_URL="https://${PUBLIC_HOSTNAME}/"
  export GF_SECURITY_CSRF_TRUSTED_ORIGINS="$PUBLIC_HOSTNAME"
fi

exec "$GRAFANA_HOME/bin/grafana" server --homepath="$GRAFANA_HOME"
