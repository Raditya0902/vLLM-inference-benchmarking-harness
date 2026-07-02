#!/usr/bin/env bash
# One-time install of Prometheus, Grafana OSS, and nvidia_gpu_exporter as
# standalone binaries — no Docker (this pod's container image doesn't
# support nested Docker). Installs to ~/observability-tools, outside the
# git repo, since these are large third-party binaries, not project code.
#
# Usage: ./observability/install_observability_stack.sh
set -euo pipefail

PROMETHEUS_VERSION="${PROMETHEUS_VERSION:-3.13.0}"
GRAFANA_VERSION="${GRAFANA_VERSION:-13.1.0}"
GPU_EXPORTER_VERSION="${GPU_EXPORTER_VERSION:-1.7.0}"
INSTALL_DIR="${INSTALL_DIR:-$HOME/observability-tools}"

mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

if [[ ! -x "prometheus-${PROMETHEUS_VERSION}.linux-amd64/prometheus" ]]; then
  echo "=== Installing Prometheus ${PROMETHEUS_VERSION} ==="
  curl -sL -o prometheus.tar.gz \
    "https://github.com/prometheus/prometheus/releases/download/v${PROMETHEUS_VERSION}/prometheus-${PROMETHEUS_VERSION}.linux-amd64.tar.gz"
  tar xzf prometheus.tar.gz
  rm prometheus.tar.gz
fi
ln -sfn "prometheus-${PROMETHEUS_VERSION}.linux-amd64" prometheus

if [[ ! -x "grafana-${GRAFANA_VERSION}/bin/grafana" ]]; then
  echo "=== Installing Grafana ${GRAFANA_VERSION} ==="
  curl -sL -o grafana.tar.gz \
    "https://dl.grafana.com/oss/release/grafana-${GRAFANA_VERSION}.linux-amd64.tar.gz"
  tar xzf grafana.tar.gz
  rm grafana.tar.gz
fi
ln -sfn "grafana-${GRAFANA_VERSION}" grafana

if [[ ! -x "nvidia_gpu_exporter-${GPU_EXPORTER_VERSION}/nvidia_gpu_exporter" ]]; then
  echo "=== Installing nvidia_gpu_exporter ${GPU_EXPORTER_VERSION} ==="
  mkdir -p "nvidia_gpu_exporter-${GPU_EXPORTER_VERSION}"
  curl -sL -o gpu_exporter.tar.gz \
    "https://github.com/utkuozdemir/nvidia_gpu_exporter/releases/download/v${GPU_EXPORTER_VERSION}/nvidia_gpu_exporter_${GPU_EXPORTER_VERSION}_linux_x86_64.tar.gz"
  tar xzf gpu_exporter.tar.gz -C "nvidia_gpu_exporter-${GPU_EXPORTER_VERSION}"
  rm gpu_exporter.tar.gz
fi
ln -sfn "nvidia_gpu_exporter-${GPU_EXPORTER_VERSION}" nvidia_gpu_exporter

echo "Installed to $INSTALL_DIR:"
ls -la "$INSTALL_DIR"
