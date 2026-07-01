#!/usr/bin/env bash
# Run once on a fresh rented GPU instance (Lambda Labs / RunPod) before
# deploying vLLM. Assumes a Debian/Ubuntu box with an NVIDIA GPU + driver
# already installed by the provider's base image.
set -euo pipefail

echo "== GPU check =="
nvidia-smi

echo "== Python env =="
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "== Done. Activate with: source .venv/bin/activate =="
