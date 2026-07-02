# vLLM Inference Benchmarking Harness

## Project Overview

Benchmark LLM serving performance using vLLM (throughput, latency
percentiles, GPU memory) for Llama-3.2-3B and AWQ/GPTQ quantized variants,
compared against a naive HuggingFace `transformers.generate()` baseline.
Runs on a rented GPU (Lambda Labs or RunPod). Deliverable: benchmark results
+ written report, an OpenAI-compatible serving endpoint, and basic
Prometheus/Grafana observability. Solo, 3-week portfolio project.

Active plan, decisions, and task checklist live in
`dev/active/vllm-benchmarking/` — read `context.md` and `tasks.md` there at
the start of a session, especially after a context reset.

## Key Commands

- One-time env setup on a fresh GPU box: `./scripts/setup_env.sh`
- Start vLLM server: `./serving/launch_vllm.sh <model-repo> [awq_marlin|gptq] [port]`
  - fp16: `./serving/launch_vllm.sh meta-llama/Llama-3.2-3B-Instruct`
  - AWQ: `./serving/launch_vllm.sh casperhansen/llama-3.2-3b-instruct-awq awq_marlin`
  - GPTQ: `./serving/launch_vllm.sh ModelCloud/Llama-3.2-3B-Instruct-gptqmodel-4bit-vortex-v3 gptq`
- One-time dataset download (ShareGPT workload): `./scripts/download_dataset.sh`
- Run smoke test (single small request, before a full benchmark run):
  `./benchmarks/smoke_test.sh <model-repo> [port]` — needs a vLLM server
  already running on that port
- Run full benchmark matrix (fp16/AWQ/GPTQ × concurrency 1/4/8/16/32,
  starts/stops each server itself): `./benchmarks/run_matrix.sh`
  (env overrides: `CONCURRENCY_LEVELS`, `NUM_PROMPTS`, `PORT`, `DATASET`,
  `RESULTS_DIR`)
- Start naive HF baseline server: `./baseline/launch_baseline.sh <model-repo> [port]`
  (e.g. `./baseline/launch_baseline.sh meta-llama/Llama-3.2-3B-Instruct`) —
  exposes the same OpenAI-compatible `/v1/completions` contract vLLM does,
  so the same benchmark client works against it unmodified
- Run baseline smoke test: `./benchmarks/smoke_test.sh <model-repo> [port]`
  (same script as vLLM — needs the baseline server already running)
- Run full baseline concurrency sweep (concurrency 1/4/8/16/32,
  starts/stops the server itself): `./benchmarks/run_baseline_matrix.sh <model-repo>`
  (env overrides: `CONCURRENCY_LEVELS`, `NUM_PROMPTS`, `PORT`, `DATASET`,
  `RESULTS_DIR` — same convention as `run_matrix.sh`)
- Start Prometheus/Grafana stack: `TBD` (Phase 3)
- Generate report: `TBD` (Phase 4)

## Environment / GPU Details

- Provider: RunPod (on-demand, secure cloud)
- Instance type / GPU model: 1x RTX A5000, 24GB VRAM, $0.27/hr
- **No pod currently provisioned** — pods are deleted once idle after each
  work session to stop billing (Phase 0/1 pod `gp5vv3dchw1t2a` deleted
  2026-07-02; its Phase 2 replacement `die22er57siiee` deleted the same day
  once the baseline sweep finished). Re-provision with, e.g.:
  `runpodctl pod create --name vllm-benchmark --image
  "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04" --gpu-id
  "NVIDIA RTX A5000" --gpu-count 1 --cloud-type SECURE
  --container-disk-in-gb 40 --ports "22/tcp,8000/http"` (get the exact
  `--gpu-id` string via `runpodctl gpu list` if this one stops matching),
  then run `./scripts/setup_env.sh` on it — proven across Phase 0 and
  Phase 2. Update this section with the new pod id/IP/port once created;
  check status with `runpodctl pod get <pod-id>`, delete when idle with
  `runpodctl pod delete <pod-id>` (aliases: `rm`, `remove` — note the
  CLI has no `terminate` subcommand, despite the name; `delete` is correct).
- Driver / CUDA: NVIDIA driver 550.127.05, max CUDA 12.4 on the Phase 0/1
  pod (see Quirks below) — re-check on any new pod, a different host may
  have a newer driver.
- vLLM version: 0.8.5 (pinned, see `requirements.txt`)
- SSH: `ssh -i ~/.runpod/ssh/runpodctl-ssh-key root@<pod-ip> -p <pod-ssh-port>`
  — get current ip/port via `runpodctl pod get <pod-id> -o json` (they can
  change if the pod restarts)
- Repo lives on the pod at `/workspace/vllm-benchmarking-harness/` (synced via
  rsync, not git-cloned)
- Exposed ports: 22/tcp (SSH), 8000/http (vLLM OpenAI-compatible endpoint).
  Prometheus/Grafana ports: `TBD` (Phase 3)

## Project Quirks

- **Driver/CUDA mismatch on this RunPod host**: the host's NVIDIA driver
  (550.127.05) only supports up to CUDA 12.4, but a plain `pip install vllm`
  pulls the latest vllm (0.24.0 at time of writing) which requires
  torch==2.11.0, a torch build with no CUDA 12.4 wheel — it fails at
  `vllm serve` runtime with "NVIDIA driver on your system is too old", not at
  install time. Fixed by pinning `vllm==0.8.5` (requires `torch==2.6.0`, the
  newest torch still published for cu124) with
  `--extra-index-url https://download.pytorch.org/whl/cu124` in
  `requirements.txt`. If re-provisioning on a different host, check
  `nvidia-smi`'s reported CUDA Version first — a newer-driver host may not
  need this pin.
- Llama-3.2-3B-Instruct (fp16) is a gated HF repo — needs `HF_TOKEN` env var
  set on the pod (license must be accepted on huggingface.co first). The
  AWQ/GPTQ community re-uploads used in this project are not gated.
- Llama-3.2-3B-Instruct's default max seq len (131072) needs more KV-cache
  memory than fits on this 24GB card — `serving/launch_vllm.sh` caps
  `--max-model-len` to 8192 by default (override via `MAX_MODEL_LEN`).
- The GPTQ checkpoint (`ModelCloud/...gptqmodel-4bit-vortex-v3`) needs
  `--dtype float16` — this vllm version's GPTQ kernel doesn't support
  bfloat16. `serving/launch_vllm.sh` sets this automatically whenever
  `quantization=gptq` (override via `DTYPE`).
- Both `serving/launch_vllm.sh` and `benchmarks/capture_gpu_stats.sh` end
  with `exec <real command>`, not a bare final command — required so that
  backgrounding either script and later doing `kill "$!"` actually stops
  the real process (`vllm serve` / `nvidia-smi`) instead of orphaning it
  under a dead wrapper shell. Keep this pattern if you add similar
  start/stop-controlled scripts.
- `benchmarks/vendor/` holds vLLM v0.8.5's own `benchmark_serving.py` +
  helpers, vendored unmodified rather than reimplemented (see
  `benchmarks/vendor/README.md` and `dev/active/vllm-benchmarking/context.md`
  for why). Don't hand-edit those files — wrapper scripts live one level up
  in `benchmarks/`.
- **AWQ needs `--quantization awq_marlin`, not `awq`**: on this Ampere
  (A5000, compute cap 8.6) card, vLLM 0.8.5 detects that
  `casperhansen/llama-3.2-3b-instruct-awq` is Marlin-compatible but only
  auto-upgrades to the fast `awq_marlin` kernel when `--quantization` is
  left unset — passing `awq` explicitly (the Phase 1 default) forces the
  slow, "not fully optimized" fallback kernel and measured ~3.5x lower
  throughput. `serving/launch_vllm.sh` and `benchmarks/run_matrix.sh` now
  default AWQ to `awq_marlin`. See `dev/active/vllm-benchmarking/context.md`
  for the re-run numbers.
- **Fresh RunPod images may not have `rsync` preinstalled**: hit this
  re-provisioning for Phase 2 (`runpod/pytorch:2.4.0-...` base image) — the
  fix is just `apt-get update && apt-get install -y rsync` on the pod
  before the first `rsync` push from the local machine; not (yet) folded
  into `scripts/setup_env.sh` since it's a one-line manual fix so far.
- The naive HF baseline (`baseline/hf_inference_server.py`) needs
  `HF_TOKEN` for the same reason vLLM's fp16 config does (gated repo) —
  copy the local `~/.cache/huggingface/token` file to the same path on a
  fresh pod via `scp` rather than exporting it as an env var or writing it
  into a shell rc file (avoids the secret sitting in a command line or a
  persisted startup script).

## Active Work

See `dev/active/vllm-benchmarking/`:
- `plan.md` — phased build plan
- `context.md` — key decisions and architecture notes
- `tasks.md` — checklist tracking progress, including the standing
  smoke-test task before every full benchmark run
