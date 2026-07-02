# Tasks

Checklist derived from `plan.md`. Check items off as completed; add new
items under the relevant phase as they surface (don't let this drift out of
sync with `plan.md`).

## Phase 0 — Deploy

- [x] Provision rented GPU (Lambda Labs or RunPod) — RunPod RTX A5000,
      pod `gp5vv3dchw1t2a`, done 2026-07-01
- [x] Install vLLM — pinned to 0.8.5 (see context.md for why)
- [x] Deploy Llama-3.2-3B fp16, confirm it loads — done 2026-07-01
- [x] Check HuggingFace Hub for existing pre-quantized Llama-3.2-3B
      checkpoints (AWQ and GPTQ) before writing any conversion scripts —
      done 2026-07-01, usable checkpoints found (see context.md), no
      conversion scripts needed
- [x] Deploy AWQ variant (`casperhansen/llama-3.2-3b-instruct-awq`) —
      done 2026-07-01, worked with default settings
- [x] Deploy GPTQ variant (`ModelCloud/Llama-3.2-3B-Instruct-gptqmodel-4bit-vortex-v3`) —
      done 2026-07-01, needed `--dtype float16` (see context.md); no need
      for the `shuyuej` fallback
- [x] Confirm OpenAI-compatible endpoint responds to a basic request —
      verified 2026-07-01 via /v1/chat/completions on the fp16 deploy

## Phase 1 — Benchmark Harness

- [x] Build load generator / request scheduler — vendored vLLM v0.8.5's
      `benchmarks/benchmark_serving.py` (see context.md) instead of writing
      one from scratch; wrapped with `benchmarks/run_matrix.sh`
- [x] Source a realistic workload for the load generator — ShareGPT
      (`scripts/download_dataset.sh`), supported natively by the vendored
      script (`--dataset-name sharegpt`)
- [x] Implement concurrency sweep support — `--max-concurrency`, swept via
      `benchmarks/run_matrix.sh` (default levels: 1/4/8/16/32)
- [x] Implement latency percentile collection (p50/p95/p99) — TTFT/TPOT/ITL
      mean/median/P99 built into the vendored script's output + saved JSON
- [x] Implement throughput collection (tokens/s, requests/s) — built into
      the vendored script's output + saved JSON
- [x] Implement GPU memory/utilization capture — `benchmarks/capture_gpu_stats.sh`
      (nvidia-smi polling, not part of the vendored script), run alongside
      each benchmark invocation by `run_matrix.sh`
- [x] **Standing task — repeat every time harness code changes:** run a
      single small request / small-batch smoke test before running the full
      benchmark matrix — `benchmarks/smoke_test.sh`, run 2026-07-01 before
      the full matrix
- [x] Full concurrency matrix run (fp16/AWQ/GPTQ x concurrency 1/4/8/16/32,
      100 prompts each) — completed 2026-07-01, 15 result files in
      `results/` (see context.md for headline numbers)
- [x] Investigate unexpectedly slow AWQ result — root-caused 2026-07-02 as a
      config bug (`--quantization awq` forcing the slow kernel instead of
      `awq_marlin`); fixed in `run_matrix.sh`/`launch_vllm.sh` (see
      context.md)
- [x] Re-run the full AWQ concurrency sweep (1/4/8/16/32) with the
      `awq_marlin` fix — done 2026-07-02, smoke test passed first,
      `results/awq-c*.json`/`.gpu.csv`/`awq.server.log` overwritten with
      corrected numbers (fp16/GPTQ results untouched); see context.md for
      the full table

## Phase 2 — Baseline Comparison

- [x] Write naive HF `transformers.generate()` baseline script —
      `baseline/hf_inference_server.py` (+ `baseline/launch_baseline.sh`
      launcher, `benchmarks/run_baseline_matrix.sh` sweep orchestrator),
      done 2026-07-02; implements the real OpenAI `/v1/completions` SSE
      contract directly (no adapter needed) so the vendored
      `benchmark_serving.py` client works unmodified; see context.md for
      design notes and the fairness audit (dtype/attention fixes made,
      no-batching/no-torch.compile documented as intentional)
- [x] Run baseline against same workloads as Phase 1 — done 2026-07-02,
      full concurrency sweep (1/4/8/16/32, 100 ShareGPT prompts each) via
      `benchmarks/run_baseline_matrix.sh`; smoke test passed first; sweep
      took ~40 min total (no need to cut prompt count/levels short); see
      context.md for the full table
- [x] Compare vLLM (fp16/AWQ/GPTQ) vs baseline results — done 2026-07-02,
      two framings in context.md: single-request (isolates kernel
      efficiency: vLLM 1.7-3.0x faster per-token) and concurrency=32
      (realistic serving: vLLM 22.6-45.5x higher throughput, baseline uses
      ~3.1-3.3x less peak GPU memory at every level)

## Phase 3 — Cost & Observability

- [x] Set up Prometheus scraping vLLM metrics endpoint — done 2026-07-02,
      confirmed vLLM 0.8.5's real `/metrics` output on a live pod (not
      assumed); `observability/prometheus.yml` scrapes it, verified `up`
      via `/api/v1/targets`; see context.md for the full metric list
- [x] Set up GPU exporter (DCGM or nvidia-smi exporter) — done 2026-07-02,
      chose `nvidia_gpu_exporter` (single static binary, no DCGM
      host-engine daemon needed); verified `/metrics` output live; see
      context.md
- [x] Build Grafana dashboards — done 2026-07-02,
      `observability/grafana/dashboards/vllm-benchmark.json` (9 panels:
      TTFT/TPOT/e2e latency, requests running/waiting, token throughput,
      KV-cache usage, GPU utilization, GPU memory, GPU power/temp),
      auto-provisioned via `observability/grafana/provisioning/`; verified
      end-to-end with live load through the RunPod proxy (screenshot
      confirmed all panels render real data) — hit and fixed a Grafana
      13.x CSRF "origin not allowed" bug along the way (see context.md)
- [x] Build $/1M-tokens cost model per configuration — done 2026-07-02,
      `benchmarks/cost_model.py`, purely retrospective against existing
      `results/*.json`; headline numbers in context.md (AWQ cheapest at
      $0.0343/1M output tokens, naive baseline ~27-45x more expensive)

## Phase 4 — Report

- [ ] Compile results tables/charts across all configurations
- [ ] Write portfolio report
