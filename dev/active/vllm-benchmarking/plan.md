# vLLM Benchmarking Harness — Build Plan

Status: Phase 0 and Phase 1 complete (2026-07-01). Phase 2 (Baseline
Comparison) is next.

## Phase 0 — Deploy ✅ done

- Provision a rented GPU (Lambda Labs or RunPod). → RunPod RTX A5000.
- Install vLLM, deploy Llama-3.2-3B in fp16.
- For AWQ/GPTQ variants: **check HuggingFace Hub first** for existing
  pre-quantized Llama-3.2-3B checkpoints. Only write model-quantization
  scripts (in `models/`) if no usable pre-quantized version exists.
  → Usable checkpoints found for both; no conversion scripts needed.
- Confirm the OpenAI-compatible endpoint responds to a basic request.

All three configurations (fp16, AWQ, GPTQ) deployed and verified via
`/v1/chat/completions`. Several environment issues came up and were fixed
along the way (CUDA/driver pin, transformers pin, context-length cap, GPTQ
dtype) — see `context.md` for details and `CLAUDE.md` Project Quirks for the
durable reference.

## Phase 1 — Benchmark Harness ✅ done

Vendored vLLM v0.8.5's own `benchmarks/benchmark_serving.py` (+ its 3
helper modules) into `benchmarks/vendor/` instead of writing a load
generator from scratch — it already covered ShareGPT sampling,
`--max-concurrency` sweeps, and TTFT/TPOT/ITL percentile + throughput
reporting. Added `benchmarks/capture_gpu_stats.sh` (nvidia-smi polling,
the one thing missing) and `benchmarks/run_matrix.sh` (orchestrates the
fp16/AWQ/GPTQ × concurrency sweep, one server config at a time). First
full matrix run completed 2026-07-01 — see `context.md` for headline
numbers and `results/` for raw output. Details, the vendoring rationale,
and an `exec`-related process-cleanup fix are in `context.md`.

## Phase 2 — Baseline Comparison

- Write a naive HuggingFace `transformers.generate()` inference script
  (`baseline/`), deliberately separate from the vLLM serving code for a
  clean A/B.
- Run the same workloads (from Phase 1) against the baseline.
- Compare vLLM (fp16/AWQ/GPTQ) vs baseline on throughput, latency, memory.

## Phase 3 — Cost & Observability

- Set up Prometheus scraping vLLM's metrics endpoint + a GPU exporter
  (e.g. DCGM or nvidia-smi exporter).
- Build Grafana dashboards for live monitoring during benchmark runs.
- Build a cost model: $/1M tokens, from GPU rental rate ÷ measured
  throughput, for each configuration (fp16 vLLM, AWQ, GPTQ, HF baseline).

## Phase 4 — Report

- Compile findings: throughput/latency/memory/cost tables and charts across
  all configurations.
- Write up as a portfolio piece targeting AI infrastructure roles.

## Explicitly out of scope for now

- No implementation code until this plan is reviewed and approved.
- No multi-GPU / multi-node serving (single rented GPU only).
- No model fine-tuning — benchmarking inference only.
