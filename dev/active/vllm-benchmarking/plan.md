# vLLM Benchmarking Harness — Build Plan

Status: Phase 0, Phase 1, and Phase 2 complete (2026-07-02). Phase 3 (Cost &
Observability) is next.

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

## Phase 2 — Baseline Comparison ✅ done

Wrote `baseline/hf_inference_server.py`, a from-scratch FastAPI server
implementing the real OpenAI `/v1/completions` SSE contract directly (no
adapter needed — the vendored `benchmark_serving.py` client works
unmodified), plus `baseline/launch_baseline.sh` and
`benchmarks/run_baseline_matrix.sh`; none of it shares code with the vLLM
path. "Naive" is scoped to no continuous batching (one `generate()` call
in flight at a time) — a fairness audit (see `context.md`) fixed two
one-line silent slow-path traps (fp32 default dtype, eager attention
default) but left no-batching and no-`torch.compile` as documented,
intentional caveats. Full concurrency sweep (1/4/8/16/32, same 100
ShareGPT prompts as Phase 1) completed 2026-07-02 — see `context.md` for
the full comparison: vLLM is 1.7-3.0x faster per-token even at
concurrency=1 (kernel efficiency alone), and 22.6-45.5x higher throughput
at concurrency=32 (kernel efficiency + continuous batching combined); the
naive baseline uses ~3.1-3.3x less peak GPU memory throughout.

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
