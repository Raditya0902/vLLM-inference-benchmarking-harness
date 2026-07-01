# Tasks

Checklist derived from `plan.md`. Check items off as completed; add new
items under the relevant phase as they surface (don't let this drift out of
sync with `plan.md`).

## Phase 0 — Deploy

- [ ] Provision rented GPU (Lambda Labs or RunPod)
- [ ] Install vLLM
- [ ] Deploy Llama-3.2-3B fp16, confirm it loads
- [ ] Check HuggingFace Hub for existing pre-quantized Llama-3.2-3B
      checkpoints (AWQ and GPTQ) before writing any conversion scripts
- [ ] If no usable pre-quantized checkpoint exists: write quantization
      scripts in `models/`
- [ ] Deploy AWQ variant
- [ ] Deploy GPTQ variant
- [ ] Confirm OpenAI-compatible endpoint responds to a basic request

## Phase 1 — Benchmark Harness

- [ ] Build load generator / request scheduler
- [ ] Source a realistic workload for the load generator (e.g. a sampled
      ShareGPT conversation dataset) instead of synthetic fixed-length
      prompts
- [ ] Implement concurrency sweep support
- [ ] Implement latency percentile collection (p50/p95/p99)
- [ ] Implement throughput collection (tokens/s, requests/s)
- [ ] Implement GPU memory/utilization capture
- [ ] **Standing task — repeat every time harness code changes:** run a
      single small request / small-batch smoke test before running the full
      benchmark matrix

## Phase 2 — Baseline Comparison

- [ ] Write naive HF `transformers.generate()` baseline script
- [ ] Run baseline against same workloads as Phase 1
- [ ] Compare vLLM (fp16/AWQ/GPTQ) vs baseline results

## Phase 3 — Cost & Observability

- [ ] Set up Prometheus scraping vLLM metrics endpoint
- [ ] Set up GPU exporter (DCGM or nvidia-smi exporter)
- [ ] Build Grafana dashboards
- [ ] Build $/1M-tokens cost model per configuration

## Phase 4 — Report

- [ ] Compile results tables/charts across all configurations
- [ ] Write portfolio report
