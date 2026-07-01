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
- Start vLLM server: `./serving/launch_vllm.sh <model-repo> [awq|gptq] [port]`
  - fp16: `./serving/launch_vllm.sh meta-llama/Llama-3.2-3B-Instruct`
  - AWQ: `./serving/launch_vllm.sh casperhansen/llama-3.2-3b-instruct-awq awq`
  - GPTQ: `./serving/launch_vllm.sh ModelCloud/Llama-3.2-3B-Instruct-gptqmodel-4bit-vortex-v3 gptq`
- Run baseline (HF) inference: `TBD` (Phase 2)
- Run benchmark suite: `TBD` (Phase 1)
- Run smoke test (single small request, before a full benchmark run): `TBD` (Phase 1)
- Start Prometheus/Grafana stack: `TBD` (Phase 3)
- Generate report: `TBD` (Phase 4)

## Environment / GPU Details

> TBD — filled in once a GPU is provisioned.

- Provider: `TBD` (Lambda Labs / RunPod)
- Instance type / GPU model: `TBD`
- CUDA / driver version: `TBD`
- vLLM version: `TBD`
- SSH / connection notes: `TBD`
- Exposed ports (serving endpoint, Prometheus, Grafana): `TBD`

## Project Quirks

> TBD — filled in as gotchas are discovered. Examples of the kind of thing
> that belongs here: vLLM/quantization library version pinning issues,
> AWQ vs GPTQ kernel support differences, thermal throttling behavior on
> rented GPUs, endpoint auth quirks.

- (none yet)

## Active Work

See `dev/active/vllm-benchmarking/`:
- `plan.md` — phased build plan
- `context.md` — key decisions and architecture notes
- `tasks.md` — checklist tracking progress, including the standing
  smoke-test task before every full benchmark run
