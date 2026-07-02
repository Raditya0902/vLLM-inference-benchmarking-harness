# Vendored from vllm-project/vllm

Source: https://github.com/vllm-project/vllm/tree/v0.8.5/benchmarks
Commit: ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d (tag v0.8.5)
License: Apache-2.0 (see SPDX header in each file)

Files copied unmodified:
- `benchmark_serving.py` — online serving load generator (async client against
  an OpenAI-compatible endpoint). Supports ShareGPT/Sonnet/Random/HF/BurstGPT
  datasets, `--max-concurrency` sweeps, and TTFT/TPOT/ITL/E2EL percentile +
  throughput reporting.
- `backend_request_func.py` — per-backend HTTP request implementations.
- `benchmark_dataset.py` — dataset loaders/samplers (ShareGPT etc).
- `benchmark_utils.py` — result formatting helpers.

Pinned to the v0.8.5 tag to match the `vllm==0.8.5` server version in root
`requirements.txt` (see CLAUDE.md Project Quirks) — later tags' benchmark
scripts may assume server-side flags/behavior this deployment doesn't have.

Do not hand-edit these files. Our own wrappers (dataset download, GPU-stats
capture, the fp16/AWQ/GPTQ x concurrency sweep) live one level up in
`benchmarks/`.
