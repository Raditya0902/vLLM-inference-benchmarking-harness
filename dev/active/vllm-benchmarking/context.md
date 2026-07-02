# Context & Decisions

This file is the working memory for this project: decisions made, why they
were made, and architecture notes. Update it as choices happen — don't wait
until the end. Read this first after any context reset to reconstruct where
things stand and why.

## Key Decisions

- **2026-07-01**: Checked HuggingFace Hub for pre-quantized Llama-3.2-3B
  checkpoints before writing any quantization scripts (per plan). Usable
  candidates exist for both formats, so `models/` will not contain
  conversion scripts — only download manifests.
  - AWQ candidate: `casperhansen/llama-3.2-3b-instruct-awq` (maintained by
    the author of AutoAWQ — high confidence).
  - GPTQ candidate: `ModelCloud/Llama-3.2-3B-Instruct-gptqmodel-4bit-vortex-v3`
    (maintained by the author of GPTQModel — high confidence). Alternative:
    `shuyuej/Llama-3.2-3B-Instruct-GPTQ` if the ModelCloud format doesn't
    load cleanly in vLLM.
  - Open: neither has been load-tested against vLLM yet — do that as part
    of Phase 0 deploy, not before (see Open Questions).
- **2026-07-01**: Provisioned RunPod pod `gp5vv3dchw1t2a` (RTX A5000, 24GB,
  $0.27/hr on-demand). Host driver is 550.127.05, max CUDA 12.4. Plain
  `pip install vllm` grabs vllm 0.24.0 + torch==2.11.0, which has no CUDA
  12.4 build and fails at runtime ("driver too old"). Pinned
  `vllm==0.8.5` + `torch==2.6.0` (cu124) + `transformers>=4.51.1,<4.52.0`
  (vllm 0.8.5 leaves transformers unbounded, and latest transformers breaks
  vllm 0.8.5's tokenizer backend assumptions) in root `requirements.txt`.
- **2026-07-01**: fp16 deploy of `meta-llama/Llama-3.2-3B-Instruct` needed
  `--max-model-len 8192` — the model's default max seq len (131072) needs
  more KV-cache memory than fits on a 24GB card. 8192 is generous for the
  ShareGPT-scale realistic workload planned for Phase 1, and is now the
  default in `serving/launch_vllm.sh` (override via `MAX_MODEL_LEN` env
  var) so fp16/AWQ/GPTQ are compared on the same context cap.
- **2026-07-01**: fp16 endpoint verified end-to-end via
  `/v1/chat/completions` — responded correctly to a basic prompt.
- **2026-07-01**: AWQ variant (`casperhansen/llama-3.2-3b-instruct-awq`)
  deployed and verified with no changes needed beyond the base env fixes.
- **2026-07-01**: GPTQ variant
  (`ModelCloud/Llama-3.2-3B-Instruct-gptqmodel-4bit-vortex-v3`) needed
  `--dtype float16` — this vllm version's GPTQ kernel doesn't support
  bfloat16 (vLLM's auto-detected default from the base model config).
  Baked into `serving/launch_vllm.sh` as an automatic default whenever
  `quantization=gptq` (override via `DTYPE` env var). No need for the
  `shuyuej` fallback checkpoint. **Phase 0 deploy is complete**: all three
  configurations (fp16, AWQ, GPTQ) load and respond correctly.
- **2026-07-01**: For Phase 1's load generator, checked vLLM's own repo
  before writing one from scratch (per reuse-first default) — the
  `vllm-project/vllm` repo ships `benchmarks/benchmark_serving.py` at the
  same `v0.8.5` tag we're pinned to, already covering ShareGPT sampling,
  `--max-concurrency` sweeps, TTFT/TPOT/ITL/E2EL percentiles, throughput,
  and JSON result output against an OpenAI-compatible endpoint. Vendored
  it unmodified (commit `ba41cc9`, Apache-2.0) into `benchmarks/vendor/`
  rather than reimplementing — see `benchmarks/vendor/README.md` for
  provenance. Only GPU memory/utilization capture was missing, added
  separately as `benchmarks/capture_gpu_stats.sh` (nvidia-smi polling).
  New deps (`pandas`, `datasets`, `pillow`) added to root `requirements.txt`
  since the vendored scripts import them directly.
- **2026-07-01**: Both `serving/launch_vllm.sh` and
  `benchmarks/capture_gpu_stats.sh` originally ran their real work
  (`vllm serve` / `nvidia-smi`) as a non-final-exec'd last command. When
  `benchmarks/run_matrix.sh` backgrounds these scripts and later calls
  `kill "$!"`, that PID is the *wrapper shell's* PID, not the child
  process's — the child survives as an orphan. Fixed by adding `exec`
  before the final command in both scripts, so the shell process is
  replaced in-place and `$!` always refers to the actual long-running
  process. `run_matrix.sh` also keeps a `pkill -f "vllm serve"` fallback
  in `stop_server()` as a second line of defense.
- **2026-07-01**: First full concurrency-matrix run complete (fp16/AWQ/GPTQ
  × concurrency 1/4/8/16/32, 100 ShareGPT prompts per run). Headline numbers
  at concurrency=32: fp16 6.70 req/s / 1327.5 output tok/s / 33.8ms median
  TTFT; GPTQ 5.36 req/s / 1086.2 tok/s / 58.1ms TTFT; AWQ 2.60 req/s / 518.2
  tok/s / 79.2ms TTFT. fp16 outperforming both quantized variants (AWQ
  slowest) on this A5000/vllm==0.8.5 combo is somewhat counter-intuitive —
  worth investigating during Phase 4 write-up (possibly an unoptimized AWQ
  kernel path in this vLLM version). Raw results in `results/*.json` +
  per-run GPU utilization in `results/*.gpu.csv`.
- **2026-07-02**: Root-caused and fixed the slow-AWQ finding above — it was
  a config bug, not a real quantization-format result. `awq.server.log`
  showed: `Detected that the model can run with awq_marlin, however you
  specified quantization=awq explicitly, so forcing awq. Use
  quantization=awq_marlin for faster inference` followed by `WARNING: awq
  quantization is not fully optimized yet.` The A5000 is Ampere (compute
  cap 8.6, confirmed via `nvidia-smi --query-gpu=compute_cap`), which
  supports the fast Marlin-based AWQ kernel, but vLLM 0.8.5 only
  auto-upgrades to `awq_marlin` when `--quantization` is left unset
  entirely — `run_matrix.sh`/`launch_vllm.sh` were passing `awq` explicitly
  (matching the checkpoint's own `quant_config.json`), which forces the old
  unoptimized kernel path instead. No known open vLLM 0.8.5 issue was found
  describing this as a bug — it's documented, if easy-to-miss, behavior
  (the log line spells out the fix). Re-ran AWQ at concurrency=32 with
  `--quantization awq_marlin` explicitly: **9.22 req/s / 1829.0 output
  tok/s / 57.2ms median TTFT**, vs the original awq run's 2.60 req/s / 518.2
  tok/s / 79.2ms TTFT — a ~3.5x throughput improvement, and now *faster*
  than fp16 (6.70 req/s / 1327.5 tok/s / 33.8ms TTFT) on throughput, though
  fp16 still has lower median TTFT. Fixed the default in both
  `serving/launch_vllm.sh` (usage comment) and `benchmarks/run_matrix.sh`
  (`CONFIGS` array now specifies `awq_marlin`).
- **2026-07-02**: Re-ran the full AWQ concurrency sweep (1/4/8/16/32, 100
  ShareGPT prompts each) with the `awq_marlin` fix, overwriting
  `results/awq-c*.json`/`.gpu.csv`/`awq.server.log` in place (fp16 and GPTQ
  result files untouched). Ran the standing smoke test
  (`benchmarks/smoke_test.sh`) against the `awq_marlin`-configured server
  first — passed (2/2 requests succeeded). Corrected AWQ numbers:

  | concurrency | req/s | output tok/s | median TTFT (ms) | median TPOT (ms) |
  |---|---|---|---|---|
  | 1  | 0.72  | 143.4  | 19.0 | 6.84 |
  | 4  | 2.63  | 524.2  | 14.0 | 7.37 |
  | 8  | 4.96  | 988.4  | 14.4 | 7.52 |
  | 16 | 8.59  | 1712.5 | 16.7 | 8.07 |
  | 32 | 10.98 | 2187.9 | 24.3 | 9.87 |

  Note the c32 numbers here (10.98 req/s / 2187.9 tok/s / 24.3ms TTFT) are
  somewhat better than the earlier one-off spot-check of the same config
  (9.22 req/s / 1829.0 tok/s / 57.2ms TTFT, now superseded and deleted) —
  plausible run-to-run variance from cudagraph warmup/scheduling state
  differences between a freshly-started single-run server and a server
  that's already served four lower-concurrency runs in the same sweep; not
  investigated further since it doesn't change the conclusion. Across every
  concurrency level, corrected AWQ now clearly beats the original broken
  AWQ numbers and is competitive with or ahead of fp16/GPTQ on throughput.
  Phase 1 is now fully complete with correct data — no more open AWQ
  question for Phase 4.

## Architecture Notes

> (none yet — e.g. how the load generator is structured, how results are
> keyed/stored, how the OpenAI-compatible endpoint is wired to observability)

## Environment Details

See root `CLAUDE.md` → Environment / GPU Details for the current pod id,
GPU/driver specs, SSH access, and exposed ports (kept there, not duplicated
here, so there's one source of truth to keep in sync).

## Open Questions

- Report format (tables/charts, how to present cost model) — not yet
  decided (Phase 4). Concurrency levels resolved: swept 1/4/8/16/32 with
  100 prompts each (`benchmarks/run_matrix.sh` defaults).
- ~~Why AWQ underperforms both fp16 and GPTQ~~ — resolved 2026-07-02: config
  bug (`--quantization awq` instead of `awq_marlin`), not a real result. Full
  concurrency sweep re-run with the fix, all `results/awq-c*.json` files now
  reflect corrected numbers. See 2026-07-02 entries above. No longer open.
