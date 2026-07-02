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
- **2026-07-02**: Deleted the Phase 0/1 pod (`gp5vv3dchw1t2a`) once idle to
  stop billing between work sessions — Phase 2 script-writing doesn't need
  a GPU. Will re-provision (new pod, `scripts/setup_env.sh`) only when it's
  time to actually deploy/run something. Also: `runpodctl pod terminate`
  (as documented in `CLAUDE.md` up to this point) isn't a real subcommand —
  the correct one is `runpodctl pod delete <id>` (aliases `rm`/`remove`).
  Fixed in `CLAUDE.md`.
- **2026-07-02**: Designed `baseline/hf_inference_server.py` (Phase 2) by
  first reading `benchmarks/vendor/backend_request_func.py`'s
  `async_request_openai_completions` (what the vendored client's
  `--backend vllm`/`openai` maps to, and the default `--endpoint
  /v1/completions`) to get the exact wire contract: `POST
  /v1/completions` with `{model, prompt, max_tokens, temperature: 0.0,
  stream: true, stream_options: {include_usage: true}}`, response as
  OpenAI-style SSE (`data: {"choices":[{"text": "..."}]}` per chunk,
  final `data: {"usage": {"completion_tokens": N}}`, terminated by `data:
  [DONE]`). This is a real, standard OpenAI-compatible contract, not
  something baseline-specific — so the baseline implements it directly
  (FastAPI + `TextIteratorStreamer` off a background thread bridged into
  an `asyncio.Queue`) rather than needing a thin adapter in front of some
  other API shape. No code is shared with `serving/launch_vllm.sh` or the
  vLLM path — separate script, separate launcher
  (`baseline/launch_baseline.sh`), separate sweep orchestrator
  (`benchmarks/run_baseline_matrix.sh`, modeled on `run_matrix.sh`'s
  structure but not sharing code with it either). "Naive" is scoped
  deliberately to one thing: **no continuous batching** — a single
  `asyncio.Lock` means only one `generate()` call runs at a time, so
  concurrent requests queue server-side exactly like a hand-rolled
  from-scratch server would, with no custom batching logic. That's the
  entire point of the vLLM comparison (PagedAttention + continuous
  batching vs. none), so it stays as designed rather than being "fixed."
- **2026-07-02**: Fairness audit for `baseline/hf_inference_server.py`
  before running anything, per the request to flag silent slow-path traps
  without artificially crippling the baseline:
  - **Fixed (one-line, standard, not a real optimization)**:
    `torch_dtype="auto"` in `from_pretrained` — without this,
    `transformers` defaults to fp32, which isn't a "naive" choice so much
    as an accidental 2x+ tensor-core slowdown nobody would intentionally
    pick; `"auto"` just loads the checkpoint's own dtype (bf16 for
    Llama-3.2-3B-Instruct), matching how the vLLM "fp16" config was
    actually also auto-dtype (see the fp16-deploy entry above — same
    parity concern).
  - **Fixed (one-line, vanilla-library default, not an optimization
    library)**: `attn_implementation="sdpa"` — PyTorch's built-in fused
    attention, shipped in vanilla `torch`/`transformers` with no extra
    dependency. Set explicitly because relying on the implicit default
    risks silently landing on the much slower pure-Python "eager"
    attention path depending on transformers/torch version. This is what
    a competent from-scratch script would set, not a hand-tuned
    optimization — deliberately did *not* reach for `flash-attn` (a
    separate optimization package) or `torch.compile` for the same
    reason.
  - **Documented as caveats, not fixed** (these are the actual point of
    the comparison, or aren't one-line/well-understood enough to bolt on
    safely):
    - **No continuous batching** (see design note above) — the main
      thing the comparison exists to measure.
    - **No `torch.compile`** — the project brief explicitly calls for a
      "naive HuggingFace `transformers.generate()` baseline"; adding
      graph compilation would blur that line and isn't a one-line change
      (needs shape-stable inputs, warmup runs, recompilation handling for
      variable-length ShareGPT prompts).
    - **`TextIteratorStreamer` chunk granularity** isn't guaranteed
      strictly one-token-per-SSE-chunk (multi-byte UTF-8 continuation
      tokens can be buffered together before being decodable) — TTFT
      should still be accurate (first chunk is still first-token
      arrival), but per-token ITL numbers may be very slightly coarser
      than vLLM's true per-token stream. Not fixable without a custom
      streamer; acceptable at this project's fidelity level.
    - **Single global lock also serializes GPU memory reuse** — since
      only one request's KV cache exists at a time, peak GPU memory
      during the baseline sweep should be *lower* than any of the vLLM
      configs (which pre-reserve a large KV-cache block pool). Expected
      and worth calling out plainly in the Phase 4 report as a
      memory-efficiency, not just speed, comparison point.
- **2026-07-02**: Re-provisioned a pod for Phase 2 (Phase 0/1 pod was
  deleted once idle — see entry above): new pod `die22er57siiee`, same
  RTX A5000/secure-cloud spec via `runpodctl pod create` with the same
  image/ports/disk as Phase 0. Notable differences from the Phase 0 pod,
  neither requiring changes: newer host driver (570.211.01, CUDA 12.8
  reported vs. 550.127.05/12.4 before) — our cu124-pinned torch/vllm stack
  is backward compatible so no requirements.txt change was needed; and
  `rsync` wasn't preinstalled on the fresh image (`apt-get install rsync`
  fixed it — worth adding to `scripts/setup_env.sh` if this recurs).
  `HF_TOKEN` from the deleted pod didn't carry over (expected, new pod);
  copied the local machine's own cached `~/.cache/huggingface/token` file
  to the pod's same path via `scp` (not via env var/`.bashrc`, to avoid
  putting the secret on a command line or in a persisted shell-startup
  file) — `huggingface_hub` picks it up automatically from there.
- **2026-07-02**: Ran the baseline concurrency sweep (1/4/8/16/32, 100
  ShareGPT prompts each, same dataset/seed as Phase 1) via
  `benchmarks/run_baseline_matrix.sh`. Smoke test passed first (2/2
  requests, ~44.6 tok/s aggregate on 890 generated tokens). Total sweep
  wall time was ~40 minutes — in line with the pre-run estimate from the
  smoke test's per-token rate, so no need to cut `NUM_PROMPTS` or
  concurrency levels short (all 5 levels ran at the full 100 prompts, same
  as the Phase 1 vLLM matrix). Results: `results/baseline-c*.json`,
  `.gpu.csv`, `baseline.server.log`.

  | concurrency | req/s | output tok/s | median TTFT (ms) | median TPOT (ms) | peak GPU mem |
  |---|---|---|---|---|---|
  | 1  | 0.24 | 48.1 | 36.8     | 21.24 | 6907 MiB |
  | 4  | 0.25 | 48.7 | 11,434.9 | 20.96 | 6743 MiB |
  | 8  | 0.24 | 48.1 | 28,035.9 | 21.08 | 6949 MiB |
  | 16 | 0.25 | 48.7 | 56,064.2 | 20.53 | 6743 MiB |
  | 32 | 0.24 | 48.3 | 111,705.0| 21.35 | 6949 MiB |

  Exactly as expected from the design (see the no-continuous-batching note
  above): output throughput is flat (~48 tok/s) across every concurrency
  level, because the baseline processes one request at a time regardless
  of how many arrive concurrently — extra concurrent requests just queue
  behind the global lock instead of being batched. Median TTFT grows
  roughly linearly with concurrency (from 37ms at c1 to ~112s at c32)
  purely from that queueing delay, not from any per-request slowdown
  (median TPOT stays ~21ms at every level). Peak GPU memory also stays
  flat (~6.7-6.9GB) since only one request's KV cache exists at a time —
  confirms the memory-efficiency caveat noted in the audit above.
- **2026-07-02**: vLLM vs. baseline comparison (Phase 2 goal). Two framings,
  both useful for the Phase 4 report:

  **Single request (concurrency=1, no batching benefit for either side)** —
  isolates raw kernel/execution efficiency from batching:

  | config | output tok/s | median TTFT (ms) | median TPOT (ms) | peak GPU mem |
  |---|---|---|---|---|
  | baseline (naive HF) | 48.1  | 36.8 | 21.24 | 6907 MiB |
  | vLLM fp16            | 80.4  | 22.9 | 12.39 | 22,303 MiB |
  | vLLM AWQ (awq_marlin) | 143.4 | 19.0 | 6.84  | 21,647 MiB |
  | vLLM GPTQ             | 146.3 | 32.7 | 6.67  | 22,431 MiB |

  Even with no batching advantage, vLLM is **1.7x (fp16) to ~3.0x
  (AWQ/GPTQ) faster per-token** than naive `generate()` — CUDA graphs,
  fused kernels, and (for AWQ/GPTQ) quantized matmul kernels all help even
  at batch size 1. This isolates "vLLM's execution engine is faster" from
  "vLLM also batches," which the concurrency=32 numbers below conflate.

  **High concurrency (concurrency=32)** — the realistic serving scenario,
  where vLLM's continuous batching compounds with its per-token speed
  advantage:

  | config | output tok/s | median TTFT (ms) | peak GPU mem |
  |---|---|---|---|
  | baseline (naive HF) | 48.3   | 111,705.0 | 6949 MiB |
  | vLLM fp16            | 1327.5 | 33.8      | 22,305 MiB |
  | vLLM AWQ (awq_marlin) | 2187.9 | 24.3      | 21,647 MiB |
  | vLLM GPTQ             | 1086.2 | 58.1      | 22,433 MiB |

  At realistic concurrency, vLLM's throughput advantage over the naive
  baseline is **~22.6x (GPTQ) to ~45.5x (AWQ)**, and TTFT is ~2000-4600x
  lower — a naive server without batching effectively falls over under
  concurrent load (TTFT climbs linearly with queue depth) while vLLM's
  TTFT barely moves. The flip side: the baseline uses ~3.1-3.3x *less*
  peak GPU memory at every concurrency level, since it never pre-reserves
  a KV-cache block pool the way vLLM's `gpu_memory_utilization=0.9`
  default does — worth noting in the report as a real memory/throughput
  tradeoff, not just "vLLM wins everything." Phase 2 is now complete.

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
