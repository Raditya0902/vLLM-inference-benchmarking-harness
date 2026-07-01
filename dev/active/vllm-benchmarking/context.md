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

## Architecture Notes

> (none yet — e.g. how the load generator is structured, how results are
> keyed/stored, how the OpenAI-compatible endpoint is wired to observability)

## Environment Details

> (none yet — actual GPU instance details once provisioned; mirror the
> summary into root `CLAUDE.md` Environment/GPU Details section too)

## Open Questions

- Does `ModelCloud/Llama-3.2-3B-Instruct-gptqmodel-4bit-vortex-v3` load
  cleanly in vLLM, or is the `shuyuej` GPTQ repo a better fit? Verify during
  Phase 0 deploy.
- GPU provider/instance choice (Lambda Labs vs RunPod), exact instance type —
  pending user decision, see plan.md Phase 0.
- Exact concurrency levels to sweep, report format — not yet decided.
