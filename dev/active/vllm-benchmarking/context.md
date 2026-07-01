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
