#!/usr/bin/env python3
"""Naive HuggingFace transformers.generate() baseline server.

Exposes a minimal OpenAI-compatible POST /v1/completions endpoint — the
exact request/response contract benchmarks/vendor/backend_request_func.py's
async_request_openai_completions expects (SSE-streamed choices[].text
chunks, a trailing usage.completion_tokens chunk, terminated by
"data: [DONE]") — so the same vendored benchmark_serving.py client used for
the vLLM sweep can be pointed at this server unmodified, no adapter needed.

This is a from-scratch minimal FastAPI wrapper around
AutoModelForCausalLM.generate(); it intentionally shares no code with
serving/launch_vllm.sh for a clean A/B. "Naive" here specifically means: no
continuous batching (one generate() call in flight at a time, enforced by
a global lock — concurrent requests queue) and no torch.compile. See
dev/active/vllm-benchmarking/context.md (Phase 2) for the fairness audit
of what was and wasn't fixed, and why.

Usage:
  python3 baseline/hf_inference_server.py <model-repo> [port]
"""
import argparse
import asyncio
import json
import sys
from threading import Thread

import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

app = FastAPI()
model = None
tokenizer = None
# Naive baseline = no continuous batching: only one generate() call runs at
# a time, matching a hand-rolled transformers.generate() loop with no
# custom batching logic. Concurrent requests queue on this lock, which is
# the whole point of the vLLM comparison (see context.md).
generation_lock = asyncio.Lock()

_DONE = object()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/v1/completions")
async def completions(request: Request):
    body = await request.json()
    prompt = body["prompt"]
    max_new_tokens = body.get("max_tokens", 16)

    async def event_stream():
        loop = asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def worker():
            try:
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                input_len = inputs["input_ids"].shape[-1]
                streamer = TextIteratorStreamer(
                    tokenizer, skip_prompt=True, skip_special_tokens=True
                )
                gen_kwargs = dict(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    streamer=streamer,
                )
                output_holder = {}

                def _generate():
                    with torch.inference_mode():
                        output_holder["ids"] = model.generate(**gen_kwargs)

                gen_thread = Thread(target=_generate)
                gen_thread.start()
                for text in streamer:
                    loop.call_soon_threadsafe(queue.put_nowait, text)
                gen_thread.join()
                n_tokens = output_holder["ids"].shape[-1] - input_len
                loop.call_soon_threadsafe(queue.put_nowait, ("usage", n_tokens))
            except Exception as exc:  # surface errors as a final SSE frame
                loop.call_soon_threadsafe(queue.put_nowait, ("error", str(exc)))
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, _DONE)

        async with generation_lock:
            Thread(target=worker, daemon=True).start()
            while True:
                item = await queue.get()
                if item is _DONE:
                    break
                if isinstance(item, tuple) and item[0] == "usage":
                    yield f"data: {json.dumps({'usage': {'completion_tokens': item[1]}})}\n\n".encode()
                elif isinstance(item, tuple) and item[0] == "error":
                    yield f"data: {json.dumps({'error': item[1]})}\n\n".encode()
                else:
                    yield f"data: {json.dumps({'choices': [{'text': item}]})}\n\n".encode()
        yield b"data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def main():
    global model, tokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="HF model repo, e.g. meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("port", nargs="?", type=int, default=8000)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: no CUDA device visible — refusing to silently fall back to CPU", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {args.model} ...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        # "auto" picks up the checkpoint's own dtype (bf16 for
        # Llama-3.2-3B-Instruct) instead of transformers' fp32 default —
        # see context.md fairness audit; without this the baseline would
        # run 2x+ slower on fp32 tensor-core math for no representative
        # reason.
        torch_dtype="auto",
        # sdpa is the standard PyTorch/transformers default fused-attention
        # path (no extra optimization library beyond vanilla
        # torch+transformers) — explicit here because relying on the
        # library's implicit default risks silently landing on the much
        # slower pure-Python "eager" attention path on some
        # transformers/torch version combos. Still a "naive" choice, not an
        # optimization: it's what a competent from-scratch script would set.
        attn_implementation="sdpa",
    ).to("cuda")
    model.eval()
    print("Model loaded.", file=sys.stderr)

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
