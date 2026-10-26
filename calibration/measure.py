"""Measure real prefill/decode step latency and GEMM throughput on a GPU.

Run on a machine with PyTorch and transformers (the Colab notebook does this):

    python calibration/measure.py --out results/

Writes:
  steps.csv     model,phase,batch,tokens,measured_s,precision   (input to `infersim calibrate`)
  gemm.csv      m,n,k,measured_s,tflops                         (single fp16 GEMMs)
  kernels.csv   model,phase,batch,tokens,cuda_kernels           (torch.profiler kernel counts per step)
  env.json      GPU, driver, library versions, settings

Timing: CUDA events around the forward pass, warmup iterations first, median of N
iterations. Prefill computes logits for the last position only (as serving engines do).
Decode uses a KV cache pre-filled to ``tokens`` and crops it back after every step, so
each timed step sees the same context length.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
import time

import torch

MODELS = {"qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B", "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B"}
DECODE = [(b, c) for b in (1, 4, 16, 32, 64) for c in (256, 1024, 2048)]
PREFILL = [(1, s) for s in (128, 256, 512, 1024, 2048)] + [(4, 512), (8, 256)]
GEMM_M = (1, 8, 32, 128, 512, 2048)
GEMM_NK = ((896, 4864), (4864, 896), (1536, 8960), (8960, 1536), (4096, 4096))


def timer(fn, device: str, warmup: int, iters: int) -> float:
    """Median seconds per call."""
    for _ in range(warmup):
        fn()
    times = []
    if device == "cuda":
        torch.cuda.synchronize()
        for _ in range(iters):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            fn()
            b.record()
            torch.cuda.synchronize()
            times.append(a.elapsed_time(b) / 1e3)
    else:
        for _ in range(iters):
            t = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t)
    return statistics.median(times)


def load(name: str, dtype, device: str, config=None):
    from transformers import AutoModelForCausalLM

    if config is not None:  # offline test path: random weights from a config
        m = AutoModelForCausalLM.from_config(config)
        return m.to(dtype).to(device).eval()
    kw = {"attn_implementation": "sdpa"}
    try:
        m = AutoModelForCausalLM.from_pretrained(MODELS[name], dtype=dtype, **kw)
    except TypeError:  # transformers < 4.56 uses torch_dtype
        m = AutoModelForCausalLM.from_pretrained(MODELS[name], torch_dtype=dtype, **kw)
    return m.to(device).eval()


def _forward(model, **kw):
    try:
        return model(**kw, logits_to_keep=1)
    except TypeError:  # older transformers
        return model(**kw, num_logits_to_keep=1)


def prefill_fn(model, batch: int, tokens: int, device: str):
    ids = torch.randint(0, model.config.vocab_size, (batch, tokens), device=device)
    return lambda: _forward(model, input_ids=ids, use_cache=False)


def decode_fn(model, batch: int, ctx: int, device: str):
    """One decode step at context ``ctx`` (ctx - 1 cached tokens + 1 new token)."""
    vocab = model.config.vocab_size
    past = torch.randint(0, vocab, (batch, ctx - 1), device=device)
    with torch.inference_mode():
        cache = _forward(model, input_ids=past, use_cache=True).past_key_values
    tok = torch.randint(0, vocab, (batch, 1), device=device)
    pos = torch.full((batch, 1), ctx - 1, device=device, dtype=torch.long)

    def step():
        _forward(model, input_ids=tok, past_key_values=cache, position_ids=pos, use_cache=True)
        extra = cache.get_seq_length() - (ctx - 1)
        if extra > 0:  # negative crop = drop that many trailing tokens (same in transformers 4.x and 5.x)
            cache.crop(-extra)

    return step


def count_kernels(fn, device: str) -> int:
    if device != "cuda":
        return -1
    from torch.profiler import ProfilerActivity, profile

    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return sum(1 for e in prof.events() if e.device_type.name == "CUDA")


def run(out: str, models, dtype_name: str, warmup: int, iters: int, config=None, decode=DECODE,
        prefill=PREFILL, gemm_m=GEMM_M, gemm_nk=GEMM_NK) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[dtype_name]
    os.makedirs(out, exist_ok=True)
    steps, kernels, gemms = [], [], []
    torch.manual_seed(0)
    with torch.inference_mode():
        for name in models:
            model = load(name, dtype, device, config)
            for phase, grid, make in (("prefill", prefill, prefill_fn), ("decode", decode, decode_fn)):
                for batch, tokens in grid:
                    try:
                        fn = make(model, batch, tokens, device)
                        t = timer(fn, device, warmup, iters)
                        k = count_kernels(fn, device)
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        print(f"  skip {name} {phase} b={batch} t={tokens}: out of memory")
                        continue
                    steps.append([name, phase, batch, tokens, f"{t:.6e}", dtype_name])
                    kernels.append([name, phase, batch, tokens, k])
                    print(f"  {name} {phase:7s} b={batch:<3d} t={tokens:<5d} {t * 1e3:8.2f} ms  kernels={k}")
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
        for n, k in gemm_nk:
            w = torch.randn(k, n, device=device, dtype=dtype)
            for m in gemm_m:
                a = torch.randn(m, k, device=device, dtype=dtype)
                t = timer(lambda a=a, w=w: a @ w, device, warmup, iters * 2)
                gemms.append([m, n, k, f"{t:.6e}", f"{2 * m * n * k / t / 1e12:.3f}"])
                print(f"  gemm m={m:<5d} n={n:<5d} k={k:<5d} {t * 1e6:8.1f} us  {2 * m * n * k / t / 1e12:6.2f} TFLOP/s")
    _write(os.path.join(out, "steps.csv"), ["model", "phase", "batch", "tokens", "measured_s", "precision"], steps)
    _write(os.path.join(out, "kernels.csv"), ["model", "phase", "batch", "tokens", "cuda_kernels"], kernels)
    _write(os.path.join(out, "gemm.csv"), ["m", "n", "k", "measured_s", "tflops"], gemms)
    env = {
        "device": torch.cuda.get_device_name(0) if device == "cuda" else platform.processor() or "cpu",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": __import__("transformers").__version__,
        "dtype": dtype_name, "warmup": warmup, "iters": iters, "attention": "sdpa",
        "timing": "CUDA events, median" if device == "cuda" else "perf_counter, median",
    }
    if device == "cuda":
        env["driver"] = os.popen("nvidia-smi --query-gpu=driver_version --format=csv,noheader").read().strip()
    with open(os.path.join(out, "env.json"), "w") as f:
        json.dump(env, f, indent=2)
    return env


def _write(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--dtype", default="fp16", help="T4 has no bf16 tensor cores: use fp16")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    a = ap.parse_args()
    print(json.dumps(run(a.out, a.models, a.dtype, a.warmup, a.iters), indent=2))
