"""Regenerate site/data/results.json: precomputed views and auto-written findings.

Run: python bench/run_bench.py
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from infersim import __version__
from infersim.dse import best_under_sla, candidate_deployments, sweep
from infersim.models import preset
from infersim.ops import StepShape
from infersim.system import GPUDeployment, LPUDeployment
from infersim.web import catalog_view, dse_view, serve_view, step_view

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site"


def decode_crossover_batch(model: str, hw: str, precision: str) -> int:
    """Smallest decode batch at which the FFN up-projection becomes compute-bound."""
    dep = GPUDeployment(model, hw, tp=1, precision=precision)
    for b in range(1, 4097):
        r = dep.step(StepShape.decode(b, 1024))
        t = next(t for t in r.timings if t.op.name == "gate_up")
        if t.bound == "compute":
            return b
    return -1


def findings(dse70: dict) -> list[dict]:
    out = []
    # 1. roofline: decode GEMMs stay memory-bound until the batch reaches the ridge point
    b_bf16 = decode_crossover_batch("llama-3.1-8b", "h100", "bf16")
    b_fp8 = decode_crossover_batch("llama-3.1-8b", "h100", "fp8")
    same = "the same batch in FP8, because FP8 halves weight bytes and doubles peak FLOPs, so the ridge " \
           "moves with it" if b_fp8 == b_bf16 else f"batch {b_fp8} in FP8"
    out.append({"title": f"Decode is memory-bound until batch ~{b_bf16}",
                "body": f"On H100, Llama-3.1-8B's FFN GEMMs cross the roofline ridge at batch {b_bf16} in BF16 and at "
                        f"{same}. Below that, each extra user costs almost nothing per step."})
    # 2. fixed costs for TP8 low-batch decode
    r = GPUDeployment("llama-3.1-70b", "h100", tp=8).step(StepShape.decode(1, 2048)).bottleneck()
    fixed = r["bound_share"].get("overhead", 0) + r["bound_share"].get("comm", 0)
    out.append({"title": "%.0f%% of 70B single-user decode on 8x H100 is fixed cost" % (100 * fixed),
                "body": "Kernel launches ({:.0f}%) and 160 latency-bound all-reduces ({:.0f}%) per token do not "
                        "shrink with TP, so splitting further stops paying off before weight streaming does.".format(100 * r["bound_share"].get("overhead", 0), 100 * r["bound_share"].get("comm", 0))})
    # 3. GPU vs LPU trade-off
    pts = [type("P", (), p) for p in dse70["points"]]
    gpu = max((p for p in pts if p.family == "gpu"), key=lambda p: p.tok_s_user)
    lpu = max((p for p in pts if p.family == "lpu"), key=lambda p: p.tok_s_user)
    best = dse70["best_under_sla"]["50"]
    out.append({"title": "LPU: {:.1f}x faster per user, {:.0f}x less throughput per chip".format(
        lpu.tok_s_user / gpu.tok_s_user, best["tok_s_device"] / lpu.tok_s_device),
        "body": f"Llama-3.1-70B: the fastest GPU layout reaches {gpu.tok_s_user:.0f} tok/s per user; "
                f"{lpu.devices} Groq 3 LPUs reach {lpu.tok_s_user:.0f} (modeled, assumed compute). At a "
                f"50 tok/s/user SLA the best layout is {best['label']} at {best['tok_s_device']:.0f} tok/s per device."})
    # 4. MLA KV-cache compression
    ds = preset("deepseek-v3").kv_bytes_per_token(1)
    l4 = preset("llama-3.1-405b").kv_bytes_per_token(1)
    out.append({"title": "MLA caches %.1fx less KV per token" % (l4 / ds),
                "body": f"DeepSeek-V3 stores a 576-dim latent per layer ({ds / 1024:.0f} KiB/token in FP8) vs "
                        f"Llama-3.1-405B's GQA K/V ({l4 / 1024:.0f} KiB/token), so far more users fit per GPU."})
    # 5. Heterogeneous decode
    lp = LPUDeployment("llama-3.1-70b", "groq3")
    out.append({"title": f"Weights alone need {lp.devices} Groq 3 LPUs for 70B",
                "body": "SRAM-only chips must hold every weight on chip; KV cache needs more chips on top. "
                        "That is why NVIDIA pairs LPUs with HBM GPUs instead of replacing them."})
    return out


def main() -> None:
    t0 = time.time()
    out = {"version": __version__, "catalog": catalog_view(), "steps": {}, "dse": {}, "serving": {}}
    for key, kw in {
        "8b_h100_decode_b1": {"model": "llama-3.1-8b", "hw": "h100", "phase": "decode", "batch": 1, "tokens": 2048},
        "8b_h100_decode_b64": {"model": "llama-3.1-8b", "hw": "h100", "phase": "decode", "batch": 64, "tokens": 2048},
        "8b_h100_prefill": {"model": "llama-3.1-8b", "hw": "h100", "phase": "prefill", "batch": 1, "tokens": 4096},
        "70b_h100tp8_decode_b1": {"model": "llama-3.1-70b", "hw": "h100", "tp": 8, "phase": "decode", "batch": 1, "tokens": 2048},
        "70b_groq3_decode_b1": {"model": "llama-3.1-70b", "hw": "groq3", "precision": "fp8", "phase": "decode", "batch": 1, "tokens": 2048},
    }.items():
        out["steps"][key] = step_view(**kw)
    for m in ("llama-3.1-8b", "llama-3.1-70b", "deepseek-v3"):
        out["dse"][m] = dse_view(m, 2048, 256)
    curve = []
    for rate in (1, 2, 4, 8, 12, 16, 20, 24):
        s = serve_view("llama-3.1-8b", "h100", rate=rate, requests=300, prompt=1024, output=256)
        curve.append({k: v for k, v in s.items() if not isinstance(v, list)} | {"rate": rate})
    out["serving"]["llama-3.1-8b_h100"] = curve
    out["findings"] = findings(out["dse"]["llama-3.1-70b"])
    out["sla_example"] = best_under_sla(sweep(candidate_deployments("llama-3.1-70b"), 2048, 256), 50).as_dict()
    out["generated_in_s"] = round(time.time() - t0, 1)
    (SITE / "data").mkdir(parents=True, exist_ok=True)
    (SITE / "data" / "results.json").write_text(json.dumps(out))
    # copy the package so the page can run it in the browser
    dst = SITE / "py" / "infersim"
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(ROOT / "infersim", dst, ignore=shutil.ignore_patterns("__pycache__"))
    files = sorted(str(p.relative_to(SITE / "py")) for p in dst.rglob("*") if p.is_file())
    (SITE / "py" / "manifest.json").write_text(json.dumps(files))
    print(f"wrote results.json ({len(json.dumps(out)) / 1e6:.1f} MB) and {len(files)} package files "
          f"in {out['generated_in_s']}s")
    for f in out["findings"]:
        print("-", f["title"])


if __name__ == "__main__":
    main()
