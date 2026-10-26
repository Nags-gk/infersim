"""Fit the model's efficiency knobs to measured latencies, and report the error.

A measurement is one timed forward pass (a prefill of ``batch`` prompts of ``tokens``
each, or a decode step of ``batch`` sequences at context ``tokens``). Calibration fits
``eff_compute``, ``eff_memory`` and ``launch_overhead`` of a ``HardwareSpec`` by
minimizing squared log error with Nelder-Mead (pure Python, no SciPy needed), and
reports MAPE before and after, overall and per phase.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass

from .hardware import HardwareSpec
from .models import preset as get_model
from .ops import StepShape
from .system import GPUDeployment

PARAMS = ("eff_compute", "eff_memory", "launch_overhead")
BOUNDS = {"eff_compute": (0.05, 1.0), "eff_memory": (0.05, 1.0), "launch_overhead": (0.0, 1e-4)}


@dataclass(frozen=True)
class Measurement:
    model: str
    phase: str  # prefill | decode
    batch: int
    tokens: int  # prompt length (prefill) or context length (decode)
    measured_s: float
    precision: str = "bf16"

    def shape(self) -> StepShape:
        if self.phase == "prefill":
            return StepShape.prefill_only(self.batch, self.tokens)
        return StepShape.decode(self.batch, self.tokens)


def load_csv(path: str) -> list[Measurement]:
    out = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out.append(Measurement(row["model"], row["phase"], int(row["batch"]), int(row["tokens"]),
                                   float(row["measured_s"]), row.get("precision") or "bf16"))
    return out


def predict(m: Measurement, hw: HardwareSpec) -> float:
    dep = GPUDeployment(get_model(m.model), hw, tp=1, precision=m.precision)
    return dep.step(m.shape()).latency_s


def mape(ms: list[Measurement], hw: HardwareSpec) -> float:
    return 100.0 * sum(abs(predict(m, hw) - m.measured_s) / m.measured_s for m in ms) / len(ms)


def _clamp(x: list[float]) -> list[float]:
    return [min(max(v, BOUNDS[p][0]), BOUNDS[p][1]) for v, p in zip(x, PARAMS)]


def _to_hw(hw: HardwareSpec, x: list[float]) -> HardwareSpec:
    return hw.with_overrides(**dict(zip(PARAMS, _clamp(x))))


def _loss(x: list[float], ms: list[Measurement], hw: HardwareSpec) -> float:
    h = _to_hw(hw, x)
    return sum(math.log(predict(m, h) / m.measured_s) ** 2 for m in ms) / len(ms)


def nelder_mead(f, x0: list[float], scale: list[float], iters: int = 400, tol: float = 1e-10) -> list[float]:
    n = len(x0)
    simplex = [list(x0)] + [[x0[j] + (scale[j] if j == i else 0.0) for j in range(n)] for i in range(n)]
    vals = [f(p) for p in simplex]
    for _ in range(iters):
        order = sorted(range(n + 1), key=lambda i: vals[i])
        simplex = [simplex[i] for i in order]
        vals = [vals[i] for i in order]
        if abs(vals[-1] - vals[0]) < tol:
            break
        centroid = [sum(p[j] for p in simplex[:-1]) / n for j in range(n)]
        worst = simplex[-1]
        refl = [centroid[j] + (centroid[j] - worst[j]) for j in range(n)]
        fr = f(refl)
        if fr < vals[0]:
            exp = [centroid[j] + 2 * (centroid[j] - worst[j]) for j in range(n)]
            fe = f(exp)
            simplex[-1], vals[-1] = (exp, fe) if fe < fr else (refl, fr)
        elif fr < vals[-2]:
            simplex[-1], vals[-1] = refl, fr
        else:
            contr = [centroid[j] + 0.5 * (worst[j] - centroid[j]) for j in range(n)]
            fc = f(contr)
            if fc < vals[-1]:
                simplex[-1], vals[-1] = contr, fc
            else:
                best = simplex[0]
                simplex = [best] + [[best[j] + 0.5 * (p[j] - best[j]) for j in range(n)] for p in simplex[1:]]
                vals = [vals[0]] + [f(p) for p in simplex[1:]]
    return simplex[min(range(n + 1), key=lambda i: vals[i])]


@dataclass
class CalibrationResult:
    fitted: dict[str, float]
    mape_before: float
    mape_after: float
    holdout_mape_before: float | None
    holdout_mape_after: float | None
    per_phase_after: dict[str, float]
    hardware: HardwareSpec


def fit(train: list[Measurement], hw: HardwareSpec, holdout: list[Measurement] | None = None) -> CalibrationResult:
    x0 = [getattr(hw, p) for p in PARAMS]
    scale = [0.2, 0.2, 5e-6]
    best = _clamp(nelder_mead(lambda x: _loss(x, train, hw), x0, scale))
    tuned = _to_hw(hw, best)
    phases = {}
    for ph in ("prefill", "decode"):
        sub = [m for m in (train + (holdout or [])) if m.phase == ph]
        if sub:
            phases[ph] = mape(sub, tuned)
    return CalibrationResult(
        fitted=dict(zip(PARAMS, best)),
        mape_before=mape(train, hw),
        mape_after=mape(train, tuned),
        holdout_mape_before=mape(holdout, hw) if holdout else None,
        holdout_mape_after=mape(holdout, tuned) if holdout else None,
        per_phase_after=phases,
        hardware=tuned,
    )


def split(ms: list[Measurement], every: int = 4) -> tuple[list[Measurement], list[Measurement]]:
    """Deterministic holdout: every ``every``-th measurement is held out."""
    train = [m for i, m in enumerate(ms) if i % every]
    hold = [m for i, m in enumerate(ms) if not i % every]
    return train, hold


# ---------------------------------------------------------------------------
# GEMM-level validation and the full calibration report


@dataclass(frozen=True)
class GemmMeasurement:
    m: int
    n: int
    k: int
    measured_s: float
    dtype: str = "fp16"


def load_gemm_csv(path: str, dtype: str = "fp16") -> list[GemmMeasurement]:
    with open(path, newline="") as f:
        return [GemmMeasurement(int(r["m"]), int(r["n"]), int(r["k"]), float(r["measured_s"]), dtype)
                for r in csv.DictReader(f)]


def predict_gemm(g: GemmMeasurement, hw: HardwareSpec) -> float:
    from .ops import PRECISIONS, gemm
    from .roofline import time_op

    prec = PRECISIONS[g.dtype]
    return time_op(gemm("g", "ffn", g.m, g.n, g.k, prec, hw.tile_m, hw.tile_n, 1), hw).time_s


def gemm_mape(gs: list[GemmMeasurement], hw: HardwareSpec) -> float:
    return 100.0 * sum(abs(predict_gemm(g, hw) - g.measured_s) / g.measured_s for g in gs) / len(gs)


def model_kernels(m: Measurement) -> int:
    """Kernel launches the model assumes for one step (all layers)."""
    from .ops import PRECISIONS, step_ops

    return int(sum(o.kernels * o.count for o in step_ops(get_model(m.model), m.shape(), 1, PRECISIONS[m.precision])
                   if o.kind not in ("allreduce", "alltoall")))


def _rows(ms: list[Measurement], before: HardwareSpec, after: HardwareSpec, held: set[int]) -> list[dict]:
    out = []
    for i, m in enumerate(ms):
        pb, pa = predict(m, before), predict(m, after)
        out.append({"model": m.model, "phase": m.phase, "batch": m.batch, "tokens": m.tokens,
                    "measured_ms": m.measured_s * 1e3, "predicted_before_ms": pb * 1e3,
                    "predicted_after_ms": pa * 1e3, "error_after_pct": 100 * (pa - m.measured_s) / m.measured_s,
                    "holdout": i in held})
    return out


def report(steps: list[Measurement], hw: HardwareSpec, gemms: list[GemmMeasurement] | None = None,
           kernels: dict | None = None, env: dict | None = None, every: int = 4) -> dict:
    """Fit on a training split and report everything the dashboard and README need."""
    train, hold = split(steps, every)
    r = fit(train, hw, hold)
    held = {i for i in range(len(steps)) if not i % every}
    by_model = {}
    for name in sorted({m.model for m in steps}):
        for ph in ("prefill", "decode"):
            sub = [m for m in steps if m.model == name and m.phase == ph]
            if sub:
                by_model[f"{name} {ph}"] = {"before": mape(sub, hw), "after": mape(sub, r.hardware), "n": len(sub)}
    out = {
        "hardware": hw.name,
        "env": env or {},
        "fitted": r.fitted,
        "defaults": {p: getattr(hw, p) for p in PARAMS},
        "n_measurements": len(steps),
        "n_holdout": len(hold),
        "mape_train_before": r.mape_before,
        "mape_train_after": r.mape_after,
        "mape_holdout_before": r.holdout_mape_before,
        "mape_holdout_after": r.holdout_mape_after,
        "by_model_phase": by_model,
        "rows": _rows(steps, hw, r.hardware, held),
    }
    if gemms:
        out["gemm"] = {
            "mape_before": gemm_mape(gemms, hw),
            "mape_after": gemm_mape(gemms, r.hardware),
            "peak_flops": hw.peak(gemms[0].dtype),
            "mem_bw": hw.mem_bw,
            "rows": [{"m": g.m, "n": g.n, "k": g.k, "measured_us": g.measured_s * 1e6,
                      "predicted_after_us": predict_gemm(g, r.hardware) * 1e6,
                      "tflops": 2 * g.m * g.n * g.k / g.measured_s / 1e12,
                      "intensity": 2 * g.m * g.n * g.k / ((g.m * g.k + g.k * g.n + g.m * g.n) * 2)} for g in gemms],
        }
    if kernels:
        rows = []
        for m in steps:
            key = (m.model, m.phase, m.batch, m.tokens)
            if key in kernels and kernels[key] > 0:
                rows.append({"model": m.model, "phase": m.phase, "batch": m.batch, "tokens": m.tokens,
                             "measured": kernels[key], "modeled": model_kernels(m)})
        out["kernels"] = rows
    return out


def load_kernels_csv(path: str) -> dict:
    with open(path, newline="") as f:
        return {(r["model"], r["phase"], int(r["batch"]), int(r["tokens"])): int(r["cuda_kernels"])
                for r in csv.DictReader(f)}


def report_from_path(path: str, hw: HardwareSpec) -> dict:
    """``path`` is a steps CSV, or a folder written by ``calibration/measure.py``."""
    import json
    import os

    if os.path.isdir(path):
        steps = load_csv(os.path.join(path, "steps.csv"))
        g = os.path.join(path, "gemm.csv")
        k = os.path.join(path, "kernels.csv")
        e = os.path.join(path, "env.json")
        env = None
        if os.path.exists(e):
            with open(e) as f:
                env = json.load(f)
        return report(steps, hw, load_gemm_csv(g, steps[0].precision) if os.path.exists(g) else None,
                      load_kernels_csv(k) if os.path.exists(k) else None, env)
    return report(load_csv(path), hw)
