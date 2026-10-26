"""Serving simulator, design space exploration and calibration."""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from infersim import calibrate as cal
from infersim.dse import DesignPoint, best_under_sla, candidate_deployments, evaluate, pareto, sweep
from infersim.hardware import get
from infersim.serving import poisson_workload, simulate
from infersim.system import Disaggregated, GPUDeployment


def test_simulation_conserves_requests_and_tokens():
    reqs = poisson_workload(rate=20, n=120, prompt=512, output=64, jitter=0.3, seed=1)
    res = simulate(GPUDeployment("llama-3.1-8b", "h100"), reqs)
    assert sorted(r.rid for r in res.requests) == [r.rid for r in reqs]
    for got, want in zip(res.requests, reqs):
        assert got.output == want.output
        assert got.arrival <= got.first_token <= got.finish
    s = res.summary()
    assert s["throughput_tok_s"] <= sum(r.output for r in reqs) / (s["makespan_s"] - 1e-9) + 1e-6


def test_ttft_at_least_isolated_prefill():
    dep = GPUDeployment("llama-3.1-8b", "h100")
    from infersim.ops import StepShape
    iso = dep.step(StepShape.prefill_only(1, 2048)).latency_s
    res = simulate(dep, poisson_workload(rate=1, n=5, prompt=2048, output=8, seed=2), max_tokens=4096)
    assert min(r.ttft for r in res.requests) >= iso * 0.999


def test_higher_load_raises_latency():
    dep = GPUDeployment("llama-3.1-8b", "h100")
    lo = simulate(dep, poisson_workload(2, 150, 1024, 128, seed=3)).summary()
    hi = simulate(dep, poisson_workload(40, 150, 1024, 128, seed=3)).summary()
    assert hi["tpot_p50_ms"] > lo["tpot_p50_ms"]
    assert hi["throughput_tok_s"] > lo["throughput_tok_s"]


def test_disaggregated_simulation_runs():
    d = Disaggregated(GPUDeployment("llama-3.1-8b", "h100"), GPUDeployment("llama-3.1-8b", "h100"))
    res = simulate(d, poisson_workload(10, 50, 1024, 64, seed=4))
    assert len(res.requests) == 50 and all(r.finish > r.first_token for r in res.requests)


point = st.builds(DesignPoint, label=st.just("x"), family=st.just("gpu"), devices=st.just(1),
                  batch=st.just(1), users=st.just(1), tpot_ms=st.just(1.0), ttft_ms=st.just(1.0),
                  tok_s_user=st.floats(1, 1e4), tok_s_device=st.floats(1, 1e4),
                  dominant_bound=st.just("memory"), dominant_category=st.just("ffn"))


@settings(max_examples=50, deadline=None)
@given(st.lists(point, min_size=1, max_size=40))
def test_pareto_front_is_non_dominated_and_covers_extremes(pts):
    front = pareto(pts)
    for p in front:
        assert not any(q.tok_s_user > p.tok_s_user and q.tok_s_device > p.tok_s_device for q in pts)
    assert max(p.tok_s_user for p in front) == max(p.tok_s_user for p in pts)
    assert max(p.tok_s_device for p in front) == max(p.tok_s_device for p in pts)


def test_dse_finds_heterogeneous_and_sla_choice():
    pts = sweep(candidate_deployments("llama-3.1-70b"), 2048, 256)
    fams = {p.family for p in pts}
    assert {"gpu", "lpu", "disaggregated", "afd"} <= fams
    best = best_under_sla(pts, 50)
    assert best is not None and best.tok_s_user >= 50
    assert evaluate(GPUDeployment("llama-3.1-70b", "h100", tp=1), 1, 2048, 256) is None  # does not fit


def _synthetic(hw):
    ms = []
    for phase, batch, tokens in [("decode", b, c) for b in (1, 4, 16, 32) for c in (256, 1024)] + \
                                [("prefill", 1, s) for s in (128, 512, 1024, 2048)]:
        m = cal.Measurement("qwen2.5-0.5b", phase, batch, tokens, 1.0)
        ms.append(cal.Measurement(m.model, phase, batch, tokens, cal.predict(m, hw)))
    return ms


def test_calibration_recovers_known_parameters():
    truth = get("t4").with_overrides(eff_compute=0.45, eff_memory=0.62, launch_overhead=9e-6)
    ms = _synthetic(truth)
    train, hold = cal.split(ms)
    r = cal.fit(train, get("t4"), hold)
    assert r.mape_after < 1.0 < r.mape_before
    assert r.holdout_mape_after < 1.0
    assert r.fitted["eff_memory"] == pytest.approx(0.62, rel=0.05)
    assert r.fitted["launch_overhead"] == pytest.approx(9e-6, rel=0.15)
