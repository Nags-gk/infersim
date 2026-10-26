"""Deployments: memory fit, parallelism, LPU sizing and heterogeneous layouts."""

import math

import pytest

from infersim.hardware import get
from infersim.models import preset
from infersim.ops import StepShape
from infersim.system import AttentionFFNSplit, Disaggregated, GPUDeployment, LPUDeployment


def test_memory_fit():
    assert not GPUDeployment("llama-3.1-70b", "h100", tp=1).fits()  # 141 GB of bf16 weights
    assert GPUDeployment("llama-3.1-70b", "h100", tp=2).fits()
    assert GPUDeployment("llama-3.1-70b", "b200", tp=1, precision="fp8").fits()


def test_kv_capacity_scales_with_tp():
    a = GPUDeployment("llama-3.1-70b", "h100", tp=4)
    b = GPUDeployment("llama-3.1-70b", "h100", tp=8)
    assert b.kv_capacity_tokens() > 2 * a.kv_capacity_tokens()


def test_tp_cuts_decode_latency_but_adds_comm():
    one = GPUDeployment("llama-3.1-8b", "h100", tp=1).step(StepShape.decode(1, 2048))
    eight = GPUDeployment("llama-3.1-8b", "h100", tp=8).step(StepShape.decode(1, 2048))
    assert eight.latency_s < one.latency_s
    assert "comm" in eight.by("category") and "comm" not in one.by("category")


def test_decode_memory_bound_prefill_compute_bound():
    d = GPUDeployment("llama-3.1-8b", "h100")
    assert d.step(StepShape.decode(1, 1024)).bottleneck()["dominant_bound"] == "memory"
    assert d.step(StepShape.prefill_only(1, 4096)).bottleneck()["dominant_bound"] == "compute"


def test_single_gpu_decode_near_weight_streaming_time():
    d = GPUDeployment("llama-3.1-8b", "h100")
    t = d.step(StepShape.decode(1, 128)).latency_s
    weights = preset("llama-3.1-8b").param_count() * 2
    floor = weights / get("h100").mem_bw
    assert floor < t < 1.6 * floor


def test_lpu_chip_count_for_70b_matches_reported_order():
    """Groq reported ~576 v1 chips for a 70B model; the SRAM arithmetic alone gives ~600-700."""
    l = LPUDeployment("llama-3.1-70b", "groqchip1", precision="bf16")
    need = preset("llama-3.1-70b").param_count() * 2 / 230e6
    assert l.devices == math.ceil(preset("llama-3.1-70b").param_count() * 2 / (230e6 * 0.9))
    assert 0.9 * 576 < need < 1.3 * 576


def test_lpu_beats_gpu_on_single_user_latency():
    gpu = GPUDeployment("llama-3.1-70b", "h100", tp=8).step(StepShape.decode(1, 2048)).latency_s
    lpu = LPUDeployment("llama-3.1-70b", "groq3").step(StepShape.decode(1, 2048)).latency_s
    assert lpu < gpu


def test_disaggregated_kv_transfer():
    d = Disaggregated(GPUDeployment("llama-3.1-8b", "h100"), GPUDeployment("llama-3.1-8b", "h100"))
    assert d.kv_transfer_s(4096) == pytest.approx(10e-6 + 4096 * 131072 / 50e9)
    assert d.devices == 2


def test_afd_moves_ffn_to_lpu():
    a = AttentionFFNSplit("llama-3.1-70b", "h100", "groq3", gpu_tp=8)
    r = a.step(StepShape.decode(16, 2048))
    cats = r.by("category")
    assert cats["ffn"] > 0 and cats["comm"] > 0
    assert a.lpu_chips > 50
    assert a.gpu_weight_bytes_per_device() < GPUDeployment("llama-3.1-70b", "h100", tp=8,
                                                           precision="fp8").weight_bytes_per_device()


def test_moe_expert_parallel_uses_alltoall():
    d = GPUDeployment("mixtral-8x7b", "h100", tp=8, expert_parallel=True)
    names = {t.op.name for t in d.step(StepShape.decode(32, 1024)).timings}
    assert {"ep_dispatch", "ep_combine"} <= names and "tp_allreduce_ffn" not in names
