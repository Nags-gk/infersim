"""Roofline and operator invariants, checked over random shapes with Hypothesis."""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from infersim.hardware import CATALOG, get
from infersim.models import preset
from infersim.ops import PRECISIONS, StepShape, gemm, step_ops, total
from infersim.roofline import allreduce_time, alltoall_time, time_op
from infersim.system import GPUDeployment

MODELS = ["llama-3.1-8b", "mixtral-8x7b", "deepseek-v3", "qwen2.5-0.5b"]


@settings(max_examples=60, deadline=None)
@given(m=st.integers(1, 8192), n=st.integers(1, 16384), k=st.integers(1, 16384),
       hw=st.sampled_from(sorted(CATALOG)))
def test_gemm_time_never_beats_roofline(m, n, k, hw):
    h = get(hw)
    op = gemm("g", "ffn", m, n, k, PRECISIONS["bf16"], h.tile_m, h.tile_n, 1)
    t = time_op(op, h)
    assert t.time_s >= t.ideal > 0
    assert op.flops_padded >= op.flops


def test_ridge_point_classifies_gemv_vs_big_gemm():
    h = get("h100")
    gemv = time_op(gemm("gemv", "ffn", 1, 8192, 8192, PRECISIONS["bf16"], 64, 128, 1), h)
    big = time_op(gemm("big", "ffn", 8192, 8192, 8192, PRECISIONS["bf16"], 64, 128, 1), h)
    assert gemv.bound == "memory"
    assert big.bound == "compute"
    assert h.ridge_point("bf16") == pytest.approx(989e12 / 3.35e12)


@settings(max_examples=40, deadline=None)
@given(model=st.sampled_from(MODELS), batch=st.integers(1, 64), ctx=st.integers(16, 8192))
def test_decode_flops_close_to_two_times_active_params(model, batch, ctx):
    """Linear-layer FLOPs per token are ~2x active parameters (attention scores excluded)."""
    m = preset(model)
    ops = step_ops(m, StepShape.decode(batch, ctx))
    linear = sum(o.flops * o.count for o in ops if o.kind == "gemm")
    per_token = linear / batch
    # embeddings are a lookup, not a GEMM; lm_head is one GEMM per sequence
    expected = 2 * (m.param_count(active=True) - m.vocab * m.hidden * (0 if m.tie_embeddings else 1))
    assert per_token == pytest.approx(expected, rel=0.03)


def test_decode_attention_bytes_grow_with_context():
    m = preset("llama-3.1-8b")
    a = total([o for o in step_ops(m, StepShape.decode(8, 1024)) if o.name == "attn_decode"], "bytes")
    b = total([o for o in step_ops(m, StepShape.decode(8, 4096)) if o.name == "attn_decode"], "bytes")
    assert b / a == pytest.approx(4.0, rel=0.01)


def test_weight_bytes_read_once_per_step():
    m = preset("llama-3.1-8b")
    w = total(step_ops(m, StepShape.decode(1, 128)), "weight_bytes")
    assert w == pytest.approx(m.param_count() * 2 - m.vocab * m.hidden * 2, rel=0.01)


def test_allreduce_formulas():
    assert allreduce_time(1e9, 1, 1e11, 1e-6) == 0
    # large message: ring wins, 2(p-1)/p of payload over the link
    assert allreduce_time(1e9, 8, 1e11, 1e-6) == pytest.approx(2 * 7 / 8 * 1e9 / 1e11 + 14e-6)
    # tiny message: one-shot wins, latency dominated
    assert allreduce_time(1e3, 8, 1e11, 1e-6) == pytest.approx(7e3 / 1e11 + 2e-6)
    assert alltoall_time(8e6, 8, 1e11, 0) == pytest.approx(7 / 8 * 8e6 / 1e11)


@settings(max_examples=30, deadline=None)
@given(batch=st.integers(1, 128))
def test_step_latency_monotonic_in_batch(batch):
    d = GPUDeployment("llama-3.1-8b", "h100")
    a = d.step(StepShape.decode(batch, 1024)).latency_s
    b = d.step(StepShape.decode(batch + 1, 1024)).latency_s
    assert b >= a


def test_moe_expected_active_experts_saturate():
    from infersim.ops import expected_active_experts
    assert expected_active_experts(1, 256, 8) == pytest.approx(8)
    assert expected_active_experts(10_000, 256, 8) == pytest.approx(256, rel=1e-6)
