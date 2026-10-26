"""Guardrails for the hardware catalog and model presets.

These run on every edit through the Claude Code hook in ``.claude/settings.json``, so a
coding agent cannot add a device or model without a source and sane numbers.
"""

import pytest

from infersim.hardware import CATALOG
from infersim.models import PRESET_FILES, preset


@pytest.mark.parametrize("key", sorted(CATALOG))
def test_every_device_is_sourced_and_plausible(key):
    h = CATALOG[key]
    assert h.sources, f"{key}: add at least one public source"
    assert all(s.strip() for s in h.sources + h.assumptions)
    assert h.kind in ("gpu", "lpu")
    assert 0 < h.eff_compute <= 1 and 0 < h.eff_memory <= 1 and 0 < h.usable_memory <= 1
    for dtype, flops in h.peak_flops.items():
        assert 1e12 <= flops <= 1e17, f"{key}.{dtype} peak looks wrong: {flops}"
    assert 1e10 <= h.mem_bw <= 1e15
    assert 1e6 <= h.mem_capacity <= 1e12
    ridge = max(h.peak_flops.values()) / h.mem_bw
    assert 1 <= ridge <= 5000, f"{key}: ridge point {ridge:.0f} FLOP/B is implausible"


@pytest.mark.parametrize("name", sorted(PRESET_FILES))
def test_every_model_preset_loads_and_is_consistent(name):
    m = preset(name)
    assert m.heads % m.kv_heads == 0
    assert m.param_count(active=True) <= m.param_count()
    assert m.kv_elems_per_token_layer() > 0
