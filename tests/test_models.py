"""Architectures: parameter counts and KV sizes against published numbers."""

import json

import pytest

from infersim.models import all_presets, from_hf_config, preset

# (total params, active params) as published by the model authors
PUBLISHED = {
    "llama-3.1-8b": (8.03e9, 8.03e9),
    "llama-3.1-70b": (70.6e9, 70.6e9),
    "llama-3.1-405b": (405.9e9, 405.9e9),
    "mixtral-8x7b": (46.7e9, 12.9e9),
    "deepseek-v3": (671e9, 37e9),
    "qwen2.5-0.5b": (0.49e9, 0.49e9),
    "qwen2.5-1.5b": (1.54e9, 1.54e9),
}


@pytest.mark.parametrize("name", sorted(PUBLISHED))
def test_param_counts_match_published(name):
    m = preset(name)
    total, active = PUBLISHED[name]
    assert m.param_count() == pytest.approx(total, rel=0.015)
    assert m.param_count(active=True) == pytest.approx(active, rel=0.02)


def test_kv_bytes_per_token():
    # Llama-3.1-8B: 32 layers x (K+V) x 8 KV heads x 128 dims x 2 bytes = 128 KiB
    assert preset("llama-3.1-8b").kv_bytes_per_token(2) == 32 * 2 * 8 * 128 * 2
    # DeepSeek-V3 MLA caches a 512-dim latent + 64-dim rope key per layer
    assert preset("deepseek-v3").kv_bytes_per_token(2) == 61 * (512 + 64) * 2


def test_gqa_and_mla_flags():
    assert preset("llama-3.1-70b").gqa_group == 8
    assert preset("deepseek-v3").is_mla and preset("deepseek-v3").is_moe
    assert preset("deepseek-v3").moe_layers == 58


def test_from_hf_config_roundtrip(tmp_path):
    cfg = {"hidden_size": 64, "num_attention_heads": 4, "num_hidden_layers": 2,
           "intermediate_size": 128, "vocab_size": 100, "architectures": ["LlamaForCausalLM"]}
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg))
    m = from_hf_config(json.loads(p.read_text()), "tiny")
    assert m.kv_heads == 4 and m.head_dim == 16 and not m.is_moe
    assert len(all_presets()) == 7
