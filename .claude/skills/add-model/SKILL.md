---
name: add-model
description: Add a Hugging Face model to InferSim's presets and verify its parameter count. Use when asked to model a new LLM architecture.
---

# Add a model preset

1. Copy the fields InferSim reads from the model's Hugging Face `config.json` into
   `infersim/data/<name>.json` (hidden_size, intermediate_size, num_hidden_layers,
   num_attention_heads, num_key_value_heads, head_dim, vocab_size, tie_word_embeddings, and the MoE /
   MLA fields if present). Keep `_source`.
2. Register it in `PRESET_FILES` in `infersim/models.py`.
3. Add the published total and active parameter counts to `PUBLISHED` in `tests/test_models.py`.
   If the count is off by more than 1.5%, the architecture is not modeled correctly: fix
   `ModelConfig`, do not loosen the tolerance.
