---
name: add-hardware
description: Add an accelerator (GPU, LPU or other) to the InferSim hardware catalog with sourced specs. Use when asked to model a new chip.
---

# Add a device to the catalog

1. Find the vendor datasheet or a reputable spec page. Record dense (non-sparse) peak FLOP/s per dtype,
   memory bandwidth, memory capacity, and scale-up link bandwidth **per direction** (NVLink totals are
   bidirectional: halve them).
2. Add a `HardwareSpec` entry to `infersim/hardware.py`:
   - put each sourced number's citation in `sources`;
   - put every number you had to estimate in `assumptions`, saying what you assumed and why.
3. Never invent a figure without listing it under `assumptions`.
4. The PostToolUse hook runs `tests/test_catalog.py`; it rejects entries with no source or with
   implausible peaks, bandwidths or ridge points. Fix the entry rather than the test.
5. Run `python bench/run_bench.py` and check the new device on the dashboard.
