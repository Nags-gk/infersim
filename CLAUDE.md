# InferSim: notes for coding agents

- Core package `infersim/` is pure Python with no dependencies, so it runs in Pyodide on the dashboard.
  Do not add imports of numpy/scipy there.
- Every hardware number needs a source or an entry in `assumptions` (see `.claude/skills/add-hardware`).
- Latencies are modeled. Never write "measured" for a number that came from the model.
- After any change: `ruff check . && pytest -q`, then `python bench/run_bench.py` to refresh `site/data/results.json`.
- A PostToolUse hook runs the catalog, model and roofline tests after every edit.
