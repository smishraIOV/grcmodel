# Project conventions

A GRC (Governance, Risk, Compliance) framework and an accompanying quantitative
model. `readme.md` states the thesis; `docs/framework.md` gives the conceptual
structure; `docs/quant-model.md` is the model spec and the current results.

## Running things

Everything goes through `uv`:

```bash
uv run pytest                                 # test suite
uv run python scripts/run_static_model.py     # Froot-Stein premium + spread sweep
uv run python -m quant.simulate               # Monte Carlo, multi-seed
uv run python -m quant.threshold              # break-even analysis
uv run python scripts/run_dynamic_model.py    # multi-period, GRC as a stock
uv run python scripts/bench_profiles.py       # where the accelerator starts paying
```

Every entry point takes `--profile {reference,cpu-fast,fast}` and prints the
profile, torch version and commit it ran under.

## Numerics

- **Every tensor is built through a `NumericsProfile`** (`quant/numerics.py`),
  passed down as an argument — never looked up globally. A bare `torch.tensor(...)`
  without a `dtype=` silently gives float32; that bug lived in `static.py` for two
  commits and `tests/test_numerics.py` now fails the build on it.
- **`reference` (CPU, float64) is the default, and the only profile whose numbers
  may be quoted in `docs/`.** The closed-form control test needs the precision;
  it runs on four states, so exactness there is free. That is not an argument
  against `fast` (MPS, float32) for large rollouts — sampling error dominates
  float32 rounding by five orders of magnitude. Pick per workload, and re-measure
  with `scripts/bench_profiles.py` rather than trusting a remembered crossover.
- **Randomness is drawn on CPU in float64, transformed there, and cast last.**
  MPS generators differ from CPU's for the same seed. Drawing on CPU keeps the
  scenario set bit-identical across profiles, so a profile change alters only
  arithmetic. Use `profile.draw_uniform(...)`, not `torch.rand`.
- **Benchmark warm.** MPS pays a one-off compilation cost on first launch; timing
  a short loop without discarding it is how this repo once concluded MPS was
  twice as slow as CPU.
- **Constrained controls go through `softplus`, never `torch.clamp`.** Clamp's
  gradient is zero exactly at the boundary, so a control initialized at zero never
  moves. See `docs/static-model-debug-notes.md` §1.
- **Keep cost functions dimensionally consistent.** Anything raised to a power
  needs a reference level carrying the same units, or results silently depend on
  the currency the firm is denominated in. §4 of the same document.
- **Check the regime before trusting a comparative static.** `optimize_policy`
  returns `constrained_fraction`; if it is 0 or 1 the model is in a degenerate
  regime and its comparative statics are meaningless. §6.

## Documentation

- Functions carry a pointer to the doc section they implement, and doc sections
  name the module implementing them. Keep both ends in sync when either moves.
- Numbers quoted in `docs/` must come from a real run. If a change moves them,
  re-run and update — a stale table is worse than no table.
- Distinguish what is implemented from what is aspirational. `quant-model.md`
  §1–§2 mark this per row; keep those marks honest as the code grows.
- Model parameters here are illustrative, not calibrated. Prefer reporting
  thresholds and break-evens over point estimates, and say which is which.
