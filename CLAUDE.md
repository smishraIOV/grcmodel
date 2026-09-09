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
```

## Numerics

- **CPU and float64 by default.** The zero-friction control test asserts against
  a closed-form benchmark and needs the precision. MPS cannot allocate float64 at
  all, and at these tensor sizes it measures slower than CPU anyway — kernel
  launch overhead dominates. `quant/device.py` has the opt-in if that changes.
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
