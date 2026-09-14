# Project conventions

A GRC (Governance, Risk, Compliance) framework and an accompanying quantitative
model. `readme.md` states the thesis; `docs/framework.md` gives the conceptual
structure; `docs/quant-model.md` is the model spec and the current results.

## Running things

Everything goes through `uv`:

```bash
uv run pytest                                 # test suite (see the note below)
uv run python scripts/run_static_model.py     # Froot-Stein premium + spread sweep
uv run python -m quant.simulate               # Monte Carlo, multi-seed
uv run python -m quant.threshold              # break-even analysis
uv run python scripts/run_dynamic_model.py    # multi-period, GRC as a stock
uv run python scripts/run_breakevens.py       # the decision-facing break-evens
uv run python scripts/run_seed_study.py       # solver comparison across seeds, out of sample
uv run python scripts/bench_profiles.py       # where the accelerator starts paying
```

If `uv run pytest` fails to spawn, `./.venv/bin/python -m pytest` works; the
suite takes about three and a half minutes.

The rejected truncated-BPTT-with-a-critic experiment is not on this branch.
It lives on `svg-critic` (`quant/solvers/svg.py`, `tests/test_svg.py`,
`scripts/compare_learners.py`), because a solver that lost at every horizon
tested should not be one of the things a reader has to rule out when a number
moves. The verdict and the numbers behind it stay in `docs/quant-model.md`
section 9 -- the finding is worth keeping even though the code is not.

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
- **Check the regime before trusting a comparative static.** Every solver
  returns a diagnostic bundle — `annual_death_probability` (usable band roughly
  0.2%–15%), `going_concern_share`, `underinvestment_fraction`, `leverage`. At
  0 or 1 the model is in a degenerate regime and its comparative statics are
  meaningless. `docs/static-model-debug-notes.md` §6.
- **`underinvestment_fraction` is the one that goes quiet.** It is the
  Froot-Stein channel, and it switches off whenever the firm's balance sheet
  outgrows its opportunity rather than when anything is broken. Adding the
  deposit stock took it from 71% to 0.0% in one commit, with every other
  diagnostic still in band and the tests still green, because the firm simply
  stopped wanting what it could now fund. If you change what the firm can fund,
  re-check what it wants to deploy (`AnnualRates.curvature_per_period`).
- **Re-solve `initial_grc_stock` after anything that touches a hazard or a loss
  channel.** It is a fixed point of the whole model — the level at which the
  firm's own optimal maintenance spend replaces depreciation — so a change that
  moves optimal spend invalidates it, and a stale value degrades into a
  plausible wrong answer rather than an error. Retiring one hazard left the firm
  opening six times above its new steady state; it spent the horizon running the
  stock down and reported a budget 78% lower, which read as a headline finding
  and was an artifact. Re-solved, the fall was 5%.
- **`optimize_constant`'s step budget has to reach the answer.** The control
  starts at `softplus(0)` and Adam moves the raw parameter by at most `lr` per
  step, so 1500 steps at lr 0.05 cannot reach an investment level near 80 —
  it converges to about 58 and looks like an economic finding. A sweep whose
  conclusions move when you raise `n_steps` was measuring the optimizer.
- **The grid solver only grades policies inside its own restriction.** It fixes
  payout and disables wind-down, so its value function assumes the firm reverts
  to that restriction after the current step. A policy using a control the grid
  holds fixed is charged against a continuation it will not follow, and scores
  badly while being better — the neural policy is worst on the grid's metric
  and best on honest evaluation. Do not read a disagreement as a learner bug
  until the learner has been confined to the same restriction.

## Documentation

- Functions carry a pointer to the doc section they implement, and doc sections
  name the module implementing them. Keep both ends in sync when either moves.
- Numbers quoted in `docs/` must come from a real run. If a change moves them,
  re-run and update — a stale table is worse than no table.
- Distinguish what is implemented from what is aspirational. `quant-model.md`
  §1–§2 mark this per row; keep those marks honest as the code grows.
- Model parameters here are illustrative, not calibrated. Prefer reporting
  thresholds and break-evens over point estimates, and say which is which.
