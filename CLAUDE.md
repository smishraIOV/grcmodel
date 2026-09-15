# Project conventions

A GRC (Governance, Risk, Compliance) framework and an accompanying quantitative
model. `readme.md` states the thesis and the current headline results;
`docs/framework.md` gives the conceptual structure; `docs/quant-model.md` is the
model spec, the full results, and an appendix of derivations and traps;
`docs/calibration.md` says which parameters could be anchored and which
conclusions have survived recalibration.

## Working with me on this repo

- **Answer the question that was asked, briefly.** A question is not a work
  order. If the answer reveals something worth fixing, say so in a sentence and
  stop there.
- **Ask before starting anything that will take a while.** Long runs, sweeps,
  refactors and multi-file rewrites need approval first, with a rough estimate
  of what they will cost. Do not turn a simple question into a task.
- Short replies by default. Detail on request.

## Running things

Everything goes through `uv`:

```bash
uv run pytest                                 # test suite (see the note below)
uv run python scripts/run_dynamic_model.py    # multi-period, GRC as a stock
uv run python scripts/run_breakevens.py       # the decision-facing break-evens
uv run python scripts/run_seed_study.py       # solver comparison across seeds, out of sample
uv run python scripts/run_static_model.py     # Froot-Stein premium + spread sweep
uv run python -m quant.simulate               # Monte Carlo, multi-seed
uv run python -m quant.threshold              # break-even analysis
uv run python scripts/bench_profiles.py       # where the accelerator starts paying
```

If `uv run pytest` fails to spawn, `./.venv/bin/python -m pytest` works; the
suite is 130 tests and takes about four and a half minutes.

Every entry point takes `--profile {reference,cpu-fast,fast}` and prints the
profile, torch version and commit it ran under.

The truncated-BPTT-with-a-critic experiment is **parked, not rejected**, and is
not on this branch. It lives on `svg-critic` (`quant/solvers/svg.py`,
`tests/test_svg.py`, `scripts/compare_learners.py`), kept off `main` so that a
solver which has not yet earned its place is not one of the things a reader has
to rule out when a number moves. It lost at every horizon tested; the numbers
are in `docs/quant-model.md` ("Truncated BPTT with a critic", under §6). The one
live reason to expect it back is that it was only ever tested to 32 steps and
the default horizon is now 60. The other reason — the learner's instability —
has gone: that was a single gradient explosion out of a converged state, not the
long-chain growth a critic damps.

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
  moves. See `docs/quant-model.md` appendix A1.
- **Keep cost functions dimensionally consistent.** Anything raised to a power
  needs a reference level carrying the same units, or results silently depend on
  the currency the firm is denominated in. Appendix A4.
- **Floor every `logit` away from 0 and 1.** The straight-through estimators
  differentiate `logit(p)`, and an unguarded zero sends a NaN through
  `shifted.sum()` into every parameter at once. Three separate instances of this
  have been found; assume a fourth exists. Appendix A7.

## Traps that have cost real time

- **Check the regime before trusting a comparative static.** Every solver
  returns a diagnostic bundle — `annual_death_probability` (usable band roughly
  0.2%–15%), `going_concern_share`, `underinvestment_fraction`, `leverage`. At
  0 or 1 the model is in a degenerate regime and its comparative statics are
  meaningless. Appendix A6.
- **`underinvestment_fraction` is the one that goes quiet.** It is the
  Froot-Stein channel, and it switches off whenever the firm's balance sheet
  outgrows its opportunity rather than when anything is broken. Adding the
  deposit stock took it from 71% to 0.0% in one commit, with every other
  diagnostic still in band and the tests still green, because the firm simply
  stopped wanting what it could now fund. If you change what the firm can fund,
  re-check what it wants to deploy (`AnnualRates.curvature_per_period`).
- **Re-solve `initial_grc_stock` after anything that touches a hazard, a loss
  channel, or the horizon.** It is a fixed point of the whole model — the level
  at which the firm's own optimal maintenance spend replaces depreciation — so a
  change that moves optimal spend invalidates it, and a stale value degrades into
  a plausible wrong answer rather than an error. Retiring one hazard left the
  firm opening six times above its new steady state; it spent the horizon running
  the stock down and reported a budget 78% lower, which read as a headline
  finding and was an artifact. Re-solved, the fall was 5%. `GRC_STEADY_STATE` in
  `quant/params.py` holds one value per `(frequency, horizon)` and
  `steady_state_firm` raises rather than guessing for a configuration nobody has
  solved.
- **Never compare across `batch` sizes and call it a refinement.**
  `CommonRandomNumbers` draws `(horizon, batch)` row-major, so changing the
  batch changes the *scenario set* rather than sampling the same one more
  finely. Three runs at 1024/2048/4096 paths were written up as a monotone
  batch-size effect and were three draws of a lottery — at a fixed batch,
  varying the seed, two of three succeed. Vary the seed to isolate anything;
  vary the batch only to talk about cost.
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
- **An absorbing action is a gradient trap.** Once probability mass exits or
  dies there is nothing left to generate a gradient for not doing it, so a
  learner cannot find its way back. This shape has appeared four times; appendix
  A8 lists them. Initialise absorbing actions by *cumulative* probability, not
  per-period.
- **A script with no test importing it can be broken for days.** An unterminated
  string in `run_seed_study.py` survived two commits because nothing imported
  it. `tests/test_frequency.py::test_every_entry_point_still_imports` now covers
  this; keep it covering new entry points.

## Documentation

- Functions carry a pointer to the doc section they implement, and doc sections
  name the module implementing them. Keep both ends in sync when either moves.
- Numbers quoted in `docs/` must come from a real run. If a change moves them,
  re-run and update — a stale table is worse than no table.
- Distinguish what is implemented from what is aspirational. `quant-model.md`
  §1–§2 mark this per row; keep those marks honest as the code grows.
- Model parameters here are illustrative, not calibrated. Prefer reporting
  thresholds and break-evens over point estimates, and say which is which.
- Lead with shapes, quote levels as "at the current calibration". Every
  directional claim in this project has survived four recalibrations; no quoted
  level has.
