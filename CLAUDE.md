# Project conventions

A GRC (Governance, Risk, Compliance) model built as a dynamic version of Froot &
Stein (1998): a bank that decides risk management, capital budgeting and capital
structure together, every period, on an infinite horizon, with GRC as a fourth
lever. `docs/design.md` is the specification and the record of why each choice
was made; `readme.md` states the thesis and the current status;
`docs/framework.md` gives the conceptual GRC structure. The previous model
(finite horizon, hazard-based failure) lives on the `legacy` branch and is not
to be merged back.

## Working with me on this repo

- **Answer the question that was asked, briefly.** A question is not a work
  order. If the answer reveals something worth fixing, say so in a sentence and
  stop there.
- **Ask before starting anything that will take a while.** Long runs, sweeps,
  refactors and multi-file rewrites need approval first, with a rough estimate
  of what they will cost. Do not turn a simple question into a task.
- Short replies by default. Detail on request.
- **Raise new concepts before using them.** Any theoretical or numerical
  concept not already discussed (a method, theorem, distribution, solver
  technique) is named, with its source — the papers, standard literature, or
  Claude's own construction — and discussed before it appears in code or docs.
- **Abstraction level:** keep the paper's core economic drivers, at the level
  of detail where the problem solves computationally as an optimization. Do not
  chase every detail of the paper.
- **Only quantitatively meaningful measurements.** No outputs that compare the
  bank with an unrealistic counterfactual (equity raised for free, forced
  recapitalization, unlimited liability). Their qualitative point is already
  known, and their numbers mean nothing. Acknowledge such effects in words.
- **Stages need approval.** `docs/design.md` §7 lists the stages; each starts only
  when agreed.

## Running things

Everything goes through `uv`:

```bash
uv run pytest
```

Entry points take `--profile {reference,cpu-fast,fast}` (`quant/cli.py`) and
print the profile, torch version and commit they ran under.

**Decision frequency.** Parameters are stated per year; `periods_per_year`
(default 4, quarterly) sets how often the bank decides, and the solver converts
on entry (`Params.per_period()`). Use 1 for quick exploration and tests, 4 for
realistic runs; `scripts/solve_v0.py --periods-per-year 4`. Quarterly runs need
a finer equity grid near the issuance kink (`docs/design.md` §2.5).

## Layout

- `quant/params.py` — every parameter, one economic driver each, rates per year
- `quant/model.py` — the bank form, one period: primitives and the transition
- `quant/vault.py` — the vault form, one period: fees, flows, the firm's books
- `quant/quadrature.py` — deterministic rules for the Student-t shocks and jumps
- `quant/grid.py` — the shared N-dimensional grid, interpolation and policy iteration
- `quant/solver.py`, `quant/vault_solver.py` — each form's controls and objective
- `quant/results.py`, `quant/vault_results.py` — what a solved firm says
- `quant/twoperiod.py` — the two-period truncation, the paper's own structure,
  used as the oracle (a known-answer check against the paper's formulas)
- `quant/numerics.py` — profiles: device, dtype, RNG
- `scripts/solve_v0.py`, `scripts/solve_vault.py` — entry points

## Numerics

- **Every tensor is built through a `NumericsProfile`**, passed down as an
  argument, never looked up globally. A bare `torch.tensor(...)` without
  `dtype=` silently gives float32; `tests/test_numerics.py` fails the build on
  it.
- **`reference` (CPU, float64) is the default, and the only profile whose numbers
  may be quoted in `docs/`.**
- **Expectations use quadrature, not Monte Carlo,** wherever the state is small
  enough for a grid. A grid solution is then deterministic: no seed noise, and
  no scenario set that changes with the batch size.
- **Randomness, when the learner needs it, is drawn on CPU in float64 and cast
  last.** MPS generators differ from CPU's for the same seed.
- **Benchmark warm.** MPS pays a one-off compilation cost on first launch.
- **Keep forms smooth and element-wise** where the learner will differentiate.
  Constrained controls go through `softplus`, not `torch.clamp`, whose gradient
  is zero at the boundary. Floor every `logit` away from 0 and 1.
- **Keep cost functions dimensionally consistent.** Money is measured in units
  of the first-best book (`docs/design.md` §3.1), so every money parameter is a
  multiple of that. "First-best" means the frictionless optimum — what the bank
  would lend if capital were free — not the model's optimal policy, which lends
  less (`quant/params.py`).

## Oracles

- **The two-period oracle runs on pinned parameters** (`oracle_params` in
  `quant/twoperiod.py`). They exist to keep every quadrature node inside the
  region where the final-period value is exactly quadratic, which is what makes
  the checks against eqs. (9) and Prop. 2 exact. They are not plausible and must
  not be recalibrated with the model.
- **Under the Student-t the paper's mean-variance rules are approximations.**
  A gap between eq. (9) and the numerical optimum under fat tails is a result,
  not a bug. Check the normal case first.

## Lessons carried over from the legacy model

- **Check the regime before trusting a comparative static.** A constraint that
  never binds, or always binds, gives flat or reversed comparative statics while
  everything else looks healthy.
- **An absorbing action is a gradient trap.** Once mass exits there is nothing
  left to generate a gradient for not exiting. Relevant to walk-away once a
  learner is involved.
- **A script with no test importing it can be broken for days.** Keep every
  entry point imported by a test.
- **An optimizer's step budget has to reach the answer.** A result that moves
  when the budget is raised was measuring the optimizer.

## Documentation

- **US spelling throughout** — docs, docstrings, comments and identifiers:
  program, optimize, capitalize, behavior, modeling, license (noun and verb).

- Functions carry a pointer to the design section they implement. Keep both
  ends in sync when either moves.
- Numbers quoted in `docs/` must come from a real run under `reference`. If a
  change moves them, re-run and update.
- Distinguish what is implemented from what is designed but not yet built.
- **`docs/overview.md` is the plain-language companion**: no formulas, written
  for non-technical readers. Update it when a stage lands or a finding changes.
- Parameters are illustrative, not calibrated. Lead with shapes and
  break-evens; quote levels as "at the current calibration".
