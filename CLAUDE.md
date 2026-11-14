# Project conventions

A GRC (Governance, Risk, Compliance) model built as a dynamic version of Froot &
Stein (1998): a crypto-native firm that decides risk management, capital budgeting and capital
structure together, every period, on an infinite horizon, with GRC as a fourth
lever. `docs/design.md` is the specification and the record of why each choice
was made; `readme.md` states the thesis and the current status;
`docs/framework.md` gives the conceptual GRC structure. The previous model
(finite horizon, hazard-based failure) lives on the `legacy` branch and is not
to be merged back.

## Working with me on this repo

- **Write plainly. Avoid using jargon, unless the term is used in everyday language or very common given the context** 
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
  firm with an unrealistic counterfactual (equity raised for free, forced
  recapitalization, unlimited liability). Their qualitative point is already
  known, and their numbers mean nothing. Acknowledge such effects in words.
- **Stages need approval.** `docs/design.md` §7 lists the stages; each starts only
  when agreed.

## Running things

Everything goes through `uv`:

```bash
uv run pytest                          # the core and the quick research checks: seconds
uv run pytest -m slow                  # the solver tests: about five minutes
uv run pytest -m "slow or not slow"    # everything
```

Entry points take `--profile {reference,cpu-fast,fast}` (`quant/cli.py`) and
print the profile, torch version and commit they ran under.

**Decision frequency.** Parameters are stated per year; `periods_per_year`
(default 4, quarterly) sets how often the firm decides, and the solver converts
on entry (`Params.per_period()`). Use 1 for quick exploration and tests, 4 for
realistic runs. Quarterly runs need
a finer equity grid near the issuance kink (`docs/design.md` §4.4).

## Layout

Two paths share one core (`docs/plan-tool.md`). **The core** is the one-period
accounting and the scenarios; the what-if tool is built on it. **Research** is
everything that searches for the best policy. The core never imports from
research (`tests/test_layout.py`).

The core:

- `quant/params.py` — every parameter, one economic driver each, rates per year
- `quant/core.py` — primitives every form shares: issuance cost, GRC, jumps
- `quant/firm.py` — the firm's own books (stablecoins, BTC, hedge, runway,
  capital floor) and BTC's market (price, crashes, scenario rule)
- `quant/lending.py` — the stablecoin lending vault, one period
- `quant/btc_vault.py` — the BTC yield vault, one period
- `quant/integrated.py` — both vaults joined through one loan book, one period
- `quant/quadrature.py` — deterministic rules for the Student-t shocks and jumps
- `quant/profile.py` — a firm entered in dollars and rates per year, mapped onto
  the parameters, the firm's state and the scale, and back
- `quant/policy.py` — the firm's policy as an input: checked against the
  model's constraints, and turned into the one-period decision
- `quant/readout.py` — what one period holds for a firm under a policy, in
  dollars, with no solver
- `quant/stress.py` — named stress scenarios: one outcome worked through the
  accounting, with what it leaves the firm, the vaults and their investors
- `quant/paths.py` — the firm over several years on random paths, under a rule
  the user sets
- `quant/compare.py` — two policies for the same firm, side by side
- `quant/numerics.py` — profiles: device, dtype, RNG
- `quant/cli.py` — what every entry point shares: `--profile` and the run header

- `scripts/whatif.py` — print that read-out, the stress scenarios and the
  multi-year paths for the example firm
  (`uv run python -m scripts.whatif --ltv 0.7 --stress --years 5`), or two
  policies side by side (`--ltv 0.5 --b-ltv 0.7 --years 5`)

Research (`quant/research/`, tests in `tests/research/`, entry points in
`scripts/research/`):

- `grid.py` — the shared N-dimensional grid, interpolation and policy iteration
- `lending_solver.py`, `btc_vault_solver.py` — each vault alone on a 3-D grid
- `integrated_solver.py` — the integrated firm on a coarse 4-D grid
- `lending_results.py`, `integrated_results.py` — what a solved firm says: the
  read-out and the cost ratios
- `twoperiod.py` — the two-period truncation of the paper's own bank,
  self-contained, used as the oracle (a known-answer check against the paper's formulas)
- `scripts/research/integrated.py` — solve the integrated firm and print its
  read-out (`uv run python -m scripts.research.integrated --periods 1`)

## Numerics

- **Every tensor is built through a `NumericsProfile`**, passed down as an
  argument, never looked up globally. A bare `torch.tensor(...)` without
  `dtype=` silently gives float32; `tests/test_numerics.py` fails the build on
  it.
- **`reference` (CPU, float64) is the default, and the only profile whose numbers
  may be quoted in `docs/`.**
- **Expectations use quadrature, not Monte Carlo,** wherever the state is small
  enough for a grid. A grid solution is then deterministic: no seed noise, and
  no scenario set that changes with the batch size. The tool's one-period
  read-out is quadrature too. Its multi-year paths (`quant/paths.py`) are
  Monte Carlo, from a seeded generator, and are checked against the read-out
  over one period.
- **Randomness, when the learner needs it, is drawn on CPU in float64 and cast
  last.** MPS generators differ from CPU's for the same seed.
- **Benchmark warm.** MPS pays a one-off compilation cost on first launch.
- **Keep forms smooth and element-wise** where the learner will differentiate.
  Constrained controls go through `softplus`, not `torch.clamp`, whose gradient
  is zero at the boundary. Floor every `logit` away from 0 and 1.
- **Keep cost functions dimensionally consistent.** Money is in dollars at a
  scale the business fixes (`docs/design.md` §4.1), so every money parameter is
  a multiple of that.

## Oracles

- **The two-period oracle runs on pinned parameters** (`oracle_params` in
  `quant/research/twoperiod.py`). They exist to keep every quadrature node inside the
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
