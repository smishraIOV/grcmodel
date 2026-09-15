# GRC

A framework and a quantitative model for **G**overnance, **R**isk and
**C**ompliance — the function that sets risk limits, runs controls and keeps a
firm on the right side of its regulators. GRC is normally treated as a cost
centre. The thesis here is that GRC done well **increases firm value**, and that
this can be made quantitative rather than asserted.

The academic anchor is Froot and Stein: even a firm indifferent to risk in
principle will pay to manage it, if losing money in bad states stops it from
funding good investments. Risk management is then not insurance — it is
protection of the firm's capacity to keep operating.

The firm modelled is a **financial intermediary on crypto rails**: it takes
deposits into vaults, lends and invests them, and runs the tokenized
infrastructure those assets live on. It carries the full set of banking risks —
credit is the largest single loss channel at 62% of expected loss — plus a
second family the rails add: operational incidents at 16%, cliff events such as
a drained bridge at 13%, compliance penalties at 9%. The claim is not that the
novel risks displace the textbook ones, but that a firm of this shape faces
both and has to budget a programme against both.

> **A note on units.** Money here has no denomination. Opening capital is 16 and
> every other quantity — a book of 77, a programme costing 0.95, a franchise of
> 26.93 — is in the same arbitrary currency. Only ratios carry meaning, which is
> deliberate: the cost functions are built so the answer cannot depend on
> whether the firm is denominated in dollars or cents. Bare numbers are not
> percentages.

## Running it

```bash
uv run python scripts/run_dynamic_model.py  # the multi-period model -- start here
uv run python scripts/run_breakevens.py     # the decision-facing break-evens
uv run python scripts/run_seed_study.py     # is a reactive policy worth its keep?
uv run python scripts/run_static_model.py   # the original two-period model
uv run python -m quant.simulate             # Monte Carlo, reported across seeds
uv run python scripts/bench_profiles.py     # where the accelerator starts paying
uv run pytest                               # 130 tests, about four minutes
```

Each entry point takes `--profile {reference,cpu-fast,fast}` and prints the
profile, torch version and commit it ran under. Numbers quoted in `docs/` come
from `reference`.

## Where it started, and where it is

The original model was **two periods** — Froot, Scharfstein and Stein's own
structure, and the smallest model in which risk management can add value at
all: one period in which risk is borne, one in which the investment opportunity
arrives.

> With external finance free the firm spends 0.77 on GRC. Facing convex
> financing costs the same firm spends 2.56. The gap is the Froot-Stein
> premium — risk management that pays for itself only because capital is
> expensive to raise in bad states.

Real but limited in three ways that turned out to matter: the firm could not
die, nothing accumulated, and the convex financing cost was a stand-in for
something never represented.

**The model is now multi-period, and the firm can die.** It decides **monthly
over five years**: how much to spend on GRC across three families, how much
capital to deploy, how much to distribute, and whether to wind down. It funds a
book near 77 with 16 of capital and a deposit base about four times that, so
leverage near 4.9 means a loss of one percent of assets is a loss of five
percent of capital. Deposits persist, pay interest whether or not the book
earns, and shrink with the capital that carries them — so this period's losses
bind *next* period's lending.

It fails through three channels whose intensities add: **capital** (equity
thins to insolvency), **liquidity** (asked for deposits it cannot pay, even
after selling its book at a 35% discount), and **licence** (a breach escalates
to revocation). Against these it holds two defences and uses both: a GRC stock
that depreciates at 25% a year, and reserves — currently 22% of the balance
sheet. Annual probability of failure: 3.0%.

## What it says

**The headline has no uncalibrated effectiveness parameter in it:**

> A programme costing 0.95 a year, against a business worth 26.93 as a going
> concern, must cut the annual probability of failure by at least **354 basis
> points** to pay for itself.

Both inputs are things a board already has a view on. As an effectiveness
threshold, a compliance programme pays if a unit of spend removes about 12% of
exposure, where pure loss-reduction arithmetic demands an impossible 992%.

- **Accumulation matters more than the annual number.** Treating GRC as a stock
  rather than an expense is worth **+131% of firm value** and takes survival
  from 40% to 86%. Over two years the same comparison was worth +31% — a
  control bought in month one is still working in month sixty, and a short
  window cannot see most of that.
- **GRC is bought about equally by survival and by smaller losses.** A firm
  budgeting only for expected-loss reduction spends 38% less; the gap costs 4.0%
  of firm value and 2.0 points of annual failure probability.
- **Controls and liquidity are complements, not alternatives.** Introducing runs
  takes reserves from 10% to 22% *and* operational GRC from 0.012 to 0.028.
  Over two years controls then fell back as runs grew frequent; over five they
  rise the whole way. A buffer protects you today, a control stock has to be
  built first — so a firm with two years to live holds cash instead.
- **Spend peaks at middling capitalisation, but it is a plateau, not a point.**
  About three times as much at equity 8–16 as at 4 or 64, on every seed. The
  argmax landed at 32, 8 and 8 on three draws of the same experiment; anyone
  quoting an optimal capitalisation from a single run is quoting the draw.
- **A cheaper failure buys a smaller programme.** Spend falls 11% as recovery in
  failure rises from nothing to 80%.
- **State feedback does not beat a fixed budget.** +0.37% against seed noise of
  0.75%; the learner converges to within 0.05% of an optimised constant policy.
  Not because the method is weak — the firm's state barely moves. Raising the
  cliff-event rate, the one shock large enough to jump the state, takes the
  advantage to +9.32% monotonically.

**Trust the shapes, not the levels.** The model has been recalibrated four
times, each for a defensible reason, and each time every underlying number
moved. Every claim about a *direction* survived; no quoted level did — the
hazard break-even ran 527 → 270 → 243 → 354 bp. One direction reversed with the
horizon and no parameter touched (cash-versus-controls above), and one headline
reversed outright: "GRC is paid for almost entirely by survival" became roughly
even, once leverage made ordinary losses expensive and an asserted death rate
was retired. [`docs/calibration.md`](docs/calibration.md) sorts every parameter
by how calibratable it is and names a source for each.

Parameters are illustrative and uncalibrated, which is why the outputs are
break-evens — the effectiveness a programme must reach to be worth running —
rather than spend recommendations.

## What is unresolved

- **A run damages the deposit base, but not the franchise.** Deposits rebuild at
  a four-month half-life whether they left in a panic or drifted away. Real
  franchises do not. This is the assumption doing most of the work in the
  controls-versus-cash finding, and a slower rebuild is the likeliest route to
  operational controls being worth more than the model says.
- **The terminal constant still carries too much weight.** A constant stands in
  for "the business continues" at the horizon, and conditional on surviving,
  nothing the firm does affects it. Moving from two years to five cut its share
  of firm value from 47% to 34%. A third is still too much — and it cannot be
  fixed by solving for it, since setting it self-consistently diverges (15 → 40
  → 58, with annual failure falling 5.5% → 3.1%: a model talking itself into
  being safe). Computing it from the grid solver would cost the project its one
  independent referee.
- **Firm value is not comparable across horizons**, for the same reason.
- **The learner is the only cold-started solver**, beginning at a firm value
  near 0.5. Its instability is resolved — a gradient explosion out of a
  converged state, from the run channel's straight-through estimator, fixed by
  clipping at norm 100 — but the clairvoyant bound is warm-started and the
  learner is not.

## Numerics: a profile per workload, not one global default

`quant/numerics.py` defines three profiles and every entry point takes
`--profile`:

| profile | device | dtype | for |
|---|---|---|---|
| `reference` | CPU | float64 | analytic oracles, anything quoted in `docs/` |
| `cpu-fast` | CPU | float32 | many small deterministic solves |
| `fast` | MPS | float32 | large rollouts and grid sweeps |

**float64 is the default because one assertion needs it**: the zero-friction
control test checks optimal GRC budgets against a closed form and matches to
~1e-15. It runs on a four-state world, so exactness is free. That is not an
argument against the GPU — the precision requirement belongs to the oracle, not
the simulation. Over 8192 paths sampling error is ~1e-2 relative against a
float32 epsilon of 1e-7, and the three profiles agree on the static solve to
four decimals.

**Warm, the accelerator is ahead from about 4k elements** for elementwise path
work and about 1k once a network is in the loop (M4 Pro, torch 2.14, 200 steps,
seconds):

| n | `reference` | `cpu-fast` | `fast` |
|---|---|---|---|
| 4 | **0.009** | 0.011 | 0.033 |
| 1,024 | 0.012 | **0.012** | 0.032 |
| 4,096 | 0.040 | 0.036 | **0.032** |
| 131,072 | 0.288 | 0.252 | **0.036** |
| 1,048,576 | 0.709 | 0.448 | **0.161** |

An earlier version of this file reported MPS as twice as slow at 20,000 paths.
That measurement did not discard MPS's one-off compilation cost, which swamps a
short timing loop. `quant/static.py` runs at n=4, so CPU is genuinely right for
it; the dynamic model is not.

**Randomness is always drawn on CPU in float64, then cast.** A `torch.Generator`
on MPS uses a different algorithm: same seed, different numbers. Drawing on CPU
keeps the scenario set bit-identical across profiles, so changing profile
changes arithmetic rounding and nothing else — which is what makes comparing
them a test rather than a confound.

## See also

* [`docs/framework.md`](docs/framework.md) — GRC pillars, risk taxonomy, operating environment, and the firm-value definition this project uses
* [`docs/quant-model.md`](docs/quant-model.md) — the model specification (state, controls, frictions, objective), the current results, and an appendix of derivations and the traps each one guards against
* [`docs/calibration.md`](docs/calibration.md) — every parameter, sorted by how calibratable it is, with a source named for each

The parked truncated-backpropagation-with-a-critic experiment lives on the
`svg-critic` branch, deliberately not on `main`: it lost at every horizon
tested, and a reader chasing a number that moved should not have to rule out a
solver that never earned its place. Its verdict is in `docs/quant-model.md` §9.
