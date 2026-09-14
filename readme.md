# GRC
Initial explorations for building a GRC framework. GRC is Governance (corporate or organizational), Risk management, and Compliance. It's wider than ERM.

Working assumption is that GRC, if done well, can increase organization's value

In a narrow, academic sense of corporate finance, this is related to the work of Froot and Stein, that even risk-neutral firms can gain value from adopting active risk management. In the theoretical models, a sort of endogenous risk-aversion arises from the concavity of cost functions (financial frictions).

A full organizational level optimal control problem of maximizing firm value through GRC seems impractical. However, this can still serve as an aspirational goal for this project.

We hope to build a pragmatic, quantitative model of optimal GRC,one that can serve as an aid to executive decision making


## operating environment

The organization is a financial intermediary, like a bank on crypto rails. It has three main arms
* deposits
* lending / investments / loans
* core technology infrastructure: this is the "crypto-rails" part. All assets (deposits, loans) exist in the form of digital cryptocurrencies or tokenized real world assets (such as bonds)

Deposits go into "vaults". Investment managers invest these assets into various strategies.

## Running the model

```bash
uv run python scripts/run_dynamic_model.py  # the multi-period model -- start here
uv run python scripts/run_breakevens.py     # the decision-facing break-evens
uv run python scripts/run_seed_study.py     # is a reactive policy worth its keep?
uv run python scripts/run_static_model.py   # Froot-Stein premium + mean-preserving spread
uv run python -m quant.simulate             # Monte Carlo, reported across seeds
uv run python scripts/bench_profiles.py     # where the accelerator starts paying
uv run pytest                               # test suite
```

Each entry point takes `--profile {reference,cpu-fast,fast}` and prints the
profile, torch version and commit it ran under. Numbers quoted anywhere in
`docs/` come from `reference`.

**The two-period result.** With external finance costless the firm spends 0.77
on GRC; facing convex financing costs the same firm spends 2.56. That gap is the
Froot-Stein premium — risk management that pays for itself only because capital
is expensive to raise in bad states. It is the quantitative form of the claim
above that GRC can increase firm value, and it is where this project started.

**The model is now multi-period, and the firm can die.** It operates over eight
quarters, funds a book of about 68 with 16 of capital and a deposit base four
times that, holds roughly a quarter of its balance sheet liquid against a
depositor run, accumulates GRC as a depreciating capital stock, and faces
competing failure hazards — thin capital, a lost licence, and being unable to
pay depositors who ask — plus rare severe "cliff" events. The corresponding
headline is a threshold rather than a budget:

> A programme costing 0.66 a year, against a business worth 27.27 as a going
> concern, must cut the annual probability of failure by at least **243 basis
> points** to pay for itself.

Parameters are illustrative and uncalibrated, which is why the outputs are
break-evens — the effectiveness a programme must reach to be worth running —
rather than spend recommendations.

[`docs/status-report.md`](docs/status-report.md) is the orientation document:
what has been built, which approaches were tried and abandoned, and what is
still unresolved. [`docs/quant-model.md`](docs/quant-model.md) is the
specification and the full results.

## Numerics: a profile per workload, not one global default

The default is **CPU in float64**, and that is deliberate — but it is a
property of *this* workload, not a property of the project. `quant/numerics.py`
defines three profiles and every entry point takes `--profile`:

| profile | device | dtype | for |
|---|---|---|---|
| `reference` | CPU | float64 | analytic oracles, anything quoted in `docs/` |
| `cpu-fast` | CPU | float32 | many small deterministic solves |
| `fast` | MPS | float32 | large rollouts and grid sweeps |

**Why float64 is the default.** One assertion needs it: the zero-friction
control test checks that optimal GRC budgets land on a closed-form analytic
benchmark, and in float64 it matches to ~1e-15. That is a real check on
whether the optimizer and the objective agree. It runs on a four-state world,
so it costs nothing to keep it exact.

**Why that is not an argument against the GPU.** The precision requirement
belongs to the oracle, not to the simulation. A Monte Carlo estimate over 8192
paths carries a sampling standard error around 1e-2 relative; float32 epsilon
is 1e-7. Sampling noise dominates rounding by five orders of magnitude, which
is why `simulate.py` already reports budgets to three decimals. Measured, the
three profiles agree on the static solve to four decimals — `tests/test_numerics.py::test_profiles_agree`
holds float32-on-MPS to 1e-5 relative on firm value.

**Where the accelerator starts paying.** MPS carries a fixed dispatch cost of
roughly 160 µs per optimizer step that does not shrink with the batch; CPU
cost grows with it. So the crossover is a batch size, and it is lower than
this file used to claim (M4 Pro, torch 2.14, 200 steps, seconds):

| n | `reference` | `cpu-fast` | `fast` |
|---|---|---|---|
| 4 | **0.009** | 0.011 | 0.033 |
| 1,024 | 0.012 | **0.012** | 0.032 |
| 4,096 | 0.040 | 0.036 | **0.032** |
| 131,072 | 0.288 | 0.252 | **0.036** |
| 1,048,576 | 0.709 | 0.448 | **0.161** |

An earlier version of this section reported MPS as twice as slow at 20,000
paths. That measurement did not discard MPS's one-off compilation cost, which
is tens of milliseconds and swamps a short timing loop. Warm, the accelerator
is ahead from about 4k elements for elementwise path work and about 1k once a
network is in the loop. `quant/static.py` runs at n=4, so CPU is genuinely
right for it; the dynamic model in `docs/quant-model.md` §5 will not be.

**Randomness is always drawn on CPU in float64, then cast.** A `torch.Generator`
on MPS uses a different algorithm from CPU's: same seed, different numbers.
Drawing on CPU makes the scenario set bit-identical across profiles, so
changing profile changes arithmetic rounding and nothing else — which is what
makes comparing them a test rather than a confound.

## See also

* [`docs/framework.md`](docs/framework.md) — GRC pillars, risk taxonomy, operating-environment structure, and the firm-value definition this project uses
* [`docs/quant-model.md`](docs/quant-model.md) — the quantitative model formulation (state, controls, frictions, objective) and the current results
* [`docs/static-model-debug-notes.md`](docs/static-model-debug-notes.md) — derivations behind the formulation, and the failure modes each one guards against
* [`docs/critical-review.md`](docs/critical-review.md) — critical review of the objective and the earlier implementation, with prioritized next steps
