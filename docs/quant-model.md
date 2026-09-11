# Quantitative GRC model — formulation and current implementation

This document turns `readme.md`'s aspirational goal into an explicit formulation
and records what is actually built. It builds on the choices in
[`framework.md`](framework.md) — in particular the firm-value definition in its §3.

## 1. State variables

| state | symbol | implemented? |
|---|---|---|
| Capital / equity | $E$ | **yes** — `FirmState.equity`, evolving |
| GRC capital stock, by family | $G_c, G_o, G_k$ | **yes** — `FirmState.grc_stock` |
| Internal wealth after the risk draw | $w$ | **yes** — derived, `StandardDynamics.step` |
| Alive / failed | — | **yes** — `FirmState.alive`, absorbing |
| Deposit volume | $D$ | no |
| Portfolio composition / risk metrics | — | no |
| Liquidity buffer | $L_{\text{buf}}$ | no |
| Operational-risk indicator | — | partial — enters as a risk-family exposure, not as evolving state |
| Compliance-risk score | — | partial — enters as a breach probability, not as evolving state |

The model is a two-period problem, so nothing here evolves over time yet. The
unimplemented rows are the honest gap between this section and the code.

## 2. Controls

| control | symbol | implemented? |
|---|---|---|
| GRC investment, split by risk family | $g_c,\ g_o,\ g_k$ | **yes** — `GrcBudgets` |
| Capital allocation to investment, state-contingent | $I$ | **yes** — per-path, chosen after the shock |
| Target liquidity buffer | — | no |
| Deposit pricing / redemption-term incentives | — | no |

## 3. Frictions, costs and the investment channel

**Convex cost of external finance.** With internal wealth $w$ and desired
investment $I$, the firm raises $e = \max(0,\ I - w)$ externally at

$$
P(e) = \sigma\, K \left( \frac{e}{K} \right)^{\gamma}
$$

where $\gamma$ is the financing convexity and $\sigma$ a switch that turns the
friction off ($\sigma = 0$) or on ($\sigma = 1$) — which is how the Froot-Stein
premium in §6 is measured.

$K$ is a reference distress level carrying money units. It is not decoration:
without it the expression is $e^{\gamma}$, which subtracts
$\text{money}^{\gamma}$ from money and makes the model's answer depend on whether
the firm is denominated in dollars or cents.

**Failure intensity.** The firm can die, and the rate at which it does depends
on how well capitalized it is:

$$
h(E) = \frac{\bar{h}}{4} \exp\!\left( \frac{\kappa^{\star} - \kappa}{s} \right),
\qquad \kappa = E / E_0
$$

Implemented in `quant/hazard.py`. Exponential rather than logistic because a
logistic saturates, and at its base rate even a firm with deeply negative
equity would survive the quarter — which is not a description of insolvency.
The exponent is a ratio over a dimensionless scale, so it stays unit-invariant.

Death is not sampled. Each path carries the *probability* it is still alive,
accumulated in log space, and value is an expectation over survival. That is
what keeps the objective differentiable in everything driving the hazard: a
sampled death is a step function of equity and carries no gradient. Measured,
$dS/dE$ is non-zero from $\kappa = 1$ down to about $\kappa = -0.5$, against
exactly zero everywhere under a hard barrier.

**The investment opportunity.** Deployed capital returns

$$
F(I) = A\, S \left( 1 - e^{-I/S} \right)
$$

concave, with unconstrained optimum $I^{\star} = S \ln A$. This is the term that
makes risk management *value-adding* rather than merely loss-avoiding. Froot,
Scharfstein & Stein (1993) requires three ingredients — costly external finance,
an investment opportunity whose funding depends on internal wealth, and a
risk-management instrument. A model with only the first and third can say
"losses are expensive, reduce them"; it cannot support `readme.md`'s claim that
GRC *increases firm value*.

**GRC acts on the moment that each risk family actually moves.** All three
families were previously summed and mitigated by one uniform factor, which made
`framework.md`'s taxonomy mathematically inert.

| family | GRC reduces | why |
|---|---|---|
| credit | the **mean** loss | better underwriting shifts the whole distribution |
| operational | the **severity** | controls contain an incident; they do not prevent it |
| compliance | the **probability** | a programme prevents breaches; it does not soften the penalty |

Mitigation is $e^{-\alpha g}$ in all three cases — diminishing returns, so each
additional unit of spend removes less than the last.

**Reducing a probability is not differentiable through a sample.** The
compliance channel cannot work by shrinking a drawn Bernoulli indicator.
Indicators are drawn once at a base probability $p_0$, and each path is reweighted
by the likelihood ratio

$$
\frac{p(g_k)}{p_0} \ \text{ on breach paths,} \qquad
\frac{1 - p(g_k)}{1 - p_0} \ \text{ otherwise,}
\qquad p(g_k) = p_0\, e^{-\alpha_k g_k}
$$

This is unbiased, differentiable in $g_k$, and keeps breaches discrete — which
matters, because it is the spread those breaches create that the convex premium
prices.

## 4. Objective

At $t_0$ the firm chooses GRC budgets $g_c, g_o, g_k$. At $t_1$ it observes the
risk draw, leaving internal wealth

$$
w = E - \sum_{f} g_f - L(g)
$$

and then chooses investment $I$, funding $e = \max(0,\ I - w)$ externally. The
problem is

$$
\max_{g,\ I(\cdot)} \ \mathbb{E}\Big[\, w - I + F(I) - P\big(\max(0,\ I - w)\big) \,\Big]
$$

The expectation is probability-weighted over paths, with the weights themselves
depending on $g_k$ per §3.

Constraints from the original formulation — solvency, a liquidity buffer against
a modelled run, and regulatory limits — are **not** imposed as hard constraints.
Distress enters through the convex financing cost instead. Adding them is future
work, not something the current code approximates.

## 5. Staged path

The destination is a multi-period model of a firm that can die, solved
numerically with a learned policy. Periods are **quarterly**. The two-period
convex-cost model is being replaced outright rather than kept as a special
case — but the closed-form benchmark in §6 survives the replacement, because
setting the discount factor to zero removes the continuation value and leaves
exactly the one-period problem that benchmark solves.

| stage | what it adds | status |
|---|---|---|
| 0 | Numerics profiles, provenance | **built** — `quant/numerics.py` |
| 1 | Environment seam: state, action, dynamics, policy, one evaluator | **built** — `quant/env/` |
| 2 | Horizon, discounting, GRC as a depreciating stock | **built** — `EnvConfig.quarterly` |
| 3a | Smooth survival hazard replacing the hard barrier | **built** — `quant/hazard.py` |
| 3b | GRC acting on the hazard, not only on losses | not started |
| 3c | Discrete cliff events; licence loss as absorbing | not started |
| 3d | Abandonment / orderly wind-down option | not started |
| 3e | Diagnostic bundle and the break-even outputs | not started |
| 4 | Grid / fitted value iteration on a reduced config | not started |
| 5 | The learner (truncated-BPTT actor-critic) | not started |

Survival moved ahead of the grid solver after stage 2: the hard insolvency
barrier turned out to be required for the multi-period problem to be finite at
all, and a grid solver's inner maximization would inherit its gradient
pathology.

Stages 1–3 keep the convex financing cost as scaffolding so each step has an
exact regression target; stage 4 is where it is replaced and where §6 is
withdrawn.

**Stage 1, as built.** `quant/env/` factors the single-expression objective in
`quant/model.py` into a per-period transition (`dynamics.py`), a state
(`state.py`), an action and the one squashing map (`actions.py`), pre-drawn
common random numbers (`shocks.py`), and a single `evaluate` that scores every
solver (`env.py`). Two properties carry the design:

- **Non-anticipativity is a property of the policy, not the dynamics.**
  `rollout` hands a policy the state and then draws the shock, so a policy that
  sees only state cannot condition on what has not happened. That is what lets
  one environment serve both an honest state-feedback policy and the
  deliberately clairvoyant bound below.
- **`optimize_policy` was never a policy optimizer.** Its free per-path
  investment vector is a clairvoyant choice — correct at one period, because
  all uncertainty resolves before the single decision, but an upper bound at
  any longer horizon. It is now `solvers/pathwise.perfect_information_bound`,
  which gives `V(any implementable policy) ≤ V* ≤ V_PI` for free. A learner
  reporting a value above `V_PI` has a bug.

### Fork not yet taken: firm maturity

The model will eventually split in two, and the split changes the horizon:

- **Mature organization** — an existing deposit franchise, survival dominated
  by run and licence hazards, infinite-horizon discounted with a stationary
  policy. This is what the stages above build.
- **Young / pre-revenue** — no deposit franchise, so continuation value is not
  an annuity on deposits. Survival is runway-driven, the terminal object is an
  exit or acquisition option rather than a perpetuity, and compliance GRC acts
  mainly as a gate on growth rather than as death-avoidance. Shorter, finite
  horizon.

Nothing built so far preempts this. `TerminalValue` (`quant/env/reward.py`) is
already a strategy object and `horizon` is configuration rather than an
assumption baked into a solver.

## 6. Results

Run `uv run python scripts/run_static_model.py`.

### The Froot-Stein premium

The same firm solved twice, with the financing friction off and on. The gap is
GRC spend that exists *only* because external finance is costly.

| budget | no friction | closed form | with friction | premium |
|---|---|---|---|---|
| credit | 0.6077 | 0.6077 | 0.9932 | 0.3854 |
| operational | 0.1626 | 0.1626 | 0.8935 | 0.7309 |
| compliance | 0.0001 | 0.0000 | 0.6686 | 0.6685 |
| **total** | **0.7705** | **0.7704** | **2.5553** | **1.7848** |

Firm value 16.0768 (no friction) → 9.8342 (friction). The constraint binds on
52.0% of probability mass.

The "closed form" column is the analytic risk-neutral optimum

$$
g_f^{\star} = \max\left( 0,\ \frac{\ln(\alpha_f X_f)}{\alpha_f} \right)
$$

for family $f$ with exposure $X_f$, which the frictionless solve reproduces to
four decimals — a check that the optimizer and the objective agree with analysis.

The compliance row is the sharpest case: a firm facing costless external finance
would run **no compliance programme at all** at these parameters, because a unit
of spend buys back less than a unit of expected loss. The same firm facing convex
financing costs spends 0.67. The entire programme is Froot-Stein premium.

### Mean-preserving spread

The mechanism itself. Credit loss mean held at 4.0 while its spread widens:

| spread | credit GRC | total GRC | firm value |
|---|---|---|---|
| 0.00 | 0.6077 | 2.1172 | 10.4183 |
| 0.50 | 0.6938 | 2.2618 | 10.2836 |
| 1.00 | 0.7891 | 2.3652 | 10.1387 |
| 1.50 | 0.8890 | 2.4572 | 9.9885 |
| 2.00 | 0.9932 | 2.5553 | 9.8342 |

At zero spread, credit GRC sits exactly on the risk-neutral benchmark (0.6077):
with a deterministic credit loss there is nothing for the convex premium to act
on through that channel. Every unit above it is bought by spread alone.

### Multi-period: GRC as a stock rather than an expense

`uv run python scripts/run_dynamic_model.py`. Eight quarters, 2048 paths,
discount 0.9809/quarter, insolvency barrier at zero. GRC capital follows
$G' = (1-\delta)G + g$ with mitigation $e^{-\alpha G}$ applied to the stock;
$\delta = 1$ recovers the static model's assumption that spend buys exactly
one period of protection.

| GRC decay | solver | value | spend/qtr | end stock | survives |
|---|---|---|---|---|---|
| 1.000 | constant | 14.1301 | 0.8091 | 0.8091 | 38.8% |
| 1.000 | neural | 14.6745 | 0.7786 | 0.5873 | 39.9% |
| 1.000 | PI bound | 26.2992 | 1.1870 | 1.6826 | 56.6% |
| 0.069 | constant | 25.1594 | 2.5778 | 16.2519 | 59.7% |
| 0.069 | neural | 35.5962 | 2.5087 | 11.8188 | 75.9% |
| 0.069 | PI bound | 44.3781 | 1.7309 | 9.1730 | 78.0% |

Persistence is worth **+143% of firm value** (14.67 → 35.60) and takes survival
from 40% to 76%. The firm also spends **more** per quarter, not less
(0.56 → 2.52): a unit of spend now protects every later quarter, so more of it
is worth buying. That is the intertemporal content the static model could not
express — it is not the one-period answer repeated.

Three solvers, one `evaluate`. `constant` is the best state-independent action
and the floor a learner must clear; `neural` is state feedback trained by
backpropagation through the rollout; `PI bound` chooses every control per path
with the future known and is therefore not implementable. The sandwich
$V(\text{constant}) \le V(\text{neural}) \le V_{PI}$ is asserted on every run.

Two things found by building this, both invisible at one period:

- **The convex financing cost is unbounded below over a horizon.** The premium
  is $K(e/K)^\gamma$ with $e = I - w$, so once equity goes negative the
  shortfall grows, the premium grows faster, and equity roughly squares each
  quarter — reaching $-2 \times 10^{7}$ by quarter three and overflowing
  float64 by quarter eight. The insolvency barrier is what makes the problem
  well posed, so survival does real work here for arithmetic reasons before it
  does for economic ones.
- **A hard barrier creates a gradient desert.** Limited liability means a
  failed firm is worth a constant, so a dead path's gradient is exactly zero
  and carries no signal about how death might have been avoided. Cold-started,
  the perfect-information solver converges below a plain constant policy (14.4
  against 25.4), which is not a bound at all; it is warm-started from the
  constant policy for that reason. The smooth hazard in §3 narrows the desert
  but does not close it — survival still underflows to zero below about
  $\kappa = -0.5$, and a cold start drives paths there within a few quarters
  (3.2 against 25.2). Warm-starting is required under either death channel.

### Monte Carlo

`uv run python -m quant.simulate` — 8192 paths × 5 seeds, reported with a 95%
confidence interval because the severity distributions are heavy-tailed and a
single-seed answer quoted to four decimals overstates what the sample supports.

### Break-even analysis

`uv run python -m quant.threshold`. Nothing here can calibrate $\alpha$, so the
decision-useful output is a threshold rather than a point estimate:

| family | exposure $X_f$ | risk-neutral break-even $\alpha$ | break-even with friction |
|---|---|---|---|
| credit | 4.00 | 0.250 | 0.227 |
| operational | 3.50 | 0.286 | 0.231 |
| compliance | 1.50 | 0.667 | 0.265 |

The risk-neutral column is $\alpha = 1 / X_f$, the point below which a unit of
spend buys back less than a unit of expected loss.

Read the compliance row as: *this programme pays provided you believe a unit of
spend removes at least 26.5% of exposure, even though pure expected-loss
reduction would demand 66.7%.* The 0.401 gap is the Froot-Stein premium expressed
as a threshold, and it is largest for compliance because that is where the tail
sits.

## 7. Superseded result

An earlier version reported that optimal GRC investment rises with financing-cost
convexity (2.92 → 4.32). That result did not survive dimensional correction of the
cost function and a parameterization that is not permanently insolvent; see
[`critical-review.md`](critical-review.md) F1–F2. The convexity comparative static
is regime-dependent and is no longer claimed.

## 8. Not done

- Calibrating any distribution or $\alpha$ to real data. Every parameter is illustrative.
- The unimplemented state variables and controls in §1–§2.
- Hard solvency / liquidity / regulatory constraints (§4).
- §5.3, the MDP.
