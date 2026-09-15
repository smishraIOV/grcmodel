# Quantitative GRC model — formulation and current implementation

This document turns `readme.md`'s aspirational goal into an explicit formulation
and records what is actually built. It builds on the choices in
[`framework.md`](framework.md) — in particular the firm-value definition in its §3.

## 1. State variables

| state | symbol | implemented? |
|---|---|---|
| Capital / equity | $E$ | **yes** — `FirmState.equity`, evolving |
| Opening GRC capital | $G_{f,0}$ | **yes** — `FirmParams.initial_grc_stock`, set to the steady state |
| GRC capital stock, by family | $G_c, G_o, G_k$ | **yes** — `FirmState.grc_stock` |
| Internal wealth after the risk draw | $w$ | **yes** — derived, `StandardDynamics.step` |
| Alive / failed | — | **yes** — `FirmState.alive`, absorbing |
| Deposit volume | $D$ | **yes** — `FirmState.deposits`, a stock that persists, costs interest and shrinks with the capital ratio |
| Portfolio composition / risk metrics | — | no |
| Liquidity buffer | $L_{\text{buf}}$ | **yes, as a residual** — reserves are what the firm chose not to lend; they earn a rate, absorb withdrawals at par, and the firm now holds them deliberately (22% of the balance sheet) because a run can arrive |
| Operational-risk indicator | — | partial — enters as a risk-family exposure, not as evolving state |
| Compliance-risk score | — | partial — enters as a breach probability, not as evolving state |

The unimplemented rows are the honest gap between this section and the code.

## 2. Controls

| control | symbol | implemented? |
|---|---|---|
| GRC investment, split by risk family | $g_c,\ g_o,\ g_k$ | **yes** — `FirmAction.grc` |
| Orderly wind-down | — | **yes** — `FirmAction.abandon` |
| Payout / retention | — | **yes** — `FirmAction.payout` |
| Capital allocation to investment, state-contingent | $I$ | **yes** — per-path, chosen after the shock |
| Target liquidity buffer | — | **yes, implicitly** — chosen through how much of the balance sheet to deploy rather than as a separate control. A bank does not pick a buffer and a book independently; it picks a book and lives with the remainder |
| Deposit pricing / redemption-term incentives | — | no — the firm takes the deposit base its capital supports at a fixed rate |

## 3. Frictions, costs and the investment channel

**The liability side.** The firm funds its book with its own capital plus a
stock of deposits, and cannot deploy more than the two together:

$$
I_t = \min\!\big(I_t^{\text{desired}},\ w_t + D_t\big)
$$

No free parameter is needed to state this, because the balance sheet already
states it. The deposit stock is what carries the economics:

$$
\bar{D}(E) = \lambda\, E \cdot \varsigma(E),
\qquad \varsigma(E) = \text{sigmoid}\!\left(\frac{E/E_0 - \kappa_{\text{mkt}}}{s}\right),
\qquad D_{t+1} = D_t + \theta\,\big(\bar{D}(E_{t+1}) - D_t\big)
$$

Three things follow, and none of them were expressible before.

**Leverage amplifies ordinary losses.** A one percent loss on a book funded
five-to-one against capital is a five percent loss of capital. That is the
textbook route by which *loan defaults* — not exotic risks — take an
intermediary down, and with no liabilities the model had no way to carry it.

**Funding has a price.** Deposits pay $r_D$ per period on the base outstanding,
whether or not the book earns. Funding the firm has not lent earns $r_R < r_D$,
so surplus deposits are parked rather than burned; without that term the model
punishes a firm for having a franchise, and measured at 2.5× leverage it cost
5% of firm value purely for carrying deposits it had no use for. The half-point
spread between the two rates is what will make a liquidity buffer a decision
rather than a free good once withdrawals exist.

**Funding withdraws as the firm weakens, with a lag.** Capacity $\bar{D}$ falls
in the capital ratio on both terms at once — a smaller multiple of a smaller
number, times a market that is closing — and the stock moves toward it at speed
$\theta$ rather than arriving. So this quarter's losses bind *next* quarter's
lending. That lag is the Froot-Stein underinvestment channel arriving through
the liability side: a bad draw does not make investment expensive, it makes
investment unavailable, so risk management protects the firm's capacity to
invest rather than only its cash.

The partial adjustment is what makes $D$ a state variable rather than a formula
in $E$. At $\theta = 1$ the liability side collapses back into the asset side
and nothing is outstanding that could run.

**The run.** A discrete event, not a smooth outflow. Its probability is

$$
p_{\text{run}} = \frac{\lambda_r}{m}\, e^{-\alpha_o G_o}
  \cdot \exp\!\left(\frac{\tau_r - \kappa}{s_r}\right),
\qquad \kappa = \frac{E - g - L}{E_0}
$$

— three separate claims. $\lambda_r$ is how often an incident becomes public at
all. The GRC term is the only place operational controls still buy survival,
and they buy it by making the run rarer rather than by being told they lower a
death rate. The capital term is evaluated on equity **after** the quarter's
losses, which is what makes this a run rather than a weather event: losses thin
the capital, thin capital draws depositors out, meeting them forces a fire sale,
and the fire sale thins the capital further. It is deliberately gentler and
earlier than the death hazard's capital term — depositors do not wait for
insolvency, they leave on the suspicion of it.

Differentiated by the same straight-through relaxation the cliff channel uses,
for the same reason: the cost of a run is convex in its size, because a small
one comes out of reserves and a large one comes out of a fire sale, so replacing
the event by its expectation deletes the state it exists to represent.

**Meeting it.** A withdrawal $W$ is paid from reserves at par and from the
unmatured book at a haircut $h$: raising $S$ costs $S/(1-h)$ of book and
destroys $S\,h/(1-h)$. That is the only loss in the model caused by the *timing*
of an obligation rather than by an asset going wrong, and it is what makes an
idle reserve worth holding — reserves earn less than the book and still cost
deposit interest, so a firm ignoring runs would hold none.

**Whether it is fatal, in closed form.** Everything the firm can raise is
$E + D - hB$, so a withdrawal of $\omega D$ is meetable exactly when
$hB \le E + D(1-\omega)$, and a *full* run when

$$B \le E/h$$

independent of the deposit base, which cancels. This is worth stating because
it is a trap: at $h = 0.20$ the boundary is a book of 80 against equity of 16,
which is precisely where the leverage limit puts the firm — so no run was ever
quite unmeetable, and the model concluded that operational controls are
worthless and one should simply hold cash. True, but only because the buffer
was a complete defence. $h = 0.35$ puts the boundary at 46, inside the firm's
actual balance sheet.

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

**Failure intensity.** The firm can die, by three competing routes whose
intensities add:

| channel | driver | GRC acting on it |
|---|---|---|
| capital | equity falls toward insolvency | indirect — GRC leaves more equity behind |
| liquidity | the firm was asked for its deposits and could not pay | indirect — operational GRC makes the run rarer |
| licence | a breach escalates to revocation | **compliance GRC, directly** |
| ~~operational~~ | ~~an incident becomes public, depositors leave~~ | **retired — see below** |

Credit has no hazard channel of its own, deliberately: bad underwriting erodes
equity, and equity is already the capital channel's argument, so giving it one
would count the same mechanism twice.

**The liquidity channel is not an assumed rate.** Its intensity is
$\lambda_\ell / m$ times the *share* of a withdrawal the firm could not meet,
which the dynamics computes from reserves and what a fire sale could raise. At
the full share it is 25 a year, so a quarter's survival is 0.2% — a bank that
gates withdrawals is finished, and the number says so. It is deliberately not
redundant with the capital channel: the shortfall also lands on equity and is
priced there, but **insolvency and illiquidity are different failures**, and a
firm can be solvent and unable to pay. That distinction is the whole reason to
model a run.

**The operational channel is retired, and that is a replacement rather than a
deletion.** It asserted three things in one parameter: that an incident becomes
public, that depositors leave, and that the firm therefore dies. There were no
depositors in the model to leave, so the middle claim had no representation and
the third followed from a rate. Depositors now leave for real, and whether the
firm dies depends on whether it can meet them.

That change cost the model a result, and §6 reports it: under the asserted rate,
operational GRC was **72% of the entire GRC budget**.

The two GRC-reducible channels are $\bar{h}_f e^{-\alpha_f G_f} / 4$, reusing
each family's existing $\alpha$ rather than introducing a second effectiveness
parameter per family — nothing here can calibrate one, let alone two (§8). This
is what makes GRC buy *survival* rather than only smaller losses, which is the
difference between a programme justified by expected-loss reduction and one
justified by the franchise it protects.

The capital channel depends on how well capitalized the firm is:

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

### Why the model is differentiable at all

Almost nothing here is naturally smooth. A firm either suffers a bridge exploit
or does not, either survives the quarter or does not, either winds down or does
not. Every one of those is a step function, and a step function has zero
gradient almost everywhere and none at the step. **The differentiability is
engineered, mechanism by mechanism**, and it is what lets the whole project use
exact gradients instead of sampled ones.

| what is naturally discrete | what the model does instead | cost |
|---|---|---|
| **Death** | Not sampled at all. Each path carries the *probability* it is still alive, accumulated in log space; value is an expectation over survival | none — this is strictly better than sampling |
| **Cliff strike, depositor run** | Straight-through estimator: forward pass is the true hard indicator, backward pass differentiates a tempered sigmoid of the same threshold | the gradient is biased, by a factor the temperature controls |
| **Compliance breach** | Likelihood-ratio reweighting: draw once at a base probability, reweight each path by $p(g)/p_0$ | unbiased, but the weights multiply along the path so the effective sample size decays |
| **The wind-down decision** | Relaxed from a binary choice to a probability in $[0,1]$ | **none** — see below |
| **Non-negative spend and investment** | `softplus`, never `clamp` | a control whose true optimum is exactly zero is only approached asymptotically |
| **Failing to pay depositors** | Deliberately *not* a second hard death branch. The unmet share feeds a hazard channel that is linear in it | none — arguably the cleanest instance of the pattern |

**Every random draw is reparameterised**, which is the precondition for all of
the above. Continuous losses are inverse-CDF transforms of a pre-drawn uniform
($-\mu \ln(1-u)$), and the Bernoullis are thresholds against *fixed* base
probabilities. So **nothing the policy controls sits inside a sampler** — the
two probabilities GRC does move are precisely the two whose raw uniforms are
handed to the dynamics rather than thresholded in the sampler, so that the
relaxation can see the control.

**Death is the one that matters most.** A sampled death is a step function of
equity, so a firm that died tells the optimiser nothing about how it might have
avoided dying. Carrying survival as a probability gives a gradient everywhere:
measured, $dS/dE$ is non-zero from $\kappa = 1$ down to about $\kappa = -0.5$,
against exactly zero everywhere under a hard barrier. This is the whole reason
the barrier was replaced by a hazard.

**A liquidity failure was the last thing that could have become a hard branch,
and did not.** A firm that cannot pay depositors even after liquidating its
whole book has plainly failed, and the obvious implementation is a second death
test. Instead the shortfall stays owed — which drives equity sharply negative
where the capital hazard already prices it — and the *share* it could not meet
feeds a hazard channel linear in that share. A discrete failure event converted
into a smooth intensity.

**Two different devices for the same problem, deliberately.** Reweighting is
unbiased but its variance compounds along the trajectory; a straight-through
estimator is biased but its bias does not. Compliance got reweighting when the
horizon was short. The cliff and the run arrived later and at longer horizons,
where compounding variance would have been fatal — so they got the estimator
whose error stays put. Neither device is better; they fail in different
directions and the horizon decides which failure is affordable.

**The wind-down relaxation is free, which is unusual.** Relaxing a binary
decision normally costs accuracy. Here firm value is **linear** in the exit
probability — it is a convex combination of exiting now and carrying on — and a
linear function on $[0,1]$ attains its maximum at an endpoint. The optimiser
therefore drives it to 0 or 1 by itself and the relaxed optimum *equals* the
discrete one. What the relaxation buys is a gradient in between, which an argmax
would not have.

**Where the gradient is dead anyway.** "Differentiable" is accurate but not
unqualified, and the quiet regions are known:

- **Limited liability.** Recovery is `clamp(equity, min=0)`, so below zero the
  term is flat. The only thing still pushing a path away from deep insolvency
  is the survival probability — which is the job the hazard was brought forward
  to do.
- **The capital hazard's exponent is clamped** at 50, so below about
  $\kappa = -9.6$ the derivative is exactly zero. Deliberate: it keeps a
  catastrophic path's intensity enormous rather than infinite.
- **The straight-through sigmoids saturate.** Only paths whose uniform lands
  within a few tenths of a logit of the threshold carry any gradient at all;
  elsewhere the tempered sigmoid's derivative underflows to exactly zero. So
  the "a handful of paths carry the whole gradient" objection raised against
  reweighting does not fully disappear under the relaxation — what disappears
  is the *compounding* of that variance along the trajectory, which is the
  trade actually being made.
- **A quarter that made no profit has no payout gradient.** Dividends come out
  of `clamp(gross − equity, min=0)`, so when the firm did not earn, the control
  is flat. Necessary (without it the payout control is a way to strip the firm)
  but a dead zone all the same.
- **The gradient desert.** From a cold start most paths die in the first few
  periods, and dead paths carry no gradient. Measured, a perfect-information
  solver started cold converges to 14.4 against a constant policy's 25.4 — not
  a bound at all. Warm-starting fixes it, and is mandatory for that solver.
  Counter-intuitively the hazard makes the *cold start* worse, not better — 3.2
  against 10.5 under a hard barrier — because it kills paths more gently and so
  kills more of them early. It buys gradient where a sensible policy operates,
  not where a bad one wanders.

That last one is the honest summary of the limitation: the gradient exists
wherever a sensible policy actually operates, and goes quiet in the regions a
badly initialised one wanders into.

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
The leverage limit in `FundingParams.deposit_capacity` is the first of them and
it binds through the deposit base rather than as a penalty; a liquidity buffer
the firm is *required* to hold, and a run to hold it against, are stage 6b.

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
| 3b | GRC acting on the hazard, not only on losses | **built** — `quant/hazard.py` |
| 3c | What failure costs, and what GRC cannot do about it | **built** — `FirmParams.failure_recovery` |
| — | Discrete cliff events as a *loss* channel | **built** — `StandardDynamics.cliff_loss`, [debug notes §7](static-model-debug-notes.md) |
| 3d | Abandonment / orderly wind-down option | **built** — out of the money, see below |
| 3e | Diagnostic bundle and the break-even outputs | **built** — `quant/studies/breakeven.py` |
| — | Funding constraint: the balance sheet gates investment | **built** — live but weak, see §6 |
| — | Payout control: dividends out of profit, keeping capital scarce | **built** — `FirmAction.payout` |
| 4 | Grid value iteration on a reduced config, and the agreement metric | **built** — `quant/solvers/gridvi.py` |
| 5 | The learner: horizon scaling, multi-seed, out-of-sample | **built** — `quant/studies/seeds.py`; verdict in §6 |
| 5b | Truncated BPTT with a critic, and a head-to-head against full BPTT | **built, then parked** — on branch `svg-critic`, not on `main`; verdict in §6, and see the note there on why it is kept |
| 6a | The liability side: deposits as a priced, persistent, procyclical stock | **built** — `quant/params.py` `FundingParams`, `FirmState.deposits` |
| 6b | Withdrawals, a liquidity buffer, and the fire-sale cost of meeting a run out of an unmatured book | **built** — `StandardDynamics.withdrawal`, `meet_withdrawal` |
| 6c | The run hazard derived from deposit flight rather than assumed as a rate | **built** — `HazardParams.annual_operational_rate` retired; `quant/hazard.py` `liquidity_intensity` |
| 6d | A run that damages the deposit *franchise*, not only its level | not built — see §6 |

**Why the liability side is a stage at all.** Until 6a the firm had no
liabilities: it funded its book out of equity plus a costless multiple of
equity, settled inside the period. That deleted three risks an intermediary
exists to manage — leverage, the price of funding, and the funding leaving —
and it made the hazard channel named "depositors leave" a label on a constant,
because there were no depositors in the model to leave. Loan defaults were
present but could not threaten the firm, because a loss on an unlevered book is
a loss of the same size on capital rather than a multiple of it.

6a is the stock; 6b is what can happen to it; 6c is the channel 6b lets us stop
asserting. 6a was separated because it alone forces a recalibration — a
bank-sized balance sheet needs a bank-sized opportunity, or the firm stops
wanting what it can now fund. **6b and 6c had to land together**: a model where
a run withdraws deposits *and* a hazard rate still asserts that depositors
leaving kills the firm charges twice for one event, and reports plausible
numbers either way.

Survival moved ahead of the grid solver after stage 2: the hard insolvency
barrier turned out to be required for the multi-period problem to be finite at
all, and a grid solver's inner maximization would inherit the same dead
gradient — a failed firm is worth a constant, so nothing about it responds to
what the firm did.

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

> **Which of these have been re-run since the liability side arrived.** The
> deposit stock took the firm's book from about 25 to about 80 against the same
> equity, and forced a recalibration, so every dynamic number moved. Re-run and
> current, at commit `aab280d`: the GRC-as-a-stock table, the survival channel,
> the failure-recovery sweep, the payout comparison, and all five break-evens.
> **Not re-run, and marked individually below:** the grid-agreement table, the
> decision-frequency comparison, the cliff-rate sweep, the horizon-comparability
> table, the horizon-versus-credit-share sweep, and the SVG head-to-head. The
> static tables that follow immediately are on the pinned oracle and did not
> move at all.

> **Two recalibrations, and what forced each.**
>
> **The first** replaced the static model's magnitudes — a quarterly operating
> surplus of 9.01 and a gross expected loss of 9.00, both against equity of 16 —
> which were one-shot quantities wearing quarterly clothing. A firm earning more
> than half its equity every quarter has a franchise worth about 473 against a
> book value of 16, and that single ratio is what left the wind-down option
> permanently out of the money, the financing friction inert, and the balance
> sheet decorative. It set `A = 1.05, S = 600`, `alpha` from 0.3 to 1.5 (it
> carries units of 1/money and had to move with the loss scale), and an opening
> GRC stock of 0.650 per family.
>
> **The second** was forced by the liability side. Four parameters had been
> calibrated against a book of 25 and none of them moved on their own when the
> book went to 80:
>
> | parameter | from | to | why it had to move |
> |---|---|---|---|
> | `curvature_per_period` | 150 | 698 | I* stayed at 29 while capacity tripled, so the firm stopped wanting what it could fund — funding-constrained quarters fell 71% → 0.0% |
> | `annual_return` | 1.2155 | 1.15 | a 12%/yr asset margin levered five times is a 33% return on equity |
> | `initial_grc_stock` | 0.650 | 1.17 | the self-maintaining stock scales with losses, which tripled with the book |
> | terminal franchise | 15 | 20 | one-shot estimate from the measured surplus; see the note below on why it is not a fixed point |
>
> The firm now runs a book near 81 on equity of 16 — leverage 4.85 — earns
> **23% a year** on that equity, and faces a 4.79% annual probability of
> failure. Funding binds in 14.7% of quarters, against 71% before: a bank with
> deposits is less often capital-rationed than one financing itself out of
> retained earnings, which is most of the reason to be a bank.
>
> **The third** was forced by the run replacing the asserted operational
> hazard, and it is a single parameter — `initial_grc_stock`, 1.17 → 0.79.
> Missing it was briefly the most misleading thing in the model. That parameter
> is a fixed point *of the whole model*: the level at which the firm's own
> optimal maintenance spend replaces depreciation. Retiring a hazard that was
> buying a large share of the budget moves optimal spend, so the old value left
> the firm opening **six times above the steady state it would now choose**,
> spending the horizon running the stock down, and reporting a budget of 0.043
> a quarter. That read as "retiring the hazard destroyed the case for GRC". At
> the correct fixed point the budget is 0.187 against 0.197 before — a fall of
> 5%, not 78%.
>
> A stale fixed point degrades into a plausible wrong answer rather than an
> error, so it is worth a standing rule: re-solve it after anything that
> touches a hazard or a loss channel.
>
> **23% overshoots the 17.6% `annual_return` was set against, and the reason is
> an ordering mistake worth recording.** The margin was chosen first, holding
> the opening GRC stock at its old value of 0.650; the stock was then re-solved
> to 1.17, which cut expected loss and lifted the return with it. The two are
> not independent and were solved as though they were. The number is inside
> what an illustrative parameter set can claim for a high-margin crypto
> intermediary, so it has not been re-solved — but a third recalibration should
> iterate the pair rather than fix one and then move the other.
>
> **The franchise is not solved for, deliberately.** Iterating it to a fixed
> point diverges — a larger franchise makes survival worth more, so the firm
> buys more GRC, so the hazard falls, so the franchise grows. Successive passes
> ran 15 → 40 → 58 with annual failure falling 5.5% → 3.1%, which is a model
> talking itself into being safe rather than a calibration converging.
>
> The four-state oracle is **not** recalibrated and never should be
> (`quant/params.py`, `ORACLE`). It exists to be exactly solvable, not
> plausible, and sharing parameters with the model it checks is how one change
> to `production_scale` took out five regression tests and both closed-form
> checks at once. It is why the two static tables below are unchanged across
> both recalibrations.

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

Monthly decisions over five years. `constant` is the best single action, with
the *level* learned by gradient ascent and only its shape held fixed — the same
action every period and every state. `PI bound` is the **perfect-information**
bound: it gives every control a free parameter per path, chosen knowing that
path's whole future, so it is not an implementable policy at all but an upper
bound on what any policy could achieve. A learner scoring above it has a bug.

| GRC decay | solver | value | spend/period | end stock | survives |
|---|---|---|---|---|---|
| 1.000 | constant | 14.3982 | 0.0001 | 0.0001 | 39.9% |
| 1.000 | neural | 12.9874 | 0.0428 | 0.0428 | 32.3% |
| 1.000 | PI bound | 22.5247 | 0.0124 | 0.0118 | 51.0% |
| 0.024 | constant | 33.3223 | 0.0912 | 3.7350 | 85.8% |
| 0.024 | neural | 33.3068 | 0.0913 | 3.7369 | 85.7% |
| 0.024 | PI bound | 36.5838 | 0.0665 | 2.2620 | 87.5% |

**Re-run with the gradient clip in place.** Both neural rows previously read
11.2 with zero survival — the absorbing wind-down. The learner now lands within
0.05% of the constant policy at the real depreciation rate, with a spend of
0.0913 against 0.0912 and an end stock of 3.7369 against 3.7350: it converges to
essentially the same policy. That is the "state feedback earns nothing" verdict
demonstrated rather than obscured by a failed solve.

The constant and PI-bound rows are unchanged to four decimals, as they must be —
the clip touches `train_pathwise` and nothing else.

**How those rows used to read, and why.** Both carried 11.20 — exactly
$0.7 \times 16$, the absorbing wind-down — with zero survival. The script still
detects and flags that shape, since the exit action is absorbing: once the
probability mass has left there is nothing still operating to generate a
gradient for staying ([debug notes §8](static-model-debug-notes.md)).

**The cause was a gradient explosion out of a converged state, now fixed.**

An earlier version of this section reported the failure as a *batch-size* effect
— 33.4 at 1024 paths, 11.2 at 2048, 1.1 at 4096, "monotone in the batch, which
rules out bad luck". **That was wrong and is withdrawn.**
`CommonRandomNumbers` draws `(horizon, batch)` row-major, so changing the batch
changes the *scenario set*: those were three different problems, not one problem
at three resolutions, and three points of a trimodal outcome is not a trend.

Holding the batch fixed and varying the seed, which is the comparison that
actually isolates anything:

| paths | seed | final | peak ‖∇‖ | survives |
|---|---|---|---|---|
| 1024 | 0, 1, 2 | 33.13, 33.91, 33.28 | ~81 | 85–87% |
| 4096 | 0 | **0.77** | **2.7 × 10⁵** | **0%** |
| 4096 | 1, 2 | 33.47, 33.53 | ~81 | 86–87% |

**It is a seed lottery — one draw in six here — and the peak gradient norm is
what separates them**, by three orders of magnitude.

Instrumenting the failing run shows the mechanism exactly. Training converges
normally for 500 steps, with the gradient norm *decaying* the whole way: value
32.97 and ‖∇‖ 0.64 at step 500. Then at step 506 the norm jumps 35× in a single
step and the policy destroys itself in eight more:

| step | 500 | 506 | 509 | 512 | 514 | 599 |
|---|---|---|---|---|---|---|
| value | 32.97 | 31.17 | 23.49 | 10.50 | 1.51 | 0.77 |
| ‖∇‖ | 0.64 | 21.9 | 67.5 | 173.1 | 22.5 | 2.33 |

**No NaN is ever produced** — every parameter gradient is finite at every step,
which rules out the unguarded `logit(0)` in `cliff_loss` as the cause. The
effective sample size is 0.47–0.50 on healthy runs, which exonerates the
self-normalised weights in `path_weights` — the only place paths couple.

**The amplifier is the run channel.** Switching it off drops the peak gradient
norm from 2.7 × 10⁵ to 82 and the failure disappears. Its straight-through
relaxation differentiates $(\text{logit}\,p - \text{logit}\,u)/\tau$, and
$d\,\text{logit}(p)/dp = 1/(p(1-p))$ is about 147 at a run probability near
0.7%; the temperature $\tau = 0.1$ multiplies that by another ten.

**What follows the spike is a runaway, not a stumble**, because
`run_probability` is exponential in the capital ratio. Leverage up, ratio down,
runs likelier, fire sales, capital down:

| step | 498–503 | 504 | 508 | 512 | 515 |
|---|---|---|---|---|---|
| reserves | 32.8% | 30.1% | 22.0% | 10.6% | 10.0% |
| leverage | 4.98 | 5.06 | 5.56 | 9.42 | **13.60** |
| p(run) | 0.69% | 1.30% | 5.29% | 19.07% | **41.71%** |
| survives | 85.5% | 85.0% | 74.8% | 17.5% | **0%** |

The exit rate stays at zero throughout: the firm levers itself to death rather
than winding down. And both endings are absorbing, so none of it is recoverable.

**Adam makes this worse rather than better.** Its update is $lr \cdot
g/\sqrt{\hat v}$, and after 500 steps of small steady gradients $\hat v$ is
tiny, so a sudden enormous $g$ produces a step far larger than the learning
rate — the variance estimate only catches up afterwards. That is why clipping is
the right tool and a smaller learning rate is not.

**Fixed: the gradient is clipped at norm 100**, chosen where the two
distributions separate rather than as the first value that worked. Steady-state
norms are 0.6–2, the cold start peaks near 82, the spike is 2.7 × 10⁵. On
the failing seed the clip fires on 2 steps of 600 and turns 0.77 into 33.25; on
a healthy seed it fires on 0 of 600 and the result is unchanged to three
decimals — provably inert except on the spike.

Everything above and below has been re-run with it. The truncated-BPTT critic
was not the answer, for a better reason than before: a critic addresses
gradients that explode or vanish as they are multiplied back through many
timesteps, and this was a single step off a cliff from an otherwise healthy
trajectory.

The detection threshold had to be made *relative*, having missed this twice on
the third decimal place: the collapsed value is not exactly
$0.7 \times 16$ because the exit probability is a sigmoid that never quite
reaches one, so a solve lands a fraction of a percent away — 11.1858 against
11.2 most recently.

(The $\delta = 1$ rows start from the same opening stock but cannot keep it, so
they describe a firm whose control function evaporates each quarter.)

**Persistence is worth +131% of firm value** (14.40 → 33.32) and takes five-year
survival from 40% to 86%. Per-period spend *rises* as it does so, from
essentially nothing to 0.091: at $\delta = 1$ a unit of spend buys one month of
protection and is barely worth anything, while at $\delta = 0.024$ it protects
every later month too, so much more of it is worth buying. That is the
intertemporal content the static model could not express — it is not the
one-period answer repeated.

**The longer the horizon, the larger this is**: +31% over two years, +131% over
five. A control bought in month one is still working in month sixty, and a
two-year window simply cannot see most of that. It is the sharpest statement of
the horizon effect the model produces.

(The direction of that last sentence has flipped twice across recalibrations,
which is why the script now prints the measured direction rather than an
asserted one. It printed "spends MORE" unconditionally once, over a pair of
numbers that had gone the other way.)

The **value of perfect information** — the gap between the bound and the best
implementable policy, i.e. what clairvoyance would be worth — is 3.26 at
$\delta = 0.024$ against 8.13 at $\delta = 1$. Persistence narrows the gap a clairvoyant policy can exploit: a
stock that carries across quarters is partly a substitute for knowing what is
coming.

### What the survival channel is worth

Both budgets scored in the *same* world — the one where GRC reduces failure
intensity as well as expected loss. Comparing them in their own worlds would be
meaningless, since a world without those channels is simply less dangerous.

| budget set for | spend/period | annual failure | value |
|---|---|---|---|
| expected loss only | 0.0563 | 5.02% | 31.9839 |
| loss **and** survival | 0.0912 | 3.03% | 33.3223 |

Budgeting as though GRC only bought smaller losses gives a programme **38%
smaller**, and costs 4.0% of firm value and 2.0 percentage points of annual
failure probability.

**This ratio has now stabilised.** It was 1.6× at quarterly over two years and
is 1.62× at monthly over five — the first time in five calibrations that this
number has stayed put. What did move is the *cost* of getting it wrong, from
0.5% of firm value to 4.0%, which is the longer horizon giving the firm more
time to die of a risk it under-insured.

**This number has now moved twice, in the same direction, for two different
reasons — and both of them were the model being made more honest.**

The ratio of the survival-aware budget to the loss-only one ran *fifty* times
before the liability side, *nine* times after it, and **1.6 times** now. The
first fall was leverage: losses on a book four times capital are worth far more
to avoid, so ordinary loss reduction carries a real share of the programme. The
second was retiring the asserted rate that claimed an operational incident kills
the firm — the single largest source of GRC's *survival* value, and the one that
turned out to assert its own conclusion.

What is left is a firm that buys GRC for both reasons in roughly equal measure.
**The original headline — that GRC is paid for almost entirely by survival and
hardly at all by loss reduction — does not survive either correction.** It was
true of an unlevered firm whose death rate was asserted rather than derived.

Read alongside the horizon caveat below, which pushes in the same direction and
has not yet been re-measured: *how much* of the programme survival pays for is
partly a statement about the two-year horizon as well.

The history is worth keeping because no single figure here was ever quotable on
its own. At the original magnitudes the gap was 27% of the budget and 0.3% of
value; the first recalibration removed a loss channel large enough to disguise
the survival effect and took it to fifty times; the liability side took it to
nine; retiring the asserted operational hazard took it to 1.6. Four values of
the project's sharpest result, each correct about the model that produced it.

### Controls or cash? What the run channel changed

`uv run python scripts/run_dynamic_model.py`. The run replaced a hazard rate
that asserted an operational incident kills the firm. Once depositors actually
leave and the firm can hold reserves against them, it has **two** defences
rather than one, and which it buys is the question this stage exists to answer.

The same firm, at its own GRC-stock fixed point in each configuration and at
identical solver settings:

| | credit | operational | compliance | **total** | reserves | book | annual failure |
|---|---|---|---|---|---|---|---|
| asserted rate, no run | 0.0204 | **0.1469** | 0.0299 | **0.1972** | 9.4% | 80.9 | 4.79% |
| run mechanism | 0.0455 | **0.0651** | 0.0767 | **0.1873** | 24.2% | 68.2 | 4.29% |

**The asserted rate was overstating operational GRC by about 2.3×.** It claimed
three things in one parameter — that an incident becomes public, that depositors
leave, and that the firm therefore dies — with no depositors in the model to
leave. Replacing it with a mechanism the firm can respond to cuts the
operational line by 56%.

**The budget does not disappear; it moves.** Total spend falls 5%, while credit
and compliance each roughly double. Retiring a channel does not make the firm
stop buying GRC — it makes it buy a different mix, and the mix it was buying
before was distorted by the channel that asserted its own conclusion.

**And it holds a liquidity buffer.** Reserves go from 9.4% of the balance sheet
to 24.2%, funded by lending 15% less. That buffer is the largest single
behavioural change in the stage and it is entirely new: before the run there was
nothing to hold reserves against.

Sweeping how often runs happen separates the two defences:

| runs/year | GRC/period | of which operational | reserves | book | p(run)/period | annual failure | value |
|---|---|---|---|---|---|---|---|
| 0.00 | 0.0782 | 0.0124 | 9.7% | 82.62 | 0.000% | 2.82% | 34.855 |
| 0.45 | 0.0912 | 0.0278 | 21.6% | 76.93 | 0.898% | 3.03% | 33.322 |
| 1.00 | 0.0912 | 0.0295 | 27.5% | 72.73 | 2.410% | 3.34% | 32.311 |
| 2.00 | 0.0902 | **0.0301** | 32.5% | 69.25 | 6.020% | 3.88% | 30.923 |

Introducing a run at all takes operational GRC from 0.0124 to 0.0278 and
reserves from 10% to 22%: **the firm buys both defences.** That part holds at
every horizon tried.

**But the substitution does not, and this is a reversal.** Over two years,
operational GRC rose and then *fell back* as runs became frequent — 0.0021,
0.0651, 0.0716, 0.0386 — and the conclusion drawn was that cash is the certain
defence and wins at the margin. Over five years it rises monotonically all the
way: 0.0124, 0.0278, 0.0295, 0.0301, while reserves also rise the whole way.
**Controls and liquidity are complements here, not alternatives.**

The two-year result was a horizon artifact. A buffer protects you now; a control
stock has to be built before it protects you at all, and over two years a firm
facing frequent runs does not have time to build one — so it holds cash instead.
Give it five years and it does both.

That makes the earlier decision-relevant reading wrong, and it is worth stating
plainly because it was stated plainly the other way: *"a firm facing frequent
runs should hold capital and liquidity rather than buy its way out with
controls"* is true only of a firm with two years to live.

The script now computes this direction from the rows rather than asserting it.
It asserted the two-year direction and printed "operational GRC falls back to
0.0301" over the largest number in its own column — the third hardcoded
direction in that file to go stale, and they are all now measured.

**What most drives this, and is not modelled.** Deposits rebuild toward capacity
at a four-month half-life, whether they left in a panic or drifted away. Real
deposit franchises do not come back that fast after a run, and a slower rebuild
would make runs costlier in a way a buffer cannot offset — pushing further in
the direction the five-year result already points. That is stage 6d (§5) and it
is not built.

### What failure costs, and what GRC cannot do about it

`--recovery-sweep`. When the firm fails, a fraction of whatever positive equity
remains is recovered — floored at zero, because a firm that died owing money is
worth nothing to its owners rather than a negative number.

**GRC does not appear in that recovery, and the asymmetry is the point.** A
programme acts on how *often* failure happens; it cannot make a failure cheaper
once it has happened. Losing a licence costs what it costs. A model in which
GRC reduced both would let one parameter buy the same protection twice, and the
second purchase would be free.

| recovery | spend/period | annual failure | value | going-concern share |
|---|---|---|---|---|
| 0.0 | 0.0969 | 2.79% | 32.702 | 1.000 |
| 0.4 | 0.0912 | 3.03% | 33.322 | 0.808 |
| 0.8 | 0.0865 | 3.35% | 34.075 | 0.624 |

The comparative static is the cleanest the survival channel produces, and it is
one an executive can argue with: **the less a failure would destroy, the less a
programme to avoid it is worth.** Spend falls by 11% across the range — 41%
before the run channel and 27% after it, all on different levels, so the
direction is the durable part and the magnitude is not — while
the firm becomes more willing to die. Every unit of that spend is bought by the
franchise at risk — the going-concern share is exactly the part of firm value
that failure would destroy, and when nothing is recovered it is 1.000 by
construction.

That share is also a regime diagnostic. If liquidation were worth nearly as
much as continuing, death would be cheap, the survival motive would vanish, and
the objective would quietly revert to expected-loss minimisation — which
[`framework.md`](framework.md) §3 rejects.

### The option to stop, and why it is currently worthless

The firm may wind down deliberately at the start of any quarter, recovering
$0.7 E$ rather than the $0.4 E$ a disorderly failure leaves. This is where
[`framework.md`](framework.md) §1's **Governance** pillar — "a named authority
who can halt activity" — finally does work. Risk and Compliance both act by
making bad outcomes rarer or smaller; Governance acts by converting one kind of
ending into another, and until now that had no representation here.

The decision is relaxed to a probability in $[0,1]$ rather than a hard choice.
That costs nothing: firm value is **linear** in it — a convex combination of
stopping now and carrying on — and a linear function on $[0,1]$ attains its
maximum at an endpoint, so the optimizer drives it to a corner on its own and
the relaxed optimum equals the discrete one. What it buys is a gradient in
between, which an argmax would not have. Asserted: fewer than 25% of exit
decisions settle in the interior.

**And at the default parameters the firm never takes it.** The exit rate is
$3.5 \times 10^{-5}$ and the option is worth $-0.0007$, which is Adam not
settling one extra parameter rather than anything economic. That holds all the
way down to opening equity of 2.0, where annual failure is 36%:

| franchise $A$ | surplus/qtr | exit rate | option value |
|---|---|---|---|
| 3.00 (default) | 9.01 | 0.000 | −0.0007 |
| 1.60 | 1.30 | 1.000 | 2.9931 |
| 1.20 | 0.18 | 1.000 | 7.0747 |
| 1.05 | 0.01 | 1.000 | 7.5778 |

The mechanism is correct — thin the franchise and the firm exits immediately —
but it is structurally out of the money, and the reason matters more than the
result. **Nothing in this model caps investment by capital.** The firm can
always deploy $I^{\star}$ and earn the same 9.01 per quarter whatever its
balance sheet, so the franchise is worth about 473 as a perpetuity against
equity of 16. Winding down is never close.

That is the same root cause as the convex financing cost going inert: with no
funding constraint, the balance sheet does not gate operations, and equity
matters only through the hazard.

### Retaining earnings versus distributing them

A funding constraint alone was not enough. Retaining everything, equity grew
from 16 to 67 over eight quarters, funding capacity grew with it, and a
constraint that bound 2.4% of the time in the first quarter bound 0.1% by the
eighth. The balance sheet was decorative again, one step removed.

The firm now chooses what to distribute. Dividends come out of the quarter's
profit, never the capital base — the ordinary accounting constraint, needing no
parameter, and without it the control is simply a way to strip the firm:
distributing all equity returns it at face value, which beats the 0.7 a
wind-down recovers, so the exit option is dominated and the balance sheet can
be emptied in a quarter. Measured, the thin-franchise firm's exit rate fell
from 1.000 to 0.0005 before this constraint was added.

| payout | value | spend/period | underinvestment | annual failure | end equity | dividend share |
|---|---|---|---|---|---|---|
| retain all | 32.964 | 0.0912 | 0.014 | 2.21% | 32.20 | 0.000 |
| optimized | 33.322 | 0.0912 | 0.021 | 3.03% | 23.87 | **0.192** |

**Over five years the firm distributes 19% of its value**, and accepts a failure
rate of 3.03% against 2.21% to do it. Retaining everything, equity would end at
32.20 against an opening 16 — the firm out-grows its own investment opportunity,
so the marginal retained unit has little left to buy and time value wins.

**Four configurations, four different answers, all from the same control:**

| configuration | dividend share | because a retained unit… |
|---|---|---|
| no deposits | 0.000 | …relieves a funding constraint binding on 82% of paths |
| deposits, no run | 0.044 | …relieves one binding on 15% — worth less |
| deposits + run, 2 years | 0.000 | …is what the liquidity buffer is held out of |
| deposits + run, 5 years | **0.192** | …has nowhere left to go; equity doubles regardless |

The control is doing real work in every one. What changes is what a retained
unit is *for*, and that is a property of the configuration rather than of the
firm's patience.

The control also responds to impatience — at a quarterly discount of 0.90 the
same firm distributes more, and `test_impatience_raises_the_payout` asserts the
direction.

This replaces an earlier reading in which the firm distributed 89% of profit.
That was an artifact of a mis-specified terminal value: writing the going
concern as `(1 + m) · equity` made a unit retained to the horizon worth `(1+m)`
against a unit distributed today worth 1 — a pure arbitrage, which also
dominated the wind-down option and left the objective so driven by terminal
equity that every other gradient was noise beside it. The franchise is now
**added**, not multiplied (`quant/env/reward.py`).

**This is where the discount rate starts to matter.** Every reward was zero
until now — all value was terminal, so $\beta$ was a scalar multiplier that
could not change any decision. A dividend is worth its face value today against
capital that pays off later in survival and funding capacity, so a more
impatient firm distributes more. That is asserted rather than assumed.

Seven-tenths of firm value is now cash actually handed over rather than capital
still being held when the horizon arrives, which is what makes the going-concern
story a claim about distributions rather than about a terminal balance sheet. That gap is the dynamic analogue of the Froot-Stein
premium: spend that pays for itself only because the firm has a franchise worth
surviving to keep.

The firm starts with a GRC stock of 5.18 per family rather than zero. That is
not a tuned number: it is the self-consistent steady state, the level at which
the firm's own optimal maintenance spend exactly replaces depreciation, found
by bisection on $G_0 = g^{\star}(G_0)/\delta$. A firm starting there neither
builds nor runs down its control function, which is what "mature going concern"
should mean.

It matters because both neighbouring regimes are degenerate, in opposite ways:

> **From the first recalibration, not re-run since.** The fixed point has been
> re-solved twice: 5.18 here, then 0.650, and 1.17 with the liability side. The
> regimes either side of it are what this table is for and they have not moved;
> the levels have.

| opening stock | annual death | spend/qtr | regime |
|---|---|---|---|
| 0 | 27.5% | 3.33 | rebuilding from nothing; dies from tail events while it does |
| 3 | 11.7% | 1.99 | usable |
| **5.18** | **7.1%** | **1.07** | **self-consistent steady state** |
| 9 | 3.8% | 0.11 | inherited so much capital that spend collapses |
| 12 | 2.1% | 0.002 | model has nothing to say about budgets |

The high end is the "never binds" trap of
[`static-model-debug-notes.md`](static-model-debug-notes.md) §6 in a new place:
survival looks excellent and the model is silent on the only question it was
built to answer.

Persistence is worth **+49% of firm value** (23.16 → 34.55) and takes two-year
survival from 67% to 91% — a far more modest claim than the +268% the original
magnitudes produced, and a believable one. Per-quarter spend *falls* while it
does so, 0.362 to 0.197: a stock that persists buys the same protection for a
smaller flow. That is the intertemporal content the static model could not
express — it is not the one-period answer repeated.

(This paragraph carried a contradiction through two recalibrations, asserting
in consecutive sentences that spend falls and that it rises. It falls. The
claim that it rises was true of the original magnitudes and should have gone
when they did.)

Three solvers, one `evaluate`. `constant` is the best state-independent action
and the floor a learner must clear; `neural` is state feedback trained by
backpropagation through the rollout; `PI bound` chooses every control per path
with the future known and is therefore not implementable. The sandwich
$V(\text{constant}) \le V(\text{neural}) \le V_{PI}$ is asserted on every run.

**The convex financing cost has gone nearly inert.** At this parameterization
the finance constraint binds on 1.9% of probability mass, against 52% in the
static model. A well-capitalized going concern funds its investment internally
almost always, and the binding channel is the hazard instead.

That is this project's own thesis appearing as a measurement rather than an
assertion. The convex financing premium was a *static reduced form* of a
curvature in the value function; a two-period model has to assume that
curvature, while a dynamic model with death derives it, because death destroys
continuation value. Once survival is explicit the reduced form stops doing
work. Removing it is §5's bite 3c, and this is the evidence for it rather than
a plan item taken on faith.

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

### Checking the solvers against something that is not a solver

> **Pre-liability figures — not re-run.** Produced by a firm with no deposits, funding a book of about 25 out of its own capital. The shape of the finding is the part to trust; the levels are all from the earlier calibration.


> **Read this before grading any learner against the grid.** The grid solves a
> *restricted* problem — payout fixed, wind-down off — so its value function
> assumes the firm reverts to that restriction after the current step. A policy
> that uses a control the grid holds fixed is charged against a continuation it
> will not actually follow, and scores badly while being better. The neural
> policy below does exactly this: worst on the grid's metric, best on honest
> evaluation. **A disagreement with this rung is not evidence of a learner bug
> until the learner has been confined to the same restriction.**

`quant/solvers/gridvi.py`. Every solver up to this point is a gradient method
on the same objective, so they can all be wrong in the same way. The grid
shares only the *dynamics* — it calls the same `StandardDynamics.step` the
rollouts do — and has nothing else in common: no gradients, no policy
parameterization, no Adam. Backward induction rather than fixed-point
iteration, because the environment is finite-horizon and a stationary solution
would be answering a different question.

It solves a **named restriction**, not the full model: state is equity and a
single GRC stock held equal across families; action is total spend and
investment; payout is fixed and wind-down is off. The binding constraint on a
grid solver here is the action space, not the state space — five continuous
controls at ten points each would be 100,000 evaluations per node. The rung's
job is to be right, not general.

**Agreement, and what it is worth.** Three policies, each an answer to "what
should the firm do each quarter?" found a different way:

- **tabular** — the grid's own policy: discretize the state, back up a value
  function, take the highest-scoring action at each node. No gradients.
- **optimized constant** — one fixed action used in every state and every
  quarter, found by Adam on the differentiable rollout. Ignores the state.
- **neural** — a small network mapping observed state to action, trained by
  backpropagation through the rollout. The only one that reacts.

The two columns answer different questions, and conflating them is the easiest
way to misread the table.

*Honest MC value* is what a policy is actually worth: run it through the real
simulator over 2048 sampled paths and take the expected discounted firm value,
using the same `evaluate` and the same draws for all three. "Honest" is in
contrast to the grid's own internal value estimate, which is computed on a
coarse grid with a finite shock sample — convenient, but not ground truth. This
column is the performance ranking, and no solver grades its own homework.

*Gap* is not a performance measure at all. It asks whether an independent
solver agrees with what a policy is doing: score the policy's chosen action
using the **grid's** value function, and compare against the action the grid
would have taken. Negative means the policy leaves value on the table by the
grid's reckoning; zero means they agree. Formally the one-step
policy-improvement gap $Q^{\text{grid}}(s, a_\pi(s)) - V^{\text{grid}}(s)$
at states the policy actually visits — both terms from the same value function,
so the grid's discretization largely cancels. Comparing $V^{\text{grid}}$
against a rollout value instead would conflate three separate errors: the
grid's discretization, the learner's optimization, and Monte Carlo noise.

| policy | honest MC value | gap vs grid |
|---|---|---|
| grid's own (tabular) | 27.6594 | −0.11% |
| optimized constant | 27.8251 | −0.81% |
| neural (state feedback) | 27.8289 | −0.81% |

**The +0.28% rows are the result.** The constant-policy solver reaches its
answer by Adam on a differentiable rollout and the grid reaches its by
enumeration and backward induction; they share only the dynamics and agree to a
third of a percent. That is the evidence this rung exists to produce. That both
entries are *identical* is itself a clue to what the residual is: the opening
state is not exactly a grid node, and bilinear interpolation of a concave value
function reads low, so $Q - V$ comes out slightly positive for any policy.

**The −1.10% row is the metric failing, not the policy.** The neural policy
scores worst here while winning on honest evaluation, because it chooses a
payout the grid's restriction holds fixed. When the grid scores that action, it
charges it against a continuation the neural policy will not actually follow —
it assumes the firm reverts to the restricted policy next quarter, and so
understates it.

Agreement with this rung is therefore only meaningful *inside* the restriction.
Worth stating plainly rather than discovering later: a metric that silently
penalizes a policy for using a control the referee cannot see would read as a
learner bug for a long time.

Direct value agreement converges with the grid's shock sample — 7.7% at 48
draws, 3.6% at 768, 2.0% at 3072, monotone — which is what identifies the
residual as sampling rather than disagreement. The severity distributions are
heavy-tailed, so a small sample misses the tail and the grid overstates value.

### Does state feedback earn its place?

`uv run python scripts/run_seed_study.py`. Every other number on this page
comes from one draw of the scenarios and one optimizer start. That is the
practice `quant/simulate.py` was corrected for in the static model, and the
dynamic model reintroduced it — a rollout looks deterministic once the common
random numbers are fixed. It is deterministic. It is also one sample.

Trained on 512 paths, scored on 4096 **held out**, across five seeds. Both the
scenario draw and the network initialization vary together, so the spread
bounds both sources at once.

Monthly decisions over five years, five seeds:

| solver | median | 95% CI | range over seeds | overfit |
|---|---|---|---|---|
| constant | 33.295 | ±0.222 | [32.925, 33.631] | 0.39% |
| neural | 33.417 | ±0.251 | [32.911, 33.632] | 0.35% |

Before the gradient clip these rows read `33.177, ±12.799, [0.618, 33.544]`. The
interval fell **51-fold** and the worst seed went from a failed solve to a
healthy one. The constant policy is unchanged, which is the control: the clip
touches only the learner.

**State feedback is worth +0.37% at the median, against measured seed noise of
0.75% — within it.** The verdict is unchanged from every earlier horizon and
calibration, but this is the first time it has been measured on five solves that
all worked. Previously it read −0.35% against noise of 38.6%, a number produced
by averaging over a wreck.

**The learner is stable now and still does not earn its place**, which are two
separate findings and were previously entangled. At the five-year horizon it
converges to within 0.05% of a policy that cannot see the state at all, with a
near-identical spend and control stock — so the state genuinely carries little
the firm can act on, rather than the learner being unable to find it.

The script now compares the advantage against the spread it measured rather than
a hard-coded 2%, and warns when any seed lands below half the best. It fired on
the pre-clip run and does not fire now.

Read this table against the one immediately above it too: at the old
magnitudes state feedback was worth nothing at eight quarters and +11.5% at
thirty-two, and the thirty-two-quarter advantage turned out to be an optimizer
trap rather than economics (see the note below).

> **The thirty-two-quarter rows used to be degenerate** — they read 11.196,
> which is exactly $0.7 \times 16$: the firm was **winding down in the first
> quarter**, which is also why the variance across seeds was zero. They are no
> longer, and the numbers above are from a firm that trades for the whole
> thirty-two quarters.
>
> **Since resolved, and it was not what it looked like.** The collapse was an
> *optimizer trap*, not economics: the exit action is absorbing, and it was
> initialized at a flat 1.8% per quarter, which is 13% cumulative over eight
> quarters but 44% over thirty-two. The firm began its search halfway out of
> the door and could not climb back, because once the mass has left there is
> nothing still operating to generate a gradient for staying. Forbidding the
> action outright was worth 76% more at twenty-four quarters. Holding the
> *cumulative* probability fixed instead removes it —
> [debug notes §8](static-model-debug-notes.md).
>
> A milder version of the horizon effect is real and remains — see below. It no
> longer blocks the horizon question, but it does mean values are not
> comparable across horizons.

A learner is a means, not a deliverable, and at every horizon currently
trustworthy the simplest policy in the ladder is not measurably beaten.

Overfitting is small and is measured rather than assumed: 0.39% for the constant
policy and 0.51% for the neural one.

### Open: firm value is not comparable across horizons

> **Pre-liability figures — not re-run.** Produced by a firm with no deposits, funding a book of about 25 out of its own capital. The shape of the finding is the part to trust; the levels are all from the earlier calibration.


With the exit trap removed, value still declines gently with the horizon —
25.5, 22.1, 19.7, 17.9 at 8, 16, 24 and 32 quarters. A going concern should not
be worth less for living longer, so this is worth understanding before any
result is quoted across horizons.

The mechanics are clean. The funding constraint binds on 98.9% of probability
mass, so retained capital is too valuable to distribute and the payout share is
zero; with no dividends, *all* value is terminal, $V(T) = \beta^{T}(E_T + F)$.
Decomposing that:

| $T$ | $\beta^{T}$ | $E_T$ | $\beta^{T} E_T$ | $\beta^{T} F$ |
|---|---|---|---|---|
| 8 | 0.857 | 18.4 | 15.78 | 12.86 |
| 16 | 0.735 | 21.0 | 15.44 | 11.03 |
| 24 | 0.630 | 24.0 | 15.12 | 9.45 |
| 32 | 0.540 | 26.5 | 14.32 | 8.10 |

$\beta^{T} E_T$ is nearly flat. **The whole decline is $\beta^{T} F$** — a
*fixed* terminal franchise discounted harder the longer the firm waits to
collect it.

The obvious repair is to choose $F$ so that $V(T) = V(T+k)$, the same
fixed-point construction used for the opening GRC stock. It does not work here,
and the reason is the finding rather than the failure: the firm's equity grows
at **1.53% a quarter against a discount rate of 1.94%**. After losses and
failure risk it barely fails to out-earn its cost of capital, so no choice of
$F$ makes value horizon-invariant — solving for one gives $F \approx 0.5$,
which is small enough to bring back the end effect the franchise was introduced
to remove.

So this is a calibration question, not a parameter to tune: either the firm
should out-earn its cost of capital by a clearer margin, or the discount rate
is too high for a business of this risk, or value genuinely is horizon-specific
and results should never be compared across $T$. Deliberately left open rather
than guessed at.

### How often the firm decides

> **Pre-liability figures — not re-run.** Produced by a firm with no deposits, funding a book of about 25 out of its own capital. The shape of the finding is the part to trust; the levels are all from the earlier calibration.


`quant/params.model_at(periods_per_year)`. The model silently assumed quarterly
decisions. The discount, the GRC depreciation and all three hazard rates were
already annual and divided down correctly; the loss means, the event
probabilities, the cliff rate and the production parameters were *per-period*
and did not. Running the same firm weekly would have handed it thirteen times
its annual losses and thirteen times its annual return with every diagnostic
reporting normally — §4 of the debug notes relocated to the time axis.

`AnnualRates` now states the economics without reference to frequency: a rate
per year for how often, a magnitude per event for how bad, since a severity
does not scale with how often you look. Production needs both directions — the
gross return compounds *down* while the curvature scales *up*, so the
unconstrained optimum $I^{\star} = S\ln A$ is the same amount of capital
however often it is re-decided (29.273 at both frequencies).

**Invariance, verified.** The same firm over the same two years:

| frequency | steps | value | GRC/year | annual failure |
|---|---|---|---|---|
| quarterly | 8 | 24.4071 | 0.9600 | 9.06% |
| weekly | 104 | 24.2268 | 0.9600 | 9.27% |

Asserted at 5%, not tighter, and for an economic reason rather than a numerical
one: thirteen weekly losses summing to the same mean as one quarterly loss are
*less* volatile, because a sum of exponentials is a gamma. The means are
invariant by construction; the distributions are not, and should not be. A
finer decision frequency genuinely carries less aggregate tail.

**Weekly decisions do not make state feedback pay.** The motivating idea was
that at quarterly steps an incident lands and the firm's next chance to respond
is three months away, so a reactive policy has nothing to react to. Measured:

| frequency | constant | neural | edge |
|---|---|---|---|
| quarterly (8 steps) | 25.4608 | 25.4739 | +0.05% |
| weekly (104 steps) | 25.1572 | 25.1700 | +0.05% |

Identical, and the reason is not the frequency. Survival-weighted over
everything the firm visits, its equity runs from **15.71 at the 5th percentile
to 18.68 at the 95th** — the middle half spans 9% of the median. Across that
whole range the trained policy's own outputs move by less than a percent
(per-period GRC 0.01682 → 0.01671).

**State feedback buys nothing because the state barely varies.** A constant
action is close to optimal everywhere the firm actually goes, and adding
decision points to a state that does not move adds nothing to react to. The
learner is not failing; it is being asked to exploit variation the model does
not generate.

### When state feedback does earn its place

> **Pre-liability figures — not re-run.** Produced by a firm with no deposits, funding a book of about 25 out of its own capital. The shape of the finding is the part to trust; the levels are all from the earlier calibration.


That suggests a test, and the test passes. Sweeping the cliff rate — the one
shock large enough to move the state sharply — while holding everything else:

| cliff strikes/year | equity p5–p95 | spread | constant | neural | edge |
|---|---|---|---|---|---|
| 0.101 (default) | 15.83–18.53 | 16% | 25.3975 | 25.4068 | **+0.04%** |
| 0.500 | 15.13–18.01 | 18% | 24.5415 | 25.5052 | **+3.93%** |
| 1.500 | 14.31–17.46 | 20% | 23.3652 | 24.9242 | **+6.67%** |
| 3.000 | 14.09–17.24 | 20% | 22.3022 | 24.3819 | **+9.32%** |

Monotone, and by a factor of two hundred from end to end. Note what does *not*
explain it: the unconditional spread of equity barely moves, 16% to 20%. So it
is not dispersion as such.

**State feedback pays when the state makes large discrete jumps a policy can
respond to, not when it merely varies.** A cliff is a recognisable event with a
sensible response; smooth quarter-to-quarter drift is not, and no amount of it
adds up to something worth reacting to. The constant policy loses 12% of firm
value across this sweep (25.40 → 22.30) while the neural policy loses 4%
(25.41 → 24.38) — the entire advantage is in handling cliffs.

That closes a circle. The cliff channel was added because the state it creates
— impaired but alive, holding a real decision — is where the decision problem
actually lives. This is the measurement of that claim: it is also the only
state in this model where having a policy rather than a budget is worth
anything.

### The two-year horizon systematically undervalues loss reduction

> **Pre-liability figures — not re-run, and this one is load-bearing.** Leverage already moved the survival-versus-loss-reduction split a long way toward loss reduction, and a longer horizon moves it further in the same direction. Until this sweep is repeated the two effects are confounded.


Credit GRC spend at the default two-year horizon is 0.0012 a quarter — half a
percent of the budget. The model appears to say that managing loan defaults, the
largest single loss channel, is not worth doing.

It is a horizon artifact. GRC is a *stock* that depreciates, so a flow of
spending reaches only about 4.3 times itself over eight quarters against a
steady-state 14.5 times. A control bought to reduce losses has not finished
accumulating before the model stops, while a control bought to reduce the
failure rate pays from the first period through the hazard. Lengthening the
horizon and re-solving:

| horizon | credit GRC/qtr | credit share of budget |
|---|---|---|
| 2 years | 0.0012 | 0.5% |
| 5 years | 0.0281 | 9.2% |
| 10 years | 0.0494 | 16.6% |

**Credit GRC spend rises fortyfold between two years and ten.**

This qualifies the headline above. "The programme is paid for almost entirely by
survival" is substantially a statement about a two-year horizon, not about GRC.
Over a horizon long enough for a control function to be built, loss reduction
recovers a sixth of the budget. The survival channel still dominates, but by
nothing like the margin a short horizon suggests.

(A twenty-year solve was also run and is not reported: every budget collapses to
near zero, which is the long-horizon degeneracy recorded above rather than a
result.)

### Truncated BPTT with a critic, measured against full BPTT

> **Pre-liability figures — not re-run.** Produced by a firm with no deposits, funding a book of about 25 out of its own capital. The shape of the finding is the part to trust; the levels are all from the earlier calibration.

> **Parked, not rejected — and the case for re-testing has strengthened twice
> since.** Everything below was measured at horizons of **at most 32 steps**,
> and the technique exists for long horizons where full backpropagation, whose
> cost is linear in the number of steps, becomes the binding constraint. The
> horizon is going to roughly triple. Separately, the neural policy has since
> become unstable at the longest horizon tested — one seed of five produced a
> firm worth 0.9 against a best of 32.7 — which is the failure mode a critic is
> meant to damp. The verdict below is sound about the regime it was measured in
> and says nothing about the regime now approaching.


`scripts/compare_learners.py` **on the `svg-critic` branch**, where the three
files this experiment needs are kept; `main` does not carry them. Stage 5
declined to build SVG(K) on the evidence available then; it was then built so the
choice rests on a head-to-head rather than on either argument.

**K is a dial, not a different algorithm.** At `K = horizon` the windowed
objective is *bitwise* the ordinary rollout value and the critic is never
consulted — asserted, along with the identity that the critic's regression
target read backwards equals `path_values` read forwards. Smaller K cuts the
gradient chain more often and leans harder on the critic.

Eight quarters, trained on 512 paths, scored on 4096 held out:

| solver | held-out | note |
|---|---|---|
| constant | 25.3969 | the floor |
| full BPTT | 25.4064 | K = horizon, no critic |
| SVG(K=4) | 24.9182 | |
| SVG(K=2) | 24.1112 | |
| SVG(K=1) | 11.1992 | collapsed to wind-down |

**Truncation costs value monotonically and buys nothing.** Full BPTT is already
stable here, so cutting the chain trades an exact gradient for a biased one
with nothing gained in return. The shorter the window, the worse.

`K=1` is the instructive failure. 11.1992 is $0.7 \times 16$: the firm winds
down in the first quarter. Early in training the critic underestimates
continuation, exiting looks better, and once the policy exits there is no
continuing mass left to generate a gradient for not exiting. The critic then
learns to predict the wind-down value very accurately — its loss falls to
0.0003 — which is a self-fulfilling bootstrap rather than convergence.

**And it does not rescue the long horizon, which was the whole case for it.**
Re-run after the exit trap of [debug notes §8](static-model-debug-notes.md) was
fixed, so full BPTT is no longer handicapped:

| quarters | constant | full BPTT | SVG(K=8) | SVG(K=4) |
|---|---|---|---|---|
| 24 | 19.5568 | **19.5927** | 17.8677 | 14.0259 |
| 32 | 17.6940 | **17.8697** | 11.1995 | 11.1996 |

Monotone in K at every horizon tested: the shorter the window, the worse. Full
backpropagation wins everywhere.

The thirty-two-quarter row carries the sharpest finding. Both truncated
variants sit at 11.1995 — the wind-down value — while full BPTT reaches 17.87
*on the same initialization*. Fixing the absorbing-action trap cured full BPTT
and did not cure SVG, because **a critic makes that trap worse**: early in
training it underestimates continuation, which is exactly the input the policy
uses to decide whether to continue, and once the mass has exited the critic
learns to predict the wind-down value accurately and the bootstrap closes on
itself.

So the verdict stands, for a better reason than before. Truncation was declined
on the grounds that full BPTT looked stable; it is now declined on the grounds
that it was built, measured at four horizons, and made things worse at all of
them — and that its one distinguishing component actively deepens the failure
mode that mattered most. Knowing its size is
the only way to know it is small.

### Break-even analysis

`uv run python scripts/run_breakevens.py [--alpha-sweep]`. Nothing here can
calibrate $\alpha$, so an optimal budget computed from an invented one implies
a precision that does not exist. The question is inverted into claims a reader
can argue with.

The survival framing improves the question rather than answering it. $\alpha$
is still uncalibrated, but the quantity a programme has to move is now the
annual probability of failure — and that has external anchors $\alpha$ never
had: bank failure rates, rating-agency default rates, observed crypto-lender
failures, and the price of the D&O and cyber cover insuring the same risk.

**1. Hazard break-even — the only one with no $\alpha$ in it.**

> A programme costing **0.95 a year**, against a franchise of **26.93**, must
> cut the annual probability of failure by at least **354 basis points** to pay
> for itself.

*(Monthly decisions over five years. At quarterly decisions over two it was
243 basis points against a cost of 0.66 — the threshold rose because the longer
horizon buys a larger programme, not because the business got riskier.)*

Both inputs are things a board already has a view on, so the whole claim can be
checked without touching an uncalibrated parameter.

**2. Capitalization band — over what range is a programme a decision?**

Annualized spend, averaged over **three seeds**, with the spread across them —
which at this horizon is the part that matters:

| equity | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|
| mean GRC/year | 0.36 | **1.09** | **1.02** | 0.88 | 0.32 |
| spread over seeds | 0.54 | 0.10 | 0.11 | 0.38 | 0.32 |

**Optimal GRC spend is non-monotone in capitalization and peaks in the middle.**
Well capitalized, the hazard is too small to be worth buying down; thinly
capitalized, there is too little franchise left to protect. A model that made
GRC monotone in capital would be saying something false about both ends, and
this is the most decision-useful shape the model produces.

**The shape is solid and the peak is not.** The middle of the band is about
three times either end on every seed, so "spend peaks at middling
capitalisation" holds. But the argmax landed at 32, 8 and 8 across three draws
of the same experiment: equity 8 and 16 sit on a plateau a few percent wide and
the sampling noise is wider than that. Quoting a peak from one run reports the
draw.

At the ends the spread is qualitative rather than quantitative. At equity 4 one
seed spent 0.544 a year and another 0.001 — the firm winding down on one
scenario draw and trading on through on another.

**It is not an optimizer budget problem**, which was the cheaper candidate and
was checked: tripling the budget to 6000 steps moves the answers by under a
percent. It is scenario sampling, so the fix is seeds and not steps.

Read against the two-year figures below, the whole band also shifted *up* — the
longer horizon makes a persisting control stock worth more everywhere.

**The peak moved down when the firm got a liability side**, from equity 16 to
8, and the shape at the thin end changed character. Before, a firm at equity 4
had almost nothing left and simply wound down — its annual *failure* rate read
0.02%, which looked like the safest firm on the page and was in fact the firm
exiting. Now the same firm still carries a deposit-funded business of about 20
in assets, does not exit, and fails at 14.3% a year. It is uneconomic to
protect for the opposite reason: not because there is nothing left, but because
it is too far gone for a programme to pull back.

The going-concern share is also much flatter across the band (0.78–0.82 against
0.43–0.81). With deposits, the amount a failure would destroy scales with the
firm, so how much is at stake stops depending on how capitalised it is.

**One correction to the study itself.** The sweep now scales the terminal
franchise with the firm, as it already scaled the opening GRC stock and for the
same reason. Held fixed at 20 against an equity of 4, the going concern is worth
five times its own book, so a barely-capitalised firm reads as having everything
to protect — and the band comes out monotone, reporting the constant rather than
the capitalisation. That confound was second-order while the balance sheet was
equity alone; once the balance sheet scaled with equity it was not.

**3. Franchise break-even — how much business must be at stake?**

| going concern | spend/qtr | firm value | verdict |
|---|---|---|---|
| 0.0 | 0.0793 | 18.192 | uneconomic |
| 3.0 | 0.0820 | 20.469 | uneconomic |
| 8.0 | 0.1130 | 24.319 | worth running |
| 15.0 | 0.1475 | 29.759 | worth running |

Swept through the going-concern value itself rather than through the
productivity that generates it — sweeping productivity leaves the terminal
franchise pinned, so the firm always has something worth protecting and every
row reports "worth running", which is what it did before this was corrected.

Read it as: *this programme pays if you believe the business is worth at least
about 8 beyond its book value.* GRC is bought by the business it protects, and
with nothing at stake it stops paying. The threshold rose from about 3 with the
liability side, which is the same effect from the other side: a bigger firm
needs a bigger franchise at stake before a programme of the corresponding size
earns its keep.

**4. Value curvature — the gambling-for-resurrection check.** Second difference
of firm value in opening equity. A convex region would mean the firm is
risk-*loving* near failure, and a learner would find it and recommend cutting
GRC in a crisis: correct inside the model and indefensible outside it.

**The convex region appeared with the recalibration and then disappeared again
when credit loss was made to scale with the book:**

| equity | 2.0 | 4.0 | 8.0 | 16.0 | 32.0 | 64.0 |
|---|---|---|---|---|---|---|
| firm value | 16.632 | 20.869 | 26.152 | 33.668 | 46.715 | 72.508 |
| $d^2V/dE^2$ | — | −0.266 | −0.064 | −0.010 | −0.0004 | — |

It was an artifact, and an instructive one. With credit loss fixed rather than
proportional, a small firm faced the *same* expected default loss as a large
one — so a thinly capitalised firm looked far more fragile than it is, its value
collapsed (2.800 at equity 4, against 13.459 now), and the value function turned
convex near the bottom. Once lending less also means losing less, that
fragility goes and the firm is risk-averse everywhere the diagnostic reaches.

**There is now no positive second difference at all** — the largest is
$-4 \times 10^{-4}$, so the firm is concave in equity everywhere the diagnostic
reaches. The faint convexity that survived the liability side has gone with the
run channel, and for a reason worth stating: a thinly capitalised firm now faces
a *higher* run probability, so taking more risk when weak makes a run likelier
rather than merely making the gamble larger. The channel that most plausibly
produces gambling-for-resurrection is the one that removed the last trace of it. The diagnostic stays in place: a genuine convex region would mean the
firm is risk-loving near failure, and a learner would find it and recommend
**cutting GRC in a crisis** — correct inside the model, indefensible outside
it.

**5. Effectiveness break-even, per family.** The original question, bisected on
the *value* a programme adds rather than on the size of its budget — optimal
spend is non-monotone in $\alpha$, since $\ln(\alpha X)/\alpha$ rises then
falls and a very effective programme needs little spending on. Testing budget
materiality reports "no $\alpha$ works" for a programme that is enormously
worthwhile, which is what it did here before being corrected.

| family | risk-neutral $1/X$ | with survival | ratio |
|---|---|---|---|
| credit | 1.725 | 0.1684 | 10× |
| operational | 5.694 | 0.1215 | 47× |
| compliance | 9.924 | 0.1215 | 82× |

Credit's exposure is a *rate on the book*, so the comparison needs the book the
firm funds rather than the bare draw. Read as money it gave a risk-neutral
break-even of 118 rather than 5.0, off by exactly the size of the book, and
credit simply looked like a family nothing could justify protecting.

**Credit is the row the liability side changed, and it is the point.** Its
risk-neutral break-even fell from 5.028 to 1.462 — because the book the rate
applies to is now four times larger, so the same effectiveness buys back four
times as much money. Operational and compliance did not move at all: their
exposures are per-event magnitudes that do not scale with the balance sheet.
**Leverage is what makes loan-default risk worth managing on expected-loss
grounds alone**, and it is why credit's ratio is 15× where the other two are
above 100×.

Read the compliance row as: *this programme pays provided you believe a unit of
spend removes about 12% of exposure, where pure expected-loss reduction would
demand 992%* — which is to say, on expected-loss grounds alone it could never
pay at all. Credit's equivalent pair is 17% against 173%: still a gap, but the
only one of the three where the expected-loss threshold is a number a
practitioner could argue about rather than dismiss. The gap is the Froot-Stein
premium expressed as a threshold.

**Every threshold rose when the asserted operational hazard was retired** —
operational from 0.043 to 0.122, compliance from 0.086 to 0.122, credit from
0.098 to 0.168. A programme now has to clear a higher bar to be worth running,
which is the direct consequence of removing a channel that had been crediting
GRC with preventing deaths it was never shown to prevent. The ratios narrowed
correspondingly: 131× to 47× for operational.

The exposures are computed rather than written in. They were hardcoded at the
static model's 4.0 / 3.5 / 1.5 and are now about twenty-five times smaller, so
the risk-neutral column was being compared against the wrong denominator
entirely.

Credit's ratio is the narrowest because it is the one family with **no hazard
channel of its own** — it reaches survival only indirectly, by leaving more
equity behind.

## 7. Superseded result

An earlier version reported that optimal GRC investment rises with financing-cost
convexity (2.92 → 4.32). That result did not survive dimensional correction of the
cost function and a parameterization that is not permanently insolvent; see
[`critical-review.md`](critical-review.md) F1–F2. The convexity comparative static
is regime-dependent and is no longer claimed.

## 8. Not done

- Calibrating any distribution or $\alpha$ to real data. Every parameter is illustrative.
- Stratified sampling of cliff strikes during training. They are rare by
  construction — about nine paths in four thousand per quarter — so a learned
  policy gets almost no gradient signal about how to behave *after* one, which
  is the hardest decision in the model and the least trained.
- The unimplemented state variables and controls in §1–§2.
- Hard solvency / liquidity / regulatory constraints (§4).
- §5.3, the MDP.
