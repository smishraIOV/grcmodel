# Model specification

What the implemented model *is*, as of the current commit. No history, no
superseded results, no narration of corrections. Where a claim here and the code
disagree, that is a bug in one of them, and `tests/test_spec.py` fails the build
on the subset it can check mechanically.

For **results** see [`quant-model.md`](quant-model.md) §6; for **why parameters
are what they are** see [`calibration.md`](calibration.md); for the **conceptual
frame** see [`framework.md`](framework.md). This file is the reference the other
three cite for definitions.

Every symbol used anywhere below is defined in §1 before it is used. Symbols are
not reused for two meanings.

---

## 1. Notation

### 1.1 Balance sheet and firm

| symbol | meaning | units | parameter | default |
|---|---|---|---|---|
| $E_t$ | equity at the start of period $t$ | money | — | — |
| $E_0$ | reference equity; fixes the money scale | money | `FirmParams.initial_equity` | 16.0 |
| $\kappa$ | capital ratio $E/E_0$ | — | — | — |
| $D_t$ | deposit stock outstanding | money | — | — |
| $\Lambda$ | deposits carried per unit of equity at full market access | — | `FundingParams.deposit_capacity` | 4.0 |
| $\kappa_D$ | capital ratio at which market access is half open | — | `FundingParams.access_ratio` | 0.15 |
| $s_D$ | width of the market-access transition | — | `FundingParams.access_scale` | 0.08 |
| $\theta$ | fraction of the gap to deposit capacity closed per period | — | `FundingParams.annual_adjustment_speed` | 2.0 |
| $r_D$ | interest paid on deposits, per year | 1/year | `FundingParams.annual_deposit_rate` | 0.02 |
| $r_R$ | interest earned on reserves, per year | 1/year | `FundingParams.annual_reserve_rate` | 0.015 |
| $m$ | decision periods per year | 1/year | `FirmParams.periods_per_year` | 4 |
| $\beta$ | discount factor per period | — | `FirmParams.annual_discount_rate` | 0.08 |

### 1.2 GRC

Families are indexed $f \in \{c, o, k\}$ — credit, operational, compliance.

| symbol | meaning | units | parameter | default |
|---|---|---|---|---|
| $g_{f,t}$ | spend on family $f$ this period | money | — | — |
| $G_{f,t}$ | GRC capital stock for family $f$ | money | — | — |
| $G_{f,0}$ | opening stock, per family | money | `FirmParams.initial_grc_stock` | 0.79 |
| $\delta$ | fraction of the stock lost per period | — | `FirmParams.annual_grc_depreciation` | 0.25 |
| $\alpha_f$ | mitigation effectiveness, family $f$ | 1/money | `GrcAlphas.credit` | 1.5 |
| | | | `GrcAlphas.operational` | 1.5 |
| | | | `GrcAlphas.compliance` | 1.5 |

$\alpha_f$ is a **rate**, not a fraction: one unit of stock removes
$1 - e^{-\alpha_f}$ of the family's exposure. See §6 for which channels each
$\alpha_f$ drives — the count differs by family.

### 1.3 Losses

| symbol | meaning | units | parameter | default |
|---|---|---|---|---|
| $\ell_c$ | credit loss rate on the deployed book, per year | 1/year | `AnnualRates.credit_loss_rate` | 0.034 |
| $\lambda_o$ | operational incidents per year | 1/year | `AnnualRates.op_events` | 1.0 |
| $\mu_o$ | mean operational loss per incident | money | `AnnualRates.op_severity` | 0.70 |
| $\lambda_k$ | compliance breaches per year | 1/year | `AnnualRates.compliance_events` | 0.20 |
| $\mu_k$ | penalty per breach | money | `AnnualRates.compliance_severity` | 2.0 |
| $\lambda_X$ | cliff strikes per year at zero GRC | 1/year | `AnnualRates.cliff_events` | 0.101272 |
| $\mu_X$ | mean cliff severity, as a share of opening equity | — | `CliffParams.mean_severity_fraction` | 0.35 |
| $L_c, L_o, L_k$ | mitigated loss by family, this period | money | — | — |
| $L_X$ | cliff loss this period | money | — | — |
| $L_{\text{cash}}$ | $L_o + L_k$ — losses paid out | money | — | — |
| $L_{\text{asset}}$ | $L_c + L_X$ — losses that impair assets held | money | — | — |

### 1.4 The run

| symbol | meaning | units | parameter | default |
|---|---|---|---|---|
| $\lambda_R$ | runs per year at zero GRC and full capitalisation | 1/year | `FundingParams.annual_run_rate` | 0.45 |
| $p_R$ | probability of a run this period | — | — | — |
| $\tau_R$ | capital ratio around which run risk turns | — | `FundingParams.run_capital_target` | 0.75 |
| $s_R$ | width of that transition | — | `FundingParams.run_capital_scale` | 0.35 |
| $\omega$ | share of the deposit base withdrawn, given a run | — | `FundingParams.mean_withdrawal_fraction` | 0.35 |
| $W$ | withdrawal demanded, $\omega D$ | money | — | — |
| $h$ | fire-sale haircut on the unmatured book | — | `FundingParams.fire_sale_haircut` | 0.35 |
| $X$ | cash raised by the fire sale | money | — | — |
| $u$ | share of the withdrawal left unmet | — | — | — |
| $R$ | reserves the firm chose to hold | money | — | — |
| $\tilde{R}$ | reserves surviving the period's cash losses | money | — | — |
| $B$ | book deployed | money | — | — |
| $\tilde{B}$ | book a fire sale can raise against | money | — | — |

### 1.5 Failure intensity

The hazard is $\eta$, never $h$ — $h$ is the haircut and nothing else.

| symbol | meaning | units | parameter | default |
|---|---|---|---|---|
| $\eta$ | total failure intensity this period | 1/period | — | — |
| $\eta_E$ | capital channel | 1/period | — | — |
| $\eta_\ell$ | liquidity channel | 1/period | — | — |
| $\eta_k$ | licence channel | 1/period | — | — |
| $\bar{\eta}_E$ | capital-channel intensity at $\kappa = \kappa^\star$, per year | 1/year | `HazardParams.annual_base_rate` | 0.5 |
| $\kappa^\star$ | capital ratio at which $\eta_E$ equals its base | — | `HazardParams.capital_target` | 0.4 |
| $s_E$ | width of the capital-hazard response | — | `HazardParams.capital_scale` | 0.2 |
| $\bar{\eta}_\ell$ | intensity at a wholly unmet withdrawal, per year | 1/year | `HazardParams.annual_liquidity_failure_rate` | 25.0 |
| $\bar{\eta}_k$ | licence-revocation intensity at zero GRC, per year | 1/year | `HazardParams.annual_licence_rate` | 0.08 |
| $S_t$ | probability the firm is still operating at $t$ | — | — | — |

### 1.6 Production and value

$S$ is the production curvature throughout; cash raised in a fire sale is $X$.

| symbol | meaning | units | parameter | default |
|---|---|---|---|---|
| $I$ | capital deployed | money | — | — |
| $A$ | gross return on deployed capital, per year | — | `AnnualRates.annual_return` | 1.15 |
| $S$ | production curvature, per period | money | `AnnualRates.curvature_per_period` | 698.0 |
| $F(I)$ | value returned by a book of $I$, principal included | money | — | — |
| $\Phi$ | terminal franchise: going-concern value beyond book | money | — | 20.0 |
| $\rho_f$ | recovery on disorderly failure, as a share of positive equity | — | `FirmParams.failure_recovery` | 0.4 |
| $\rho_o$ | recovery on orderly wind-down | — | `FirmParams.orderly_recovery` | 0.7 |
| $a_t$ | probability of winding down at the start of $t$ | — | — | — |
| $\pi_t$ | share of the period's profit distributed | — | — | — |
| $T$ | horizon, in periods | — | — | — |

### 1.7 Differentiability devices

| symbol | meaning | units | parameter | default |
|---|---|---|---|---|
| $\tau_X$ | straight-through temperature, cliff | — | `CliffParams.relaxation_temperature` | 0.1 |
| $\tau_R^{\text{st}}$ | straight-through temperature, run | — | `FundingParams.relaxation_temperature` | 0.1 |

### 1.8 Annual rates to per-period quantities

`quant/params.model_at(m)` converts §1.3's annual statement into the per-period
parameters the sampler uses. Rates that count events convert as Poisson,
$1 - e^{-\lambda/m}$; severities do not convert at all, because an incident is
the size it is however often you look; gross return compounds down,
$A^{1/m}$; and the production curvature scales *up* as $S \cdot m$, so that
$I^\star = S \ln A$ is the same amount of capital at any frequency.

The discount and every hazard rate divide by $m$ directly.

---

## 2. Timing within a period

The order matters and is not recoverable from the equations alone. This is
`StandardDynamics.step` in `quant/env/dynamics.py`, in sequence.

1. **Spend.** $g_{f}$ is committed. $w^- = E_t - \sum_f g_f$.
2. **Accumulate.** $G_{f,t+1} = (1-\delta)G_{f,t} + g_f$. Mitigation everywhere
   below uses this *post-spend* stock, so a control bought now protects now.
3. **Size the book.** $I = \min(I^{\text{desired}},\, w^- + D_t)$ — the balance
   sheet is the only constraint, and it binds *before* losses are known, because
   a bank sets its book from the capital it has at the start of the period.
4. **Reserves.** $R = \max(w^- + D_t - I,\, 0)$: what was fundable and not lent.
5. **Losses land**, on the book just chosen. $L = L_c + L_o + L_k + L_X$ and
   $w = w^- - L$.
6. **The run is drawn on post-loss equity** $w$, so a bad period draws the run
   that then makes it worse.
7. **Meet the withdrawal** out of $\tilde{R}$ first, then by selling $\tilde{B}$
   at the haircut.
8. **Produce** on the book that survived the fire sale.
9. **Close the books**, distribute, and evaluate the hazard on closing equity.
10. **Deposits adjust** toward the capacity closing equity supports.

Steps 3 and 5 in that order are what make the Froot–Stein channel bite through
the *next* period rather than this one: a bad draw does not shrink today's book,
it shrinks tomorrow's capacity.

---

## 3. State and controls

**State** $(E_t,\ D_t,\ G_{c,t}, G_{o,t}, G_{k,t},\ \text{alive}_t)$ —
`FirmState`. `deposits` is `None`, not zero, in configurations with no liability
side; a firm with no liability side pays no interest and has no capacity beyond
its own equity, which is a different object from one funding nothing.

**Controls** $(g_c, g_o, g_k,\ I,\ a,\ \pi)$ — `FirmAction`. Spend and
investment pass through `softplus`; $a$ and $\pi$ through `sigmoid`. Never
`clamp`: its gradient is exactly zero at the boundary, so a control initialised
at zero would never move.

A policy sees only the state. `rollout` hands the policy the state and *then*
draws the shock, so non-anticipativity is a property of the policy's type rather
than of the transition.

---

## 4. The balance sheet

**Deposit capacity and its adjustment.** Capacity is a leverage limit times a
market-access sigmoid, both in the capital ratio:

$$
\bar{D}(E) = \Lambda \cdot \max(E, 0) \cdot \varsigma(E),
\qquad
\varsigma(E) = \operatorname{sigmoid}\!\left(\frac{\kappa - \kappa_D}{s_D}\right)
$$

with $\kappa = E/E_0$ as in §1.1. Both terms fall together when the firm
weakens — a smaller multiple of a smaller number, times a market that is
closing.

The stock moves toward capacity rather than arriving at it:

$$
D_{t+1} = \max\!\big(D_t' + \theta_m\,(\bar{D}(E_{t+1}) - D_t'),\ 0\big),
\qquad \theta_m = 1 - e^{-\theta/m}
$$

where $D_t' = \max(D_t - \text{paid},\, 0)$ is the base after any withdrawal was
met. **The partial adjustment is what makes $D$ a state variable**: at
$\theta_m = 1$ it collapses to a function of $E$ and nothing is outstanding that
could run.

**Interest.** $r_D$ is paid on the opening base whether or not the book earns;
$r_R$ is earned on reserves that survived the period. Both compound down from
the annual figure, $(1+r)^{1/m} - 1$. $r_R < r_D$, so an idle balance sheet
carries a small negative spread — which is what makes a liquidity buffer a
decision rather than a free good.

**Funding capacity.** $I \le w^- + D_t$. No free parameter is needed; the
balance sheet already states it.

---

## 5. Losses, and where they land

Each family is mitigated on the moment it actually moves:

$$
L_c = \ell_c^{(m)} \, e^{-\alpha_c G_c} \cdot I,
\qquad
L_o = \mathbb{1}_o \cdot \mu_o \, e^{-\alpha_o G_o},
\qquad
L_k = \mathbb{1}_k \cdot \mu_k
$$

Credit is a **rate on the book**, so lending twice as much costs twice the
expected defaults. Operational is an indicator times a severity GRC *contains*.
Compliance is an indicator times a fixed penalty — GRC does not soften it, it
makes it rarer, and that acts through the path weight in §8 rather than here.

**The cliff** is a severe, survivable hit whose *frequency* GRC reduces and
whose severity it does not:

$$
L_X = \mathbb{1}_X \cdot \phi \cdot \max(E_t, 0),
\qquad
p_X = p_X^{(m)} e^{-\alpha_o G_o},
\qquad
\phi \sim \min(\operatorname{Exp}(\mu_X),\ 1)
$$

**Where a loss lands decides whether the firm can still pay depositors**, and
this is the distinction the single total could not carry:

$$
L_{\text{cash}} = L_o + L_k \quad\text{(paid out)},
\qquad
L_{\text{asset}} = L_c + L_X \quad\text{(assets impaired)}
$$

$$
\tilde{R} = \max(R - L_{\text{cash}},\, 0),
\qquad
\sigma = \max(L_{\text{cash}} - R,\, 0),
\qquad
\tilde{B} = \max(I - L_{\text{asset}} - \sigma,\, 0)
$$

Cash losses come out of the buffer; what they overrun ($\sigma$) must be found
from the book, so it is not available to sell either. **Neither is a second
charge** — the whole loss is taken once, against equity, in $w = w^- - L$.

---

## 6. Which channels each $\alpha_f$ drives

One $\alpha$ per family, but the families do not buy the same bundle. Pinned by
`tests/test_alphas.py`.

| family | channels | where |
|---|---|---|
| credit | **1** — the loss mean | §5 |
| compliance | **2** — the breach probability, the licence intensity | §8, §7 |
| operational | **3** — the loss severity, the cliff frequency, the run frequency | §5, §5, §7 |

So the three break-evens in `quant-model.md` §6.5 are not like-for-like: the
operational figure is the effectiveness at which *three* mitigations together
pay for themselves. Reusing one $\alpha$ per family is deliberate — nothing here
can calibrate one, let alone three.

A fourth operational channel exists in `quant/hazard.py` and is inert, because
$\bar{\eta}_o = 0$: it was replaced by the run mechanism rather than deleted.

---

## 7. The run, and failure

**Run probability**, evaluated on equity *after* the period's losses:

$$
p_R = \frac{\lambda_R^{(m)}}{1} \, e^{-\alpha_o G_o}
      \cdot \exp\!\left(\frac{\tau_R - \kappa_w}{s_R}\right),
\qquad \kappa_w = w/E_0
$$

with $\lambda_R^{(m)} = 1 - e^{-\lambda_R/m}$, and the product floored off 0 and
capped off 1 because §9 takes its logit. Three separate claims: how often an
incident becomes public; that operational controls make it rarer; and that thin
capital draws depositors out. The capital term is deliberately gentler and
earlier than $\eta_E$'s — depositors leave on the suspicion of insolvency, not
on its confirmation.

**Meeting it.** A withdrawal $W = \omega D_t$ is paid from $\tilde{R}$ at par,
then by liquidating book. Raising $X$ costs $X/(1-h)$ of book and destroys
$X h/(1-h)$:

$$
\text{from reserves} = \min(W, \tilde{R}),
\qquad
\text{liquidated} = \min\!\left(\frac{W - \text{from reserves}}{1-h},\ \tilde{B}\right)
$$

$$
u = \frac{\max(W - \text{paid},\ 0)}{\max(W,\ \epsilon)}
$$

**A full run is meetable exactly when**

$$
h\,I \le (E_t - \textstyle\sum_f g_f) - L_{\text{cash}} - (1-h)\,L_{\text{asset}}
$$

so a cash loss costs more liquidity than an asset loss of the same size, by
exactly the haircut. With no losses this reduces to $I \le E/h$, independent of
the deposit base, which cancels. Asserted path-by-path in `tests/test_run.py`.

**Failure intensity.** Three competing risks, composed additively because
intensities add while survival probabilities multiply:

$$
\eta = \underbrace{\frac{\bar{\eta}_E}{m}\exp\!\left(\frac{\kappa^\star - \kappa'}{s_E}\right)}_{\eta_E}
     + \underbrace{\frac{\bar{\eta}_\ell}{m}\,u}_{\eta_\ell}
     + \underbrace{\frac{\bar{\eta}_k}{m}\,e^{-\alpha_k G_k}}_{\eta_k}
$$

evaluated on closing equity $\kappa' = E_{t+1}/E_0$. $\eta_E$ is exponential
rather than logistic because a logistic saturates: at its base rate even a firm
with deeply negative equity would survive the period, which is not a description
of insolvency. Its exponent is clamped at 50, so an intensity is enormous rather
than infinite.

$\eta_\ell$ is **linear** in $u$, which is already a fraction, and the rate is
calibrated at its top end: a firm that meets none of a withdrawal survives the
period with probability $e^{-25/m}$. Credit has no channel of its own — bad
underwriting erodes equity and equity is $\eta_E$'s argument, so a fourth channel
would count one mechanism twice.

**Death is not sampled.** Each path carries $S_{t+1} = S_t(1-a_t)e^{-\eta}$,
accumulated in log space, and value is an expectation over survival. A sampled
death is a step function of equity and carries no gradient.

---

## 8. Production, accounting and the objective

**Production.** $F(I) = A^{(m)} S\left(1 - e^{-I/S}\right)$, computed on the book
that survived the fire sale. Concave, with $F'(0) = A$ and unconstrained optimum
$I^\star = S \ln A$. Written with `expm1`, because the quantity that matters is
$F(I) - I$ — about 2% of $I$ at the calibrated curvature — and computing it as
`1 - exp(-x)` discards most of its significant digits.

**The accounting identity**, asserted elementwise to $10^{-10}$ in three tests:

$$
E_{t+1} = w - I + F(B) + X - P - r_D^{(m)}D_t + r_R^{(m)}\tilde{R}^- - \text{dividend}
$$

where $\tilde{R}^-$ is reserves net of what depositors took and $P$ is the convex
financing premium, which is **switched off** in every current configuration
(`financing_scale = 0`).

**Dividends** come out of the period's profit, never the capital base:
$\text{dividend} = \pi \cdot \max(\text{gross} - E_t,\, 0)$. Without that
constraint the control is a way to strip the firm.

**The objective.** Maximise expected discounted value over the horizon:

$$
V = \mathbb{E}\left[\sum_{t=0}^{T-1} \beta^t \Big(
      S_t a_t \rho_o E_t^+
    + \beta S_{t+1}\,\text{dividend}_t
    + \beta\,(S_t(1-a_t) - S_{t+1})\,\rho_f E_{t+1}^+
    \Big) + \beta^T S_T (E_T + \Phi)\right]
$$

with $x^+ = \max(x, 0)$ by limited liability. Exits are collected at the date
they happen; a wind-down in period one and a failure in period sixty are not
worth the same.

**The compliance channel acts on the expectation, not the dynamics.** A breach
is discrete, so its budget cannot shrink a drawn indicator. Indicators are drawn
once at a base probability $p_k$ and each path is reweighted by the likelihood
ratio $p_k(g)/p_k$ on breach paths and $(1-p_k(g))/(1-p_k)$ otherwise. Unbiased,
differentiable in $g_k$, and it keeps breaches discrete. The cost is that
weights multiply along a trajectory, so effective sample size decays with
*calendar time* rather than step count.

---

## 9. Why the model is differentiable

Almost nothing here is naturally smooth, and the differentiability is
engineered, mechanism by mechanism.

| naturally discrete | what the model does | cost |
|---|---|---|
| death | carries survival as a probability, in log space | none — strictly better than sampling |
| cliff strike, run | straight-through: forward is the hard indicator, backward differentiates a tempered sigmoid of the same threshold | biased gradient; the temperature controls the bias, which does **not** compound along the path |
| compliance breach | likelihood-ratio reweighting | unbiased; variance compounds along the path |
| wind-down | relaxed to a probability in $[0,1]$ | none — value is linear in it, so the optimum is at an endpoint |
| non-negative spend | `softplus` | a true optimum of exactly zero is approached asymptotically |
| failing to pay depositors | not a second death branch; $u$ feeds $\eta_\ell$ linearly | none |

**Two devices for one problem, deliberately.** Reweighting is unbiased but its
variance compounds; a straight-through estimator is biased but its bias does
not. Compliance got reweighting when the horizon was short; the cliff and the
run arrived later and at longer horizons, where compounding variance would have
been fatal. Neither is better — they fail in different directions and the
horizon decides which failure is affordable.

**Where the gradient is dead**, known rather than discovered: below zero equity
(limited liability floors recovery); below $\kappa \approx -9.6$ (the clamped
hazard exponent); wherever the straight-through sigmoids saturate; in a period
that made no profit (the payout control is flat); once the liquidity buffer is
exhausted ($\tilde{R}$ stops responding to the lending decision); and across the
whole region a badly initialised policy wanders into, which is why the
perfect-information bound must be warm-started.

---

## 10. Deliberately not in this specification

**Parameters that exist in `quant/params.py` and describe something other than
the current model.** `tests/test_spec.py` requires every one of these to be
listed here with a reason, so nothing is omitted silently.

| parameter | why not |
|---|---|
| `FirmParams.financing_convexity` | superseded convex-financing friction; `financing_scale = 0` in every current configuration |
| `FirmParams.distress_reference` | as above |
| `FirmParams.external_funding_multiple` | pre-liability funding rule, replaced by the deposit stock |
| `FirmParams.market_access_ratio` | as above |
| `FirmParams.market_access_scale` | as above |
| `FirmParams.production_scale` | derived from `AnnualRates.annual_return` by `model_at` |
| `FirmParams.production_curvature` | derived from `AnnualRates.curvature_per_period` by `model_at` |
| `HazardParams.annual_operational_rate` | retired; replaced by the run mechanism, kept at 0.0 so the channel can be revived |
| `CliffParams.period_probability` | derived from `AnnualRates.cliff_events` by `model_at` |
| `SamplerParams.*` | per-period image of `AnnualRates`, produced by `model_at` |
| `ShockParams.*` | the four-state oracle's world, deliberately pinned and never recalibrated |

**Mechanisms not modelled at all.** A run damages the deposit *level* but not
the franchise — deposits rebuild on a four-month half-life whether they left in
a panic or drifted away. Operational and compliance losses do not scale with the
balance sheet. GRC cannot reduce the *cost* of a failure, only its probability.
The book matures within the period, so the only maturity mismatch is the one the
run creates inside a single period. Portfolio composition, deposit pricing and
an evolving operational-risk score are state variables the model does not carry.
