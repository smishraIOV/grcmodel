# Design: a dynamic Froot–Stein (1998) bank, with GRC as a fourth lever

Status: **draft for discussion**. Nothing here is implemented. It replaces the
current model rather than extending it; the current `main` is to be frozen on a
`legacy` branch before any code lands.

Decisions already taken (2026-11-10):

- The modelling objective is the **integrated framework of Froot & Stein
  (1998)**: risk management, capital budgeting and capital structure decided
  jointly, all priced by one endogenous risk aversion. Froot, Scharfstein &
  Stein (1993) supplies the mechanism that makes the bank care about risk.
- **Infinite horizon, stationary policy**, as the main approach. A finite
  horizon only where something forces it.
- **Discrete time.**
- The practical setting is the one in `readme.md`: a **financial intermediary
  on crypto rails** that takes deposits into vaults, invests them, and runs the
  tokenized infrastructure they live on.
- How that firm's risks split into hedgeable and non-hedgeable is **not
  settled**, and is to be worked out iteratively (§5).
- **Abstraction level:** keep only the paper's core economic drivers, at the
  level of detail where the problem can be solved computationally as an
  optimisation. Matching every detail of the paper is not a goal.
- **Walk-away is in:** shareholders can abandon the firm, and the firm fails.
- **"Partly tradeable" means hedgeable at a price above fair.** GRC mechanism D
  works by lowering that price.
- **`quant/` is replaced, not kept alongside.**
- **Depositors are in from v0**, as a fixed deposit base $D$ at a fixed rate
  $r_D$, rolled over each period. Interest is senior to dividends. Depositors
  do not price risk and there is no capital requirement. If gambling for
  resurrection appears, it is reported, not suppressed.
- Deposits and withdrawals become **dynamic later** (v3): flows in and out, and
  runs, as they behave on crypto rails.

---

## 1. What the 1998 paper actually builds

Read from NBER working paper w5403 (1996), the version of the JFE paper that is
freely available. Equation numbers below are that version's. The JFE version
adds the RAROC comparison; that section still needs checking against the
published text.

**Three dates.**

- **t = 0: capital structure.** The bank holds an existing portfolio paying
  $Z_P = \mu_P + \varepsilon_P$ at t = 2, which needs no financing of its own.
  It chooses equity capital $K$, held as T-bills. Holding it carries a
  deadweight cost $\tau K$; the paper uses taxes as the concrete example.
- **t = 1: hedging and capital budgeting.** The bank picks a size $\alpha$ for
  a new product paying $Z_N = \mu_N + \varepsilon_N$, and a hedge $H$.
  Time-2 wealth is

  $$w = Z_P + \alpha Z_N + Z_H + K(1-\tau) \qquad (1)$$

- **t = 2: the FSS problem.** A non-stochastic investment $F(I)$, concave,
  funded by internal wealth plus external finance $e$ at convex cost $C(e)$:

  $$P(w) = \max_I\ F(I) - I - C(e), \quad I = w + e \qquad (2)$$

  $P$ is increasing and concave. **That concavity is the whole reason the bank
  manages risk.**

**Valuation.** The market prices with one factor $M$:
$V = \{E P(w) - \gamma\,\mathrm{cov}(P(w), M)\}/(1+r)$ (3).

**Results.**

| result | statement | equation |
|---|---|---|
| Prop. 2: hedging | Tradeable risk is always hedged fully: $Z_H = -\varepsilon_P^T - \alpha\varepsilon_N^T + k$ | — |
| Endogenous risk aversion | $G \equiv -E P_{ww} / E P_w$. It falls in $K$, and is 0 as $K \to \infty$ | — |
| Capital budgeting | $\alpha^* = \{\mu_N - \gamma\,\mathrm{cov}(\varepsilon_N^T, M) - G\,\mathrm{cov}(\varepsilon_N^N, \varepsilon_P^N)\} / G\,\mathrm{var}(\varepsilon_N^N)$ | (9) |
| Two-factor hurdle, small position | $\mu_N^* = \gamma\,\mathrm{cov}(\varepsilon_N^T, M) + G\,\mathrm{cov}(\varepsilon_N^N, \varepsilon_P^N)$ | (10) |
| Many products | $\alpha^* = \Omega^{-1}(\pi - G\,C_{NP})/G$. Two kinds of interdependence: a *covariance spillover*, and a *bank-wide cost of capital* acting through $G$ | (11)–(12) |
| Capital structure | $E P_w = 1/(1-\tau)$: hold capital until the expected shadow value of internal funds equals the cost of carrying it | (14) |

Two things the authors name as missing (§V): **a dynamic element**, in which
headquarters pools information about the bank's risks only at intervals and
desk managers price with the last $G$ they were given; and **agency at the desk
level**, the reason banks impose position limits. The first of these is exactly
what this project adds.

---

## 2. Where GRC enters

Froot & Stein give the bank three levers: $K$, $H$ and $\alpha$. GRC is a
**fourth**: a costly, persistent way to change the **non-tradeable** risk the
bank cannot sell in the market. That is the right home for it. Tradeable risk
is hedged at zero NPV, so GRC spent on it is wasted. Non-tradeable risk is
exactly what operational controls, compliance programmes and underwriting act
on.

Four ways a unit of GRC could act. They are not mutually exclusive. Which of
them apply to which risk is the brainstorm in §5.

| mechanism | effect on the non-tradeable risk | what it is worth |
|---|---|---|
| **A. mean** | lowers the expected loss | worth it to a risk-neutral firm; no Froot–Stein content |
| **B. spread** | lowers the variance or tail | worth $\approx \tfrac12 G\,\Delta\mathrm{var}$ in value terms. **This is the Froot–Stein premium** |
| **C. covariance** | lowers the correlation between a product's risk and the existing book's | lowers the hurdle in (10) *for other products*. GRC as an enabler of capital budgeting |
| **D. tradeability** | turns non-tradeable risk into tradeable risk: audited contracts become insurable, a documented book becomes saleable | moves risk from the $G$-priced bucket to the $\gamma$-priced one. The most distinctively Froot–Stein mechanism |

**The headline output this makes possible:** split a GRC programme's value into

- **A**, which any firm would pay for;
- **B + C + D**, which only a capital-constrained firm pays for.

Measure that split as a function of how much capital the bank holds. It is the
"Froot–Stein premium" the old model tried to report, now defined from the
paper's own objects rather than as a gap between two solves.

---

## 3. The dynamic model

### 3.1 Units

Money is measured in units of the investment opportunity's scale: $S = 1$ in
$F$. Every quantity is then a multiple of "one year's worth of the lending
opportunity", which retires the arbitrary 16 in the current readme.

### 3.2 State, controls, timing

**State:** $(w_t, G_t)$, internal wealth and the GRC stock. One GRC family in
v0.

**Within period t:**

1. **Payout / retention (capital structure).** Pay $d_t \in [0, w_t^+]$ and
   retain $K_t = w_t - d_t$. Retained capital costs $\tau K_t$ per period. This
   is (14)'s trade-off, made every period instead of once.
2. **GRC.** Spend $g_t$, so $G_{t+1} = (1-\delta)G_t + g_t$. Mitigation uses
   the post-spend stock, as now.
3. **Positions and hedge (capital budgeting and risk management).** Choose the
   new-product sizes $\alpha_t$ and the hedge $H_t$ on tradeable factors.
4. **Shocks land.** Internal funds become

   $$\tilde w = K_t(1-\tau) - g_t + \varepsilon_P^N(G_{t+1}) + \alpha_t(\pi + \varepsilon_N^N) + (\text{tradeable residual after } H_t)$$

5. **Fund the opportunity (the FSS step).** Choose $I_t$. Equity and deposits
   fund the book. Any shortfall is raised as new equity,
   $e_t = (I_t - \tilde w - D)^+$, at cost $C(e_t)$. Unlent funds are held as
   cash.
6. **Pay depositors, close the books.** Interest $r_D D$ is paid out of the
   period's proceeds, ahead of any dividend. The principal is rolled over:

   $$w_{t+1} = F(I_t) + (\tilde w + D + e_t - I_t) - (1 + r_D)\,D$$

   that is, gross assets less what is owed to depositors. Leverage is
   $D / w$. A loss of a given size is a larger share of equity the thinner the
   capital, which is part of why $G$ rises as capital falls.
7. **Walk-away.** If continuing is worth less than nothing to shareholders,
   they abandon the firm and receive 0. The firm fails. Death is then a
   **decision**, not a hazard parameter. With limited liability, depositors
   receive the remaining assets and bear the shortfall,
   $\big((1 + r_D)D - \text{assets}\big)^+$.

**Gambling for resurrection.** Depositors are paid a fixed rate whatever the
bank's condition, and shareholders are protected by limited liability. So
close to the walk-away boundary, the floor at zero makes equity value *convex*
in $w$: shareholders keep the upside and depositors take the downside. This
works against the Froot–Stein concavity, and the model is free to show
either effect winning. The diagnostic is the sign of $G(w)$. Where $G < 0$
the bank seeks risk, under-hedges and over-sizes positions. Where that region
starts, as a function of capital, is itself a result.

### 3.3 The Bellman equation

$$
V(w, G) = \max_{d, g, \alpha, H}\ \Big\{ d + \beta\, E\big[\, \max\{0,\ W(\tilde w, G')\} \big] \Big\},
\qquad
W(\tilde w, G') = \max_{I}\ \big\{ -e - C(e) + V(w', G') \big\}
$$

Issuance is charged to shareholders at face value plus its deadweight cost.
Concavity of $V$ in $w$ comes out of $F$, $C$ and the default floor. It is not
assumed.

**Dynamic counterparts of the paper's results.** Each one is a test:

- $G(w) = -E[W''] / E[W']$ is a **state-dependent** effective risk aversion.
  The hurdle (10) becomes a schedule in capital, $\mu^*(w)$.
- (14) becomes the **payout boundary** $\bar w$, where $V_w = 1$, with the
  carry cost $\tau$ setting where it sits. As $\tau \to 0$ the bank hoards and
  $G \to 0$.
- Prop. 2 should still hold where $V$ is concave: with fair hedges, $H$ hedges
  fully. Leave $H$ as a free control and check that the solver picks it, rather
  than imposing it. **Near the walk-away boundary it may fail.** That is the
  risk-shifting region above, so the oracle applies only where $G > 0$.

### 3.4 Functional forms (first pass)

- $F(I) = A\,(1 - e^{-I})$ in scale units: concave, with $F'(0) = A$. Kept from
  the current model.
- $C(e) = \lambda e + \tfrac{\kappa}{2} e^2$. The scale is already 1, so this
  is dimensionally clean. The convex part is what generates $G$; the linear
  part produces a pecking order.
- Non-tradeable risk:
  $\varepsilon^N = -\big(m\,e^{-a_A G} + s\,e^{-a_B G}\,\xi\big)$, with
  $\xi$ standardised. That gives **one effectiveness per mechanism**, not one
  per family. Break-evens are then reported over $(a_A, a_B)$ jointly, and the
  old problem of one $\alpha$ driving several channels disappears.

---

## 4. How it is solved

**Exact, for v0.** The state $(w, G)$ is two-dimensional, so value iteration on
a grid is exact up to grid error, and the discount guarantees convergence.

Expectations come from **Gauss–Hermite quadrature** for the Gaussian parts and
a finite mixture for jumps, not Monte Carlo. The grid solution is then
deterministic. No seed noise, no scenario sets that differ by batch size, no
"10% across draws". That removes the whole class of problems in `CLAUDE.md`'s
traps list that came from sampling.

**Oracles:**

1. $\beta \to 0$, one period: recovers (9), (10) and (14) exactly, with $P$ in
   closed form for quadratic $C$.
2. $\tau \to 0$: $G \to 0$ and the bank hoards.
3. Prop. 2: full hedging of tradeable risk.
4. No non-tradeable risk: GRC's value reduces to mechanism A alone.

**The learner** comes in when the state outgrows a grid: several GRC families,
then deposits. It is trained on the same stationary problem and **graded
against the grid** wherever the grid can still reach. An infinite horizon is
precisely where a critic belongs. The `svg-critic` branch's reason for being
parked — it lost to full backpropagation through time at short finite horizons
— does not carry over to this setting.

**Carried over from the current code:** `quant/numerics.py` and the profile
rule, CPU-drawn randomness for the learner's rollouts, the softplus/logit
guards, and the testing discipline.

**Not carried over:** the hazard stack, the terminal franchise,
`initial_grc_stock` as a hand-solved fixed point (it becomes the stationary
distribution), the finite-horizon rollout as the main objective, and the
static-model lineage.

---

## 5. Risk classification for the crypto-rails bank (open, iterative)

A first, deliberately tentative cut. Each row is a question, not an answer.

| risk | tradeable? | instrument, if any | GRC mechanism |
|---|---|---|---|
| Crypto price risk on collateral and treasury | **mostly yes** | perps, futures, options | none needed, per Prop. 2 |
| Interest-rate / funding-rate risk | **yes** | T-bills, rate swaps, on-chain fixed-rate markets | none |
| Stablecoin depeg | **partly** | thin depeg cover, diversification across issuers | B (exposure limits), D? |
| Idiosyncratic credit (crypto borrowers, protocols) | **no** | — | A, B (underwriting); C (concentration limits) |
| Smart-contract exploit, bridge drain | **partly** | on-chain cover protocols: thin, costly, capacity-limited | B (audits shrink the tail), **D (audits make cover purchasable)** |
| Custody / key management | **partly** | custody insurance, limited capacity | A, B, D |
| Operational incidents | **no** | — | A, B |
| Compliance / licence | **no** | — | A (fewer breaches); B (a lost licence is the tail) |

Questions for the next pass:

- **Decided:** the "partly" bucket is tradeable at a **cost**, a hedge priced
  above fair, which the paper does not do. GRC through D lowers that cost.
  Still open: whether the cost is linear or convex in the amount hedged.
- Is the existing book $\varepsilon_P^N$ the vault/lending business, with each
  new strategy a new product $\alpha_i$? That reading makes eq. (12) the
  question of how much the bank allocates to each strategy.
- The paper's open dynamic question — how often headquarters should refresh
  $G$ — has a direct reading here: how often the risk function re-prices
  strategy hurdles. Worth keeping in view as a v1 or v2 result.

---

## 6. Staging

| stage | adds | solved by | done when |
|---|---|---|---|
| **v0** | one non-tradeable risk, one tradeable factor, one new product, one GRC family, a fixed deposit base at a fixed rate, walk-away; state $(w, G)$ | grid VI with quadrature | the four oracles in §4 pass, and the sign of $G(w)$ is mapped |
| **v1** | several products with a covariance matrix: eqs (11)–(12), the hurdle schedule $\mu^*(w)$, the A versus B+C+D split | grid | the split is reported across capital levels |
| **v2** | GRC by family (credit, operational, compliance), mechanisms C and D | learner, checked against the grid on the v1 slice | learner within tolerance of the grid where both run |
| **v3** | deposits as a dynamic stock: inflows, withdrawals and runs, as on crypto rails; then depositors who price risk, or a minimum capital ratio, as alternatives to the fixed rate | learner ($D$ becomes a state variable) | the extension changes some answer; otherwise it is not merged |

Each stage needs approval before it starts. v0 is a few sessions.

---

## 7. Open questions

1. ~~Default~~ — decided: walk-away is in.
2. **Payout.** Needed, because it is the dynamic form of the capital-structure
   choice; see §3.2 step 1. One payout point, at the start of the period. It is
   likely to reduce to a threshold $\bar w$; test that rather than impose it.
3. ~~Package layout~~ — decided: replace `quant/`.
4. **RAROC.** Check the JFE text for the comparison section, and decide whether
   it becomes a v1 output: standalone-risk pricing against the
   covariance-with-book pricing in (10).
