# Design: a dynamic Froot–Stein (1998) bank, with GRC as a fourth lever

Status: **v0 built; v1 under way.** Built (`quant/`): the one-period model,
the quadrature (a deterministic way of computing expected values from a small
set of weighted scenarios, in place of random simulation), the two-period
oracle (a known-answer check), the infinite-horizon grid solver, the outputs
(effective risk aversion, hurdles, thresholds, GRC's value to the bank), and
v1's jump losses, all at a configurable decision frequency (quarterly by
default, §3.6). Still to come in v1: several products, and GRC lowering the
cost of hedging. The previous model is frozen on the `legacy` branch.

Decisions already taken (2026-11-10):

- The modeling objective is the **integrated framework of Froot & Stein
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
  optimization. Matching every detail of the paper is not a goal.
- **A practical model, not a replication.** The paper's simplifying
  assumptions exist to keep its analysis tractable; a computational model does
  not need them. Departures are fine, but each one is justified and listed in
  §8.
- **Walk-away is in:** shareholders can abandon the firm, and the firm fails.
- **"Partly tradeable" means hedgeable at a price above fair.** GRC mechanism D
  works by lowering that price.
- **`quant/` is replaced, not kept alongside.**
- **Depositors are in from v0**, as a fixed deposit base $D$ at a fixed rate
  $r_D$, rolled over each period. Interest is senior to dividends. Depositors
  do not price risk. If gambling for resurrection appears, it is reported, not
  suppressed.
- **No negative equity, and a minimum capital ratio.** A bank below its
  requirement either recapitalizes or shrinks its book. If neither is worth
  it, it shuts down. There is no operating with negative equity.
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
exactly what operational controls, compliance programs and underwriting act
on.

Four ways a unit of GRC could act. They are not mutually exclusive. Which of
them apply to which risk is the brainstorm in §5.

| mechanism | effect on the non-tradeable risk | what it is worth |
|---|---|---|
| **A. mean** | lowers the expected loss | worth it to a risk-neutral firm; no Froot–Stein content |
| **B. spread** | lowers the variance or tail | worth $\approx \tfrac12 G\,\Delta\mathrm{var}$ in value terms: the Froot–Stein channel |
| **C. covariance** | lowers the correlation between a product's risk and the existing book's | lowers the hurdle in (10) *for other products*. GRC as an enabler of capital budgeting |
| **D. tradeability** | turns non-tradeable risk into tradeable risk: audited contracts become insurable, a documented book becomes saleable | moves risk from the $G$-priced bucket to the $\gamma$-priced one. The most distinctively Froot–Stein mechanism |

**What is reported:** GRC's value to this bank — the bank with a program
against the same bank without one — and how that value moves with its capital,
its risks and the size of its losses.

**What is deliberately not measured: a "Froot–Stein premium".** The appeal was
to split GRC's value into what any firm would pay and what only a
capital-constrained firm pays. Two ways were built and both were removed
(2026-11-11):

- *By mechanism* (mean versus spread). Stopped working once GRC prevents jumps,
  which removes expected loss and tail risk at once.
- *Against a frictionless twin*, the same bank able to raise equity for free.
  Nobody raises equity for free, so the comparison is an academic construction
  with no quantitative meaning, and the qualitative point it would make is
  already known from the paper. It also mixed in a second effect: the real bank
  walks away from losses large enough to sink it, so preventing those losses
  protects depositors rather than shareholders. Separating the two would need a
  bank forced to recapitalize after any loss — unlimited liability, which is no
  more realistic.

**Acknowledged, not measured: limited liability weakens the incentive to
invest in GRC.** For losses that threaten survival, shareholders bear only
part of the damage and depositors the rest, so the bank may spend less on
preventing them than their full cost would justify. The model shows this
qualitatively; it does not put a number on it.

---

## 3. The dynamic model

### 3.1 Units

Money is measured so that the **first-best book is 1**: the amount a bank with
unlimited free capital would lend each period. Every quantity is then a
multiple of that, which retires the arbitrary 16 in the old readme.

### 3.2 State, controls, timing

**State:** $(w_t, G_t)$ — book equity at the start of the period, and the GRC
stock. One GRC family in v0. The deposit base $D$ is a fixed parameter in v0,
not state.

**One decision per period, taken before the period's shocks.** Earlier drafts
split the period in two, as the paper does: positions first, then a riskless
funding stage after the shocks. Here the lending book itself carries the
non-tradeable risk, because for a bank it *is* the main non-tradeable risk. So
everything is chosen together and the shocks land on all of it. The
Froot–Stein channel runs through the *next* period: a bad draw leaves less
equity, and next period the bank must either raise costly equity or lend less.

**Controls at the start of period $t$:**

| control | meaning | paper's lever |
|---|---|---|
| walk-away | abandon the firm; shareholders get 0 | — |
| $d_t \ge 0$ | dividend paid | capital structure, $K$ |
| $e_t \ge 0$ | new equity raised, at cost $C(e_t)$ | the FSS friction |
| $g_t \ge 0$ | GRC spend | the new lever |
| $I_t \ge 0$ | size of the lending book | FSS investment |
| $\alpha_t$ | size of the new product | capital budgeting |
| $H_t$ | hedge on the tradeable factor | risk management |

Paying and raising in the same period is dominated, so only one of $d, e$ is
non-zero in practice. The solver is not told this.

**Balance sheet after the decision:**

$$
K_t = w_t - d_t + e_t \ \ge\ k\,I_t,
\qquad
R_t = K_t - g_t + D - I_t \ \ge 0
$$

$K_t$ is the equity the bank operates with. $k$ is a **minimum capital
ratio** against the book, the regulatory constraint of an ordinary bank:
risk-weighted assets with a single weight. $k = 0$ leaves only "no negative
equity". A bank that a bad period leaves below the requirement has three
choices:

- recapitalize, at cost $C(e)$;
- deleverage, by shrinking $I_t$ until $K_t \ge k I_t$;
- shut down, by walking away.

It is never forced to shut. Shutting is chosen when the other two are worth
less than nothing.

$R_t$ is cash, what was funded and not lent. These two constraints are what
make capital scarce. The capital ratio adds a second route for the Froot–Stein
channel: a loss now shrinks the book the bank is *allowed* to lend, not only
the book it can fund.

Later: risk weights per asset (the new product $\alpha$, the hedge), and a
liquidity requirement on $R_t$. In v0, $\alpha$ carries no capital charge.

**GRC stock:** $G_{t+1} = (1-\delta)G_t + g_t$. Mitigation uses the
post-spend stock $G_{t+1}$, so spend protects in the period it is made.

**Shocks land, and the books close:**

$$
w_{t+1} =
\underbrace{F(I_t) - I_t\,\ell_t}_{\text{book}}
+ \underbrace{(1+r)\,R_t}_{\text{cash}}
+ \underbrace{\alpha_t(\pi + s_N\,\xi_N)}_{\text{new product}}
+ \underbrace{(I_t\,b_M - H_t)\,\varepsilon_M - h(H_t)}_{\text{tradeable, hedged}}
- \tau K_t
- (1 + r_D)\,D
$$

- $\ell_t = \phi_A(G_{t+1})\,m + \phi_B(G_{t+1})\,s_P\,\xi_P$ is the book's
  non-tradeable loss rate. It scales with the book: lending twice as much
  carries twice the credit risk.
- $b_M$ is the book's loading on the tradeable factor (crypto collateral
  prices, say), and $H_t$ sells it off. $h(H)$ is the hedging cost, zero when
  the factor is fairly tradeable.
- $\tau K_t$ is an optional extra deadweight cost of carrying equity. Its
  default is zero: the cost of holding equity comes from shareholders
  discounting at $r_E > r$ while the cash that equity funds earns only $r$
  (§3.3, §8).
- Depositors are owed $(1+r_D)D$ at the close, so $w_{t+1}$ is assets less
  that. The base is rolled over: the re-deposited $D$ arrives as cash next
  period through $R_{t+1}$. Adding it back here as well would count it twice;
  an earlier draft did. Interest is senior to everything the shareholders
  receive.

**Walk-away.** At the start of any period, shareholders may abandon the firm
and receive 0. Depositors then receive the assets and bear the shortfall,
$(-w_t)^+$. With a negative $w_t$ the choice is whether recapitalizing to
$K \ge 0$, at cost $C(e)$, is worth more than nothing. Failure is therefore a
decision, not a hazard parameter.

**Gambling for resurrection.** Depositors are paid a fixed rate whatever the
bank's condition, and shareholders are protected by limited liability. So
close to the walk-away boundary, the floor at zero makes equity value *convex*
in $w$: shareholders keep the upside and depositors take the downside. This
works against the Froot–Stein concavity, and the model is free to show
either effect winning. The capital ratio limits how far this can go, because a
bank near the boundary is also near its requirement. How much risk-shifting
survives a given $k$ is a result too. The diagnostic is the sign of $G(w)$. Where $G < 0$
the bank seeks risk, under-hedges and over-sizes positions. Where that region
starts, as a function of capital, is itself a result.

### 3.3 The Bellman equation

$$
V(w, G) = \max\Big\{\, 0,\ \max_{d, e, g, I, \alpha, H}\ \big[\, d - e - C(e) + \beta\, E\, V(w', G') \,\big] \Big\}
$$

subject to the two balance-sheet constraints of §3.2 (capital ratio and
non-negative cash), with
$\beta = 1/(1+r_E)$, the shareholders' required return. The outer
$\max\{0, \cdot\}$ is the walk-away. A single
stage per period, so the stationary problem is one fixed point.

Concavity of $V$ in $w$ is not assumed. It comes out of $F$, $C$, the
constraint $K \ge 0$, and the cost of carrying equity, $r_E - r$ (plus $\tau$
if set). The walk-away floor pushes
the other way.

**Dynamic counterparts of the paper's results.** Each one is a test or an
output:

- $G(w, G) = -E[V_{ww}(w', G')] / E[V_w(w', G')]$, evaluated on the
  next-period distribution, is a **state-dependent** effective risk aversion.
  The hurdle (10) becomes a schedule in capital, $\mu^*(w)$.
- (14) becomes the **payout boundary** $\bar w$: the bank pays out above it,
  where $V_w = 1$. The cost of carrying equity sets where it sits; as
  $r_E \to r$ (with $\tau = 0$) the bank hoards and $G \to 0$. Expect a
  threshold policy. Test for it rather than
  impose it.
- Prop. 2 should hold where $V$ is concave: with $h \equiv 0$, $H$ hedges
  fully. $H$ stays a free control and the test checks the solver picks it.
  **Near the walk-away boundary it may fail.** That is the risk-shifting
  region, so the oracle applies only where $G > 0$.

### 3.4 Functional forms

Chosen so that each parameter is one economic driver, the one-period limit has
closed forms for the oracles, and every form is smooth and element-wise. That
last property is for the later learner under PyTorch or JAX; the exact solver
does not need it.

**Return on the book: quadratic.**

$$F(I) = A\,I - \tfrac{b}{2} I^2$$

$A$ is how good the opportunity is; $b$ is how fast it runs out. The
units of §3.1 fix $b = A - (1+r)$, so the first-best book is $I^* = 1$. It
replaces the exponential form, which had no closed form against a quadratic
issuance cost.

**Issuance cost: linear plus quadratic.**

$$C(e) = \lambda e + \tfrac{\kappa}{2} e^2$$

$\lambda$ is the proportional cost and produces a pecking order. $\kappa$ is
the convex part, which in the paper is what generates $G$. **$\kappa$ is a
switch, not a calibration:** in a model with no end date, $V$ may be concave
through the walk-away and issuance boundaries alone. Whether the Froot–Stein
effect survives $\kappa = 0$ is a result in itself. A fixed issuance cost is
more realistic and is left out because it is a discontinuity the learner
cannot differentiate through.

**Shocks: multivariate Student-$t$.** $(\xi_P, \xi_N, \varepsilon_M)$ is a
scale mixture of normals,

$$
\xi = \frac{Z}{\sqrt{W/\nu}}, \qquad Z \sim N(0, \Sigma), \qquad W \sim \chi^2_\nu,
$$

with **one $W$ shared** across all components, rescaled to unit variance.

- $\nu$ is a parameter. As $\nu \to \infty$ the model is normal, and that is
  where the oracle tests run. The main configuration uses $\nu \approx 4$–$5$.
  It needs $\nu > 2$ for covariances to exist, which (10) needs, and $\nu > 4$
  for finite kurtosis.
- The shared $W$ is what gives **tail dependence**: when it is small, every
  risk is large at once. That is the crisis property a normal model misses, and
  it is distinct from fat tails in each margin alone. Copulas, which separate
  the two, come later.
- $\Sigma$ carries the correlations the paper prices: $\rho_{NP}$ between the
  new product and the book, the term in (10).
- Symmetric in v0, so it fattens the tail of gains as well as losses.
  One-sided heavy-tailed losses (event counts times heavy-tailed severities, for
  exploits and operational events) arrive with the jump component in v1.

**GRC mitigation: exponential with an irreducible floor.**

$$\phi_j(G) = \rho_j + (1 - \rho_j)\,e^{-a_j G}, \qquad j \in \{A, B\}$$

$a_A$ acts on the mean loss and $a_B$ on its spread: mechanisms A and B of §2,
one effectiveness each. $\rho_j$ is the share of the risk no program removes.
Without it, enough spend drives any risk to zero, which is how the old model
ended up reading an effectiveness as "removes 992%". Break-evens are reported
over $(a_A, a_B)$, and $\rho$ is reported as an assumption rather than hidden in
the form.

**Hedging cost: quadratic.** $h(H) = \tfrac{c_H}{2} H^2$. With $c_H = 0$ the
factor is fairly tradeable and Prop. 2 applies. With $c_H > 0$ it is "partly
tradeable" (§5), and the quadratic gives a partial hedge rather than an
all-or-nothing one. Mechanism D enters in v1 as $c_H(G)$, falling in the GRC
stock.

**Cost of carrying equity:** the spread $r_E - r$, through the discount
factor. $\tau K$ stays available as an extra linear friction, zero by default.

### 3.5 Jump losses (v1)

The continuous shocks of v0 are frequent and moderate. The risks that matter
most on crypto rails — a drained bridge, a protocol exploit, a custodian
failure — are rare and large. v1 adds them as **jump losses**.

- **At most one jump per period**, with probability
  $p_J = 1 - e^{-\lambda_J \phi_J(G')}$. $\lambda_J$ is the rate at zero GRC.
- **Severity is a share of the book**, Pareto with tail index $\zeta$, scaled to a
  mean $\mu_J$, and capped at 1: a jump cannot take more than the whole book. The
  loss is $I \cdot J$. Like credit losses, it impairs an asset rather than
  paying cash out.
- **GRC acts on frequency** (prevention), through
  $\phi_J(G) = \rho_J + (1 - \rho_J) e^{-a_J G}$, with the same irreducible
  floor as the other mechanisms. Acting on severity (containment) is a later
  option.
- **Integration.** The severity uses the same tail-placed quadrature as the
  $t$'s mixing variable, so nodes crowd into the far tail. The jump
  probability enters the quadrature **weights**, so it is exactly
  differentiable in GRC spend, with no sampling device needed.

Defaults (illustrative): $\lambda_J = 0.1$ a year, $\mu_J = 5\%$ of the book,
$\zeta = 2.5$, $a_J = 100$, $\rho_J = 0.3$. A 5% loss of a book of 0.8 is 0.04,
about two-thirds of the bank's minimum capital; the tail goes far beyond it.

### 3.6 Decision frequency

The bank decides **quarterly** in the main model, and the frequency is a
parameter, `periods_per_year`. Every rate is stated **per year**, and
`Params.per_period()` converts the annual statement into what one period sees.
The conventions were agreed before they were coded:

| what | annual → per period ($n$ periods a year) | why |
|---|---|---|
| interest rates and the required return ($r$, $r_D$, $r_E$) | $(1+r)^{1/n} - 1$ | money compounds |
| the book's gross return $A$ | $A^{1/n}$; $b$ then still puts the first-best book at 1 | keeps the units of §3.1 at every frequency |
| mean loss rate, product's excess return, carrying cost $\tau$ | $\div n$ | flows that accrue with time |
| volatilities ($s_P$, $s_N$, $s_M$) | $\div \sqrt{n}$ | independent periods' variances add up to the year's. Under fat tails this is a convention, not an identity |
| jump rate | $\div n$ | Poisson events per unit of time |
| GRC depreciation | $1 - (1-\delta)^{1/n}$ | the same share lost over a year |
| issuance costs, capital ratio, deposit base, jump severity, GRC effectiveness and floors, position bounds, correlations, $\nu$ | unchanged | sizes and per-event amounts, not rates |

**Use 1 to explore, 4 for realistic runs.** At quarterly frequency the
discount factor per period is closer to one, so a solve needs more policy
evaluation steps; the solver scales them with $1/(1-\beta)$. The tests run
annually, apart from `tests/test_frequency.py`, which checks the conversion and
solves the quarterly model, including the exact no-shock oracle.

One trap found on the way: a quarter's profit is a quarter of a year's, so the
steady state sits much closer to the kink where issuance starts. On a coarse
grid, linear interpolation across that kink understated the value by 0.006.
Quarterly runs need a finer grid in equity than annual ones.

---

## 4. How it is solved

**Exact, for v0.** The state $(w, G)$ is two-dimensional, so value iteration on
a grid is exact up to grid error, and the discount guarantees convergence.

**Expectations by quadrature, not Monte Carlo.** The scale-mixture form of the
$t$ makes this two nested rules: one over the shared variance factor $W$, then
Gauss–Hermite over $Z$. The rule over $W$ places its points by probability,
uniformly in log-probability, so they crowd into the tail where the large
shocks live. The textbook choice for this integral, generalized
Gauss–Laguerre, was tried first and measured to understate the tail badly: a
kurtosis of 7.3 against a true 9 at $\nu = 5$, even with 64 points, against 8.8
with 12 for the rule now used (`quant/quadrature.py`). The jump severities use
the same tail-placed construction. The grid solution is then deterministic. No seed noise,
no scenario sets that differ by batch size, no "10% across draws". That removes
the whole class of sampling problems in the old `CLAUDE.md` traps list.

**The cost is in the inner maximization.** It runs over six controls at every
grid point. That is the part where array tooling matters: a gradient-based
inner maximization under PyTorch or JAX, vectorized over the grid. The grid
solver and the later learner should share one backend, so the learner can be
graded against the grid on identical arithmetic.

**As built (`quant/solver.py`).** PyTorch, policy iteration: Adam on every grid
point's controls at once, then repeated evaluation of the fixed policy. Three
choices were forced by what the first solves did:

- **Linear interpolation in $w$, not cubic.** Catmull-Rom bulged into a small
  convex patch near the payout kink and the solver gambled on it: hedge errors
  of 0.03 in matched $\pm$ pairs. Linear interpolation of a concave function is
  concave, and hedged exactly.
- **Two starts per improvement.** The product size and the hedge are bounded
  through `sigmoid`/`tanh`; a control pushed to its bound early has almost no
  gradient left to come back. Each step also optimizes from a fresh start and
  keeps the better per grid point. The first solve also starts from a linear,
  risk-neutral value rather than $\max(w, 0)$, whose kink invited gambling.
- **Below the grid the bank issues equity back onto it**, rather than having
  its value clamped. Clamping capped the downside of any gamble and the first
  solve diverged.

The GRC stock is written $S$ in the code, because $G$ is also the paper's
effective risk aversion.

**Oracles:**

1. **Two-period truncation, normal shocks.** With one period left the value is
   the paper's $P(w)$ of (2), and the first period's choices of $\alpha$ and
   $K$ reproduce (9), (10) and (14). Exact under normality by Stein's lemma.
   Under the $t$ it is not exact, and how far the mean-variance hurdle (10) is
   from the true one becomes a **result**: how wrong the paper's rule is when
   tails are fat.
2. **No shocks, inert GRC.** The infinite-horizon value is known exactly above
   the steady state: the bank holds its capital floor, lends a fixed book and
   pays out the rest. Checks the solver's *level*, not only its shape. Built;
   matches to 2e-8.
3. $r_E = r$ with $\tau = 0$: equity is free to carry, so the bank holds a
   buffer well above its floor. Above some level it is indifferent between
   holding and paying out, so $K$ is not pinned there. Built.
4. Prop. 2: full hedging of the tradeable factor where the bank is risk-averse.
   Built; exact to 1e-3 at every state with $w \ge 0$.
5. ~~No non-tradeable risk: no premium over a frictionless twin.~~ Removed
   with the twin (§2).

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
| Compliance / license | **no** | — | A (fewer breaches); B (a lost license is the tail) |

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
| **v1** | **built first:** one-sided jump losses with heavy-tailed severity (§3.5). **Still to come:** several products with a covariance matrix (eqs (11)–(12)), the hurdle schedule $\mu^*(w)$, mechanism D as $c_H(G)$ | grid | GRC's value is reported across capital levels and jump settings |
| **v2** | GRC by family (credit, operational, compliance), mechanisms C and D | learner, checked against the grid on the v1 slice | learner within tolerance of the grid where both run |
| **v3** | deposits as a dynamic stock: inflows, withdrawals and runs, as on crypto rails; then depositors who price risk, as an alternative to the fixed rate; risk weights per asset and a liquidity requirement | learner ($D$ becomes a state variable) | the extension changes some answer; otherwise it is not merged |

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

---

## 8. Departures from the paper

Each is a deliberate change from Froot & Stein (1998), made because this is a
computational model for practical analysis rather than a closed-form argument.

| departure | the paper | here | why |
|---|---|---|---|
| Horizon | three dates | infinite, stationary | a bank is a going concern; a finite horizon needs a terminal value, and in the legacy model that constant carried a third of firm value |
| Timing | riskless investment after the risk is borne | one decision per period; the book itself is risky | for a bank the lending book *is* the main non-tradeable risk, and its credit risk grows with its size |
| Shocks | normal | multivariate Student-$t$, shared variance factor | fat tails and tail dependence are what crises are made of; the normal case is kept as a limit ($\nu \to \infty$) for the oracles |
| Liabilities | none explicit; $\tau$ stands for the cost of equity relative to debt | a fixed deposit base at a fixed rate, senior to dividends | the practical setting is a deposit-taking intermediary, and leverage is what makes ordinary losses dangerous |
| Failure | none | walk-away under limited liability; depositors bear the shortfall | failure becomes a decision rather than an asserted hazard rate, and risk-shifting can be observed rather than ruled out |
| Regulation | none | minimum capital ratio $K \ge kI$ | ordinary bank regulation; it gives the Froot–Stein channel a second route (a loss shrinks the book the bank may lend) |
| Hedging | tradeable risk hedged at fair prices | hedging at a cost $c_H$ for "partly tradeable" risk | crypto-rails risks (exploit cover, custody insurance) are hedgeable only thinly and expensively |
| GRC | absent | a fourth lever acting on non-tradeable risk | the project's purpose |
| **Discounting** | at the riskless rate $r$; the cost of equity is a separate deadweight $\tau K$ | at the shareholders' required return $r_E > r$; cash earns $r$; $\tau = 0$ by default | discounting an infinite stream of bank profits at the riskless rate capitalizes them into an implausibly large franchise. In the first solves the bank was worth over 1 at zero equity, so it would recapitalize from any loss and walk-away never occurred. Discounting at a cost of equity gives the carrying cost of capital ($r_E - r$) and a franchise of realistic size from one parameter. $r_E$ is to be set against a target **market-to-book ratio**, which is observable, rather than chosen directly |


---

## 9. Methods and concepts introduced without prior discussion

Concepts brought into the code or this document by Claude, not taken from the
papers and not discussed before use. Accepted for now (2026-11-11), and kept
here so any of them can be revisited. From now on a new concept is raised
before it is used (`CLAUDE.md`).

**Source** says where each comes from: *standard* means textbook or widely used
technique; *own construction* means assembled for this project, with no single
reference.

### Numerical methods

| concept | source | where | what it does | if revisited |
|---|---|---|---|---|
| Gauss–Hermite quadrature | standard | `quant/quadrature.py` | weighted scenario set for the normal part of the shocks | sparse grids if the shock dimension grows |
| Tail-placed mixing rule | own construction | `quant/quadrature.py` | scenario set for the $t$'s shared volatility factor: Gauss–Legendre nodes uniform in log-probability, placed through the inverse CDF | replaced generalized Gauss–Laguerre (standard), which understated kurtosis (7.3 against 9 at $\nu = 5$, 64 points); accuracy is checked by tests, not by a reference |
| Tail-placed jump-severity rule | own construction | `quant/quadrature.py` | the same construction for the Pareto jump severity, capped at the whole book | the dropped far tail and the cap shave the mean slightly; tested to within 5% |
| Cholesky factorization | standard | `quant/quadrature.py` | builds correlated shocks from independent ones | — |
| Policy iteration with repeated policy evaluation (Howard improvement) | standard | `quant/solver.py` | how the infinite-horizon solve converges | plain value iteration is slower but simpler |
| Adam, vectorized over the grid | standard | `quant/solver.py` | chooses every grid point's controls at once by gradient ascent; two starts per step, and a slow final pass | the step-budget test guards against it stopping short |
| Linear interpolation in equity (Catmull-Rom optional) | standard | `quant/solver.py` | reads the value function between grid points | Catmull-Rom invented convexity near kinks and the bank gambled on it; a shape-preserving smooth spline is the alternative |
| Off-grid rules | own construction | `quant/solver.py` | above the grid, pay out the excess; below it, issue equity back onto it | clamping below the grid made the first solve diverge |
| Golden-section search | standard | `quant/twoperiod.py`, `quant/solver.py` | one-dimensional optimizer for the oracles | — |
| Squashed controls (`sigmoid`, `tanh`, `softplus`) | standard | `quant/solver.py` | keep every constraint satisfied by construction | saturated controls lose their gradient; the two-start step is the guard |

### Modeling and theory

| concept | source | where | what it does | if revisited |
|---|---|---|---|---|
| Stein's lemma | standard probability | `quant/twoperiod.py`, `quant/results.py` | why the paper's formulas (9), (10) are exact under normal shocks and approximate under fat tails; the basis of the two-period oracle | — |
| Student-$t$ as a scale mixture of normals, one shared factor | standard | `quant/quadrature.py`, §3.4 | fat tails and tail dependence from one parameter $\nu$ | copulas would separate marginal tails from dependence |
| Pareto jump severity | standard in operational-risk modeling | `quant/quadrature.py`, §3.5 | heavy-tailed one-sided losses | lognormal, or an empirical exploit-loss distribution |
| Effective risk aversion from first derivatives | own construction | `quant/results.py` | measures the paper's $G$ as the price of risk, $-E[V_w (w' - Ew')] / (E[V_w] \mathrm{Var}(w'))$; equals the paper's $G$ under normality by Stein's lemma | the right generalization under fat tails is a choice, not a theorem |
| Hurdle as a measured price of risk | own construction | `quant/results.py` | $-E[V_w \varepsilon] / E[V_w]$ for a small position with payoff shock $\varepsilon$; equals eq. (10) under normality, exact in general | — |
| Frictionless twin | own construction | **removed** | a premium measured against a bank raising equity for free | removed as an academic construction with no quantitative meaning (§2) |
| GRC value split by mechanism | own construction | **removed** | the first premium measure (mean versus spread) | dropped once jumps made prevention act on both (§2) |
