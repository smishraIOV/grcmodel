# Design: a dynamic Froot–Stein (1998) bank, with GRC as a fourth lever

**Status:** v0 built; v1 under way. Built and planned are marked section by
section, and summarized in §5.

---

## 1. The bank's decisions, and where GRC enters

Following Froot & Stein (1998; summarized in Appendix A), the bank has three
levers:

- **Capital, $K$**: how much equity it holds. More equity absorbs more loss, but
  equity is costly to carry and costly to raise.
  @todo: In the current context, equity will take the shape of a combination of 
  crypto assets (such as Bitcoin) and stablecoins like USDC - which can be thought of as tokenized treasuries
- **Hedging, $H$**: how much of its *tradeable* risk — risk the market will take
  at a fair price, such as crypto prices — it sells off.
- **New business, $\alpha$**: how much it takes on of a new product line whose
  risk it cannot sell and must carry.

The paper's point is that these are one decision, not three. All depend on how
much the bank fears a loss at the margin, its effective risk aversion $G$, and
that depends on how much capital it has.

GRC is a **fourth** lever: a costly, persistent way to change the
**non-tradeable** risk the bank cannot sell in the market. Tradeable risk is
hedged at zero NPV, so GRC spent on it is wasted. Non-tradeable risk is what
operational controls, compliance programs and underwriting act on.

Ways a unit of GRC can act. They are not mutually exclusive; which apply to
which risk is the subject of §4.

| mechanism | effect on the non-tradeable risk | why it is worth something | status |
|---|---|---|---|
| **A. mean** | lowers the expected loss on the book | worth it to any firm, constrained or not | **built** (§2.4) |
| **B. spread** | lowers the variance of the book's loss | worth $\approx \tfrac12 G\,\Delta\mathrm{var}$: the Froot–Stein channel | **built** (§2.4) |
| **J. prevention** | lowers the frequency of jump losses | removes expected loss and tail risk at once | **built** (§2.5) |
| **C. covariance** | lowers the correlation between a product's risk and the book's | lowers the hurdle (10) for other products: GRC as an enabler of capital budgeting | planned (v2) |
| **D. tradeability** | turns non-tradeable risk into tradeable risk: audited contracts become insurable, a documented book becomes saleable | moves risk from the $G$-priced bucket to the $\gamma$-priced one; the most distinctively Froot–Stein mechanism | planned (v1), as $c_H(S)$ |

**What is reported:** GRC's value to this bank — the bank with a program
against the same bank without one — and how that value moves with its capital,
its risks and the size of its losses.

---

## 2. The model

**Notation.** $G$ is the paper's effective risk aversion. The GRC stock is $S$,
in this document and in the code.

### 2.1 Units

All amounts of money are measured relative to one benchmark: the size of the
lending book the bank would choose if capital were no constraint at all. That
book is 1. Equity of 0.08, for example, is equity worth 8% of it.

### 2.2 State, controls, timing

**What the bank knows at the start of a period** — the *state*:

- $w_t$, its equity;
- $S_t$, its GRC stock.

Nothing else carries over between periods. The deposit base $D$ is fixed; GRC
by family (v2) and dynamic deposits (v3) are planned and would add to the state.

**What it decides each period** — the *controls*:

| control | meaning | paper's lever |
|---|---|---|
| walk-away | abandon the firm; shareholders get 0 | — |
| $d_t \ge 0$ | dividend paid | capital, $K$ |
| $e_t \ge 0$ | new equity raised, at cost $C(e_t)$ | the cost of outside finance |
| $g_t \ge 0$ | GRC spend | the new lever |
| $I_t \ge 0$ | size of the lending book | the investment opportunity |
| $\alpha_t$ | size of the new product | new business |
| $H_t$ | hedge on the tradeable factor | hedging |

Paying and raising in the same period is never worth it, so only one of $d, e$
is non-zero in practice. The solver is not told this.

**When.** All of it is decided at once, at the start of the period, before the
period's shocks. The lending book itself carries the non-tradeable risk —
for a bank it *is* the main non-tradeable risk — so the shocks land on
everything chosen. The Froot–Stein channel runs through the *next* period: a
bad draw leaves less equity, and next period the bank must raise costly equity
or lend less.

**How its capital evolves.** The firm's capital evolves as follows. It starts
the period with equity $w_t$, pays a dividend or raises new equity, and
operates with

$$
K_t = w_t - d_t + e_t \ \ge\ k\,I_t,
\qquad
R_t = K_t - g_t + D - I_t \ \ge 0
$$

$K_t$ is the equity the bank operates with. $k$ is a **minimum capital ratio**
against the book, the regulatory constraint of an ordinary bank: risk-weighted
assets with a single weight. $k = 0$ leaves only "no negative equity". A bank
left below its requirement by a bad period has three choices:

- recapitalize, at cost $C(e)$;
- deleverage, by shrinking $I_t$ until $K_t \ge k I_t$;
- shut down, by walking away.

It is never forced to shut. Shutting is chosen when the other two are worth
less than nothing.

$R_t$ is cash: what was funded and not lent. These two constraints make capital
scarce. The capital ratio gives the Froot–Stein channel a second route: a loss
shrinks the book the bank is *allowed* to lend, not only the book it can fund.

Planned (v3): risk weights per asset (the new product, the hedge), and a
liquidity requirement on $R_t$. The new product currently carries no capital
charge.

**GRC stock:** $S_{t+1} = (1-\delta)S_t + g_t$. Mitigation uses the
post-spend stock $S_{t+1}$, so spend protects in the period it is made.

**Shocks land, and the books close:**

$$
w_{t+1} =
\underbrace{F(I_t) - I_t\,\ell_t - I_t J_t}_{\text{book}}
+ \underbrace{(1+r)\,R_t}_{\text{cash}}
+ \underbrace{\alpha_t(\pi + s_N\,\xi_N)}_{\text{new product}}
+ \underbrace{(I_t\,b_M - H_t)\,s_M\,\varepsilon_M - h(H_t)}_{\text{tradeable, hedged}}
- \tau K_t
- (1 + r_D)\,D
$$

- $\ell_t = \phi_A(S_{t+1})\,m - \phi_B(S_{t+1})\,s_P\,\xi_P$ is the book's
  continuous non-tradeable loss rate, where $\xi_P$ is the book's payoff shock.
  It scales with the book: lending twice as much carries twice the credit risk.
- $J_t$ is the jump loss, a share of the book: zero in most periods (§2.5).
- $b_M$ is the book's loading on the tradeable factor (crypto collateral
  prices, say), and $H_t$ sells it off. $h(H)$ is the hedging cost, zero when
  the factor is fairly tradeable.
- $\tau K_t$ is an optional extra deadweight cost of carrying equity, zero by
  default: the cost of holding equity comes from shareholders discounting at
  $r_E > r$ while the cash that equity funds earns only $r$ (§2.3).
- Depositors are owed $(1+r_D)D$ at the close, so $w_{t+1}$ is assets less
  that. The base is rolled over: the re-deposited $D$ arrives as cash next
  period through $R_{t+1}$, and is not added back here. Interest is senior to
  everything shareholders receive.

**Walk-away.** At the start of any period, shareholders may abandon the firm
and receive 0. Depositors then receive the assets and bear the shortfall,
$(-w_t)^+$. With a negative $w_t$ the choice is whether recapitalizing to
$K \ge 0$, at cost $C(e)$, is worth more than nothing. Failure is a decision,
not a hazard parameter.

**Gambling for resurrection.** Depositors are paid a fixed rate whatever the
bank's condition, and shareholders have limited liability. Close to the
walk-away boundary, the floor at zero makes equity value *convex* in $w$:
shareholders keep the upside and depositors take the downside. This works
against the Froot–Stein concavity, and the model is free to show either effect
winning. The capital ratio limits how far it can go, because a bank near the
boundary is also near its requirement. Where the effective risk aversion falls
— or turns negative — as capital approaches the boundary is an output.

### 2.3 The Bellman equation

$$
V(w, S) = \max\Big\{\, 0,\ \max_{d, e, g, I, \alpha, H}\ \big[\, d - e - C(e) + \beta\, E\, V(w', S') \,\big] \Big\}
$$

subject to the balance-sheet constraints of §2.2 (capital ratio and
non-negative cash), with $\beta = 1/(1+r_E)$, the shareholders' required return
per period. The outer $\max\{0, \cdot\}$ is the walk-away. One stage per
period, so the stationary problem is one fixed point.

Concavity of $V$ in $w$ is not assumed. It comes from $F$, $C$, the capital
constraints, and the cost of carrying equity, $r_E - r$ (plus $\tau$ if set).
The walk-away floor pushes the other way.

**Dynamic counterparts of the paper's results.**

| paper | here | status |
|---|---|---|
| $G = -E P_{ww} / E P_w$ | a **state-dependent** effective risk aversion $G(w, S)$, measured as the price of risk on the next-period distribution (§8) | **built**, `quant/results.py` |
| hurdle (10) | a schedule in capital: the required return on a small position, measured as a price of risk (§8) | **built** for the single product; the multi-product schedule is planned (v1) |
| capital structure (14) | the **payout boundary** $\bar w$: the bank pays out above it. The cost of carrying equity sets where it sits; as $r_E \to r$ with $\tau = 0$ the bank holds a buffer | **built**, found by the solver rather than imposed |
| Prop. 2 | full hedging where $V$ is concave and $h \equiv 0$; $H$ stays a free control. May fail near the walk-away boundary, the risk-shifting region | **built**, an oracle (§3) |

### 2.4 Functional forms

Each parameter is one economic driver, the one-period limit has closed forms for
the oracles, and every form is smooth and element-wise, for the later learner.

**Return on the book: quadratic.**

$$F(I) = A\,I - \tfrac{b}{2} I^2$$

$A$ is how good the opportunity is; $b$ is how fast it runs out. The units of
§2.1 fix $b = A - (1+r)$, so the first-best book is $I^* = 1$.

**Issuance cost: linear plus quadratic.**

$$C(e) = \lambda e + \tfrac{\kappa}{2} e^2$$

$\lambda$ is the proportional cost and produces a pecking order. $\kappa$ is the
convex part, which in the paper is what generates $G$. **$\kappa$ is a switch,
not a calibration:** with no end date, $V$ may be concave through the walk-away
and issuance boundaries alone, and whether the Froot–Stein effect survives
$\kappa = 0$ is open (§6). A fixed issuance cost is more realistic but is left
out, because it is a discontinuity the learner cannot differentiate through.

**Continuous shocks: multivariate Student-$t$.** $(\xi_P, \xi_N, \varepsilon_M)$
is a scale mixture of normals,

$$
\xi = \frac{Z}{\sqrt{W/\nu}}, \qquad Z \sim N(0, \Sigma), \qquad W \sim \chi^2_\nu,
$$

with **one $W$ shared** across all components, rescaled to unit variance.

- $\nu$ is a parameter. As $\nu \to \infty$ the model is normal, which is where
  the oracles run. The main configuration uses $\nu \approx 4$–$5$. Covariances
  need $\nu > 2$, which (10) needs; finite kurtosis needs $\nu > 4$.
- The shared $W$ gives **tail dependence**: when it is small, every risk is
  large at once. That is the crisis property a normal model misses, distinct
  from fat tails in each margin alone. Copulas, which separate the two, are a
  later option.
- $\Sigma$ carries the correlations the paper prices: $\rho_{NP}$ between the
  new product and the book, the term in (10).
- Symmetric, so it fattens the tail of gains as well as losses. One-sided
  heavy-tailed losses are the jump component (§2.5).

**GRC mitigation: exponential with an irreducible floor.**

$$\phi_j(S) = \rho_j + (1 - \rho_j)\,e^{-a_j S}, \qquad j \in \{A, B, J\}$$

$a_A$ acts on the mean loss, $a_B$ on its spread, and $a_J$ on jump frequency:
one effectiveness per mechanism (§1). $\rho_j$ is the share of the risk no
program removes; without it, enough spend drives any risk to zero.

**Hedging cost: quadratic.** $h(H) = \tfrac{c_H}{2} H^2$. With $c_H = 0$ the
factor is fairly tradeable and Prop. 2 applies. With $c_H > 0$ it is "partly
tradeable" (§4), and the quadratic gives a partial hedge rather than an
all-or-nothing one. Planned (v1): mechanism D as $c_H(S)$, falling in the GRC
stock.

**Cost of carrying equity:** the spread $r_E - r$, through the discount factor.
$\tau K$ is available as an extra linear friction, zero by default.

### 2.5 Jump losses

**Built (v1).** The continuous shocks are frequent and moderate. The risks that
matter most on crypto rails — a drained bridge, a protocol exploit, a custodian
failure — are rare and large.

- **At most one jump per period**, with probability
  $p_J = 1 - e^{-\lambda_J \phi_J(S')}$. $\lambda_J$ is the rate at zero GRC.
- **Severity is a share of the book**, Pareto with tail index $\zeta$, scaled to
  a mean $\mu_J$, and capped at 1: a jump cannot take more than the whole book.
  The loss is $I \cdot J$. Like credit losses, it impairs an asset rather than
  paying cash out.
- **GRC acts on frequency** (prevention), through $\phi_J$. Acting on severity
  (containment) is a later option.
- **Integration.** The severity uses the same tail-placed quadrature as the
  $t$'s mixing variable (§3). The jump probability enters the quadrature
  **weights**, so it is exactly differentiable in GRC spend.

Defaults (illustrative): $\lambda_J = 0.1$ a year, $\mu_J = 5\%$ of the book,
$\zeta = 2.5$, $a_J = 100$, $\rho_J = 0.3$. A 5% loss of a book of 0.8 is 0.04,
about two-thirds of the bank's minimum capital; the tail goes far beyond it.

### 2.6 Decision frequency

**Built.** The bank decides **quarterly** in the main model, and the frequency
is a parameter, `periods_per_year`. Every rate is stated **per year**, and
`Params.per_period()` converts the annual statement into what one period sees:

| what | annual → per period ($n$ periods a year) | why |
|---|---|---|
| interest rates and the required return ($r$, $r_D$, $r_E$) | $(1+r)^{1/n} - 1$ | money compounds |
| the book's gross return $A$ | $A^{1/n}$; $b$ then still puts the first-best book at 1 | keeps the units of §2.1 at every frequency |
| mean loss rate, product's excess return, carrying cost $\tau$ | $\div n$ | flows that accrue with time |
| volatilities ($s_P$, $s_N$, $s_M$) | $\div \sqrt{n}$ | independent periods' variances add up to the year's. Under fat tails this is a convention, not an identity |
| jump rate | $\div n$ | Poisson events per unit of time |
| GRC depreciation | $1 - (1-\delta)^{1/n}$ | the same share lost over a year |
| issuance costs, capital ratio, deposit base, jump severity, GRC effectiveness and floors, position bounds, correlations, $\nu$ | unchanged | sizes and per-event amounts, not rates |

**Use 1 to explore, 4 for realistic runs.** At quarterly frequency the discount
factor per period is closer to one, so a solve needs more policy-evaluation
steps; the solver scales them with $1/(1-\beta)$. A quarter's profit is small,
so the steady state sits close to the kink where issuance starts, and quarterly
runs need a finer equity grid than annual ones.

---

## 3. How it is solved

**Expectations by quadrature, not Monte Carlo.** Quadrature computes an
expected value from a small set of weighted scenarios, deterministically. The
$t$'s scale-mixture form makes this two nested rules: one over the shared
variance factor $W$, then Gauss–Hermite over $Z$. The rule over $W$ places its
points uniformly in log-probability, so they crowd into the tail where the
large shocks live (§8). The jump severities use the same construction. A solve
is therefore deterministic: no seed noise, and no scenario set that changes
with the batch size.

**The grid solver** (`quant/solver.py`, **built**). The state $(w, S)$ is
two-dimensional, so the infinite-horizon problem is solved on a grid, exact up
to grid error.

- **Policy iteration.** At every grid point the controls are chosen by gradient
  ascent (Adam), for all points at once; the fixed policy is then evaluated
  repeatedly. The two alternate until the value stops changing.
- **Controls are squashed** (`sigmoid`, `tanh`, `softplus`) so every constraint
  holds by construction.
- **Two starts per improvement**: from the last policy, and from a fresh one,
  keeping the better at each grid point. A squashed control pushed to its bound
  has almost no gradient left to come back.
- **Linear interpolation in $w$**, linear in $S$. Linear interpolation of a
  concave function stays concave; a smoother cubic can invent convex patches
  near kinks, and the solver gambles on them.
- **Off the grid:** above it, extra equity is paid out (slope one); below it,
  the bank issues equity back onto the grid, which charges the downside of a
  gamble properly.
- **Start** from a linear, risk-neutral value.

The reasons behind each of these are in §9.

**Oracles** — known-answer checks, all built and run in the test suite:

1. **Two-period truncation, normal shocks.** With one period left the value is
   the paper's $P(w)$ of (2), and the first period's choices reproduce Prop. 2
   and eq. (9), with the hurdle (10) taking the sign of the covariance. Exact
   under normality by Stein's lemma (§8); under the $t$ it is an approximation.
2. **No shocks, inert GRC.** The infinite-horizon value is known exactly above
   the steady state: the bank holds its capital floor, lends a fixed book and
   pays out the rest. Checks the solver's *level*, not only its shape, at annual
   and quarterly frequency.
3. **Free equity.** With $r_E = r$ and $\tau = 0$, carrying equity costs
   nothing, so the bank holds a buffer well above its floor. Above some level it
   is indifferent between holding and paying out, so $K$ is not pinned there.
4. **Prop. 2 in the dynamic model**: full hedging of the tradeable factor
   wherever the bank is not near walk-away.

Other tests check correctness rather than a known answer: the accounting, the
frequency conversion, quadrature accuracy, the hurdle's first-order condition,
and that more optimizer steps do not move the answer.

**The learner** (planned, v2) comes in when the state outgrows a grid: GRC by
family, then dynamic deposits. It is trained on the same stationary problem and
**graded against the grid** wherever the grid can still reach. With an infinite
horizon, a critic is natural; the `svg-critic` branch's reason for being parked
— it lost to full backpropagation through time at short finite horizons — does
not carry over.

**Carried over from the legacy model:** `quant/numerics.py` and the profile
rule, CPU-drawn randomness for the learner's rollouts, the softplus/logit
guards, and the testing discipline.

---

## 4. Risk classification for the crypto-rails bank (open)

A first, deliberately tentative cut. Each row is a question, not an answer.
"Partly tradeable" means hedgeable at a price above fair.

| risk | tradeable? | instrument, if any | GRC mechanism |
|---|---|---|---|
| Crypto price risk on collateral and treasury | **mostly yes** | perps, futures, options | none needed, per Prop. 2 |
| Interest-rate / funding-rate risk | **yes** | T-bills, rate swaps, on-chain fixed-rate markets | none |
| Stablecoin depeg | **partly** | thin depeg cover, diversification across issuers | B (exposure limits), D? |
| Idiosyncratic credit (crypto borrowers, protocols) | **no** | — | A, B (underwriting); C (concentration limits) |
| Smart-contract exploit, bridge drain | **partly** | on-chain cover protocols: thin, costly, capacity-limited | J (audits prevent), **D (audits make cover purchasable)** |
| Custody / key management | **partly** | custody insurance, limited capacity | A, J, D |
| Operational incidents | **no** | — | A, B |
| Compliance / license | **no** | — | J (fewer breaches; a lost license is the tail) |

---

## 5. Staging

| stage | adds | solved by | status |
|---|---|---|---|
| **v0** | one non-tradeable risk, one tradeable factor, one new product, one GRC stock, a fixed deposit base at a fixed rate, walk-away, a minimum capital ratio; state $(w, S)$ | grid, quadrature | **built**; oracles 1–4 pass |
| **v1** | jump losses with heavy-tailed severity (§2.5); decision frequency (§2.6) | grid | **built** |
| | several products with a covariance matrix (eqs (11)–(12)) and their hurdle schedule; mechanism D as $c_H(S)$ | grid | planned |
| **v2** | GRC by family (credit, operational, compliance); mechanism C | learner, graded against the grid on the v1 slice | planned |
| **v3** | deposits as a dynamic stock: inflows, withdrawals and runs, as on crypto rails; depositors who price risk; risk weights per asset and a liquidity requirement | learner ($D$ becomes state) | planned; merged only if it changes some answer |

Each stage needs approval before it starts.

---

## 6. Open questions

1. **RAROC.** Check the JFE text for its comparison section, and decide whether
   it becomes an output: standalone-risk pricing against the covariance-with-
   book pricing of (10).
2. **Cost of "partly tradeable" hedging.** Linear or convex in the amount
   hedged? Quadratic is assumed (§2.4); linear would give all-or-nothing hedges.
3. **What the book and the products are.** Is the book the vault/lending
   business, with each new strategy a product $\alpha_i$? That reading makes
   eq. (12) the question of how much the bank allocates to each strategy.
4. **How often to re-price.** The paper's open dynamic question — how often
   headquarters should refresh $G$ — reads here as how often the risk function
   re-prices strategy hurdles.
5. **Does the effect survive $\kappa = 0$?** Whether the Froot–Stein channel
   needs a convex issuance cost in a model with no end date.
6. **Calibration.** $r_E$ against a target market-to-book ratio; the rest of the
   parameters against data, once the model is realistic enough for its numbers
   to be worth reporting.

---

## 7. Departures from the paper

Each is a deliberate change from Froot & Stein (1998), made because this is a
computational model for practical analysis rather than a closed-form argument.

| departure | the paper | here | why |
|---|---|---|---|
| Horizon | three dates | infinite, stationary | a bank is a going concern; a finite horizon needs a terminal value, which carried a third of firm value in the legacy model |
| Timing | riskless investment after the risk is borne | one decision per period; the book itself is risky | for a bank the lending book *is* the main non-tradeable risk, and its credit risk grows with its size |
| Shocks | normal | multivariate Student-$t$ with a shared variance factor, plus heavy-tailed jumps | fat tails and tail dependence are what crises are made of; the normal case is kept as a limit for the oracles |
| Liabilities | none explicit; $\tau$ stands for the cost of equity relative to debt | a fixed deposit base at a fixed rate, senior to dividends | the setting is a deposit-taking intermediary, and leverage is what makes ordinary losses dangerous |
| Failure | none | walk-away under limited liability; depositors bear the shortfall | failure is a decision rather than an asserted hazard rate, and risk-shifting can be observed rather than ruled out |
| Regulation | none | minimum capital ratio $K \ge kI$ | ordinary bank regulation; it gives the Froot–Stein channel a second route |
| Hedging | tradeable risk hedged at fair prices | hedging at a cost $c_H$ for "partly tradeable" risk | crypto-rails risks (exploit cover, custody insurance) are hedgeable only thinly and expensively |
| GRC | absent | a fourth lever acting on non-tradeable risk | the project's purpose |
| Discounting | at the riskless rate $r$; the cost of equity is a separate deadweight $\tau K$ | at the shareholders' required return $r_E > r$; cash earns $r$; $\tau = 0$ by default | at the riskless rate an infinite stream of bank profits capitalizes into an implausibly large franchise (§9). A cost of equity gives the carrying cost of capital and a franchise of realistic size from one parameter, to be set against an observable market-to-book ratio |

---

## 8. Methods and concepts introduced without prior discussion

Concepts brought into the code or this document by Claude, not taken from the
papers and not discussed before use. Accepted for now (2026-11-11), and kept
here so any of them can be revisited. New concepts are now raised before they
are used (`CLAUDE.md`).

*Standard* means a textbook or widely used technique; *own construction* means
assembled for this project, with no single reference.

### Numerical methods

| concept | source | where | what it does | if revisited |
|---|---|---|---|---|
| Gauss–Hermite quadrature | standard | `quant/quadrature.py` | weighted scenario set for the normal part of the shocks | sparse grids if the shock dimension grows |
| Tail-placed mixing rule | own construction | `quant/quadrature.py` | scenario set for the $t$'s shared volatility factor: Gauss–Legendre nodes uniform in log-probability, placed through the inverse CDF | replaced generalized Gauss–Laguerre (standard), which understated kurtosis (7.3 against 9 at $\nu = 5$, 64 points); accuracy is checked by tests, not by a reference |
| Tail-placed jump-severity rule | own construction | `quant/quadrature.py` | the same construction for the Pareto jump severity, capped at the whole book | the dropped far tail and the cap shave the mean slightly; tested to within 5% |
| Cholesky factorization | standard | `quant/quadrature.py` | builds correlated shocks from independent ones | — |
| Policy iteration with repeated policy evaluation (Howard improvement) | standard | `quant/solver.py` | how the infinite-horizon solve converges | plain value iteration is slower but simpler |
| Adam, vectorized over the grid | standard | `quant/solver.py` | chooses every grid point's controls at once by gradient ascent; two starts per step, and a slow final pass | the step-budget test guards against it stopping short |
| Linear interpolation in equity (Catmull-Rom optional) | standard | `quant/solver.py` | reads the value function between grid points | a shape-preserving smooth spline is the alternative |
| Off-grid rules | own construction | `quant/solver.py` | above the grid, pay out the excess; below it, issue equity back onto it | — |
| Golden-section search | standard | `quant/twoperiod.py`, `quant/solver.py` | one-dimensional optimizer for the oracles | — |
| Squashed controls (`sigmoid`, `tanh`, `softplus`) | standard | `quant/solver.py` | keep every constraint satisfied by construction | saturated controls lose their gradient; the two-start step is the guard |

### Modeling and theory

| concept | source | where | what it does | if revisited |
|---|---|---|---|---|
| Stein's lemma | standard probability | `quant/twoperiod.py`, `quant/results.py` | why the paper's formulas (9), (10) are exact under normal shocks and approximate under fat tails; the basis of oracle 1 | — |
| Student-$t$ as a scale mixture of normals, one shared factor | standard | `quant/quadrature.py`, §2.4 | fat tails and tail dependence from one parameter $\nu$ | copulas would separate marginal tails from dependence |
| Pareto jump severity | standard in operational-risk modeling | `quant/quadrature.py`, §2.5 | heavy-tailed one-sided losses | lognormal, or an empirical exploit-loss distribution |
| Effective risk aversion from first derivatives | own construction | `quant/results.py` | measures the paper's $G$ as the price of risk, $-E[V_w (w' - Ew')] / (E[V_w] \mathrm{Var}(w'))$; equals the paper's $G$ under normality by Stein's lemma | the right generalization under fat tails is a choice, not a theorem |
| Hurdle as a measured price of risk | own construction | `quant/results.py` | $-E[V_w \varepsilon] / E[V_w]$ for a small position with payoff shock $\varepsilon$; equals eq. (10) under normality, exact in general | — |
| Frictionless twin | own construction | **removed** | a premium measured against a bank raising equity for free | removed as an academic construction with no quantitative meaning (§9) |
| GRC value split by mechanism | own construction | **removed** | the first premium measure (mean versus spread) | removed once jumps made prevention act on both (§9) |

---

## 9. Decisions log

What was decided, when, and why. Newest last.

**2026-11-10 — direction**

- **The model is the integrated framework of Froot & Stein (1998):** risk
  management, capital budgeting and capital structure decided jointly, priced
  by one endogenous risk aversion. The legacy model had drifted from that
  thesis: its financing friction was switched off and GRC's value came from
  asserted hazard rates.
- **Infinite horizon, stationary policy; discrete time.** The legacy model's
  finite horizon needed a terminal constant that carried a third of firm value
  and caused most of its instabilities.
- **A practical model, not a replication; keep only the paper's core economic
  drivers,** at the level of detail where the problem solves as an
  optimization. Departures are justified and listed (§7).
- **The setting stays the crypto-rails intermediary** of `readme.md`. How its
  risks split into hedgeable and not is worked out iteratively (§4).
- **`quant/` was replaced,** not kept alongside; the legacy model is on its own
  branch.
- **Walk-away is in;** failure is the shareholders' decision. Limited liability
  comes with it, and is the baseline.
- **Payout is in,** at one point, the start of the period: it is the dynamic
  form of the capital-structure decision. Without it, capital either piles up
  and $G$ vanishes, or a fixed rule assumes the target capital the model is
  meant to find.
- **Depositors from the start:** a fixed base at a fixed rate, senior to
  dividends, not pricing risk. Gambling for resurrection, if it appears, is
  reported rather than suppressed. Dynamic deposits and runs are later (v3).
- **No negative equity, and a minimum capital ratio.** A bank below its
  requirement recapitalizes, deleverages or shuts.
- **"Partly tradeable" means hedgeable at a price above fair;** GRC can lower
  that price (mechanism D).
- **Functional forms** (§2.4): quadratic return and issuance cost, so the
  one-period limit has closed forms; an irreducible floor on mitigation, so no
  spend removes a risk entirely; a quadratic hedging cost, so hedging is
  partial rather than all-or-nothing.
- **One decision per period, with the book itself risky,** replacing the
  paper's riskless final stage: for a bank the book is the main non-tradeable
  risk.
- **Student-$t$ shocks with a shared variance factor,** because models that
  assumed normal tails and independent risks underestimated past crises.
- **An accounting correction:** an early draft added the rolled-over deposits
  back into closing equity, counting them twice. Closing equity is assets less
  what depositors are owed.

**2026-11-10 — building v0**

- **Discount at the shareholders' required return $r_E$,** not the riskless
  rate. At the riskless rate the bank was worth more than the whole first-best
  book even at zero equity, so it recapitalized from any loss and walk-away
  never occurred. The cost of carrying equity becomes $r_E - r$, replacing
  $\tau$ (§7).
- **Linear interpolation, not Catmull-Rom.** Catmull-Rom bulged into a convex
  patch near the payout kink; the solver gambled on it, leaving hedge errors of
  0.03 in matched $\pm$ pairs, and it missed the exact no-shock answer by
  1.7e-4. Linear interpolation hedged exactly and matched to 2e-8.
- **Below the grid the bank issues equity back onto it.** Clamping the value
  there capped the downside of any gamble while the upside grew, and the first
  solve took an unbounded position and diverged. The unhedged factor exposure
  is also bounded.
- **Two starts per improvement, and a risk-neutral starting value.** The first
  solves pushed the product size and hedge to their bounds early — the
  starting value $\max(w, 0)$ has a convex kink that makes gambling look
  attractive — and squashed controls could not come back.
- **The GRC grid was widened:** at its first range, spend hit the cap.
- **The tail-placed quadrature replaced generalized Gauss–Laguerre,** which
  understated the $t$'s kurtosis (§8).
- **Effective risk aversion and hurdles are measured from first derivatives,**
  because a linearly interpolated value function has no curvature between grid
  points (§8).

**2026-11-11 — v1 and reporting**

- **Jump losses added** (§2.5), so GRC has the rare, severe losses that
  threaten survival to act on.
- **No Froot–Stein premium is measured.** First split by mechanism (mean versus
  spread), which stopped isolating anything once GRC prevented jumps — that
  removes expected loss and tail risk at once. Then measured against a
  frictionless twin able to raise equity for free, which also mixed in limited
  liability: the real bank walks away from the worst losses, so preventing them
  protects depositors. Separating those would need a bank forced to
  recapitalize — unlimited liability. Both comparisons rest on banks that do
  not exist, so their numbers mean nothing, and their qualitative point is
  known. Removed. Limited liability's drag on GRC incentives is acknowledged
  in words, in the overview.
- **No model results in the docs yet.** The model is not robust enough for its
  numbers to be taken seriously; results are reported once it is more
  realistic and the learner has run large scenario sets.
- **Decision frequency is a parameter, quarterly by default** (§2.6), with the
  conversion conventions agreed before coding. Annual runs are for quick
  exploration. Policy-evaluation steps scale with the discount factor, because
  a fixed budget left the quarterly solve unconverged.
- **New concepts are raised before they are used,** after Gauss–Laguerre went
  into the code without discussion. §8 lists everything introduced that way.
- **US spelling throughout.**

---

## Appendix A. What the 1998 paper builds

*Optional reading.* This describes the model in the paper; ours adapts it to an
infinite horizon (§2.2). Anything striking here that our model lacks can be
raised elsewhere. Read from the free NBER working paper w5403 (1996); the JFE
version is paywalled, and equation numbers below are the working paper's.


**Three dates.**

- **t = 0: capital structure.** The bank holds an existing portfolio paying
  $Z_P = \mu_P + \varepsilon_P$ at t = 2, which needs no financing of its own.
  It chooses equity capital $K$, held as T-bills. Holding it carries a
  deadweight cost $\tau K$; the paper uses taxes as the concrete example.
- **t = 1: hedging and capital budgeting.** The bank picks a size $\alpha$ for
  a new product paying $Z_N = \mu_N + \varepsilon_N$, and a hedge $H$.
  Time-2 wealth is

  $$w = Z_P + \alpha Z_N + Z_H + K(1-\tau) \qquad (1)$$

- **t = 2: the Froot–Scharfstein–Stein (FSS) problem.** A non-stochastic
  investment $F(I)$, concave, funded by internal wealth plus external finance
  $e$ at convex cost $C(e)$:

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
level**, the reason banks impose position limits. The first is what this
project adds.
