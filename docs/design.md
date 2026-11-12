# Design: a dynamic Froot–Stein (1998) crypto firm, with GRC as a fourth lever

**Status:** the bank form is built (§3). The vault form's first stage is built
(§4); it is the main model from here. Built and planned are marked section by
section, and summarized in §7.

---

## 1. The firm's decisions, and where GRC enters

The firm is **crypto-native**. BTC is its home asset: it holds BTC long-term,
out of conviction and because much of its business arrives in BTC — deposits,
and borrowers' collateral. It also holds dollar stablecoins (USDC, USDT), which
work like tokenized treasuries: to offset BTC's price risk, for liquidity, and
to pay operating costs, which are in dollars. **Money is measured in dollars.**

Following Froot & Stein (1998; summarized in Appendix A), the firm has three
levers:

- **Capital, $K$**: how much equity it holds. More equity absorbs more loss, but
  equity is costly to carry and costly to raise. Here equity is held as a mix
  of BTC and stablecoins.
- **Hedging, $H$**: how much of its *tradeable* risk — risk the market will take
  at a fair price — it sells off. The main tradeable risk is the BTC price,
  hedged with short perps. Because the firm *believes* BTC will outperform, it
  does not hedge all of it away: it keeps a net long position, and the model
  prices what that costs.
- **New business, $\alpha$**: how much it takes on of a product line whose risk
  it cannot sell and must carry.

The paper's point is that these are one decision, not three. All depend on how
much the firm fears a loss at the margin, its effective risk aversion $G$, and
that depends on how much capital it has.

GRC is a **fourth** lever: a costly, persistent way to change the
**non-tradeable** risk the firm cannot sell in the market. Tradeable risk can be
hedged at zero NPV, so GRC spent on it is wasted. Non-tradeable risk is what
operational controls, compliance programs and underwriting act on.

Ways a unit of GRC can act. They are not mutually exclusive; which apply to
which risk is the subject of §6.

| mechanism | effect on the non-tradeable risk | why it is worth something | status |
|---|---|---|---|
| **A. mean** | lowers the expected loss on the book | worth it to any firm, constrained or not | **built** (§2.3) |
| **B. spread** | lowers the variance of the book's loss | worth $\approx \tfrac12 G\,\Delta\mathrm{var}$: the Froot–Stein channel | **built** (§2.3) |
| **J. prevention** | lowers the frequency of jump losses | removes expected loss and tail risk at once | **built** (§2.4) |
| **C. covariance** | lowers the correlation between a product's risk and the book's | lowers the hurdle (10) for other products: GRC as an enabler of capital budgeting | planned |
| **D. tradeability** | turns non-tradeable risk into tradeable risk: audited contracts become insurable, a documented book becomes saleable | moves risk from the $G$-priced bucket to the $\gamma$-priced one; the most distinctively Froot–Stein mechanism | planned, as $c_H(S)$ |

In the vault version GRC also protects the firm's **franchise**: incidents
trigger withdrawals, and withdrawals take fee income with them (§4).

**What is reported:** GRC's value to the firm — the firm with a program against
the same firm without one — and how that value moves with its capital, its
risks and the size of its losses.

**Two business forms.** A crypto-native firm can run its business like a
**bank** — taking deposits at a fixed rate and investing them — or like a
**fund** — running vaults whose depositors share in the gains and losses, for
fees. The bank form was built first (§3). Its runs showed that, with a strong
view on BTC, fixed-rate deposits let the firm gamble with depositors' money
(§11). The fund form, the **vault**, is the main model from here (§4); the bank
form will be revisited later.

---

## 2. Common to both forms

What follows applies to the bank form (§3) and the vault form (§4) alike.

**Notation.** $G$ is the paper's effective risk aversion. The GRC stock is $S$,
in this document and in the code.

### 2.1 Units

Money is in **dollars**, the numeraire. Each form fixes its own scale: in the
bank form, the size of the lending book the bank would choose if capital were
no constraint is 1 (§3.1); the vault form will set its own (§4.7).

### 2.2 Timing, and what every firm decides

**One decision per period, taken before the period's shocks.** Everything is
chosen at once, at the start of the period, and the shocks land on all of it.
The Froot–Stein channel runs through the *next* period: a bad draw leaves less
equity, and next period the firm must raise costly equity or scale back.

**Decisions common to both forms:**

| control | meaning | paper's lever |
|---|---|---|
| walk-away | abandon the firm; shareholders get 0 | — |
| $d_t \ge 0$ | dividend paid | capital, $K$ |
| $e_t \ge 0$ | new equity raised, at cost $C(e_t)$ | the cost of outside finance |
| $g_t \ge 0$ | GRC spend | the new lever |
| $q_t \ge 0$ | BTC treasury, bought spot | capital held in BTC |
| $H_t$ | BTC hedged with short perps | hedging |

Each form adds its own business decisions (§3.2, §4.2). Paying and raising in
the same period is never worth it, so only one of $d, e$ is non-zero in
practice. The solver is not told this.

**Capital.** The firm starts the period with equity $w_t$, pays a dividend or
raises new equity, and operates with $K_t = w_t - d_t + e_t$. Equity is costly
to carry: shareholders require $r_E$ while the cash it funds earns only $r$.

**Walk-away.** At the start of any period, shareholders may abandon the firm and
receive 0: limited liability. Failure is a decision, not a hazard parameter.
Who bears what is left depends on the form (§3.5, §4.5).

**GRC stock.** $S_{t+1} = (1-\delta)S_t + g_t$. Mitigation uses the post-spend
stock $S_{t+1}$, so spend protects in the period it is made.

**BTC.** The firm's net BTC exposure is whatever its business brings, plus its
treasury, less its hedge:

$$x_t = (\text{business exposure}) + q_t - H_t$$

It earns $x_t(\mu_{BTC} + s_{BTC}\,\varepsilon_{BTC})$ over the riskless rate,
where $\mu_{BTC}$ is the excess return the firm **believes** in; hedging gives
up that believed premium on what is hedged. The treasury is bought spot, so it
uses cash and capital. The hedge can reduce exposure but not create long
exposure; otherwise synthetic exposure through perps would always win and the
treasury would never be held.

### 2.3 Functional forms

Each parameter is one economic driver, the one-period limit has closed forms for
the oracles, and every form is smooth and element-wise, for the later learner.

**Issuance cost: linear plus quadratic.**

$$C(e) = \lambda e + \tfrac{\kappa}{2} e^2$$

$\lambda$ is the proportional cost and produces a pecking order. $\kappa$ is the
convex part, which in the paper is what generates $G$. **$\kappa$ is a switch,
not a calibration:** with no end date, $V$ may be concave through the walk-away
and issuance boundaries alone, and whether the Froot–Stein effect survives
$\kappa = 0$ is open (§8). A fixed issuance cost is more realistic but is left
out, because it is a discontinuity the learner cannot differentiate through.

**Continuous shocks: multivariate Student-$t$.** The shocks — to the firm's
non-tradeable risks and to BTC — are a scale mixture of normals,

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
- $\Sigma$ carries the correlations the paper prices (§3.4). BTC is
  uncorrelated with the others in $\Sigma$, but shares the tail through $W$.
- Symmetric, so it fattens the tail of gains as well as losses. One-sided
  heavy-tailed losses are the jump component (§2.4).

**GRC mitigation: exponential with an irreducible floor.**

$$\phi_j(S) = \rho_j + (1 - \rho_j)\,e^{-a_j S}, \qquad j \in \{A, B, J\}$$

$a_A$ acts on the mean loss, $a_B$ on its spread, and $a_J$ on jump frequency:
one effectiveness per mechanism (§1). $\rho_j$ is the share of the risk no
program removes; without it, enough spend drives any risk to zero.

**Hedging cost: quadratic.** $h(H) = \tfrac{c_H}{2} H^2$, standing for perp
funding and basis. With $c_H = 0$ hedging is fair. With $c_H > 0$ the risk is
"partly tradeable" (§6), and the quadratic gives a partial hedge rather than an
all-or-nothing one. Planned: mechanism D as $c_H(S)$, falling in the GRC stock.

**Cost of carrying equity:** the spread $r_E - r$, through the discount factor
$\beta = 1/(1+r_E)$. $\tau K$ is available as an extra linear friction, zero by
default.

**BTC defaults (illustrative):** volatility 60% a year, a believed excess return
of 10% a year, a capital charge of 10% on the treasury.

### 2.4 Jump losses

**Built (bank form).** The continuous shocks are frequent and moderate. The
risks that matter most on crypto rails — a drained bridge, a protocol exploit,
a custodian failure — are rare and large.

- **At most one jump per period**, with probability
  $p_J = 1 - e^{-\lambda_J \phi_J(S')}$. $\lambda_J$ is the rate at zero GRC.
- **Severity is a share of the assets exposed**, Pareto with tail index
  $\zeta$, scaled to a mean $\mu_J$, and capped at 1: a jump cannot take more
  than everything exposed. In the bank form the exposed assets are the lending
  book, and the loss is the bank's. In the vault form they are the vault's, the
  depositors bear the loss, and the firm compensates part of it (§4.3).
- **GRC acts on frequency** (prevention), through $\phi_J$. Acting on severity
  (containment) is a later option.
- **Integration.** The severity uses the same tail-placed quadrature as the
  $t$'s mixing variable (§5). The jump probability enters the quadrature
  **weights**, so it is exactly differentiable in GRC spend.

Defaults (illustrative): $\lambda_J = 0.1$ a year, $\mu_J = 5\%$ of the exposed
assets, $\zeta = 2.5$, $a_J = 100$, $\rho_J = 0.3$.

### 2.5 Decision frequency

**Built.** The firm decides **quarterly** in the main model, and the frequency
is a parameter, `periods_per_year`. Every rate is stated **per year**, and
`Params.per_period()` converts the annual statement into what one period sees:

| what | annual → per period ($n$ periods a year) | why |
|---|---|---|
| interest rates and the required return ($r$, $r_D$, $r_E$) | $(1+r)^{1/n} - 1$ | money compounds |
| the bank's book's gross return $A$ | $A^{1/n}$; $b$ then still puts the first-best book at 1 | keeps the bank's scale (§3.1) at every frequency |
| mean loss rate, excess returns ($\pi$, $\mu_{BTC}$), carrying cost $\tau$ | $\div n$ | flows that accrue with time |
| volatilities ($s_P$, $s_N$, $s_{BTC}$) | $\div \sqrt{n}$ | independent periods' variances add up to the year's. Under fat tails this is a convention, not an identity |
| jump rate | $\div n$ | Poisson events per unit of time |
| GRC depreciation | $1 - (1-\delta)^{1/n}$ | the same share lost over a year |
| issuance costs, capital charges, deposit base, jump severity, GRC effectiveness and floors, position bounds, correlations, $\nu$ | unchanged | sizes and per-event amounts, not rates |

**Use 1 to explore, 4 for realistic runs.** At quarterly frequency the discount
factor per period is closer to one, so a solve needs more policy-evaluation
steps; the solver scales them with $1/(1-\beta)$. A quarter's profit is small,
so the steady state sits close to the kink where issuance starts, and quarterly
runs need a finer equity grid than annual ones.

---

## 3. The bank form (built)

The firm takes deposits at a fixed rate and lends and invests them. Only what is
specific to the bank form is here; the rest is §2.

### 3.1 Scale

The scale is fixed by one benchmark: the size of the lending book the bank would
choose if capital were no constraint at all. That book is 1. Equity of 0.08,
for example, is equity worth 8% of it.

### 3.2 State and controls

**What the bank knows at the start of a period** — the *state*:

- $w_t$, its equity;
- $S_t$, its GRC stock.

Nothing else carries over between periods. The deposit base $D$ is fixed.

**What it decides each period**, beyond the common decisions of §2.2:

| control | meaning | paper's lever |
|---|---|---|
| $I_t \ge 0$ | size of the lending book | the investment opportunity |
| $\alpha_t$ | size of the new product | new business |

The book itself carries the non-tradeable risk — for a bank it *is* the main
non-tradeable risk. Its business BTC exposure is $b_{BTC}\,I_t$ (BTC-backed
lending), so $x_t = b_{BTC}\,I_t + q_t - H_t$.

### 3.3 Capital and cash

$$
K_t \ \ge\ k\,I_t + k_{BTC}\,q_t,
\qquad
R_t = K_t - g_t + D - I_t - q_t \ \ge 0
$$

$k$ and $k_{BTC}$ are **capital charges**: for every dollar lent, at least $k$ of
it must be the shareholders' own money, and for every dollar of BTC held, at
least $k_{BTC}$. The rest can be funded by deposits. This is the regulatory
constraint of an ordinary bank, there to keep a cushion of the owners' money
between losses and depositors; it is not a cost in itself, but it forces the
bank to hold more of its costly equity. $k = k_{BTC} = 0$ leaves only "no
negative equity". A bank left below its requirement by a bad period has three
choices:

- recapitalize, at cost $C(e)$;
- deleverage, by shrinking its book or its BTC;
- shut down, by walking away.

It is never forced to shut. Shutting is chosen when the other two are worth
less than nothing.

$R_t$ is cash, held in stablecoins: what was funded and not lent or put into
BTC. These two constraints make capital scarce. The capital charges give the
Froot–Stein channel a second route: a loss shrinks what the bank is *allowed*
to hold, not only what it can fund.

### 3.4 Closing equity

$$
w_{t+1} =
\underbrace{F(I_t) - I_t\,\ell_t - I_t J_t}_{\text{book}}
+ \underbrace{(1+r)(R_t + q_t)}_{\text{cash, and BTC's riskless part}}
+ \underbrace{\alpha_t(\pi + s_N\,\xi_N)}_{\text{new product}}
+ \underbrace{x_t(\mu_{BTC} + s_{BTC}\,\varepsilon_{BTC}) - h(H_t)}_{\text{BTC, net of the hedge}}
- \tau K_t
- (1 + r_D)\,D
$$

- $\ell_t = \phi_A(S_{t+1})\,m - \phi_B(S_{t+1})\,s_P\,\xi_P$ is the book's
  continuous non-tradeable loss rate, where $\xi_P$ is the book's payoff shock.
  It scales with the book: lending twice as much carries twice the credit risk.
- $J_t$ is the jump loss, a share of the book: zero in most periods (§2.4).
- $\Sigma$ carries $\rho_{NP}$, the correlation between the new product and the
  book: the term priced in (10).
- Depositors are owed $(1+r_D)D$ at the close, so $w_{t+1}$ is assets less
  that. The base is rolled over: the re-deposited $D$ arrives as cash next
  period through $R_{t+1}$, and is not added back here. Interest is senior to
  everything shareholders receive.

### 3.5 Walk-away and gambling for resurrection

When the bank's shareholders walk away, depositors receive the assets and bear
the shortfall, $(-w_t)^+$. With a negative $w_t$ the choice is whether
recapitalizing to $K \ge 0$, at cost $C(e)$, is worth more than nothing.

Depositors are paid a fixed rate whatever the bank's condition. Close to the
walk-away boundary, limited liability makes equity value *convex* in $w$:
shareholders keep the upside and depositors take the downside. This works
against the Froot–Stein concavity, and the model is free to show either effect
winning. The capital charges limit how far it can go. With a strong view on BTC
and a small charge on BTC, the gamble wins outright (§11), which is why the
vault form (§4) is the main model.

### 3.6 The Bellman equation, and the paper's results

$$
V(w, S) = \max\Big\{\, 0,\ \max_{d, e, g, I, \alpha, q, H}\ \big[\, d - e - C(e) + \beta\, E\, V(w', S') \,\big] \Big\}
$$

subject to the constraints of §3.3. The outer $\max\{0, \cdot\}$ is the
walk-away. One stage per period, so the stationary problem is one fixed point.

Concavity of $V$ in $w$ is not assumed. It comes from $F$, $C$, the capital
constraints, and the cost of carrying equity, $r_E - r$ (plus $\tau$ if set).
The walk-away floor pushes the other way.

**Dynamic counterparts of the paper's results.**

| paper | here | status |
|---|---|---|
| $G = -E P_{ww} / E P_w$ | a **state-dependent** effective risk aversion $G(w, S)$, measured as the price of risk on the next-period distribution (§10) | **built**, `quant/results.py` |
| hurdle (10) | a schedule in capital: the required return on a small position, measured as a price of risk (§10) | **built** for the single product |
| capital structure (14) | the **payout boundary** $\bar w$: the bank pays out above it. The cost of carrying equity sets where it sits; as $r_E \to r$ with $\tau = 0$ the bank holds a buffer | **built**, found by the solver rather than imposed |
| Prop. 2 | with no view on BTC ($\mu_{BTC} = 0$) and $h \equiv 0$, full hedging where $V$ is concave; may fail near the walk-away boundary | **built**, an oracle (§5) |
| a view on the priced factor (15) | with $\mu_{BTC} > 0$, the bank keeps net BTC exposure $x^* = \mu_{BTC} / (G\,s_{BTC}^2)$ in the two-period case | **built**, an oracle (§5) |

### 3.7 Return on the lending book

$$F(I) = A\,I - \tfrac{b}{2} I^2$$

Quadratic: $A$ is how good the opportunity is; $b$ is how fast it runs out. The
scale of §3.1 fixes $b = A - (1+r)$, so the first-best book is $I^* = 1$.

---

## 4. The vault form (stage A built)

**Status: stage A built** (`quant/vault.py`, `quant/vault_solver.py`,
`quant/vault_results.py`); stage B, the high-water mark, is designed (§4.6).
The main model from here. Everything in §2 carries over. Only what is specific
to the vault is here.

### 4.1 Two balance sheets

- **The vault**, owned by depositors through vault shares. It holds the
  investments: BTC-backed loans, BTC, stablecoins. Its gains and losses pass
  through to depositors. Its size is the **assets under management**, $A_t$.
  There is no fixed claim to default on, so the firm cannot shift losses onto
  depositors the way the bank form could.
- **The firm**, owned by shareholders. It runs the vault for fees. Its own
  equity is held in BTC (the conviction position, with the believed premium)
  and stablecoins. It pays dollar operating costs and GRC spend.

### 4.2 How the vault is invested

The firm **hires managers** to run strategies, and decides how much of the
vault each strategy gets. Each strategy has its own expected return and risk,
taken as given. This is the paper's capital-budgeting problem with several
products — allocation across strategies, priced by their covariance with the
rest of the vault (eqs. (11)–(12)).

Out of scope: the managers' own incentives (agency at the desk level, and why
firms impose position limits).

### 4.3 How the firm earns and spends

- a **management fee**: a share of AUM a year;
- a **performance fee**: a share of the vault's gains, paid only above a
  **high-water mark** — after a loss, no performance fee is earned until the
  loss is recovered. Poor performance means no compensation;
- **operating costs**, fixed in dollars, paid from stablecoins;
- **GRC spend**;
- **compensation after incidents:** depositors bear an incident's direct loss,
  but the firm compensates them for a share of it, for reputation;
- **fines** for compliance failures.

The firm's business BTC exposure (§2.2) is the sensitivity of its fee income to
BTC, through AUM.

### 4.4 How the franchise moves

AUM next period is this period's AUM, grown by the vault's return after fees,
plus net flows:

- **inflows**: steady growth toward a capacity, the size the franchise could
  reach;
- **outflows after poor returns**: depositors leave when performance is bad
  (the flow–performance relation);
- **outflows after incidents**: a hack, a depeg or a compliance failure
  triggers withdrawals beyond the direct loss.

### 4.5 Capital, failure, and where Froot–Stein and GRC act

- **A capital requirement**: the firm's own funds must be at least a share of
  AUM, as some crypto regimes require.
- **Walk-away** (§2.2): if keeping the firm alive is worth less than nothing,
  shareholders abandon it, and the vault is wound down or transferred.
- **Froot–Stein.** The firm's own capital is costly to raise. Its revenue moves
  with AUM, and AUM moves with BTC and with incidents. A bad period shrinks
  fees just when the firm needs cash for operating costs and GRC. So the firm
  should pay to protect its franchise even from risks depositors bear directly.
- **GRC** acts on incident frequency (prevention) and on the vault's loss
  variance (protecting performance, and so flows).
- **The three levers:** capital — the firm's own equity; hedging — its own BTC,
  and the BTC sensitivity of its fee income; new business — the allocation
  across strategies.

### 4.6 Staged build

| stage | adds | state | solved by |
|---|---|---|---|
| **A** | vault, management and performance fees *without* the high-water mark, flows, compensation, capital requirement, allocation across strategies | $(w, S, A)$ | 3-D grid |
| **B** | the high-water mark | $(w, S, A, \text{drawdown})$ | the learner, checked against the 3-D grid where the two overlap |

Stage A pays a performance fee on any positive period, including the recovery
from an earlier loss; stage B corrects that. Staging keeps each step checkable
against something known.

### 4.7 Scale and parameters

**Scale:** the franchise's capacity is 1; AUM and the firm's equity are
measured against it.

**Parameters** (illustrative, all per year where they are rates; converted per
period as in §2.5):

| | value |
|---|---|
| strategies: BTC-backed lending, market-neutral, tokenized treasuries | expected excess return 6%, 4%, 0; mean loss GRC can reduce 1%, 0.5%, 0; volatility 5%, 3%, 0; BTC sensitivity 0.1, 0, 0; jump exposure 1, 0.5, 0; correlation of the two risky ones 0.3 |
| fees | management 2% of AUM; performance 20% of a period's positive excess return |
| flows | inflow closes 50% of the gap to capacity a year; outflow of 1 unit of AUM per unit of negative excess return; 20% of AUM leaves after an incident |
| the firm | operating costs 1% of capacity; compensates 50% of incident losses; own funds of at least 2% of AUM |

**The firm's controls** (`quant/vault_solver.py`): dividends or issuance, GRC
spend, the BTC treasury, a perp hedge, and the allocation across strategies (a
softmax, so the shares sum to one). The hedge covers the treasury and the
vault's BTC sensitivity, so the firm can hedge the BTC exposure of the fee
income it earns on the vault.

**Two findings that need decisions** (§8): with the default view on BTC the
firm becomes a BTC holding company, and at these parameters it buys no GRC.

---

## 5. How it is solved

**Expectations by quadrature, not Monte Carlo.** Quadrature computes an
expected value from a small set of weighted scenarios, deterministically. The
$t$'s scale-mixture form makes this two nested rules: one over the shared
variance factor $W$, then Gauss–Hermite over $Z$. The rule over $W$ places its
points uniformly in log-probability, so they crowd into the tail where the
large shocks live (§10). The jump severities use the same construction. A solve
is therefore deterministic: no seed noise, and no scenario set that changes
with the batch size.

**The grid solver** (`quant/grid.py`, **built**; shared by both forms). The
state is a product grid whose first axis is equity: $(w, S)$ for the bank,
$(w, S, A)$ for the vault. The infinite-horizon problem is solved on it, exact
up to grid error. Each form supplies only its controls and objective
(`quant/solver.py`, `quant/vault_solver.py`).

- **Policy iteration.** At every grid point the controls are chosen by gradient
  ascent (Adam), for all points at once; the fixed policy is then evaluated
  repeatedly. The two alternate until the value stops changing.
- **Controls are squashed** (`sigmoid`, `tanh`, `softplus`) so every constraint
  holds by construction. A control whose optimum is a full hedge is given room
  past it (a hedge ratio up to 1.2), so the optimum is interior.
- **Two starts per improvement**: from the last policy, and from a fresh one,
  keeping the better at each grid point. A squashed control pushed to its bound
  has almost no gradient left to come back.
- **Multilinear interpolation.** Linear interpolation of a concave function
  stays concave; a smoother cubic can invent convex patches near kinks, and the
  solver gambles on them.
- **Off the grid:** in equity, above the grid extra equity is paid out (slope
  one); below it, the firm issues equity back onto the grid, which charges the
  downside of a gamble properly. Other axes are clamped.
- **Start** from a linear, risk-neutral value.

The reasons behind each of these are in §11.

**Oracles** — known-answer checks, all built and run in the test suite:

1. **Two-period truncation, normal shocks.** With one period left the value is
   the paper's $P(w)$ of (2), and the first period's choices reproduce Prop. 2,
   eq. (9), the sign of the hurdle (10), and — with a view on BTC — the
   eq. (15) case: net exposure $\mu_{BTC} / (G\,s_{BTC}^2)$. Exact under
   normality by Stein's lemma (§10); under the $t$ it is an approximation.
2. **No shocks, inert GRC, no view on BTC.** The infinite-horizon value is known
   exactly above the steady state: the bank holds its capital floor, lends a
   fixed book and pays out the rest. Checks the solver's *level*, not only its
   shape, at annual and quarterly frequency.
3. **Free equity.** With $r_E = r$ and $\tau = 0$, carrying equity costs
   nothing, so the bank holds a buffer well above its floor. Above some level it
   is indifferent between holding and paying out, so $K$ is not pinned there.
4. **Prop. 2 in the dynamic model**: with no view on BTC, a full hedge wherever
   the bank is not near walk-away.
5. **The riskless vault.** With no shocks, inert GRC and strategies earning the
   same return, AUM settles where inflows balance growth net of fees, the firm
   holds its own-funds floor and pays out the rest, and its value has a closed
   form. Matched at annual and quarterly frequency.
6. **Prop. 2 for the vault**: with no view and no BTC in the vault, the firm
   holds no net BTC.

The paper's allocation formula, eq. (12), is not an oracle for the vault: the
allocation reaches the firm only through future fees and AUM, not through a
payoff linear in the period's return, so the formula is not exact there.
Instead the allocation is checked against its first-order condition — every
strategy held has the highest marginal value.

Other tests check correctness rather than a known answer: the accounting
(bank and vault, against the formulas written out by hand), the frequency
conversion, quadrature accuracy, the N-dimensional interpolation, the hurdle's
and the allocation's first-order conditions, the BTC treasury's use of cash and
capital, and that more optimizer steps do not move the answer.

**The learner** (planned) comes in when the state outgrows a grid: the vault's
high-water mark (§4.6), GRC by family, dynamic deposits. It is trained on the
same stationary problem and **graded against the grid** wherever the grid can
still reach. With an infinite horizon, a critic is natural; the `svg-critic`
branch's reason for being parked — it lost to full backpropagation through time
at short finite horizons — does not carry over.

**Carried over from the legacy model:** `quant/numerics.py` and the profile
rule, CPU-drawn randomness for the learner's rollouts, the softplus/logit
guards, and the testing discipline.

---

## 6. Risk classification for the crypto-native firm (open)

A first, deliberately tentative cut. Each row is a question, not an answer.
"Partly tradeable" means hedgeable at a price above fair.

| risk | tradeable? | instrument, if any | how the firm treats it |
|---|---|---|---|
| BTC price risk on collateral and treasury | **mostly yes** | perps, futures, options | held partly unhedged by conviction (§2.2) |
| Interest-rate / funding-rate risk | **yes** | T-bills, rate swaps, on-chain fixed-rate markets | hedged |
| Stablecoin depeg | **partly** | thin depeg cover, diversification across issuers | GRC: B (exposure limits), D? |
| Idiosyncratic credit (crypto borrowers, protocols) | **no** | — | GRC: A, B (underwriting); C (concentration limits) |
| Smart-contract exploit, bridge drain | **partly** | on-chain cover protocols: thin, costly, capacity-limited | GRC: J (audits prevent), **D (audits make cover purchasable)** |
| Custody / key management | **partly** | custody insurance, limited capacity | GRC: A, J, D |
| Operational incidents | **no** | — | GRC: A, B |
| Compliance / license | **no** | — | GRC: J (fewer breaches; a lost license is the tail) |

---

## 7. Staging

| stage | adds | solved by | status |
|---|---|---|---|
| **v0** | bank version: one non-tradeable risk, one tradeable factor, one new product, one GRC stock, a fixed deposit base at a fixed rate, walk-away, capital charges; state $(w, S)$ | grid, quadrature | **built**; oracles pass |
| **v1** | jump losses (§2.4); decision frequency (§2.5); BTC as the home asset, with a view, a spot treasury and perp hedges (§2.2) | grid | **built** |
| **vault A** | the vault version without the high-water mark (§4.6); three strategies | 3-D grid | **built** |
| **vault B** | the high-water mark | learner, graded against the grid | designed |
| later | GRC by family (credit, operational, compliance); mechanisms C and D; the bank form revisited, with BTC deposits, depositors who price risk, and a liquidity requirement | learner | planned |

Each stage needs approval before it starts.

---

## 8. Open questions

1. **RAROC.** Check the JFE text for its comparison section, and decide whether
   it becomes an output: standalone-risk pricing against the covariance-with-
   book pricing of (10).
2. **Cost of "partly tradeable" hedging.** Linear or convex in the amount
   hedged? Quadratic is assumed (§2.3); linear would give all-or-nothing hedges.
3. **How often to re-price.** The paper's open dynamic question — how often
   headquarters should refresh $G$ — reads here as how often the risk function
   re-prices strategy allocations.
4. **Does the effect survive $\kappa = 0$?** Whether the Froot–Stein channel
   needs a convex issuance cost in a model with no end date.
5. **Calibration.** $r_E$ against a target market-to-book ratio; the vault's
   parameters (§4.7); the rest against data, once the model is realistic enough
   for its numbers to be worth reporting.
6. **How shareholders price BTC risk.** Shareholders discount everything at one
   required return $r_E$, whatever the firm holds. With a believed BTC premium
   above $r_E - r$, every dollar of BTC then looks like a positive-NPV project
   and its risk goes unpriced: in both forms the solve turns the firm into a
   BTC holding company. The paper prices the market factor in its valuation
   (Appendix A, eq. (3)); the model dropped that. Options: price BTC risk in
   shareholders' valuation, so the view becomes an excess over the market's
   price for BTC; read $\mu_{BTC}$ as that excess and set it small; or raise the
   capital charge on BTC.
7. **Why the vault firm buys no GRC.** At the illustrative parameters the vault
   firm spends nothing on GRC. Depositors bear the vault's losses; the firm
   feels them only through compensation and outflows, and those are not large
   enough here to pay for controls. Whether that is the parameters or the
   mechanism is the next question for the thesis: the franchise channel
   (outflows after incidents, compensation, fees lost) may need to be larger,
   longer-lasting, or reputational rather than one-off.

---

## 9. Departures from the paper

Each is a deliberate change from Froot & Stein (1998), made because this is a
computational model for practical analysis rather than a closed-form argument.

| departure | the paper | here | why |
|---|---|---|---|
| Horizon | three dates | infinite, stationary | a firm is a going concern; a finite horizon needs a terminal value, which carried a third of firm value in the legacy model |
| Timing | riskless investment after the risk is borne | one decision per period; the book itself is risky | for a bank the lending book *is* the main non-tradeable risk, and its credit risk grows with its size |
| Shocks | normal | multivariate Student-$t$ with a shared variance factor, plus heavy-tailed jumps | fat tails and tail dependence are what crises are made of; the normal case is kept as a limit for the oracles |
| Business | a bank with an existing portfolio | a crypto-native firm: first a bank with fixed-rate deposits, then a vault run for fees | the setting is a crypto-native intermediary; fixed-rate deposits plus a view on BTC let a bank gamble with depositors' money |
| Failure | none | walk-away under limited liability | failure is a decision rather than an asserted hazard rate, and risk-shifting can be observed rather than ruled out |
| Regulation | none | capital charges (bank); own funds against AUM (vault) | ordinary regulation; it gives the Froot–Stein channel a second route |
| Hedging | tradeable risk hedged at fair prices | hedging at a cost $c_H$ for "partly tradeable" risk | crypto-native risks (exploit cover, custody insurance, perp funding) are hedgeable only thinly and expensively |
| GRC | absent | a fourth lever acting on non-tradeable risk, and on the franchise | the project's purpose |
| Discounting | at the riskless rate $r$; the cost of equity is a separate deadweight $\tau K$ | at the shareholders' required return $r_E > r$; cash earns $r$; $\tau = 0$ by default | at the riskless rate an infinite stream of profits capitalizes into an implausibly large franchise (§11). A cost of equity gives the carrying cost of capital and a franchise of realistic size from one parameter, to be set against an observable market-to-book ratio |

---

## 10. Methods and concepts introduced without prior discussion

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
| Multilinear interpolation on an N-dimensional grid | standard | `quant/grid.py` | reads the value function between grid points | a shape-preserving smooth spline is the alternative; Catmull-Rom was tried and dropped |
| Softmax for the vault's allocation | standard | `quant/vault_solver.py` | maps unconstrained parameters to shares that sum to one | a corner allocation is only approached, not reached |
| A hedge bounded by the franchise's BTC sensitivity | own construction | `quant/vault_solver.py` | lets the firm hedge the BTC exposure of fees it earns on the vault, not only its own treasury | the fee income's true BTC sensitivity is not linear; the bound is a proxy |
| The allocation's first-order check | own construction | `quant/vault_results.py` | every held strategy has the highest marginal value; replaces eq. (12) as the vault's allocation check | — |
| Off-grid rules | own construction | `quant/solver.py` | above the grid, pay out the excess; below it, issue equity back onto it | — |
| Golden-section search | standard | `quant/twoperiod.py`, `quant/solver.py` | one-dimensional optimizer for the oracles | — |
| Squashed controls (`sigmoid`, `tanh`, `softplus`) | standard | `quant/solver.py`, `quant/vault_solver.py` | keep every constraint satisfied by construction | saturated controls lose their gradient; the two-start step and the room past a full hedge are the guards |

### Modeling and theory

| concept | source | where | what it does | if revisited |
|---|---|---|---|---|
| Stein's lemma | standard probability | `quant/twoperiod.py`, `quant/results.py` | why the paper's formulas (9), (10), (15) are exact under normal shocks and approximate under fat tails; the basis of oracle 1 | — |
| Student-$t$ as a scale mixture of normals, one shared factor | standard | `quant/quadrature.py`, §2.3 | fat tails and tail dependence from one parameter $\nu$ | copulas would separate marginal tails from dependence |
| Pareto jump severity | standard in operational-risk modeling | `quant/quadrature.py`, §2.4 | heavy-tailed one-sided losses | lognormal, or an empirical exploit-loss distribution |
| Effective risk aversion from first derivatives | own construction | `quant/results.py` | measures the paper's $G$ as the price of risk, $-E[V_w (w' - Ew')] / (E[V_w] \mathrm{Var}(w'))$; equals the paper's $G$ under normality by Stein's lemma | the right generalization under fat tails is a choice, not a theorem |
| Hurdle as a measured price of risk | own construction | `quant/results.py` | $-E[V_w \varepsilon] / E[V_w]$ for a small position with payoff shock $\varepsilon$; equals eq. (10) under normality, exact in general | — |
| Frictionless twin | own construction | **removed** | a premium measured against a bank raising equity for free | removed as an academic construction with no quantitative meaning (§11) |
| GRC value split by mechanism | own construction | **removed** | the first premium measure (mean versus spread) | removed once jumps made prevention act on both (§11) |

---

## 11. Decisions log

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
  optimization. Departures are justified and listed (§9).
- **The setting stays the crypto-rails intermediary** of `readme.md`. How its
  risks split into hedgeable and not is worked out iteratively (§6).
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
  reported rather than suppressed.
- **No negative equity, and a minimum capital ratio.** A bank below its
  requirement recapitalizes, deleverages or shuts.
- **"Partly tradeable" means hedgeable at a price above fair;** GRC can lower
  that price (mechanism D).
- **Functional forms** (§2.3): quadratic return and issuance cost, so the
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
  $\tau$ (§9).
- **Linear interpolation, not Catmull-Rom.** Catmull-Rom bulged into a convex
  patch near the payout kink; the solver gambled on it, leaving hedge errors of
  0.03 in matched $\pm$ pairs, and it missed the exact no-shock answer by
  1.7e-4. Linear interpolation hedged exactly and matched to 2e-8.
- **Below the grid the bank issues equity back onto it.** Clamping the value
  there capped the downside of any gamble while the upside grew, and the first
  solve took an unbounded position and diverged. Risky positions are also
  bounded.
- **Two starts per improvement, and a risk-neutral starting value.** The first
  solves pushed the product size and hedge to their bounds early — the
  starting value $\max(w, 0)$ has a convex kink that makes gambling look
  attractive — and squashed controls could not come back.
- **The GRC grid was widened:** at its first range, spend hit the cap.
- **The tail-placed quadrature replaced generalized Gauss–Laguerre,** which
  understated the $t$'s kurtosis (§10).
- **Effective risk aversion and hurdles are measured from first derivatives,**
  because a linearly interpolated value function has no curvature between grid
  points (§10).

**2026-11-11 — v1 and reporting**

- **Jump losses added** (§2.4), so GRC has the rare, severe losses that
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
- **Decision frequency is a parameter, quarterly by default** (§2.5), with the
  conversion conventions agreed before coding. Annual runs are for quick
  exploration. Policy-evaluation steps scale with the discount factor, because
  a fixed budget left the quarterly solve unconverged.
- **New concepts are raised before they are used,** after Gauss–Laguerre went
  into the code without discussion. §10 lists everything introduced that way.
- **US spelling throughout.**

**2026-11-11 — BTC, and the move to a vault**

- **The firm is crypto-native, and BTC is its home asset.** It holds BTC
  long-term by conviction; deposits and collateral arrive in BTC; it holds
  stablecoins to offset price risk, for liquidity, and for dollar operating
  costs. **Dollars are the numeraire;** the scale stays at "first-best book =
  1".
- **BTC as a believed excess return, not a constraint.** The firm believes BTC
  beats the riskless rate, so it keeps a net long position — the paper's
  eq. (15) case, a bank with a view on the priced factor.
- **The treasury is bought spot, and hedges are short perps only.** Spot
  holdings compete with lending for cash and carry a capital charge, kept small
  but non-zero (10%). If perps could create long exposure, synthetic exposure
  would always win and the treasury would never be held. The hedge ratio may
  run slightly past 1 so that a full hedge is reachable: capped at exactly 1,
  the solver stalled at 0.987.
- **The bank form gambles with depositors' money once it has a view on BTC.**
  With fixed-rate deposits, limited liability, a small capital charge, and a
  believed BTC premium larger than the lending margin, the solved bank stopped
  lending and put almost everything into BTC, up to its bound: a leveraged BTC
  fund financed by depositors. A degenerate regime, and the reason for the
  next decision.
- **Move to a vault** (§4): depositors hold vault shares and bear gains and
  losses directly, so there is no fixed claim to shift losses onto. The firm
  earns a management fee and a performance fee above a high-water mark,
  compensates depositors for a share of incident losses, holds own funds
  against AUM, and allocates the vault across strategies run by hired
  managers. Managers' incentives are out of scope. The bank form is kept and
  revisited later.
- **The vault is built in two stages:** without the high-water mark first, on a
  3-D grid; then with it, using the learner, checked against the grid. The
  high-water mark adds a fourth state variable, which a grid cannot carry at
  useful resolution.


**2026-11-11 — building vault stage A** (while the author was away; for review)

- **The solver was generalized to any number of state dimensions**
  (`quant/grid.py`), and the bank form rebuilt on it with every test unchanged.
  The Catmull-Rom option was dropped.
- **The vault form's stage A was built** with the Phase 0 parameters as
  proposed (§4.7).
- **The allocation is checked by its first-order condition, not by eq. (12):**
  the formula is not exact when allocation reaches the firm only through fees
  and future AUM.
- **The firm's hedge covers the franchise's BTC sensitivity** as well as its
  treasury.
- **With the default BTC view the vault firm also becomes a BTC holding
  company** — the same cause as the bank form, and now traced to shareholders'
  valuation not pricing BTC risk (§8). The vault script defaults to no view
  until this is decided.
- **At the illustrative parameters the vault firm buys no GRC** (§8). Recorded
  as a question, not a result.

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
| A view on the priced factor | A bank that chooses not to hedge $M$ because it has a view on it prices the rest with a modified risk aversion $G^M$: $\mu_N^* = \gamma\,\mathrm{cov}(\varepsilon_N, M) + G^M\,\mathrm{cov}(\varepsilon_N, \varepsilon_P^N)$ | (15) |
| Capital structure | $E P_w = 1/(1-\tau)$: hold capital until the expected shadow value of internal funds equals the cost of carrying it | (14) |

Two things the authors name as missing (§V): **a dynamic element**, in which
headquarters pools information about the bank's risks only at intervals and
desk managers price with the last $G$ they were given; and **agency at the desk
level**, the reason banks impose position limits. The first is what this
project adds.
