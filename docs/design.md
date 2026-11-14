# Design: a dynamic Froot–Stein (1998) crypto firm, with GRC as a fourth lever

This project is a research exercise on risk management. The academic
literature used as scaffolding here was developed for traditional finance:
banks, insurance companies, hedge funds. Ordinary banking and insurance do not
fit crypto rails well; funds and collateralized lending do. The model follows
the second pair.

**Status:** being redesigned around two vaults (§2). The shared machinery (§4,
§5) is built; the two vaults are designed, and the stablecoin lending vault is
the next stage. An earlier single-vault version is built. What is built and
what is planned is summarized in §7.

---

## 1. The firm, its business, and where GRC enters

The firm is a **crypto-native financial intermediary**. It holds BTC long-term,
out of conviction or because its founding covenants require it. It also holds
some dollar stablecoins (USDC, USDT): they offset some of BTC's price risk,
provide liquidity, and pay operating costs. **Money is measured in US
dollars.**

Its business has two lines, each run as a vault (§2):

- **A stablecoin lending vault** — investors provide dollar stablecoins, which
  the firm lends to borrowers who post BTC as collateral worth more than the
  loan. The borrowers' interest is the providers' yield, and a source of the
  firm's income. The firm sets the rate it charges borrowers.
- **A BTC yield vault** — a fund for qualified institutions holding BTC.
  Investors deposit BTC; the vault earns a BTC yield; investors share its gains
  and losses.

The firm earns fees on both.

Following Froot & Stein (1998; summarized in Appendix A), the firm has three
levers to manage risk:

- **Capital, $K$**: how much equity it holds. More equity absorbs more loss, but
  equity is costly to carry and costly to raise. Here equity is held as a mix
  of BTC and stablecoins.
- **Hedging, $H$**: how much of its *tradeable* risk — risk the market will take
  at a fair price — it sells off. The main tradeable risk is the BTC price,
  hedged with short perpetual futures (perps). The firm expects more from BTC
  than the market does (§3.4), so it does not hedge all of it away.
- **New business, $\alpha$**: how much risk it takes on in its business lines —
  here, the size and terms of the loan book and the BTC vault's strategies.

The paper's point is that these are one joint decision, not three separate
ones. All depend on the firm's effective risk aversion, $G$ in the paper, and
that depends on how much capital it has.

GRC is a **fourth** lever: a costly, lasting way to reduce the
**non-tradeable** risk the firm cannot sell in the market. This is what
operational controls, compliance programs and underwriting act on. GRC also
protects the vaults' **size and fee income**: incidents trigger withdrawals,
and withdrawals take fee income with them.

**How GRC acts** — one effect per letter, each with its own effectiveness
(§4.2):

| | effect | status |
|---|---|---|
| A | lowers average operating losses | built |
| B | narrows the spread of those losses | built |
| J | makes rare large incidents (exploits, custody failures) less frequent | built |
| L | narrows the discount at which collateral sells in a liquidation, so less bad debt (§2.2) | designed |
| C | concentration limits | later |
| D | makes cover for partly tradeable risks cheaper (§6) | later |

**Metric:** GRC's value to the firm — the firm with a program against the same
firm without one — and how that value moves with its capital, its risks and the
size of its losses.

---

## 2. The business: two vaults (designed)

**Status: designed; not built.** An earlier version — one dollar vault with
three strategies, one of them BTC-backed lending — is built (`quant/vault.py`,
`quant/vault_solver.py`, `quant/vault_results.py`) and is the starting point:
its fees, flows, compensation, co-investment and 3-D solver carry over. The
lending vault is built first, on its own (§7).

### 2.1 Three balance sheets

- **The stablecoin lending vault**, owned by its providers. It holds dollar
  stablecoins, lent out to borrowers or held idle. Its size is $A^L_t$
  dollars.
- **The BTC yield vault**, owned by its investors through vault shares. It
  holds BTC and earns a BTC yield. Its size $A^B_t$ is its BTC holdings valued
  in dollars (§2.3).
- **The firm**, owned by shareholders. It runs both vaults for fees, sets the
  borrower rate, holds its own equity in BTC and stablecoins, and pays
  operating costs and GRC.

Investors in both vaults share their vault's gains and losses. They have no
fixed claim on the firm, and the firm offers no guarantee.

### 2.2 The stablecoin lending vault

Providers deposit dollar stablecoins. The firm lends them to borrowers who post
BTC collateral.

**Terms set by the firm each period:**

- **the borrower rate**. Loan demand is a **straight line** in the rate:
  highest at a rate of zero, falling to nothing at some maximum rate. The
  loan-to-value shifts it: more generous terms attract more borrowers. The BTC
  price does not affect demand for now. The loan book is the smaller of demand
  and the stablecoins available. **Stablecoins not lent earn nothing**: holding
  USDC or USDT pays no interest; the issuer keeps it.
- **the loan-to-value at origination** (LTV): the loan as a share of the
  collateral's value, for example 50%.

**Liquidation.** Loans last one period. If BTC falls far enough that the loan
reaches the **liquidation threshold** — for example 80% of the collateral's
value — the firm sells **only as much collateral as is needed** to repay the
loan, its interest and a **liquidation penalty** (for example 5%, which goes
to the vault), and returns the rest to the borrower.

**A smoothed trigger.** The penalty makes what the vault recovers jump at the
threshold, which would give the solver a wrong slope in the LTV and the rate.
The share of a loan liquidated therefore rises along a smooth S-shaped
(logistic) curve over a narrow band of the BTC price around the threshold, 1%
of the starting price; well past the threshold it is all liquidated.

**The sale discount.** In a crash, collateral sells somewhat below the market
price: markets are thin and price feeds lag. The discount is a parameter;
GRC mechanism L (better price feeds, a more reliable liquidation process)
narrows it.

**Bad debt** arises only when the whole collateral, sold at that discount, is
worth less than the loan and its interest. With a 50% LTV it takes a fall of
roughly half the BTC price in one period, so it is rare, and it grows as the
firm raises the LTV. It arrives exactly when BTC crashes, which is also when
the firm's own BTC and its BTC-vault fees fall: risks that pile up together.
Bad debt is borne by the providers, less the share the firm repays (§2.4).

**The providers' yield** is the interest earned, plus penalties, less bad debt
and the firm's share of the interest.

### 2.3 The BTC yield vault

Investors deposit BTC and want more BTC. The vault earns a **BTC yield** from
strategies the firm's hired managers run — for example basis trades, lending
BTC, market-making — each with an expected yield, a volatility, and exposure to
large incidents (exploits, custodian failures), all in BTC terms. The firm
decides how the vault is allocated across its strategies, as in the paper's
capital-budgeting problem. Managers' own incentives are out of scope.

**Fees are percentages of the vault's BTC, paid in BTC**: a 2% management fee
on 100 BTC is 2 BTC. The firm's fee income is therefore worth more or less in
dollars as the BTC price moves — a **business exposure to BTC** (§3.4) during
the period. At the period's end the fees join the firm's equity and are
rebalanced with the rest of it (§3.3).

**The vault's size is carried in dollars** — its BTC times the price. This is
the same quantity, and it avoids making the BTC price a fifth state variable.
The vault's capacity is set in dollars, which is an approximation.

### 2.4 How the firm earns and spends

- **fees on the BTC vault**: a management fee on its BTC, and a performance fee
  on its yield, paid only above a **high-water mark** — after a loss, no
  performance fee until the loss is recovered;
- **income from the lending vault**: a share of the interest the borrowers pay;
- **operating costs**, fixed in dollars, paid from the firm's own stablecoins
  (not the vault's);
- **GRC spend**;
- **compensation**: investors bear an incident's direct loss and the lending
  vault's bad debt, but the firm repays them a fixed share of it, for
  reputation. One share, a parameter, for both;
- **co-investment**, on by default: the firm keeps its own money in each vault,
  a fixed share of the vault's size, and earns what investors earn. It gives
  the firm a direct stake in the vaults' losses.

**Insurance** against some kinds of incident is a later option: cover bought at
a price, which GRC can lower (mechanism D, §6).

### 2.5 How the vaults grow and shrink

Each vault's size next period is its size this period, grown by its return
after fees (and, for the BTC vault, by the BTC price), plus net flows:

- **inflows**: steady growth toward the vault's capacity;
- **flows that follow the yield**: net flows rise and fall with the gap
  between what investors earn and an **outside rate** — what they could earn
  elsewhere (for example on major DeFi lending markets). This is what makes
  the borrower rate costly to the firm on the providers' side: a low yield
  shrinks the vault;
- **outflows after incidents**: a hack, a depeg, a large bad-debt event or a
  compliance failure triggers withdrawals beyond the direct loss — from the
  vault hit, and possibly from both, since both carry the firm's name.

### 2.6 Capital requirement and walk-away

- **A capital requirement**: the firm's own funds must be at least a share of
  the vaults' combined size, as some crypto regimes require, plus a charge on
  the firm's own BTC.
- **Walk-away** (§3.1): the vaults are wound down or transferred, and their
  investors bear what is left.

### 2.7 Scale and parameters (to settle)

**Scale:** the lending vault's capacity is 1.

To set, all illustrative until calibrated: the demand line (its height, the
rate at which it reaches zero, how LTV shifts it); the liquidation threshold,
penalty and sale discount, and how GRC narrows the discount; the outside rate; the firm's share of
the interest; flow sensitivities; the compensation share; the co-investment
share; operating costs; the capital requirement. For the BTC vault later: its
capacity, its strategies, its fees.

---

## 3. The firm's goal and decisions

### 3.1 The objective

The firm acts for its shareholders. Each period it chooses everything at once
to maximize what they receive: the dividend paid now, less any new equity
raised and the cost of raising it, plus the value of carrying on, discounted at
the shareholders' required return $r_E$. At the start of any period they may
instead **walk away** and receive nothing: limited liability. Failure is a
decision, not an assumed rate.

$$
V(\text{state}) = \max\Big\{0,\ \max_{\text{decisions}} \big[\, d - e - C(e) + \beta\, E\, V(\text{next state}) \,\big]\Big\},
\qquad \beta = \frac{1}{1+r_E}
$$

The grid solver finds $V$ and the decisions now (§5); the learner will optimize
the same thing where the grid cannot reach. The project's question is how much
GRC adds to $V$ (§1).

**One decision per period, taken before the period's shocks.** The Froot–Stein
effect runs through the *next* period: a bad draw leaves less equity, and next
period the firm must raise costly equity or scale back.

### 3.2 The decisions

| decision | meaning | paper's lever |
|---|---|---|
| walk-away | abandon the firm; shareholders get 0 | — |
| $d_t \ge 0$ | dividend paid | capital, $K$ |
| $e_t \ge 0$ | new equity raised, at cost $C(e_t)$ | the cost of outside finance |
| $g_t \ge 0$ | GRC spend | the new lever |
| $q_t \ge 0$ | the firm's own BTC, bought spot | capital held in BTC |
| — | the firm's own stablecoins: what is left of equity after GRC spend, BTC and co-investment | capital held in stablecoins |
| $H_t$ | BTC hedged with short perps | hedging |
| borrower rate, LTV | the lending vault's terms (§2.2) | new business |
| allocation | the BTC vault's strategy shares (§2.3) | new business |

Paying and raising in the same period is never worth it, so only one of $d, e$
is non-zero in practice. The solver is not told this.

### 3.3 Capital

The firm starts the period with equity $w_t$, pays a dividend or raises new
equity, and operates with $K_t = w_t - d_t + e_t$. It holds that equity as BTC
and stablecoins (and its co-investment in the vaults). **Stablecoins earn
nothing.** So holding equity costs shareholders their required return $r_E$ on
every dollar kept in stablecoins, and $r_E$ less BTC's expected return on every
dollar kept in BTC. This is what makes capital costly to carry.

**Capital is decided by three controls:** how much equity to keep (the
dividend or new equity), how to split it between stablecoins and BTC, and how
much of the BTC to hedge against a fall. Only part of the BTC is hedged; the
hedge never exceeds it. BTC earned as fees joins the firm's equity at the
period's end and is rebalanced with the rest. Hedged and unhedged BTC carry the
same capital charge.

**GRC stock.** $S_{t+1} = (1-\delta)S_t + g_t$. Mitigation uses the
post-spend stock $S_{t+1}$, so spending protects in the period it is made. The
depreciation $\delta$ reflects how controls, staffing, tooling, security, audit
and compliance coverage wear off as people leave, systems change and threats
evolve. Equity does not depreciate: it is money, and its cost is the carrying
cost above.

### 3.4 BTC: the market's expectation, the firm's bias, and hedging

The firm's net BTC exposure is whatever its business brings, plus its own BTC,
less its hedge:

$$x_t = (\text{business exposure}) + q_t - H_t$$

**The market's expectation.** BTC's expected return, as the market prices it,
is what the market requires for holding BTC's risk. In this model shareholders
require $r_E$ of everything, so the market's expected BTC return is set to
$r_E$: on its own, BTC held by the firm is worth exactly what it costs, and an
unbiased firm gains nothing from holding it beyond what its business needs.
Valuing an asset at the market's price of its risk is standard pricing theory.

**The firm's bias.** The firm expects BTC to beat the market's expectation by a
constant $b$ a year, a parameter. It **decides** with that expectation. Its
value is **measured** at the market's expectation, which shows what the bias
costs or gains shareholders; both numbers are kept. The bias is the paper's
eq. (15) case — a firm with a view on a priced risk — and the firm's
effective risk aversion limits how much BTC it holds on that view.

**Holding BTC.** The firm's own BTC is bought spot, so it spends stablecoins
and counts against the capital requirement with a charge. BTC's price moves are
fat-tailed (§4.2).

**Hedging.** Short perps reduce the firm's BTC exposure but cannot create long
exposure; otherwise exposure through perps would always beat holding BTC, and
the firm would never hold any. Hedging gives up BTC's expected excess return
on what is hedged, and costs extra in perp funding (§4.2): a short perp gains
when BTC falls and loses when it rises, and pays funding near the riskless
rate $r$, so hedged BTC earns about $r$. The hedge covers
only the firm's own BTC, $H_t \le q_t$; fee income's BTC exposure is not
hedged during the period, and reaches the hedge once the fees join the firm's
equity.

### 3.5 State

**State:** the firm's equity $w$, its GRC stock $S$, the lending vault's size
$A^L$, and the BTC vault's size $A^B$ — four dimensions with both vaults, three
for the lending stage. A high-water mark for the BTC vault's performance fee
would add a fifth. Loans last one period and the collateral is posted fresh,
so the BTC price itself need not be state.

### 3.6 Where Froot–Stein and GRC act

- **Froot–Stein.** The firm's own capital is costly to raise. Its income moves
  with the vaults' sizes, the BTC price and incidents, and the lending vault's
  bad debt arrives in a BTC crash, together with losses on the firm's own BTC,
  its co-investment and its BTC-vault fees. A bad period shrinks income just
  when the firm needs cash for operating costs, compensation and GRC. So the
  firm should pay to protect its vaults' size and fee income even from risks
  its investors bear directly.
- **GRC** acts on average losses (A), their spread (B), incident frequency (J)
  and the sale discount in a liquidation (L).

How the paper's results carry over is in Appendix B.

---

## 4. Model details

### 4.1 Units and scale

Money is in **US dollars**. The scale is fixed by the business: the lending
vault's capacity is 1 (§2.7), and every other money quantity — the BTC vault's
dollar value, the firm's equity — is measured against it.

### 4.2 Functional forms

These forms are smooth on purpose: the learner (§5) differentiates through
them.

**Issuance cost: linear plus quadratic.**

$$C(e) = \lambda e + \tfrac{\kappa}{2} e^2$$

$\lambda$ is the cost per dollar raised. It makes the firm's own earnings
cheaper than new equity, so the firm uses them first and raises equity only
when it must. (This is not the pecking-order theory of corporate finance,
which explains the same order by managers knowing more than investors; the
model has neither that nor debt.)

$\kappa$ is the convex part, which in the papers generates the firm's
effective risk aversion $G$. **$\kappa$ is a switch, not a calibration:** with
no end date, $V$ may be concave through the walk-away and issuance boundaries
alone, and whether the Froot–Stein effect survives $\kappa = 0$ is open (§8). A
fixed issuance cost is more realistic but is left out, because it is a jump the
learner cannot differentiate through.

**Continuous shocks: multivariate Student-$t$.** The shocks — to the firm's
non-tradeable risks and to BTC — are a scale mixture of normals,

$$
\xi = \frac{Z}{\sqrt{W/\nu}}, \qquad Z \sim N(0, \Sigma), \qquad W \sim \chi^2_\nu,
$$

with **one $W$ shared** across all components, rescaled to unit variance.

- $\nu$ is a parameter. As $\nu \to \infty$ the model is normal, which is where
  the oracles run. The main configuration uses $\nu \approx 4$–$5$. Covariances
  need $\nu > 2$; finite kurtosis needs $\nu > 4$.
- The shared $W$ gives **tail dependence**: when it is small, every risk is
  large at once. That is the crisis property a normal model misses, distinct
  from fat tails in each margin alone. Copulas, which separate the two, are a
  later option.
- $\Sigma$ carries the correlations between the firm's non-tradeable risks. BTC
  is uncorrelated with the others in $\Sigma$, but shares the tail through $W$.
- Symmetric, so it fattens the tail of gains as well as losses. One-sided
  heavy-tailed losses are the jump component (§4.3).

**GRC mitigation: exponential with an irreducible floor.**

$$\phi_j(S) = \rho_j + (1 - \rho_j)\,e^{-a_j S}, \qquad j \in \{A, B, J, L\}$$

$a_j$ is mechanism $j$'s effectiveness (§1). $\rho_j$ is the share of the risk
no program removes; without it, enough spend drives any risk to zero. For L,
$\phi_L$ multiplies the sale discount in a liquidation (§2.2).

**Hedging cost: quadratic.** $h(H) = \tfrac{c_H}{2} H^2$, standing for perp
funding and basis. With $c_H = 0$ hedging is fair. With $c_H > 0$ the risk is
"partly tradeable" (§6), and the quadratic gives a partial hedge rather than an
all-or-nothing one. Planned: mechanism D as $c_H(S)$, falling in the GRC stock.

**Cost of carrying equity:** through the discount factor
$\beta = 1/(1+r_E)$ and what equity earns while held (§3.3). $\tau K$ is
available as an extra linear cost, zero by default.

**BTC defaults (illustrative):** volatility 60% a year, a capital charge of 10%
on the firm's own BTC. The firm's bias $b$ is to be set.

### 4.3 Jump losses

The continuous shocks are frequent and moderate. The risks that matter most on
crypto rails — a drained bridge, a protocol exploit, a custodian failure — are
rare and large.

- **At most one jump per period**, with probability
  $p_J = 1 - e^{-\lambda_J \phi_J(S')}$. $\lambda_J$ is the rate at zero GRC.
- **Severity is a share of the assets exposed**, Pareto with tail index
  $\zeta$, scaled to a mean $\mu_J$, and capped at 1: a jump cannot take more
  than everything exposed. The exposed assets are a vault's; its investors
  bear the loss, and the firm repays a share of it (§2.4).
- **GRC acts on frequency** (prevention), through $\phi_J$. Acting on severity
  (containment) is a later option.
- **Integration.** The severity uses the same tail-placed quadrature as the
  $t$'s mixing variable (§5). The jump probability enters the quadrature
  **weights**, so it is exactly differentiable in GRC spend.

Defaults (illustrative): $\lambda_J = 0.1$ a year, $\mu_J = 5\%$ of the exposed
assets, $\zeta = 2.5$, $a_J = 100$, $\rho_J = 0.3$.

### 4.4 Decision frequency

**Built.** The firm decides **quarterly** in the main model, and the frequency
is a parameter, `periods_per_year`. Every rate is stated **per year**, and
`Params.per_period()` converts the annual statement into what one period sees:

| what | annual → per period ($n$ periods a year) | why |
|---|---|---|
| interest rates, the required return $r_E$, the borrower rate | $(1+r)^{1/n} - 1$ | money compounds |
| mean loss rates, excess returns and the BTC bias, fees charged per year, operating costs, carrying cost $\tau$ | $\div n$ | flows that accrue with time |
| volatilities | $\div \sqrt{n}$ | independent periods' variances add up to the year's. Under fat tails this is a convention, not an identity |
| jump rate | $\div n$ | Poisson events per unit of time |
| GRC depreciation, inflow speed | $1 - (1-\delta)^{1/n}$ | the same share over a year |
| issuance costs, capital requirements, jump severity, GRC effectiveness and floors, performance-fee share, flow sensitivities, compensation share, co-investment share, capacities, loan-to-value terms, liquidation sale discount, position bounds, correlations, $\nu$ | unchanged | sizes, shares and per-event amounts, not rates |

**Use 1 to explore, 4 for realistic runs.** At quarterly frequency the discount
factor per period is closer to one, so a solve needs more policy-evaluation
steps; the solver scales them with $1/(1-\beta)$. A quarter's profit is small,
so the steady state sits close to the kink where issuance starts, and quarterly
runs need a finer equity grid than annual ones.

---

## 5. How it is solved

**The grid solver** (`quant/grid.py`, **built**) solves the objective of §3.1
exactly, up to grid error, for up to three state variables at useful
resolution. The state is a product grid whose first axis is equity. Each model
supplies only its decisions and objective.

- **Expectations by quadrature, not Monte Carlo.** Quadrature computes an
  expected value from a small set of weighted scenarios, deterministically.
  The $t$'s scale-mixture form makes this two nested rules: one over the
  shared variance factor $W$, then Gauss–Hermite over $Z$. The rule over $W$
  places its points uniformly in log-probability, so they crowd into the tail
  where the large shocks live (§10). The jump severities use the same
  construction. A solve is therefore deterministic. Directions that need more
  detail get more points: liquidations happen only after large BTC falls, so
  the BTC direction gets about 15 points where the others get 3, and a test
  checks that more points do not move the answer.
- **Policy iteration.** At every grid point the decisions are chosen by
  gradient ascent (Adam), for all points at once; the fixed policy is then
  evaluated repeatedly. The two alternate until the value stops changing.
- **Decisions are squashed** (`sigmoid`, `tanh`, `softplus`, `softmax`) so every
  constraint holds by construction. A decision whose optimum is a full hedge is
  given room past it (a hedge ratio up to 1.2), so the optimum is interior.
- **Two starts per improvement**: from the last policy, and from a fresh one,
  keeping the better at each grid point. A squashed decision pushed to its
  bound has almost no gradient left to come back.
- **Multilinear interpolation.** Linear interpolation of a concave function
  stays concave; a smoother cubic can invent convex patches near kinks, and the
  solver gambles on them.
- **Off the grid:** in equity, above the grid extra equity is paid out (slope
  one); below it, the firm issues equity back onto the grid, which charges the
  downside of a gamble properly. Other axes are clamped.
- **Start** from a linear, risk-neutral value.

The reasons behind each of these are in §11.

**The learner: actor-critic** (decided; not built). With both vaults the state
has four dimensions, and the number of quadrature scenarios grows quickly with
the number of shocks (BTC, the BTC vault's strategies, the lending vault's
losses): a grid at useful resolution is out of reach. The learner replaces the
grid with two neural networks: a **critic** that estimates the value of a
state, and an **actor** that chooses the decisions. It is trained the way the
grid solver works — one step at a time through the known model, against the
critic's estimate of the next state's value — so it fits an infinite horizon
without a value at the end of a path. It is **graded against the grid**
wherever the grid can reach. Robustness is preferred over computing time.
Expectations inside the learner will use random scenarios; how they are drawn
is open (§8). The `svg-critic` branch was parked because it lost to full
backpropagation through time at short finite horizons; that reason does not
carry over to an infinite horizon.

**Oracles** — known-answer checks:

1. **Two-period truncation, normal shocks.** The paper's own structure: with one
   period left the value is the paper's $P(w)$ of (2), and the first period's
   choices reproduce Prop. 2, eq. (9), the sign of the hurdle (10), and — with a
   view on BTC — the eq. (15) case. Exact under normality by Stein's lemma
   (§10). **Built**, but in code it rides on the bank form, and needs untangling
   from it when the bank code is removed.
2. **The riskless vault.** With no shocks, inert GRC and equal strategy
   returns, a vault's size settles where inflows balance growth net of fees,
   the firm holds its own-funds floor and pays out the rest, and its value has
   a closed form. **Built** for the earlier vault, at annual and quarterly
   frequency; to be redone for the lending vault.
3. **Prop. 2 for the vaults**: with no bias and no BTC in the business, the
   firm holds no net BTC. **Built** for the earlier vault.

The paper's allocation formula, eq. (12), is not an oracle for a vault: the
allocation reaches the firm only through fees and future vault size, not
through a payoff linear in the period's return. The allocation is checked
instead against its first-order condition — every strategy held has the
highest marginal value.

Other tests check correctness rather than a known answer: the accounting
against the formulas written out by hand, the frequency conversion, quadrature
accuracy, the N-dimensional interpolation, the first-order conditions, the
firm's BTC use of stablecoins and capital, and that more optimizer steps do not
move the answer.

**Carried over from the legacy model:** `quant/numerics.py` and the profile
rule, CPU-drawn randomness for the learner, the softplus/logit guards, and the
testing discipline.

---

## 6. Risk classification for the crypto-native firm (open)

A first, deliberately tentative cut. Each row is a question, not an answer.
"Partly tradeable" means hedgeable at a price above fair.

| risk | tradeable? | instrument, if any | how the firm treats it |
|---|---|---|---|
| BTC price risk on the firm's BTC and the BTC vault's fees | **mostly yes** | perps, futures, options | held partly unhedged on the firm's bias (§3.4) |
| Interest-rate / funding-rate risk | **yes** | T-bills, rate swaps, on-chain fixed-rate markets | hedged |
| Liquidation discount and bad debt in a BTC crash | **partly** | puts on BTC, thin and costly | GRC: L (price feeds, liquidation process); a lower loan-to-value |
| Stablecoin depeg | **partly** | thin depeg cover, diversification across issuers | GRC: B (exposure limits), D? |
| Borrower and counterparty credit | **no** | — | GRC: A, B (due diligence); C (concentration limits) |
| Smart-contract exploit, bridge drain | **partly** | on-chain cover protocols: thin, costly, capacity-limited | GRC: J (audits prevent), **D (audits make cover purchasable)** |
| Custody / key management, including collateral | **partly** | custody insurance, limited capacity | GRC: A, J, D |
| Operational incidents | **no** | — | GRC: A, B |
| Compliance / license | **no** | — | GRC: J (fewer breaches; a lost license is the tail) |

Insurance for the partly tradeable rows is a later option (§2.4).

---

## 7. Staging

| stage | adds | solved by | status |
|---|---|---|---|
| **machinery** | quadrature, fat tails, jumps, decision frequency, BTC, the firm's own BTC and perp hedges; the N-dimensional grid solver; the two-period oracle | grid | **built** |
| **single vault** | one dollar vault, three strategies, fees, flows, compensation, co-investment, a guarantee | 3-D grid | **built**; superseded by §2, and the starting point for it |
| **lending vault** | the stablecoin lending vault alone (§2.2): borrower rate and LTV, liquidation and bad debt, mechanism L; stablecoins earning nothing; BTC at the market's expectation plus the firm's bias; the bank code removed and the two-period oracle moved off it | 3-D grid | **next**; a build plan comes first |
| **BTC vault** | the BTC yield vault alongside the lending vault | actor-critic, graded against the grid | designed |
| **high-water mark** | the BTC vault's performance fee above a high-water mark | actor-critic | designed |
| later | insurance; GRC by family (credit, operational, compliance); mechanisms C and D | actor-critic | planned |

Each stage needs approval before it starts.

---

## 8. Open questions

1. **How the learner draws its scenarios.** Plain random draws, a fixed set
   reused across training steps, or quasi-random points spread evenly over the
   shocks — to be raised with the learner stage.
2. **Why the earlier vault firm bought no GRC.** Investors bore the vault's
   losses; the firm felt them only through compensation and outflows, and
   raising equity was cheap enough that its capital never became scarce. The
   lending vault's bad debt in a crash and co-investment are what should make
   the firm's own capital matter now.
3. **Crashes within a period.** Liquidation is driven by the period's fall, but
   BTC can fall further during a quarter than it ends down. A crash term for
   BTC, or a sale discount that reflects it, may be needed.
4. **Linking the two vaults.** Could borrowers' BTC collateral be deposited in
   the BTC yield vault to earn yield (rehypothecation)? It would tie the two
   vaults' risks together; it is excluded unless decided otherwise.
5. **RAROC.** Check the JFE text for its comparison section, and decide whether
   it becomes an output: standalone-risk pricing against the covariance-with-
   book pricing of (10).
6. **Cost of "partly tradeable" hedging.** Linear or convex in the amount
   hedged? Quadratic is assumed (§4.2); linear would give all-or-nothing hedges.
7. **How often to re-price.** The paper's open dynamic question — how often
   headquarters should refresh $G$ — reads here as how often the risk function
   re-prices the vaults' allocations and the loan terms.
8. **Does the effect survive $\kappa = 0$?** Whether the Froot–Stein effect
   needs a convex issuance cost in a model with no end date.
9. **Calibration.** $r_E$ against a target market-to-book ratio; the vaults'
   parameters (§2.7); the rest against data, once the model is realistic
   enough for its numbers to be worth reporting.

---

## 9. Departures from the paper

Each is a deliberate change from Froot & Stein (1998), made because this is a
computational model for practical analysis rather than a closed-form argument.

| departure | the paper | here | why |
|---|---|---|---|
| Horizon | three dates | infinite, stationary | a firm is a going concern; a finite horizon needs a terminal value, which carried a third of firm value in the legacy model |
| Timing | riskless investment after the risk is borne | one decision per period; the business itself is risky | the firm's main non-tradeable risks sit in what it runs, and grow with its size |
| Shocks | normal | multivariate Student-$t$ with a shared variance factor, plus heavy-tailed jumps | fat tails and tail dependence are what crises are made of; the normal case is kept as a limit for the oracles |
| Business | a bank with an existing portfolio | a crypto-native firm running a stablecoin lending vault and a BTC yield vault for fees | banking and insurance fit crypto rails poorly; funds and collateralized lending fit well. A bank form with fixed-rate deposits was built and dropped (§11) |
| Failure | none | walk-away under limited liability | failure is a decision rather than an assumed hazard rate, and risk-shifting can be observed rather than ruled out |
| Regulation | none | own funds against the vaults' size | ordinary regulation of asset managers and lenders; it gives the Froot–Stein effect a second route |
| Hedging | tradeable risk hedged at fair prices | hedging at a cost $c_H$ for "partly tradeable" risk | crypto-native risks (exploit cover, custody insurance, perp funding) are hedgeable only thinly and expensively |
| GRC | absent | a fourth lever acting on non-tradeable risk, and on the vaults' size | the project's purpose |
| Discounting | at the riskless rate $r$; the cost of equity is a separate deadweight $\tau K$ | at the shareholders' required return $r_E$; stablecoins earn nothing; BTC is expected to earn $r_E$ at the market's price, plus the firm's bias; $\tau = 0$ by default | at the riskless rate an infinite stream of profits capitalizes into an implausibly large firm (§11). A cost of equity gives the carrying cost of capital and a firm value of realistic size from one parameter, to be set against an observable market-to-book ratio |

---

## 10. Methods and concepts introduced without prior discussion

Concepts brought into the code or this document by Claude, not taken from the
papers and not discussed before use. Accepted for now (2026-11-11), and kept
here so any of them can be revisited. New concepts are now raised before they
are used (`CLAUDE.md`), and recorded in the decisions log (§11) when agreed.

*Standard* means a textbook or widely used technique; *own construction* means
assembled for this project, with no single reference.

### Numerical methods

| concept | source | where | what it does | if revisited |
|---|---|---|---|---|
| Gauss–Hermite quadrature | standard | `quant/quadrature.py` | weighted scenario set for the normal part of the shocks | sparse grids if the shock dimension grows |
| Tail-placed mixing rule | own construction | `quant/quadrature.py` | scenario set for the $t$'s shared volatility factor: Gauss–Legendre nodes uniform in log-probability, placed through the inverse CDF | replaced generalized Gauss–Laguerre (standard), which understated kurtosis (7.3 against 9 at $\nu = 5$, 64 points); accuracy is checked by tests, not by a reference |
| Tail-placed jump-severity rule | own construction | `quant/quadrature.py` | the same construction for the Pareto jump severity, capped at the whole book | the dropped far tail and the cap shave the mean slightly; tested to within 5% |
| Cholesky factorization | standard | `quant/quadrature.py` | builds correlated shocks from independent ones | — |
| Policy iteration with repeated policy evaluation (Howard improvement) | standard | `quant/grid.py` | how the infinite-horizon solve converges | plain value iteration is slower but simpler |
| Adam, vectorized over the grid | standard | `quant/grid.py` | chooses every grid point's decisions at once by gradient ascent; two starts per step, and a slow final pass | the step-budget test guards against it stopping short |
| Multilinear interpolation on an N-dimensional grid | standard | `quant/grid.py` | reads the value function between grid points | a shape-preserving smooth spline is the alternative; Catmull-Rom was tried and dropped |
| Softmax for the vault's allocation | standard | `quant/vault_solver.py` | maps unconstrained parameters to shares that sum to one | a corner allocation is only approached, not reached |
| A hedge bounded by the vault's BTC sensitivity | own construction | `quant/vault_solver.py` | lets the firm hedge the BTC exposure of fees it earns on the vault, not only its own BTC | superseded (2026-11-13): the hedge covers only the firm's own BTC; fees are rebalanced once they join its equity |
| The allocation's first-order check | own construction | `quant/vault_results.py` | every held strategy has the highest marginal value; replaces eq. (12) as the vault's allocation check | — |
| Off-grid rules | own construction | `quant/grid.py` | above the grid, pay out the excess; below it, issue equity back onto it | — |
| Golden-section search | standard | `quant/twoperiod.py`, `quant/solver.py` | one-dimensional optimizer for the oracles | — |
| Squashed decisions (`sigmoid`, `tanh`, `softplus`) | standard | `quant/solver.py`, `quant/vault_solver.py` | keep every constraint satisfied by construction | saturated decisions lose their gradient; the two-start step and the room past a full hedge are the guards |

### Modeling and theory

| concept | source | where | what it does | if revisited |
|---|---|---|---|---|
| Stein's lemma | standard probability | `quant/twoperiod.py`, `quant/results.py` | why the paper's formulas (9), (10), (15) are exact under normal shocks and approximate under fat tails; the basis of oracle 1 | — |
| Student-$t$ as a scale mixture of normals, one shared factor | standard | `quant/quadrature.py`, §4.2 | fat tails and tail dependence from one parameter $\nu$ | copulas would separate marginal tails from dependence |
| Pareto jump severity | standard in operational-risk modeling | `quant/quadrature.py`, §4.3 | heavy-tailed one-sided losses | lognormal, or an empirical exploit-loss distribution |
| Effective risk aversion from first derivatives | own construction | `quant/results.py` | measures the paper's $G$ as the price of risk, $-E[V_w (w' - Ew')] / (E[V_w] \mathrm{Var}(w'))$; equals the paper's $G$ under normality by Stein's lemma | the right generalization under fat tails is a choice, not a theorem |
| Hurdle as a measured price of risk | own construction | `quant/results.py` | $-E[V_w \varepsilon] / E[V_w]$ for a small position with payoff shock $\varepsilon$; equals eq. (10) under normality, exact in general | — |
| Frictionless twin | own construction | **removed** | a premium measured against a bank raising equity for free | removed as an academic construction with no quantitative meaning (§11) |
| GRC value split by mechanism | own construction | **removed** | the first premium measure (mean versus spread) | removed once jumps made prevention act on both (§11) |

---

## 11. Decisions log

What was decided, when, and why. Newest last. Section numbers in older entries
are today's.

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
- **Functional forms** (§4.2): quadratic return and issuance cost, so the
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

- **Jump losses added** (§4.3), so GRC has the rare, severe losses that
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
- **Decision frequency is a parameter, quarterly by default** (§4.4), with the
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
  eq. (15) case, a bank with a view on the priced factor. (Replaced on
  2026-11-13 by the market's expectation plus a bias.)
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
- **Move to a vault:** depositors hold vault shares and bear gains and
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
  proposed.
- **The allocation is checked by its first-order condition, not by eq. (12):**
  the formula is not exact when allocation reaches the firm only through fees
  and future AUM.
- **The firm's hedge covers the vault's BTC sensitivity** as well as its
  treasury.
- **With the default BTC view the vault firm also becomes a BTC holding
  company** — the same cause as the bank form, traced to shareholders'
  valuation not pricing BTC risk. The vault script defaults to no view until
  this is decided. (Decided 2026-11-13.)
- **At the illustrative parameters the vault firm buys no GRC** (§8). Recorded
  as a question, not a result.

**2026-11-12 — two vaults; the bank form dropped**

- **The bank form is dropped from the design.** It was built first because it
  fits the academic papers, which were written for traditional finance: banks,
  insurers, hedge funds. Ordinary banking and insurance fit crypto rails
  poorly; funds and collateralized lending fit well. Its code is to be removed
  in a separate step; its history stays in this log.
- **The firm runs two vaults** (§2): a **BTC-denominated yield vault** for
  qualified institutions holding BTC, and a **stablecoin lending vault**, whose
  providers' dollars are lent to borrowers over-collateralized by BTC. The
  borrowers' interest is the providers' yield and a source of the firm's
  income.
- **The borrower rate is set by the firm** each period.
- **Lending's mechanics** — loan-to-value, a liquidation threshold and penalty,
  the shortfall in a crash and bad debt — were raised and agreed before being
  written down. They are standard in crypto lending, not from the papers.
- **The single-vault model built as stage A** (three strategies in one dollar
  vault) is superseded by this design. Its machinery — accounting, fees, flows,
  compensation, co-investment, the 3-D solver — is the starting point for the
  rework.
- **Both vaults are in the model,** so the state has four dimensions: the
  firm's equity, its GRC stock, and the two vaults' sizes.

**2026-11-13 — review of the two-vault design**

- **Stablecoins earn nothing,** in the lending vault and in the firm's own
  holdings: holders of USDC or USDT receive no interest. Holding equity in
  stablecoins therefore costs the full $r_E$.
- **BTC at the market's expectation plus a bias.** The market's expected BTC
  return is what it requires for BTC's risk, set to $r_E$ here, so BTC alone
  adds nothing to firm value. The firm expects a constant bias $b$ above that,
  decides with it, and its value is measured at the market's expectation (§3.4).
  This settles why every earlier version became a BTC holding company.
  Valuing at the market's price of risk is standard pricing theory, raised
  before use.
- **Fees are percentages of the vault's BTC, paid in BTC;** the vault's size is
  carried in dollars so the BTC price need not be state (§2.3).
- **No guarantees.** The guarantee is out of scope; the code's (off by default)
  goes with the rework.
- **Co-investment is on by default:** the firm has some of its own money in
  each vault.
- **Bad debt is repaid at the same share as incident losses,** one parameter,
  not a decision.
- **Loan demand is a straight line in the borrower rate,** not linked to the
  BTC price for now.
- **Liquidation sells only what is needed** to repay the loan, its interest
  and the penalty; the rest goes back to the borrower. Bad debt arises only
  when the whole collateral cannot cover the loan. (A first version, in which
  collateral sold below the threshold by a share of the period's fall, was
  dropped the same day.)
- **GRC mechanism L narrows the sale discount** in a liquidation.
- **Agreed for the lending build:** providers' flows follow the gap between
  their yield and an outside rate (own construction); more quadrature points
  in the BTC direction than in the others (standard); a short perp pays the
  BTC fall less funding near $r$ (standard); the Prop. 2 test alone lets the
  hedge run to 1.2 times the BTC held, so a full hedge is reachable.
- **The liquidation trigger is smoothed** over a 1% band of the BTC price, so
  the penalty does not make the payoff jump (standard numerical smoothing,
  raised and agreed). A BTC price cannot fall below zero.
- **Operating costs are paid from the firm's own stablecoins,** separate from
  the vault's.
- **Insurance is a later option** (mechanism D).
- **The firm's capital is held as stablecoins and BTC, part of the BTC hedged
  against a fall.** Capital is decided by three controls: how much equity to
  keep, the split between stablecoins and BTC, and how much of the BTC to
  hedge (§3.3). Hedged and unhedged BTC carry the same capital charge.
- **The hedge covers only the firm's own BTC.** BTC earned as fees joins the
  firm's equity at the period's end and is rebalanced with it, so fee income
  is not hedged during the period.
- **The learner is actor-critic,** graded against the grid; robustness is
  preferred over computing time. How it draws scenarios is open (§8).
- **The lending vault is built first,** alone, on a 3-D grid (§7).
- **The document was reordered:** the business and the firm's goal come before
  the technical details, and the paper's results moved to Appendix B. "Pecking
  order" was dropped as a term: the issuance cost only makes the firm's own
  earnings cheaper than new equity.

---

## Appendix A. What the 1998 paper builds

*Optional reading.* This describes the model in the paper; ours adapts it to an
infinite horizon (§3). Anything striking here that our model lacks can be
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

---

## Appendix B. The paper's results here

*Optional reading.* How the paper's results (Appendix A) carry over, and what
is built.

| paper | here | status |
|---|---|---|
| $G = -E P_{ww} / E P_w$ | a **state-dependent** effective risk aversion, measured as the price of risk on the next-period distribution of the firm's equity (§10) | **built** |
| hurdle (10) | the required return on a small position, measured as a price of risk (§10) | **built** for the earlier vault's strategies |
| many products (11)–(12) | the allocation of the BTC vault across strategies, and the size and terms of the loan book; checked by its first-order condition, since the formula is not exact here (§5) | **built** for the earlier vault |
| capital structure (14) | the **payout boundary**: the firm pays out above it, found by the solver rather than imposed | **built** |
| Prop. 2 | with no bias on BTC and fair hedging, the firm holds no net BTC | **built**, an oracle (§5) |
| a view on the priced factor (15) | with a bias, the firm keeps net BTC exposure $b / (G\,s_{BTC}^2)$ in the two-period case | **built**, an oracle (§5) |
