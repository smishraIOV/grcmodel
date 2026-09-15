# Calibration: what could be anchored, what cannot, and what has moved

Every number in this model is illustrative. That is stated honestly throughout,
and it is also the standing objection to the whole exercise: *if none of the
parameters are calibrated, why should any of the conclusions be believed?*

Three parts. **§1** sorts every parameter by how calibratable it actually is,
with a target statistic and a source for each — "uncalibrated" is doing too much
work as a single label, and most of these have external anchors that have simply
never been used. **§2** lists the modelling assumptions that carry the same risk
but are not parameters at all, the more dangerous category because nothing flags
them. **§3** is the empirical answer: how each headline behaved across four
successive recalibrations, which is the only direct evidence available about
which conclusions survive a parameter change.

The short version of §3: **the shapes held and the levels did not**, and one
headline reversed outright.

---

## 1. The parameters, by how calibratable they are

### Tier 1 — directly observable

These should simply be set from data and cited. Nothing about them is
controversial; they have not been sourced because no one has done it yet.

| parameter | current | target statistic | where it comes from |
|---|---|---|---|
| `FundingParams.deposit_capacity` | 4.0 | deposits ÷ equity | bank call reports; disclosed balance sheets of crypto lenders |
| `FundingParams.annual_deposit_rate` | 0.02 | rate paid on demandable balances | published deposit/stablecoin yields |
| `FundingParams.annual_reserve_rate` | 0.015 | yield on liquid assets | T-bill / policy rate |
| `AnnualRates.credit_loss_rate` | 0.034 | net charge-offs ÷ loans | FRED charge-off series by loan category; 2022 crypto-lending loss experience |
| `FirmParams.annual_discount_rate` | 0.08 | cost of equity | CAPM or comparable-firm estimate |
| `FirmParams.initial_equity` | 16.0 | — | **units only.** Money here has no denomination; this fixes the scale and nothing else |

### Tier 2 — observable in aggregate, needs judgement to map onto *this* firm

Real data exists, but it describes a population, and using it means arguing that
this firm resembles that population. That argument is the calibration work.

| parameter | current | target statistic | where it comes from |
|---|---|---|---|
| `HazardParams.annual_base_rate` | 0.5 | failure rate of maximally distressed firms | FDIC failure statistics; rating-agency default rates at the lowest grades |
| `HazardParams.capital_target` / `capital_scale` | 0.4 / 0.2 | how failure rate varies with the capital ratio | failure rates conditioned on capital, by bucket |
| `HazardParams.annual_licence_rate` | 0.08 | authorisations withdrawn ÷ authorisations held | regulator enforcement registers |
| `HazardParams.annual_liquidity_failure_rate` | 25.0 | survival after gating withdrawals | near-deterministic; a firm that gates is finished. Low information content, high confidence |
| `FirmParams.failure_recovery` | 0.4 | recovery in disorderly failure | FDIC loss-given-failure; crypto bankruptcy estates (Celsius, Voyager, BlockFi, FTX) |
| `FirmParams.orderly_recovery` | 0.7 | recovery in solvent wind-down | solvent liquidations; the gap to the row above is the parameter that matters |
| `FundingParams.fire_sale_haircut` | 0.35 | discount on forced sale of an unmatured book | 2023 bank securities-sale losses; crypto collateral liquidation discounts |
| `FundingParams.annual_run_rate` | 0.45 | frequency of acute deposit flight | rare-event; small sample, wide interval |
| `FundingParams.mean_withdrawal_fraction` | 0.35 | share of deposits withdrawn in a run | 2023 bank failures give actual per-day fractions; crypto withdrawal freezes |
| `FundingParams.annual_adjustment_speed` | 2.0 | how fast a deposit base rebuilds | deposit betas / franchise recovery after distress. See §2 — the weakest row here |
| `AnnualRates.op_events` / `op_severity` | 1.0 / 0.70 | operational loss frequency and severity | operational-risk loss consortia; Basel's business-indicator approach |
| `AnnualRates.compliance_events` / `compliance_severity` | 0.20 / 2.0 | enforcement frequency and penalty size | public enforcement actions, scaled to revenue |
| `AnnualRates.cliff_events` / `CliffParams.mean_severity_fraction` | 0.101 / 0.35 | exploit frequency and share of assets lost | exploit databases — **the best-documented row in this table**, and the most novel risk |

### Tier 3 — not externally observable

| parameter | current | why not | what the model does instead |
|---|---|---|---|
| `GrcAlphas.credit` / `.operational` / `.compliance` | 1.5 each | no public data links control spend to loss reduction; the firms that hold it do not publish it | **reports a break-even** — the effectiveness a programme must reach to be worth running. This is the whole reason the break-even framing exists |
| terminal franchise (`PerpetuityValue`) | 20.0 | it is a stand-in for everything past the horizon | partially anchorable via price-to-book of comparable firms; properly fixed by computing continuation value from the model itself |
| `AnnualRates.annual_return` / `curvature_per_period` | 1.15 / 698 | not independently observable | **jointly** pinned so the firm reproduces a target ROE and a target funding-constraint frequency. Identified as a pair, never separately |
| `FirmParams.initial_grc_stock` | 1.12 | it is not a free parameter at all | a **fixed point of the whole model** — the level at which optimal maintenance spend replaces depreciation. Depends on the horizon as well as the hazards (0.79 quarterly over two years, 1.12 monthly over five), so `GRC_STEADY_STATE` holds one per configuration and re-solving is mandatory after any change that moves optimal spend |
| relaxation temperatures | 0.1 | numerical, not economic | chosen by measuring gradient bias against an exact answer |
| `market_access_*`, `access_*`, `run_capital_*` | — | shape parameters of smooth transitions | chosen for smoothness, not level. Results should not be sensitive to them; that is testable and has not been tested |

### Retired or pinned, listed so they are not mistaken for live parameters

`HazardParams.annual_operational_rate` is **0.0** — retired, replaced by the run
mechanism. `financing_convexity`, `distress_reference` and
`external_funding_multiple` belong to the superseded convex-financing friction.
`ShockParams` and the `ORACLE` parameter set are **deliberately pinned** and must
never be recalibrated: they exist to be exactly solvable, and an oracle that
moves when the thing it checks moves is not an oracle.

---

## 2. Assumptions that are not parameters

More dangerous than an uncalibrated parameter, because nothing marks them as
choices. Each of these would change conclusions if wrong, and none has a number
attached that anyone would think to question.

**A run cuts the deposit level, not the deposit franchise.** Deposits rebuild
toward capacity on a four-month half-life *regardless of why they fell* —
panic and ordinary drift are the same process. Real franchises do not recover
that fast after a run. This is the single assumption doing the most work in the
"cash beats controls" finding: a slower rebuild would make runs costlier in a
way a liquidity buffer cannot offset, which is the most likely route to
operational controls being worth more than the model currently says.

**Operational and compliance losses do not scale with the balance sheet.** They
are per-event magnitudes. Credit does scale, and the cliff scales with equity.
So a firm that doubles its book doubles its credit risk and not its operational
risk. Defensible for a fixed institution; wrong for comparisons across sizes,
which is exactly what the capitalisation-band study does.

**GRC cannot reduce the cost of a failure, only its probability.** A deliberate
asymmetry — modelling GRC as reducing both would let one parameter buy the same
protection twice — but it is an assumption, not a finding.

**The book matures within the period.** There is no multi-period asset, so the
only maturity mismatch is the one the run channel creates inside a single
period.

**Mitigation is exponential in the GRC stock**, `exp(-αG)`. The functional form
is never questioned, only α. A different curvature would change the optimal
spend even at the same break-even effectiveness.

---

## 3. What actually survived four recalibrations

The direct evidence. The model has been recalibrated four times, each time for
a defensible reason, and each time every number moved. This table is what that
did to the conclusions.

**C0** original static magnitudes · **C1** first recalibration (plausible
profitability, GRC as a stock) · **C2** the liability side (deposits, leverage) ·
**C3** the run channel (withdrawals, buffer, fire sale; asserted operational
hazard retired).

**C4** is not a recalibration but a horizon change — monthly decisions over five
years rather than quarterly over two — included because it moved results as much
as any parameter change did, and because the two that it *reversed* were both
claims the earlier columns had marked as stable.

| headline | C0 | C1 | C2 | C3 | C4 | verdict |
|---|---|---|---|---|---|---|
| Spend peaks at middling capitalisation | — | peak at 16 | peak at 8 | peak at 16 | plateau 8–16 | **shape held**; location is inside the seed noise (§below) |
| GRC as a stock beats GRC as an expense | +268% | +33% | +49% | +31% | **+131%** | **sign held**, and it grows with the horizon |
| Less destroyed in failure → less spend | — | −41% | −41% | −27% | −11% | **held**, magnitude fading |
| Firm is risk-averse near failure | — | trace convex | trace convex | none | none | **held, and strengthened** |
| State feedback beats a fixed policy | — | +0.03% | −0.02% | −0.50% | +0.37% vs 0.75% noise | **held: it does not** |
| Hazard break-even (basis points) | — | 527 | 270 | 243 | 354 | level unstable |
| Survival vs loss-reduction spend ratio | ~1.3× | ~55× | ~9× | **1.6×** | 1.62× | **reversed**, then finally stable |
| Cost of ignoring survival (firm value) | 0.3% | 4.9% | 2.1% | 0.5% | 4.0% | level unstable |
| Credit effectiveness break-even (α) | — | 0.383 | 0.098 | 0.168 | — | level unstable |

### How to read this

**A caveat on the band, found by re-running it across seeds.** The "location
moved" column above is not a finding — it is sampling. At the five-year horizon
the same experiment on three scenario draws put the peak at 32, 8 and 8, because
equity 8 and 16 sit on a plateau narrower than the noise. The *shape* survives
every seed (the middle is ~3x either end); the argmax survives none of them.
Tripling the optimizer budget changes nothing, so this is scenario sampling
rather than convergence. Any row of this table that reports a *location* rather
than a shape deserves the same treatment before it is believed.

**Directional findings are robust.** Every claim about a *shape* — that spend is
non-monotone in capitalisation, that persistence is worth a lot, that a cheaper
failure buys a smaller programme, that the firm is risk-averse, that state
feedback earns nothing — survived four changes that moved every underlying
number. These are the claims the model is entitled to make.

**Levels are not.** The headline break-even halved. The cost of ignoring
survival moved by an order of magnitude in both directions. Any single quoted
level is a statement about one calibration.

**A second conclusion reversed at C4, and it was not a parameter that did it.**
"Given frequent runs the firm substitutes cash for controls" held over two years
and fails over five, where it buys more of both. A liquidity buffer protects you
today; a control stock has to be built before it protects you at all, so a firm
with two years to live cannot use one. That was a statement about the horizon
wearing the clothes of a statement about risk — which is a failure mode no
amount of parameter calibration would have caught.

**And one conclusion reversed.** "GRC is paid for almost entirely by survival
and hardly at all by loss reduction" was the project's sharpest claim at C1,
where a loss-only budget was ~55× smaller. It is now 1.6×. The two corrections
that did it were both the model becoming *more* honest — leverage (losses on a
levered book are worth far more to avoid) and retiring an asserted death rate
that was crediting GRC with preventing deaths it never demonstrated. Neither was
a recalibration for taste.

That reversal is the strongest argument in this document, in both directions: it
shows the framework catching its own error, and it shows that a confidently
stated level from this model has about a 1-in-4 record.

### What follows for how results are reported

1. **Lead with shapes; quote levels as "at the current calibration".**
2. **Prefer break-evens to budgets** — already the practice, and the reason the
   α problem is survivable.
3. **Re-run the robustness row when a headline is added**, rather than asserting
   stability.
4. **Sensitivity has not been tested one-parameter-at-a-time.** Everything above
   is evidence from four *joint* recalibrations, which confounds parameters. A
   one-at-a-time sweep of the Tier 2 rows against the headline break-even is the
   obvious next piece of work and has not been done.
