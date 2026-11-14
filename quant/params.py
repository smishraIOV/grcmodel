"""Parameters of the firm (docs/design.md sections 2-4).

Plain floats, frozen. Every parameter is one economic driver; the comment on
each says which. All values are illustrative, not calibrated.

**Every rate is stated per year.** The model runs at `periods_per_year`
decisions a year, and `per_period()` converts the annual statement into what
one period sees (design.md section 4.4). Sizes and per-event amounts do not
convert. With `periods_per_year = 1` the conversion is the identity, so it is
safe to apply twice; the solver applies it once, on entry.

Use 1 for fast exploratory runs and tests, and 4 (the default, quarterly) for
realistic ones. Each extra period a year brings the discount factor closer to
one, so a solve takes more iterations to converge.

Units (design.md section 4.1): money is in dollars, scaled so that a vault's
capacity is 1.
"""

import math
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Params:
    # -- frequency --------------------------------------------------------
    periods_per_year: int = 4  # decisions a year: 4 is quarterly, 1 is annual (fast, for testing)

    # -- rates (per year) ---------------------------------------------------
    r: float = 0.02          # riskless rate; cash earns it
    r_E: float = 0.06        # shareholders' required return; beta = 1/(1+r_E). Departure from the
                             # paper, design.md section 9: to be set against a target market-to-book

    # -- frictions: what makes the firm care about risk ------------------
    lam: float = 0.05        # proportional cost of issuing equity (per unit raised)
    kappa: float = 1.0       # convex cost of issuing equity. A switch, not a calibration
    tau: float = 0.0         # extra deadweight cost of carrying equity, per year; the main cost is r_E - r

    # -- BTC ----------------------------------------------------------------
    # Money is in dollars (the numeraire); BTC is the firm's home asset.
    # mu_BTC is the single-vault form's believed excess return over r
    # (design.md section 3.4 replaces it with the market's expectation plus a
    # bias, in the lending form).
    s_BTC: float = 0.60      # BTC's dollar volatility, per year
    mu_BTC: float = 0.10     # single-vault form: believed excess return on BTC, per year
    k_BTC: float = 0.10      # capital charge on the firm's own BTC
    q_max: float = 1.0       # bound on the firm's own BTC, so a risk-seeking region stays finite
    c_H: float = 0.0         # cost of hedging with short perps, c_H/2 * H^2; zero is fair

    # -- shocks ----------------------------------------------------------
    nu: float = 5.0          # Student-t degrees of freedom; math.inf is the normal case

    # -- jump losses (v1, design.md section 4.3) --------------------------
    jump_rate: float = 0.1   # jumps per year at zero GRC (Poisson; at most one a period)
    jump_mean: float = 0.05  # mean share of the exposed assets lost in a jump
    jump_tail: float = 2.5   # Pareto tail index of the severity; > 2 keeps the variance finite

    # -- GRC -------------------------------------------------------------
    delta: float = 0.25      # share of the GRC stock lost per year
    a_A: float = 50.0        # effectiveness against the mean loss, per unit of stock
    a_B: float = 50.0        # effectiveness against the spread
    rho_A: float = 0.3       # share of the mean loss no program removes
    rho_B: float = 0.3       # share of the spread no program removes
    a_J: float = 100.0       # effectiveness against jump frequency (prevention)
    rho_J: float = 0.3       # share of jump frequency no program removes

    # -- the single-vault form (superseded; the start of the BTC vault) -----
    # Three strategies run by hired managers; the firm sets the allocation.
    # Order: BTC-backed lending, market-neutral (basis, market-making),
    # tokenized treasuries. Returns are excess over the riskless rate.
    vault_mu: tuple[float, float, float] = (0.06, 0.04, 0.0)      # expected excess return, per year
    vault_loss: tuple[float, float, float] = (0.01, 0.005, 0.0)   # mean loss rate GRC can reduce, per year
    vault_vol: tuple[float, float, float] = (0.05, 0.03, 0.0)     # volatility, per year
    vault_btc: tuple[float, float, float] = (0.1, 0.0, 0.0)       # sensitivity to BTC's return
    vault_jump: tuple[float, float, float] = (1.0, 0.5, 0.0)      # exposure to jump losses
    vault_rho: float = 0.3          # correlation of the two risky strategies
    capacity: float = 1.0           # the vault's full size; fixes the scale
    fee_mgmt: float = 0.02          # management fee, share of AUM per year
    fee_perf: float = 0.20          # performance fee, share of a period's positive excess return
    opex: float = 0.02              # operating costs, dollars per year, fixed (payroll is sticky)
    payroll_share: float = 0.60     # share of operating costs that is payroll
    runway: float = 1.0             # operating costs held in stablecoins, in years (lending form); per period after conversion
    inflow_speed: float = 0.5       # share of the gap to capacity closed per year
    outflow_perf: float = 1.0       # extra outflow per unit of negative excess return
    outflow_incident: float = 0.2   # share of AUM that leaves after an incident
    compensation: float = 0.5       # share of incident losses the firm makes good
    k_A: float = 0.02               # own funds required per unit of AUM
    co_invest: float = 0.0          # the firm's own stake in the vault, share of AUM ("skin in the game")

    # -- the stablecoin lending vault (design.md section 2.2) ----------------
    # Shares with the single vault: inflow_speed, outflow_incident,
    # compensation (also applied to bad debt), k_A, opex, and the jump rule.
    btc_bias: float = 0.0           # the firm's expected BTC return above the market's (r_E), per year
    btc_holding_cost: float = 0.015 # securing the firm's own BTC (custody, keys, insurance), share per year
    btc_crash_rate: float = 0.5     # sudden BTC crashes per year (a market event; GRC does not change it)
    btc_crash_mean: float = 0.25    # mean share of BTC's price a crash takes
    btc_crash_tail: float = 2.5     # Pareto tail index of the crash size
    lend_capacity: float = 1.0      # the vault's full size; fixes the scale
    loan_demand: float = 1.5        # loans wanted at a zero rate and the reference LTV
    rate_max: float = 0.20          # borrower rate at which demand reaches zero, per year
    ltv_ref: float = 0.50           # reference loan-to-value for loan_demand
    ltv_demand: float = 1.0         # demand shift per unit of LTV: +10% per 10 points
    ltv_min: float = 0.30           # bounds on the LTV the firm may offer
    ltv_max: float = 0.75
    liq_threshold: float = 0.80     # loan (with interest) to collateral value at which liquidation starts
    liq_penalty: float = 0.05       # paid by a liquidated borrower to the vault, share of what is owed
    liq_band: float = 0.01          # width of the smoothed liquidation trigger, in BTC price (start = 1)
    sale_discount: float = 0.10     # how far below market liquidated collateral sells, at zero GRC
    a_L: float = 50.0               # GRC effectiveness against the sale discount (mechanism L)
    rho_L: float = 0.3              # share of the discount no program removes
    interest_share: float = 0.40    # the firm's share of the interest collected; high for now (design.md 11)
    outside_rate: float = 0.05      # what providers could earn elsewhere, per year
    flow_yield: float = 2.0         # net flows, share of the vault, per unit of yield gap
    lend_loss: float = 0.002        # mean operating loss, share of the vault per year, at zero GRC
    lend_vol: float = 0.005         # its volatility, per year
    lend_co_invest: float = 0.05    # the firm's own stake in the lending vault, share of its size

    def __post_init__(self):
        if not (isinstance(self.periods_per_year, int) and self.periods_per_year >= 1):
            raise ValueError("periods_per_year must be a positive integer")
        if not self.nu > 2.0:
            raise ValueError("nu must exceed 2 for covariances to exist")
        if not self.r_E >= self.r:
            raise ValueError("shareholders cannot require less than cash earns: r_E >= r")
        if self.btc_crash_rate > 0 and not self.btc_crash_tail > 2.0:
            raise ValueError("btc_crash_tail must exceed 2 for the crash variance to exist")
        if not self.sale_discount < 1.0 - self.liq_threshold:
            raise ValueError("sale_discount must be below 1 - liq_threshold, or selling at the threshold leaves bad debt")
        if self.jump_rate > 0 and not self.jump_tail > 2.0:
            raise ValueError("jump_tail must exceed 2 for the jump variance to exist")
        if not (0.0 <= self.fee_perf < 1.0 and 0.0 <= self.compensation <= 1.0):
            raise ValueError("fee_perf must be in [0, 1) and compensation in [0, 1]")
        if not 0.0 < self.ltv_min <= self.ltv_max < self.liq_threshold <= 1.0:
            raise ValueError("need 0 < ltv_min <= ltv_max < liq_threshold <= 1")
        if not (0.0 <= self.sale_discount < 1.0 and 0.0 <= self.lend_co_invest < 1.0):
            raise ValueError("sale_discount and lend_co_invest must be in [0, 1)")

    def per_period(self) -> "Params":
        """The same firm, with every rate restated for one period (design.md 4.4).

        Returned with `periods_per_year = 1`, so applying it again changes
        nothing. The conventions:

        - interest rates and the required return compound: (1 + r)^(1/n) - 1;
        - flows that accrue with time divide by n: BTC's excess return, the
          carrying cost tau;
        - volatilities divide by sqrt(n), the square-root-of-time rule.
          Independent periods' variances add up to the year's. Under fat
          tails this is a convention, not an identity: a fat-tailed year is
          not four fat-tailed quarters summed;
        - the jump rate divides by n (Poisson events per unit of time);
        - GRC depreciation keeps the same annual loss: 1 - (1 - delta)^(1/n).

        Sizes and per-event amounts do not convert: issuance costs, jump severity, GRC effectiveness and
        floors, position bounds, correlations, nu. For the vault: the
        management fee, operating costs and strategies' returns and losses are
        flows (divide by n); strategy volatilities follow sqrt(n); the inflow
        speed keeps its annual effect; the performance fee, flow sensitivities, compensation share, capacity,
        capital requirement and co-investment share are shares or sizes and do
        not convert. For the lending vault: the rate at which demand reaches
        zero and the outside rate compound; the BTC bias, the BTC holding cost and the
        operating loss are flows; the BTC crash rate divides by n; the runway, stated in
        years, becomes a number of periods, so runway x opex is unchanged; the loss volatility follows sqrt(n); the demand line's
        height and LTV shift, LTV bounds, liquidation threshold, penalty, trigger
        band, sale discount, interest share, flow response (a share of the vault per unit
        of yield gap, both per period) and capacity do not convert.
        """
        n = self.periods_per_year
        if n == 1:
            return self
        root = math.sqrt(n)

        def compound(rate: float) -> float:
            return (1.0 + rate) ** (1.0 / n) - 1.0

        return replace(
            self,
            periods_per_year=1,
            r=compound(self.r),
            r_E=compound(self.r_E),
            tau=self.tau / n,
            s_BTC=self.s_BTC / root,
            mu_BTC=self.mu_BTC / n,
            jump_rate=self.jump_rate / n,
            delta=1.0 - (1.0 - self.delta) ** (1.0 / n),
            vault_mu=tuple(x / n for x in self.vault_mu),
            vault_loss=tuple(x / n for x in self.vault_loss),
            vault_vol=tuple(x / root for x in self.vault_vol),
            fee_mgmt=self.fee_mgmt / n,
            opex=self.opex / n,
            runway=self.runway * n,
            inflow_speed=1.0 - (1.0 - self.inflow_speed) ** (1.0 / n),
            btc_bias=self.btc_bias / n,
            btc_holding_cost=self.btc_holding_cost / n,
            btc_crash_rate=self.btc_crash_rate / n,
            rate_max=compound(self.rate_max),
            outside_rate=compound(self.outside_rate),
            lend_loss=self.lend_loss / n,
            lend_vol=self.lend_vol / root,
        )

    @property
    def beta(self) -> float:
        """Discount factor per period. Read it off `per_period()`."""
        return 1.0 / (1.0 + self.r_E)

    @property
    def normal(self) -> bool:
        return math.isinf(self.nu)


DEFAULTS = Params()
