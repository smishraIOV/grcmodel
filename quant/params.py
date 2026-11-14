"""Parameters of the bank (docs/design.md sections 2-4).

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

Units (design.md section 4.1): money is measured so that the first-best book is
exactly 1. That pins the curvature of the return on the book, `b`, at every
frequency: F'(1) must equal the return on cash over one period.

**"First-best"** is the economist's term for the optimum with no frictions at
all: here, what the bank would lend if capital were free and unlimited, where
the last unit lent earns exactly what cash earns (F'(I) = 1 + r). It is a
reference point for the units, not the model's solution. The model's optimal
policy lends less, because capital is costly and scarce -- the "second-best" --
and the gap between the two is the underinvestment Froot-Stein is about. Used
in this sense throughout the code.
"""

import math
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Params:
    # -- frequency --------------------------------------------------------
    periods_per_year: int = 4  # decisions a year: 4 is quarterly, 1 is annual (fast, for testing)

    # -- rates and the opportunity (per year) -----------------------------
    r: float = 0.02          # riskless rate; cash earns it
    r_E: float = 0.06        # shareholders' required return; beta = 1/(1+r_E). Departure from the
                             # paper, design.md section 9: to be set against a target market-to-book
    A: float = 1.08          # gross return on the first unit lent over a year, F'(0)

    # -- frictions: what makes the bank care about risk ------------------
    lam: float = 0.05        # proportional cost of issuing equity (per unit raised)
    kappa: float = 1.0       # convex cost of issuing equity. A switch, not a calibration
    tau: float = 0.0         # extra deadweight cost of carrying equity, per year; the main cost is r_E - r

    # -- liabilities and regulation --------------------------------------
    D: float = 0.9           # deposit base, fixed for now, rolled over each period
    r_D: float = 0.015       # rate paid on deposits, per year, senior to dividends
    k: float = 0.08          # minimum capital ratio: K >= k * I

    # -- the book's non-tradeable risk -----------------------------------
    m: float = 0.01          # mean loss rate on the book at zero GRC, per year
    s_P: float = 0.02        # standard deviation of the book's return per unit lent, per year

    # -- the new product (capital budgeting) -----------------------------
    pi: float = 0.01         # expected excess payoff per unit of the product, per year
    s_N: float = 0.10        # its standard deviation per unit, per year
    rho_NP: float = 0.5      # correlation of its payoff with the book's
    alpha_max: float = 1.0   # bound on its size, so a risk-seeking region stays finite

    # -- BTC: the tradeable factor, and the firm's view on it --------------
    # Money is in dollars (the numeraire). BTC is the firm's home asset: its
    # book is partly BTC-backed, and it holds a BTC treasury out of conviction.
    # It believes BTC beats the riskless rate by mu_BTC a year, so it does not
    # hedge all of it away -- the paper's eq. (15) case, a bank with a view on
    # the priced factor (design.md section 3.4).
    b_BTC: float = 0.5       # the book's loading on BTC (BTC-backed lending)
    s_BTC: float = 0.60      # BTC's dollar volatility, per year
    mu_BTC: float = 0.10     # the firm's believed excess return on BTC, per year
    k_BTC: float = 0.10      # capital charge on the BTC treasury: K >= k I + k_BTC q
    q_max: float = 1.0       # bound on the BTC treasury, so a risk-seeking region stays finite
    c_H: float = 0.0         # cost of hedging with short perps, c_H/2 * H^2; zero is fair

    # -- shocks ----------------------------------------------------------
    nu: float = 5.0          # Student-t degrees of freedom; math.inf is the normal case

    # -- jump losses (v1, design.md section 4.3) --------------------------
    jump_rate: float = 0.1   # jumps per year at zero GRC (Poisson; at most one a period)
    jump_mean: float = 0.05  # mean share of the book lost in a jump
    jump_tail: float = 2.5   # Pareto tail index of the severity; > 2 keeps the variance finite

    # -- GRC -------------------------------------------------------------
    delta: float = 0.25      # share of the GRC stock lost per year
    a_A: float = 50.0        # effectiveness against the mean loss, per unit of stock
    a_B: float = 50.0        # effectiveness against the spread
    rho_A: float = 0.3       # share of the mean loss no program removes
    rho_B: float = 0.3       # share of the spread no program removes
    a_J: float = 100.0       # effectiveness against jump frequency (prevention)
    rho_J: float = 0.3       # share of jump frequency no program removes

    # -- the vault form (design.md section 2) ------------------------------
    # Three strategies run by hired managers; the firm sets the allocation.
    # Order: BTC-backed lending, market-neutral (basis, market-making),
    # tokenized treasuries. Returns are excess over the riskless rate.
    vault_mu: tuple[float, float, float] = (0.06, 0.04, 0.0)      # expected excess return, per year
    vault_loss: tuple[float, float, float] = (0.01, 0.005, 0.0)   # mean loss rate GRC can reduce, per year
    vault_vol: tuple[float, float, float] = (0.05, 0.03, 0.0)     # volatility, per year
    vault_btc: tuple[float, float, float] = (0.1, 0.0, 0.0)       # sensitivity to BTC's return
    vault_jump: tuple[float, float, float] = (1.0, 0.5, 0.0)      # exposure to jump losses
    vault_rho: float = 0.3          # correlation of the two risky strategies
    capacity: float = 1.0           # the franchise's full size; fixes the vault's scale
    fee_mgmt: float = 0.02          # management fee, share of AUM per year
    fee_perf: float = 0.20          # performance fee, share of a period's positive excess return
    opex: float = 0.01              # operating costs, dollars per year
    inflow_speed: float = 0.5       # share of the gap to capacity closed per year
    outflow_perf: float = 1.0       # extra outflow per unit of negative excess return
    outflow_incident: float = 0.2   # share of AUM that leaves after an incident
    compensation: float = 0.5       # share of incident losses the firm makes good
    k_A: float = 0.02               # own funds required per unit of AUM
    co_invest: float = 0.0          # the firm's own stake in the vault, share of AUM ("skin in the game")
    guarantee: float | None = None  # minimum return promised to depositors, per year; None is no guarantee

    def __post_init__(self):
        if not (isinstance(self.periods_per_year, int) and self.periods_per_year >= 1):
            raise ValueError("periods_per_year must be a positive integer")
        if not self.A > 1.0 + self.r:
            raise ValueError("the book must beat cash at the margin: A > 1 + r")
        if not self.nu > 2.0:
            raise ValueError("nu must exceed 2 for covariances to exist")
        if not self.r_E >= self.r:
            raise ValueError("shareholders cannot require less than cash earns: r_E >= r")
        if self.jump_rate > 0 and not self.jump_tail > 2.0:
            raise ValueError("jump_tail must exceed 2 for the jump variance to exist")
        if not (0.0 <= self.fee_perf < 1.0 and 0.0 <= self.compensation <= 1.0):
            raise ValueError("fee_perf must be in [0, 1) and compensation in [0, 1]")
        if not 0.0 <= self.k < 1.0:
            raise ValueError("capital ratio k must be in [0, 1)")

    def per_period(self) -> "Params":
        """The same bank, with every rate restated for one period (design.md 4.4).

        Returned with `periods_per_year = 1`, so applying it again changes
        nothing. The conventions:

        - interest rates and the required return compound: (1 + r)^(1/n) - 1;
        - the book's gross return compounds, A^(1/n), so b still puts the
          first-best book at 1;
        - flows that accrue with time divide by n: the mean loss rate, the
          product's and BTC's excess returns, the carrying cost tau;
        - volatilities divide by sqrt(n), the square-root-of-time rule.
          Independent periods' variances add up to the year's. Under fat
          tails this is a convention, not an identity: a fat-tailed year is
          not four fat-tailed quarters summed;
        - the jump rate divides by n (Poisson events per unit of time);
        - GRC depreciation keeps the same annual loss: 1 - (1 - delta)^(1/n).

        Sizes and per-event amounts do not convert: issuance costs, the
        capital ratio, the deposit base, jump severity, GRC effectiveness and
        floors, position bounds, correlations, nu. For the vault: the
        management fee, operating costs and strategies' returns and losses are
        flows (divide by n); strategy volatilities follow sqrt(n); the inflow
        speed keeps its annual effect; the guaranteed return compounds; the
        performance fee, flow sensitivities, compensation share, capacity,
        capital requirement and co-investment share are shares or sizes and do
        not convert.
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
            r_D=compound(self.r_D),
            A=self.A ** (1.0 / n),
            tau=self.tau / n,
            m=self.m / n,
            pi=self.pi / n,
            s_P=self.s_P / root,
            s_N=self.s_N / root,
            s_BTC=self.s_BTC / root,
            mu_BTC=self.mu_BTC / n,
            jump_rate=self.jump_rate / n,
            delta=1.0 - (1.0 - self.delta) ** (1.0 / n),
            vault_mu=tuple(x / n for x in self.vault_mu),
            vault_loss=tuple(x / n for x in self.vault_loss),
            vault_vol=tuple(x / root for x in self.vault_vol),
            fee_mgmt=self.fee_mgmt / n,
            opex=self.opex / n,
            inflow_speed=1.0 - (1.0 - self.inflow_speed) ** (1.0 / n),
            guarantee=None if self.guarantee is None else compound(self.guarantee),
        )

    @property
    def b(self) -> float:
        """Curvature of the return on the book, fixed by the units: I* = 1.

        Per period: read it off `per_period()`, not off an annual statement.
        """
        return self.A - (1.0 + self.r)

    @property
    def beta(self) -> float:
        """Discount factor per period. Read it off `per_period()`."""
        return 1.0 / (1.0 + self.r_E)

    @property
    def normal(self) -> bool:
        return math.isinf(self.nu)


DEFAULTS = Params()
