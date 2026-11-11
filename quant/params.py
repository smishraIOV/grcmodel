"""Parameters of the bank (docs/design.md section 3).

Plain floats, frozen. Every parameter is one economic driver; the comment on
each says which. All values are illustrative, not calibrated.

**Every rate is stated per year.** The model runs at `periods_per_year`
decisions a year, and `per_period()` converts the annual statement into what
one period sees (design.md section 3.6). Sizes and per-event amounts do not
convert. With `periods_per_year = 1` the conversion is the identity, so it is
safe to apply twice; the solver applies it once, on entry.

Use 1 for fast exploratory runs and tests, and 4 (the default, quarterly) for
realistic ones. Each extra period a year brings the discount factor closer to
one, so a solve takes more iterations to converge.

Units (design.md section 3.1): money is measured so that the first-best book --
what a bank with unlimited free capital would lend -- is exactly 1. That pins
the curvature of the return on the book, `b`, at every frequency: F'(1) must
equal the return on cash over one period.
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
                             # paper, design.md section 8: to be set against a target market-to-book
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

    # -- the tradeable factor (risk management) --------------------------
    b_M: float = 0.5         # the book's loading on the factor (crypto collateral prices)
    s_M: float = 0.10        # the factor's standard deviation, per year
    c_H: float = 0.0         # hedging cost, c_H/2 * H^2. Zero: fairly tradeable, Prop. 2 applies
    x_max: float = 1.0       # bound on the unhedged factor exposure, |b_M I - H|, for the same reason

    # -- shocks ----------------------------------------------------------
    nu: float = 5.0          # Student-t degrees of freedom; math.inf is the normal case

    # -- jump losses (v1, design.md section 3.5) --------------------------
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
        if not 0.0 <= self.k < 1.0:
            raise ValueError("capital ratio k must be in [0, 1)")

    def per_period(self) -> "Params":
        """The same bank, with every rate restated for one period (design.md 3.6).

        Returned with `periods_per_year = 1`, so applying it again changes
        nothing. The conventions:

        - interest rates and the required return compound: (1 + r)^(1/n) - 1;
        - the book's gross return compounds, A^(1/n), so b still puts the
          first-best book at 1;
        - flows that accrue with time divide by n: the mean loss rate, the
          product's excess return, the carrying cost tau;
        - volatilities divide by sqrt(n), the square-root-of-time rule.
          Independent periods' variances add up to the year's. Under fat
          tails this is a convention, not an identity: a fat-tailed year is
          not four fat-tailed quarters summed;
        - the jump rate divides by n (Poisson events per unit of time);
        - GRC depreciation keeps the same annual loss: 1 - (1 - delta)^(1/n).

        Sizes and per-event amounts do not convert: issuance costs, the
        capital ratio, the deposit base, jump severity, GRC effectiveness and
        floors, position bounds, correlations, nu.
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
            s_M=self.s_M / root,
            jump_rate=self.jump_rate / n,
            delta=1.0 - (1.0 - self.delta) ** (1.0 / n),
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
