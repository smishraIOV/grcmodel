"""Parameters of the v0 bank (docs/design.md section 3).

Plain floats, frozen. Every parameter is one economic driver; the comment on
each says which. All values are illustrative, not calibrated.

Units (design.md section 3.1): money is measured so that the first-best book --
what a bank with unlimited free capital would lend each period -- is exactly 1.
That pins the curvature of the return on the book, `b`, rather than leaving it
free: F'(1) must equal the return on cash, 1 + r.

One period is one year in v0.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Params:
    # -- rates and the opportunity ---------------------------------------
    r: float = 0.02          # riskless rate per period; cash earns it
    r_E: float = 0.06        # shareholders' required return; beta = 1/(1+r_E). Departure from the
                             # paper, design.md section 8: to be set against a target market-to-book
    A: float = 1.08          # gross return on the first unit lent, F'(0)

    # -- frictions: what makes the bank care about risk ------------------
    lam: float = 0.05        # proportional cost of issuing equity
    kappa: float = 1.0       # convex cost of issuing equity. A switch, not a calibration
    tau: float = 0.0         # extra deadweight cost of carrying equity; the main cost is r_E - r

    # -- liabilities and regulation --------------------------------------
    D: float = 0.9           # deposit base, fixed in v0, rolled over each period
    r_D: float = 0.015       # rate paid on deposits, senior to dividends
    k: float = 0.08          # minimum capital ratio: K >= k * I

    # -- the book's non-tradeable risk -----------------------------------
    m: float = 0.01          # mean loss rate on the book at zero GRC
    s_P: float = 0.02        # standard deviation of the book's return per unit lent

    # -- the new product (capital budgeting) -----------------------------
    pi: float = 0.01         # expected excess payoff per unit of the product
    s_N: float = 0.10        # its standard deviation per unit
    rho_NP: float = 0.5      # correlation of its payoff with the book's
    alpha_max: float = 1.0   # bound on its size, so a risk-seeking region stays finite

    # -- the tradeable factor (risk management) --------------------------
    b_M: float = 0.5         # the book's loading on the factor (crypto collateral prices)
    s_M: float = 0.10        # the factor's standard deviation per period
    c_H: float = 0.0         # hedging cost, c_H/2 * H^2. Zero: fairly tradeable, Prop. 2 applies
    x_max: float = 1.0       # bound on the unhedged factor exposure, |b_M I - H|, for the same reason

    # -- shocks ----------------------------------------------------------
    nu: float = 5.0          # Student-t degrees of freedom; math.inf is the normal case

    # -- jump losses (v1, design.md section 3.5) --------------------------
    jump_rate: float = 0.1   # jumps per period at zero GRC (Poisson rate; at most one counted)
    jump_mean: float = 0.05  # mean share of the book lost in a jump
    jump_tail: float = 2.5   # Pareto tail index of the severity; > 2 keeps the variance finite

    # -- GRC -------------------------------------------------------------
    delta: float = 0.25      # share of the GRC stock lost per period
    a_A: float = 50.0        # effectiveness against the mean loss, per unit of stock
    a_B: float = 50.0        # effectiveness against the spread
    rho_A: float = 0.3       # share of the mean loss no program removes
    rho_B: float = 0.3       # share of the spread no program removes
    a_J: float = 100.0       # effectiveness against jump frequency (prevention)
    rho_J: float = 0.3       # share of jump frequency no program removes

    def __post_init__(self):
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

    @property
    def b(self) -> float:
        """Curvature of the return on the book, fixed by the units: I* = 1."""
        return self.A - (1.0 + self.r)

    @property
    def beta(self) -> float:
        return 1.0 / (1.0 + self.r_E)

    @property
    def normal(self) -> bool:
        return math.isinf(self.nu)


DEFAULTS = Params()
