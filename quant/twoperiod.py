"""The two-period truncation: the paper's own structure, as an oracle.

An oracle here is a known-answer check: a case where the paper's formulas give
the exact result, so the numerical code can be tested against them
(design.md section 5). It checks the shared machinery -- quadrature, the
paper's G, Stein's lemma -- on the paper's own bank, which it carries itself.

With one period left, the bank faces the Froot-Scharfstein-Stein problem of
Froot & Stein (1998) eq. (2): a riskless opportunity F(I), funded from internal
wealth plus costly external equity. Its value P(w) is concave, and that is what
makes the bank care about the risk it takes in the period before.

The first period then picks the new product's size alpha and the hedge H
against E[P(w')]. Under normal shocks, Stein's lemma turns the first-order
conditions into the paper's results exactly:

    Prop. 2   H* = b_BTC I                     -- tradeable risk is fully hedged
    eq. (9)   alpha* = (pi - G cov_NP) / (G var_N),   G = -E[P''] / E[P']
    eq. (15)  x* = mu_BTC / (G s_BTC^2)        -- with a view on BTC, the bank
                                                  keeps this much net exposure

Prop. 2 is the mu_BTC = 0 case of the third line: with no view, it hedges
everything.

`tests/test_twoperiod.py` checks both against a numerical optimum that knows
nothing about either formula. Under the Student-t, Stein's lemma fails, and the
gap between eq. (9) and the true optimum is a result rather than an error.

**Deliberately not the full model.** The last period here is riskless, carries
no GRC and no product, and has no walk-away floor unless asked for: the paper's
P(w) has no default. The oracle configuration keeps the first period's outcomes
inside the region where the bank issues equity, so P is exactly quadratic there
and Gauss-Hermite integrates it without error.
"""

import math
from dataclasses import dataclass, replace

import numpy as np
import torch
from scipy.optimize import minimize

from quant.core import issuance_cost, mitigation, next_grc
from quant.numerics import NumericsProfile
from quant.quadrature import ShockRule

GOLDEN = (np.sqrt(5.0) - 1.0) / 2.0


# -- the paper's bank -------------------------------------------------------
#
# Self-contained: the main model no longer has a bank form (design.md section
# 11), but the paper's results are stated for one, so the oracle keeps its own.


@dataclass(frozen=True)
class PaperParams:
    """The paper's bank, one period per date. Defaults are the oracle's.

    These numbers exist to make the oracle exact, not to be plausible, and must
    not be recalibrated with the model.
    """

    r: float = 0.02          # riskless rate; cash earns it
    r_E: float = 0.02        # discounting at r: the paper's, beta (1 + r) = 1
    A: float = 1.08          # gross return on the first unit lent, F'(0)
    lam: float = 0.05        # proportional cost of issuing equity
    kappa: float = 1.0       # convex cost of issuing equity
    tau: float = 0.01        # deadweight cost of carrying equity
    D: float = 0.95          # deposit base
    r_D: float = 0.015       # rate paid on deposits
    k: float = 0.08          # minimum capital ratio: K >= k I
    m: float = 0.1           # mean loss rate on the book at zero GRC
    s_P: float = 0.005       # standard deviation of the book's return per unit lent
    pi: float = 0.0002       # expected excess payoff per unit of the new product
    s_N: float = 0.03        # its standard deviation per unit
    rho_NP: float = 0.5      # correlation of its payoff with the book's
    alpha_max: float = 1.0   # bound on its size
    b_BTC: float = 0.5       # the book's loading on BTC
    s_BTC: float = 0.10      # BTC's volatility
    mu_BTC: float = 0.0      # the bank's view: BTC's excess return; 0 makes Prop. 2 the check
    k_BTC: float = 0.10      # capital charge on the BTC treasury
    c_H: float = 0.0         # cost of hedging, c_H/2 H^2; zero is fair
    nu: float = math.inf     # normal shocks: where the paper's formulas are exact
    delta: float = 0.25      # GRC depreciation
    a_A: float = 50.0        # GRC effectiveness against the mean loss
    a_B: float = 50.0        # and against the spread
    rho_A: float = 0.3       # irreducible floors
    rho_B: float = 0.3

    @property
    def b(self) -> float:
        """Curvature of the return on the book, fixed so the first-best book is 1."""
        return self.A - (1.0 + self.r)

    @property
    def beta(self) -> float:
        return 1.0 / (1.0 + self.r_E)


def shock_correlation(p: PaperParams) -> np.ndarray:
    """Shock order (xi_P, xi_N, eps_BTC). BTC, the tradeable factor, is
    uncorrelated with the non-tradeable shocks: that is what "tradeable" means
    in Froot-Stein."""
    return np.array(
        [
            [1.0, p.rho_NP, 0.0],
            [p.rho_NP, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def production(p: PaperParams, I: torch.Tensor) -> torch.Tensor:
    """Gross return on a book of I, principal included: F(I) = A I - b/2 I^2."""
    return p.A * I - 0.5 * p.b * I * I


def capital_floor(
    p: PaperParams, I: torch.Tensor, g: torch.Tensor, q: torch.Tensor | None = None
) -> torch.Tensor:
    """The least equity the bank may hold: the capital ratio K >= k I + k_BTC q,
    non-negative cash K - g + D - I - q >= 0, and K >= 0."""
    if q is None:
        q = torch.zeros_like(I)
    zero = torch.zeros_like(I)
    ratio = p.k * I + p.k_BTC * q
    cash = I + g + q - p.D
    return torch.maximum(torch.maximum(ratio, cash), zero)


@dataclass(frozen=True)
class Controls:
    """K: equity. g: GRC spend. I: the book. alpha: the new product.
    H: BTC hedged with short perps. q: the BTC treasury."""

    K: torch.Tensor
    g: torch.Tensor
    I: torch.Tensor
    alpha: torch.Tensor
    H: torch.Tensor
    q: torch.Tensor | None = None


def next_wealth(
    p: PaperParams, c: Controls, S_next: torch.Tensor, shocks: torch.Tensor
) -> torch.Tensor:
    """Equity at the close of the period: assets less what depositors are owed.

        w' = F(I) - I phi_A m + I phi_B s_P xi_P       the book
           + (1 + r)(K - g + D - I)                    cash, and the treasury's riskless part
           + alpha (pi + s_N xi_N)                     the new product
           + x (mu_BTC + s_BTC eps_BTC) - c_H/2 H^2    BTC: net exposure x = b_BTC I + q - H
           - tau K - (1 + r_D) D
    """
    xi_P, xi_N, eps_BTC = shocks[:, 0], shocks[:, 1], shocks[:, 2]

    def col(x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(-1)

    q = c.q if c.q is not None else torch.zeros_like(c.I)
    I, K, g, alpha, H = col(c.I), col(c.K), col(c.g), col(c.alpha), col(c.H)
    x = col(p.b_BTC * c.I + q - c.H)
    S_next = col(S_next)
    phi_A = mitigation(S_next, p.a_A, p.rho_A)
    phi_B = mitigation(S_next, p.a_B, p.rho_B)

    book = production(p, I) - I * phi_A * p.m + I * phi_B * p.s_P * xi_P
    cash = (1.0 + p.r) * (K - g + p.D - I)
    product = alpha * (p.pi + p.s_N * xi_N)
    btc = x * (p.mu_BTC + p.s_BTC * eps_BTC) - 0.5 * p.c_H * H * H
    return book + cash + product + btc - p.tau * K - (1.0 + p.r_D) * p.D



def final_objective(p: PaperParams, w: torch.Tensor, I: torch.Tensor) -> torch.Tensor:
    """Value of the last operating period at book I, with K at its floor.

    The bank moves equity from w to K, lends I, earns F(I) and the return on
    cash, repays depositors and liquidates one period later:

        J = w - L - C((L - w)^+) + beta [F(I) + (1 + r)(L + D - I) - tau L - (1 + r_D) D]

    K sits at its floor L(I): a unit of equity held costs 1 now and returns
    beta (1 + r - tau) <= 1 later, so nothing above the floor is held. J is
    concave in I, because L is convex and C is convex and increasing.

    Written for any r_E. Under the paper's discounting, beta (1 + r) = 1, it
    reduces to w - C - beta tau L + beta F(I) - I + D (1 - beta (1 + r_D)),
    which is the form `issuance_threshold` is derived from.
    """
    beta = p.beta
    L = capital_floor(p, I, torch.zeros_like(I))
    terminal = (
        production(p, I)
        + (1.0 + p.r) * (L + p.D - I)
        - p.tau * L
        - (1.0 + p.r_D) * p.D
    )
    return w - L - issuance_cost(p, torch.relu(L - w)) + beta * terminal


def final_value(
    p: PaperParams, w: torch.Tensor, walk_away: bool = False, iterations: int = 120
) -> torch.Tensor:
    """P(w): the last period's value, maximized over I in [0, 1] by golden section.

    The first-best book is 1 by the choice of units, and every friction only
    lowers it, so [0, 1] brackets the optimum. 120 iterations shrink the bracket
    by 0.618^120, about 1e-25, so the value is exact to rounding: the objective
    is flat at its maximum and the error is quadratic in the bracket.
    """
    lo = torch.zeros_like(w)
    hi = torch.ones_like(w)
    a = hi - GOLDEN * (hi - lo)
    b = lo + GOLDEN * (hi - lo)
    fa, fb = final_objective(p, w, a), final_objective(p, w, b)
    for _ in range(iterations):
        left = fa > fb
        hi = torch.where(left, b, hi)
        lo = torch.where(left, lo, a)
        a_new = hi - GOLDEN * (hi - lo)
        b_new = lo + GOLDEN * (hi - lo)
        a, b = a_new, b_new
        fa, fb = final_objective(p, w, a), final_objective(p, w, b)
    value = final_objective(p, w, 0.5 * (lo + hi))
    return torch.clamp(value, min=0.0) if walk_away else value


def final_marginals(
    p: PaperParams, w: torch.Tensor, h1: float = 1e-4, h2: float = 1e-3
) -> tuple[torch.Tensor, torch.Tensor]:
    """P'(w) and P''(w) by central differences.

    Central differences are exact for a quadratic at any step, so the steps are
    chosen large enough that rounding (1e-16 / h) stays negligible: at 1e-6
    the second difference carried 1e-7 of noise. Exact up to rounding in the
    region the oracle runs in. Near a kink -- where the capital floor switches constraint, or where
    issuance starts -- P'' has a jump and these are local averages.
    """
    def d1(x):
        return (final_value(p, x + h1) - final_value(p, x - h1)) / (2.0 * h1)

    return d1(w), (d1(w + h2) - d1(w - h2)) / (2.0 * h2)


def issuance_threshold(p: PaperParams) -> float:
    """The wealth below which the last period raises equity, when K = k I binds.

    From the first-order condition in I with e = k I - w > 0, I is linear in w:

        I(w) = (c + k kappa w) / (beta b + k^2 kappa),   c = beta A - 1 - beta tau k - k lam

    and issuance happens while k I(w) > w, i.e. below w = k c / (beta b).
    Valid when the capital ratio, not the cash constraint, sets the floor:
    D >= (1 - k) I.

    Derived under the paper's discounting, beta (1 + r) = 1, which the oracle
    configuration pins.
    """
    if abs(p.beta * (1.0 + p.r) - 1.0) > 1e-12:
        raise ValueError("issuance_threshold assumes beta (1 + r) = 1, i.e. r_E = r")
    beta, k = p.beta, p.k
    c = beta * p.A - 1.0 - beta * p.tau * k - k * p.lam
    den = beta * p.b + k * k * p.kappa
    return k * c / (den - k * k * p.kappa)


@dataclass(frozen=True)
class FirstPeriod:
    """The first period's fixed decisions, around which alpha and H are chosen."""

    K: float
    I: float
    g: float = 0.0
    G: float = 0.0


def first_period_wealth(
    p: PaperParams,
    rule: ShockRule,
    profile: NumericsProfile,
    fixed: FirstPeriod,
    alpha: float,
    H: float,
) -> torch.Tensor:
    """w' at every quadrature node, for a given alpha and H."""
    c = Controls(
        K=profile.tensor(fixed.K),
        g=profile.tensor(fixed.g),
        I=profile.tensor(fixed.I),
        alpha=profile.tensor(alpha),
        H=profile.tensor(H),
    )
    G_next = next_grc(p, profile.tensor(fixed.G), c.g)
    return next_wealth(p, c, G_next, rule.nodes)


def optimal_positions(
    p: PaperParams, rule: ShockRule, profile: NumericsProfile, fixed: FirstPeriod
) -> tuple[float, float]:
    """(alpha*, H*) maximizing E[P(w')], found numerically.

    Knows nothing about eq. (9) or Prop. 2. E[P(w')] is concave in (alpha, H)
    because P is concave and w' is linear in both, so L-BFGS-B from any start
    finds the one optimum.
    """
    def negative_value(x):
        w_next = first_period_wealth(p, rule, profile, fixed, x[0], x[1])
        return -float(rule.expect(final_value(p, w_next)))

    result = minimize(
        negative_value,
        x0=np.array([0.1, 0.0]),
        method="L-BFGS-B",
        bounds=[(0.0, p.alpha_max), (None, None)],
        options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 500},
    )
    return float(result.x[0]), float(result.x[1])


@dataclass(frozen=True)
class FrootStein:
    """The paper's quantities, evaluated at a first-period decision."""

    G: float          # effective risk aversion, -E[P''] / E[P']
    alpha: float      # eq. (9): alpha* = (pi - G cov_NP) / (G var_N)
    hurdle: float     # eq. (10): mu* = G cov_NP, the hurdle for a small position
    btc: float        # eq. (15) analogue: net BTC exposure x* = mu_BTC / (G s_BTC^2)


def froot_stein(
    p: PaperParams,
    rule: ShockRule,
    profile: NumericsProfile,
    fixed: FirstPeriod,
    alpha: float,
    H: float,
) -> FrootStein:
    """G, and what eqs. (9) and (10) say alpha and the hurdle should be.

    `cov_NP` is the covariance of a unit of the product with the existing
    non-tradeable book: s_N times the book's payoff volatility, I phi_B s_P,
    times their correlation.
    """
    w_next = first_period_wealth(p, rule, profile, fixed, alpha, H)
    d1, d2 = final_marginals(p, w_next)
    G = -float(rule.expect(d2)) / float(rule.expect(d1))
    G_next = (1.0 - p.delta) * fixed.G + fixed.g
    phi_B = float(mitigation(profile.tensor(G_next), p.a_B, p.rho_B))
    cov_NP = p.s_N * fixed.I * phi_B * p.s_P * p.rho_NP
    var_N = p.s_N**2
    btc = p.mu_BTC / (G * p.s_BTC**2) if p.s_BTC > 0 else float("nan")
    return FrootStein(
        G=G, alpha=(p.pi - G * cov_NP) / (G * var_N), hurdle=G * cov_NP, btc=btc
    )


def oracle_params(**overrides) -> PaperParams:
    """A configuration in which the two-period oracle is exact.

    D >= 1 - k makes the capital ratio, not the cash constraint, the binding
    floor at every book up to the first-best, so the floor has no kink in
    range. A heavy mean loss in the first period (m = 0.1) leaves the bank
    thinly capitalized, so with small shocks every quadrature node lands below
    the issuance threshold. P is then one quadratic over every node and
    Gauss-Hermite integrates it exactly. `pi` is set so alpha* is interior.
    """
    return replace(PaperParams(), **overrides)


ORACLE_FIRST_PERIOD = FirstPeriod(K=0.08, I=1.0)
