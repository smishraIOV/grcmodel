"""Primitives every business form shares (design.md sections 3 and 4).

Element-wise and branch-free, so they vectorize over grid points, decisions and
quadrature nodes, and stay differentiable for the learner. Each takes the
parameters it needs from any object carrying them.
"""

import torch


def issuance_cost(p, e: torch.Tensor) -> torch.Tensor:
    """Deadweight cost of raising e of new equity: C(e) = lam e + kappa/2 e^2."""
    return p.lam * e + 0.5 * p.kappa * e * e


def issuance_cost_after(p, e: torch.Tensor, btc_fall: torch.Tensor, incident: torch.Tensor) -> torch.Tensor:
    """The cost of raising e straight after a period in which BTC fell by the
    share `btc_fall` (zero or less if it rose) and there was an incident or
    not (`incident`, 1 or 0): C(e) (1 + a fall^+ + b incident).

    Investors are hardest to find when the firm most needs them
    (design.md section 4.2). The solvers do not see this: they raise at C(e).
    """
    return issuance_cost(p, e) * (1.0 + p.issue_btc_fall * torch.relu(btc_fall) + p.issue_incident * incident)


def shareholder_flow(p, w: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """What shareholders receive this period for moving equity from w to K.

    K < w is a dividend of w - K. K > w is an issue of e = K - w, which costs
    shareholders e + C(e). One expression covers both: w - K - C((K - w)^+).
    """
    return w - K - issuance_cost(p, torch.relu(K - w))


def mitigation(S: torch.Tensor, a: float, rho: float) -> torch.Tensor:
    """Share of an exposure left after a GRC stock S: rho + (1 - rho) exp(-a S).

    `rho` is the irreducible floor no program removes (design.md section 4.2).
    """
    return rho + (1.0 - rho) * torch.exp(-a * S)


def next_grc(p, S: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """S' = (1 - delta) S + g. Mitigation uses this post-spend stock."""
    return (1.0 - p.delta) * S + g


def jump_probability(p, S_next: torch.Tensor) -> torch.Tensor:
    """Chance of a jump this period: 1 - exp(-rate phi_J(S')) (design.md 4.3).

    GRC acts on frequency -- prevention -- with the same irreducible floor as
    the other mechanisms. The probability enters the quadrature *weights*, so
    it is exactly differentiable in the spend; no sampling device is needed.
    """
    return 1.0 - torch.exp(-p.jump_rate * mitigation(S_next, p.a_J, p.rho_J))
