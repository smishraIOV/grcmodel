"""Failure intensity: the channel through which the firm can actually die.

Replaces the hard insolvency barrier as the binding survival channel, for two
reasons that arrived independently.

**Economic.** A regulated crypto intermediary does not fail by crossing an
accounting threshold on a particular Tuesday. It fails because a loss becomes
public and depositors leave, or because a regulator withdraws a licence. Those
are hazards whose intensity rises as the firm weakens, not a cliff at book
equity zero.

**Numerical.** A hard barrier plus limited liability is not differentiable in
any useful sense: a failed firm is worth a constant, so a dead path's gradient
is exactly zero and carries no information about how death might have been
avoided. A hazard restores it -- the gradient flows through the *probability*
of crossing rather than dying at the crossing. Measured on this model, with
opening equity 16:

    equity      16      8      4      1     0.1    -2      -8
    kappa      1.00   0.50   0.25   0.06   0.01  -0.12   -0.50
    dS/dE     0.0019 0.0220 0.0635 0.1074 0.1143 0.0960  0.000046

against exactly zero at every one of those points under a hard barrier.

What this does *not* fix, and an earlier commit message in this repo claimed it
would: a cold start from a badly wrong initialization. Survival still
underflows to exactly zero once a path runs far enough past the barrier --
below roughly kappa = -0.5 here -- and the gradient goes with it. A
perfect-information solver started at softplus(0) drives paths there in the
first few quarters and converges below a plain constant policy under either
death channel. Warm-starting remains necessary; the hazard buys gradient in the
region an already-sensible policy actually operates in, not everywhere.

Everything here is a rate per unit time, and every argument to an exponential
is dimensionless. Survival is accumulated in log space by the caller, which
matters once horizons get long enough for a product of per-period
probabilities to underflow float32.
"""

import torch

from quant.params import HazardParams


def capital_ratio(equity: torch.Tensor, reference_equity: float) -> torch.Tensor:
    """Equity as a fraction of the firm's opening capital. Dimensionless, so
    the model's answer does not depend on the currency it is denominated in."""
    return equity / reference_equity


def failure_intensity(
    equity: torch.Tensor,
    reference_equity: float,
    params: HazardParams,
    periods_per_year: int,
) -> torch.Tensor:
    """Per-period hazard rate, exponential in the capital shortfall.

        h = (base / periods_per_year) * exp((target - kappa) / scale)

    Exponential rather than logistic, and the difference matters. A logistic
    saturates: at its base rate even a firm with deeply negative equity would
    survive the quarter with high probability, which is not a description of
    insolvency. An unbounded intensity lets survival go to zero continuously as
    equity does, so certain death is represented as a limit rather than as a
    cliff -- which is the entire point of replacing the barrier.

    The exponent is a ratio over a dimensionless scale, so it stays
    unit-invariant; that was the objection to a power of a money quantity
    (docs/static-model-debug-notes.md section 4), not to exponentials as such.

    Falling equity raises the intensity smoothly. That dependence is the whole
    point and is asserted in the tests: a hazard that does *not* respond to
    state leaves firm value linear in equity, which removes the Froot-Stein
    content entirely and turns the hazard into a discount-rate adjustment. It
    is the dynamic form of the deterministic-loss trap in
    docs/static-model-debug-notes.md section 2, and just as silent -- the model
    runs, converges, and means nothing.
    """
    kappa = capital_ratio(equity, reference_equity)
    # Clamped so a catastrophically negative path gives an intensity that is
    # merely enormous rather than inf. exp(-1e20) underflows to exactly zero
    # survival, which is what we want; inf would produce NaN when multiplied by
    # a frozen equity.
    exponent = torch.clamp((params.capital_target - kappa) / params.capital_scale, max=50.0)
    return (params.annual_base_rate / periods_per_year) * torch.exp(exponent)


def log_survival(intensity: torch.Tensor) -> torch.Tensor:
    """log P(survive the period) = -h. Kept in logs so it can be summed."""
    return -intensity
