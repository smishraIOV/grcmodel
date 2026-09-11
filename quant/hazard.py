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

**Three competing risks, composed additively.** Intensities add, survival
probabilities multiply -- which is why this module returns intensities and
lets the caller accumulate their sum in log space:

    capital       rises as equity falls. GRC reaches it only indirectly, by
                  leaving more equity behind.
    operational   an incident becomes public and depositors leave. Operational
                  GRC reduces it directly.
    licence       a breach escalates to revocation. Compliance GRC reduces it
                  directly.

That split is what makes GRC buy *survival* rather than only buying smaller
losses, and it is the difference between a programme justified by expected-loss
reduction and one justified by the franchise it protects. Credit has no hazard
channel of its own on purpose: bad underwriting erodes equity, and equity is
already the capital channel's argument. Adding a fourth channel for it would be
counting the same mechanism twice.
"""

import torch

from quant.frictions import exponential_mitigation
from quant.params import GrcAlphas, HazardParams

OPERATIONAL, COMPLIANCE = 1, 2  # columns of the GRC stock


def capital_ratio(equity: torch.Tensor, reference_equity: float) -> torch.Tensor:
    """Equity as a fraction of the firm's opening capital. Dimensionless, so
    the model's answer does not depend on the currency it is denominated in."""
    return equity / reference_equity


def capital_intensity(
    equity: torch.Tensor,
    reference_equity: float,
    params: HazardParams,
    periods_per_year: int,
) -> torch.Tensor:
    """Per-period hazard from being thinly capitalized.

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


def grc_reduced_intensity(
    annual_rate: float, stock: torch.Tensor, alpha: float, periods_per_year: int
) -> torch.Tensor:
    """Per-period intensity of a hazard a GRC stock acts on directly.

        h = (rate / periods_per_year) * exp(-alpha * G)

    Same mitigation curve and the same alpha the loss channels use. Giving each
    channel its own effectiveness parameter would be more general and less
    honest: nothing in this repo can calibrate one alpha, let alone two per
    family (docs/quant-model.md section 8).

    alpha carries units of 1/money and G is money, so the exponent is
    dimensionless.
    """
    return exponential_mitigation(
        torch.as_tensor(annual_rate / periods_per_year, dtype=stock.dtype, device=stock.device),
        stock,
        alpha,
    )


def intensity_components(
    equity: torch.Tensor,
    grc_stock: torch.Tensor,
    reference_equity: float,
    params: HazardParams,
    alphas: GrcAlphas,
    periods_per_year: int,
) -> dict[str, torch.Tensor]:
    """The three channels separately, for diagnostics and for the tests.

    Reported apart because "slow death by attrition" and "a licence was pulled"
    are different failures with different remedies, and a single number cannot
    tell a risk owner which one is binding.
    """
    return {
        "capital": capital_intensity(equity, reference_equity, params, periods_per_year),
        "operational": grc_reduced_intensity(
            params.annual_operational_rate,
            grc_stock[..., OPERATIONAL],
            alphas.operational,
            periods_per_year,
        ),
        "licence": grc_reduced_intensity(
            params.annual_licence_rate,
            grc_stock[..., COMPLIANCE],
            alphas.compliance,
            periods_per_year,
        ),
    }


def failure_intensity(
    equity: torch.Tensor,
    grc_stock: torch.Tensor,
    reference_equity: float,
    params: HazardParams,
    alphas: GrcAlphas,
    periods_per_year: int,
) -> torch.Tensor:
    """Total per-period hazard. Competing risks add in intensity."""
    return sum(
        intensity_components(
            equity, grc_stock, reference_equity, params, alphas, periods_per_year
        ).values()
    )


def log_survival(intensity: torch.Tensor) -> torch.Tensor:
    """log P(survive the period) = -h. Kept in logs so it can be summed."""
    return -intensity
