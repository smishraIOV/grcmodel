"""Model parameters, as plain Python floats.

Separated from the tensor code on purpose. Parameters are scalars that belong
to the economics; tensors belong to a NumericsProfile (quant/numerics.py).
Keeping them apart is what lets a profile own *every* tensor construction,
which is how the silent-float32 bug is prevented rather than merely fixed.

Nothing here may hold a torch.Tensor, and nothing here may import torch.

Every value is illustrative, not calibrated — which is why quant/threshold.py
reports the effectiveness a programme must reach rather than a spend
recommendation (docs/quant-model.md section 8).
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GrcAlphas:
    """Mitigation effectiveness per risk family: how much of the family's
    exposure one unit of GRC spend removes.

    These are the parameters nothing in this repo can currently calibrate,
    which is why quant/threshold.py reports break-even values for them rather
    than trusting a point estimate.
    """

    credit: float = 0.3
    operational: float = 0.3
    compliance: float = 0.3


@dataclass(frozen=True)
class FirmParams:
    """Balance-sheet and technology parameters shared by every stage.

    These lived in quant/static.py, which meant the Monte Carlo stage imported
    them from the four-state model it was meant to be independent of.

    A caveat that matters now that there is a horizon: the loss and production
    parameters are *per period*, not annual rates. Only the discount and the
    GRC depreciation below are expressed annually and converted. Until the
    rest are rate-based, changing `periods_per_year` does not rescale the
    economics and the model cannot be checked for time-scale invariance --
    the same class of silent bug as the dimensional one in
    docs/static-model-debug-notes.md section 4, one axis over.
    """

    initial_equity: float = 16.0
    financing_convexity: float = 2.0
    distress_reference: float = 5.0
    production_scale: float = 3.0
    production_curvature: float = 10.0

    periods_per_year: int = 4  # quarterly
    annual_discount_rate: float = 0.08
    # A control installed today still works in a year's time, but not forever:
    # staff turn over, systems age, regulations move. Illustrative, like
    # everything else here.
    annual_grc_depreciation: float = 0.25

    def discount(self) -> float:
        """Per-period discount factor."""
        return (1.0 + self.annual_discount_rate) ** (-1.0 / self.periods_per_year)

    def grc_depreciation(self) -> float:
        """Per-period fraction of the GRC stock that decays away."""
        return 1.0 - (1.0 - self.annual_grc_depreciation) ** (1.0 / self.periods_per_year)


@dataclass(frozen=True)
class ShockParams:
    """The four-state shock in quant/static.py.

    `credit_spread` widens the gap between the good and bad credit outcome
    while holding the mean at `credit_mean`, so sweeping it is a
    mean-preserving spread. Compliance failure is licence-to-operate risk
    (docs/framework.md section 1): rare, and severe enough to dwarf a bad
    credit year when it lands.
    """

    credit_mean: float = 4.0
    credit_spread: float = 2.0
    op_severity: float = 7.0
    compliance_probability: float = 0.05
    compliance_severity: float = 30.0


@dataclass(frozen=True)
class SamplerParams:
    """Monte Carlo distribution parameters (quant/simulate.py).

    Chosen to match the expected loss of the four-state shock above
    (4.0 credit + 3.5 operational + 1.5 compliance) so the two stages are
    comparable. Credit is a continuous loss; operational incidents are rare
    but heavy-tailed when they land; a compliance breach is rarer still and
    carries a fixed severity, because losing a licence costs what it costs.
    """

    credit_loss_mean: float = 4.0
    op_probability: float = 0.25
    op_severity_mean: float = 14.0
    compliance_probability: float = 0.05
    compliance_severity: float = 30.0


@dataclass(frozen=True)
class ModelParams:
    """Everything the static and Monte Carlo stages need, in one place."""

    firm: FirmParams = field(default_factory=FirmParams)
    alphas: GrcAlphas = field(default_factory=GrcAlphas)
    shock: ShockParams = field(default_factory=ShockParams)
    sampler: SamplerParams = field(default_factory=SamplerParams)


DEFAULTS = ModelParams()
SPREAD_SWEEP = [0.0, 0.5, 1.0, 1.5, 2.0]
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
