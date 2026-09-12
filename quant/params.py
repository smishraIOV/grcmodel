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

    # GRC capital the firm already has, per family, at the start of a run.
    # Zero describes a firm building a control function from nothing, which is
    # a real case -- it is close to the pre-revenue variant of the maturity
    # fork in docs/quant-model.md section 5 -- but it is not the mature going
    # concern the current stages are about. Starting from zero left the firm
    # facing un-mitigated base hazards for several quarters while it built a
    # stock, which put the annual failure probability at 27.5%, outside the
    # band where survival comparative statics mean anything.
    #
    # The value is not tuned to taste: it is the self-consistent steady state,
    # the level at which the firm's own optimal maintenance spend exactly
    # replaces depreciation, so G_0 = g*(G_0) / delta. Found by bisection at
    # 5.18; a firm starting here neither builds nor runs down its control
    # function, which is what "mature going concern" should mean. It puts the
    # annual failure probability at 7.2%, inside the band, with a material
    # 1.07 per quarter of spend.
    #
    # Both neighbouring regimes are degenerate and it is worth knowing why.
    # Below about 3 the firm is rebuilding from a hole and dies from tail
    # events while it does. Above about 9 it has inherited so much capital that
    # optimal spend collapses to ~0.002 per quarter and the model has nothing
    # to say about budgets at all -- a "never binds" regime of the kind
    # docs/static-model-debug-notes.md section 6 warns about, in a new place.
    initial_grc_stock: float = 5.18

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
class HazardParams:
    """Intensity of firm failure, as a function of how well capitalized it is.

    A logistic in the capital ratio rather than a power law. Every argument to
    the exponential must be dimensionless, and a ratio over a dimensionless
    scale is that by construction; a power of a money quantity is how the
    financing cost acquired its units bug (docs/static-model-debug-notes.md
    section 4).

    The base rate is the intensity a maximally distressed firm faces, not the
    one a healthy firm faces. At `capital_target` the hazard is half of it.
    Illustrative, like everything else here -- but unlike alpha, these have
    external anchors that could be used: bank failure rates by capital ratio,
    rating-agency default rates, observed crypto-lender failures.
    """

    annual_base_rate: float = 0.5
    capital_target: float = 0.4   # ratio of opening equity at which hazard equals its base
    capital_scale: float = 0.2    # how sharply hazard responds around that point

    # Two further channels, each reducible by the GRC stock of its family.
    # Competing risks compose additively in intensity, so these sum with the
    # capital channel above rather than multiplying.
    #
    # Operational: an incident becomes public and depositors leave. This is how
    # a crypto intermediary actually fails -- a bridge exploit or a custody
    # failure, not an accounting threshold crossed on a particular Tuesday.
    # Licence: a compliance breach escalates to revocation, which ends the firm
    # regardless of its balance sheet.
    #
    # Rates below are at ZERO GRC stock, so they are what the firm faces with no
    # programme at all, not what it faces in practice.
    annual_operational_rate: float = 0.15
    annual_licence_rate: float = 0.08


@dataclass(frozen=True)
class ModelParams:
    """Everything the static and Monte Carlo stages need, in one place."""

    firm: FirmParams = field(default_factory=FirmParams)
    alphas: GrcAlphas = field(default_factory=GrcAlphas)
    shock: ShockParams = field(default_factory=ShockParams)
    sampler: SamplerParams = field(default_factory=SamplerParams)
    hazard: HazardParams = field(default_factory=HazardParams)


DEFAULTS = ModelParams()
SPREAD_SWEEP = [0.0, 0.5, 1.0, 1.5, 2.0]
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
