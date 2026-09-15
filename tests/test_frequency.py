"""Does it still mean the same thing at a different decision frequency?

The model is meant to be invariant to how often the firm decides: rates convert,
per-event magnitudes do not, and the economics should be unchanged. That
invariance has been broken four separate times, always in the same way and
always silently -- a quantity stated per *period* compared against a threshold
stated per *quarter*, or a literal `4` standing in for `periods_per_year`.

None of those produced an error. Each produced a confident wrong number, which
is why they get their own file: the failure mode is not "the code raises", it is
"the study reports uneconomic for every row and nobody notices".
"""

import pytest
import torch

from quant.env.env import EnvConfig, FirmEnv
from quant.env.shocks import CommonRandomNumbers, MonteCarloSampler
from quant.numerics import REFERENCE
from quant.params import DEFAULTS, MONTHLY, model_at
from quant.solvers.gridvi import GridSolver, ReducedSpec
from quant.studies.breakeven import ANNUAL_MATERIALITY, is_material


class FakeResult:
    """Just enough of an EvalResult to ask about materiality."""

    def __init__(self, total_grc: float):
        self.total_grc = total_grc


def test_materiality_is_a_verdict_about_a_year_not_a_period():
    """The same annual programme must get the same verdict however often it is
    decided.

    Stated per period, the threshold silently retunes itself with the decision
    frequency: a programme costing 0.6 a year is material at quarterly (0.15 a
    period) and immaterial at monthly (0.05 a period), and every row of every
    break-even sweep flips to "uneconomic" with nothing to show for it.
    """
    annual = ANNUAL_MATERIALITY * 1.5          # comfortably material
    for ppy in (4, 12, 26, 52):
        firm = model_at(ppy).firm
        assert is_material(FakeResult(annual / ppy), firm), ppy

    trivial = ANNUAL_MATERIALITY * 0.5         # comfortably not
    for ppy in (4, 12, 26, 52):
        firm = model_at(ppy).firm
        assert not is_material(FakeResult(trivial / ppy), firm), ppy


def test_the_grid_draws_its_shocks_from_its_own_environment():
    """The grid solver built its shock from `DEFAULTS.sampler`, ignoring the
    environment it was handed.

    Harmless for as long as every caller was quarterly, and wrong by a factor of
    three the moment one was not: a monthly environment would have been given a
    quarter's losses every month while its discount, depreciation and hazards
    were all monthly. That is the mistake `standard_env` exists to prevent, and
    the seam was open here.
    """
    spec = ReducedSpec(n_equity=4, n_stock=3, n_spend=3, n_investment=3, n_shocks=64)
    monthly = FirmEnv(
        EnvConfig.monthly(3, profile=REFERENCE), MonteCarloSampler(MONTHLY.sampler)
    )
    quarterly = FirmEnv(
        EnvConfig.quarterly(3, profile=REFERENCE), MonteCarloSampler(DEFAULTS.sampler)
    )

    monthly_loss = GridSolver(monthly, spec).shock.credit_loss.mean().item()
    quarterly_loss = GridSolver(quarterly, spec).shock.credit_loss.mean().item()

    assert monthly_loss < quarterly_loss, "a month cannot lose as much as a quarter"
    assert monthly_loss == pytest.approx(quarterly_loss / 3.0, rel=0.05), (
        "the grid is not using the frequency of the environment it was given"
    )


def test_the_grids_spend_axis_follows_the_decision_frequency():
    """Investment is a stock and needs no conversion; spend is a flow and does.

    With the spend ceiling fixed in per-quarter units, a monthly firm's entire
    optimum falls inside the first grid cell -- and the module's own warning
    applies: a grid whose action range does not cover the optimum reports a
    confident answer to a different problem.
    """
    spec = ReducedSpec()
    quarterly_spend = max(s for s, _ in spec.actions(REFERENCE, 4))
    monthly_spend = max(s for s, _ in spec.actions(REFERENCE, 12))
    assert monthly_spend == pytest.approx(quarterly_spend / 3.0)

    quarterly_invest = max(i for _, i in spec.actions(REFERENCE, 4))
    monthly_invest = max(i for _, i in spec.actions(REFERENCE, 12))
    assert monthly_invest == quarterly_invest, "investment is a stock; it must not scale"


def test_the_scripts_annualization_takes_the_frequency_as_an_argument():
    """`annual_death` hardcoded a 4. Over sixty monthly periods that annualizes
    across fifteen years instead of five, and the wrong figure is printed in
    five separate tables."""
    from scripts.run_dynamic_model import annual_death

    # A firm that survives 90% of five years, described three ways. All three
    # must agree, because they are the same firm.
    survival = 0.90
    quarterly = annual_death(survival, periods=20, per_year=4)
    monthly = annual_death(survival, periods=60, per_year=12)
    weekly = annual_death(survival, periods=260, per_year=52)

    assert quarterly == pytest.approx(monthly, rel=1e-12)
    assert quarterly == pytest.approx(weekly, rel=1e-12)
    assert quarterly == pytest.approx(1.0 - survival ** 0.2, rel=1e-12)


def test_a_year_of_economics_is_the_same_at_any_frequency():
    """The end-to-end property the three tests above are pieces of.

    One year of losses, discounting and GRC decay must come to the same thing
    whether the firm decides four times or fifty-two. This is the check that
    would have caught every historical instance at once.
    """
    for ppy in (4, 12, 26, 52):
        params = model_at(ppy)
        firm = params.firm

        # Discounting and depreciation compound back to their annual figures.
        assert firm.discount() ** ppy == pytest.approx(
            1.0 / (1.0 + firm.annual_discount_rate), rel=1e-12
        )
        assert (1.0 - firm.grc_depreciation()) ** ppy == pytest.approx(
            1.0 - firm.annual_grc_depreciation, rel=1e-12
        )
        # Expected loss per year is frequency-free: rates divide down, and
        # per-event severities are untouched.
        assert params.sampler.credit_loss_mean * ppy == pytest.approx(0.034, rel=1e-12)
        assert params.sampler.op_probability * ppy == pytest.approx(1.0, rel=1e-12)
        assert params.sampler.op_severity_mean == pytest.approx(0.70, rel=1e-12)
        # The unconstrained optimum I* = S ln A is a stock and must not move.
        istar = firm.production_curvature * torch.log(
            torch.tensor(firm.production_scale, dtype=torch.float64)
        )
        assert istar.item() == pytest.approx(97.6, rel=0.02), ppy


def test_the_grc_steady_state_is_a_property_of_the_configuration():
    """`initial_grc_stock` is named as a property of the firm and is not one.

    It is the level at which optimal maintenance spend replaces depreciation,
    which is a fixed point of the whole model -- so it moves with the horizon as
    well as with the hazards. A firm looking five years ahead maintains a
    materially larger control stock than the same firm looking two years ahead.
    """
    from quant.params import GRC_STEADY_STATE, steady_state_firm

    two_year = steady_state_firm(DEFAULTS, 8).initial_grc_stock
    five_year = steady_state_firm(MONTHLY, 60).initial_grc_stock
    assert five_year > two_year * 1.2, (two_year, five_year)


def test_an_unsolved_configuration_is_refused_rather_than_guessed():
    """The failure mode this guards is silent: a stale opening stock does not
    error, it produces a firm that spends the horizon running down controls it
    would never have built and reports that as its budget. That has happened
    once already and read as a headline finding."""
    from quant.params import steady_state_firm

    with pytest.raises(KeyError, match="no GRC steady state"):
        steady_state_firm(MONTHLY, 24)
