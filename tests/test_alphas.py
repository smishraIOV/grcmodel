"""Which channels each family's effectiveness parameter actually drives.

`GrcAlphas` looks like three comparable numbers, and `docs/quant-model.md` §6.5
tabulates their break-evens side by side as though they were. They are not
comparable in the way that table implies, because a unit of stock buys a
different *number of things* in each family:

    credit         1 channel   -- the loss mean
    compliance     2 channels  -- the breach probability, the licence intensity
    operational    3 channels  -- the loss severity, the cliff frequency,
                                  the run frequency

Reusing one alpha per family was a deliberate choice and a defensible one:
nothing in this repo can calibrate one alpha, let alone two or three per family
(docs/quant-model.md §8). The problem was never the choice, it was that the
break-even table did not say the families were buying different bundles and
then compared the numbers.

This file exists so that map cannot go stale. If someone adds a fourth channel
to operational GRC, or gives credit a hazard channel of its own, a test here
fails and the documentation has to be updated with it. That is the same
contract `tests/test_frequency.py` holds for the decision frequency.
"""

import pytest
import torch

from quant.env.actions import FirmAction
from quant.env.env import EnvConfig, FirmEnv
from quant.env.shocks import CommonRandomNumbers, MonteCarloSampler
from quant.hazard import intensity_components
from quant.numerics import REFERENCE
from quant.params import DEFAULTS, GrcAlphas

BATCH = 64
STOCK = 1.0  # per family, near the steady state


def probe(alphas: GrcAlphas) -> dict[str, torch.Tensor]:
    """Every quantity any alpha could touch, at a fixed stock and shock.

    Deliberately reaches into the sub-steps rather than running a rollout. A
    rollout would answer "did firm value move", which every alpha does; the
    question here is *through which channel*, and only the pieces can say.
    """
    env = FirmEnv(
        EnvConfig.quarterly(1, alphas=alphas, profile=REFERENCE),
        MonteCarloSampler(DEFAULTS.sampler),
    )
    dynamics = env.dynamics
    state = env.reset(BATCH)
    stock = REFERENCE.full((BATCH, 3), STOCK)
    shock = env.sampler(CommonRandomNumbers(0, 1, BATCH, REFERENCE).at(0), REFERENCE)

    # A cliff fires or does not; the straight-through estimator's *forward* pass
    # is the hard indicator, so the loss only responds to alpha where the draw
    # sits between the mitigated and unmitigated probabilities. Half the base
    # rate is inside that window for any alpha at or above the default, since
    # exp(-1.5) is 0.22.
    base = DEFAULTS.cliff.period_probability
    shock = type(shock)(
        **{**vars(shock), "cliff_uniform": REFERENCE.full((BATCH,), base * 0.5)}
    )

    losses = dynamics.loss_components(stock, shock, state.equity)
    hazards = intensity_components(
        state.equity, stock, DEFAULTS.firm.initial_equity,
        DEFAULTS.hazard, alphas, DEFAULTS.firm.periods_per_year,
    )
    return {
        "credit loss": losses.credit,
        "operational loss": losses.operational,
        "compliance loss": losses.compliance,
        "cliff loss": dynamics.cliff_loss(state, stock, shock),
        "run probability": dynamics.run_probability(stock, state.equity),
        "breach reweighting": dynamics.path_weight(stock, shock),
        "licence intensity": hazards["licence"],
        "operational intensity": hazards["operational"],
    }


def channels_driven_by(family: str) -> set[str]:
    """Which quantities move when this family's alpha is switched off."""
    live = probe(DEFAULTS.alphas)
    dead = probe(GrcAlphas(**{**vars(DEFAULTS.alphas), family: 0.0}))
    return {
        name
        for name in live
        if not torch.allclose(live[name], dead[name], atol=1e-12)
    }


def test_credit_alpha_drives_one_channel():
    """Credit reaches survival only indirectly, by leaving more equity behind,
    which is why it has no hazard channel of its own -- giving it one would
    count the same mechanism twice (quant/hazard.py)."""
    assert channels_driven_by("credit") == {"credit loss"}


def test_compliance_alpha_drives_two_channels():
    """A programme makes a breach rarer *and* makes revocation rarer. Both are
    real and they are different events -- a breach is not a lost licence -- but
    one parameter buys both, which the break-even table has to say."""
    assert channels_driven_by("compliance") == {
        "breach reweighting",
        "licence intensity",
    }


def test_operational_alpha_drives_three_channels():
    """The widest bundle, and the reason the operational break-even is not
    comparable with the credit one: a unit of spend contains an incident,
    makes a bridge exploit rarer, and makes a depositor run rarer."""
    assert channels_driven_by("operational") == {
        "operational loss",
        "cliff loss",
        "run probability",
    }


def test_the_fourth_operational_channel_is_retired_not_missing():
    """`hazard.py` still has an operational intensity reduced by the same alpha.
    It does not appear above because its *rate* is zero -- the channel was
    replaced by the run mechanism, not deleted. Set the rate and it comes back,
    which is what makes this a retirement rather than an absence."""
    assert DEFAULTS.hazard.annual_operational_rate == 0.0

    stock = REFERENCE.full((BATCH, 3), STOCK)
    equity = REFERENCE.full((BATCH,), DEFAULTS.firm.initial_equity)
    revived = type(DEFAULTS.hazard)(
        **{**vars(DEFAULTS.hazard), "annual_operational_rate": 0.5}
    )

    def operational(alphas):
        return intensity_components(
            equity, stock, DEFAULTS.firm.initial_equity, revived, alphas,
            DEFAULTS.firm.periods_per_year,
        )["operational"]

    assert torch.all(
        operational(DEFAULTS.alphas) < operational(GrcAlphas(operational=0.0))
    )


@pytest.mark.parametrize(
    "family,expected", [("credit", 1), ("compliance", 2), ("operational", 3)]
)
def test_the_documented_channel_counts_are_the_measured_ones(family, expected):
    """The table in docs/quant-model.md §6.5 and the docstring on `GrcAlphas`
    both quote these counts. This is what stops them drifting from the code."""
    assert len(channels_driven_by(family)) == expected
