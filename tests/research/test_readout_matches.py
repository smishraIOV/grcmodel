"""The tool's read-out against the research one (docs/plan-tool.md step 3):
given a solved firm's own decisions at a state, the two agree. They are
written separately, so this checks both."""

from dataclasses import replace

import pytest

from quant.params import INTEGRATED_DEFAULTS
from quant.policy import to_policy
from quant.profile import FirmState, ModelFirm
from quant.numerics import REFERENCE
from quant.readout import ScenarioSet, read_out
from quant.research.integrated_results import readout, state_index
from quant.research.integrated_solver import IntegratedSpec, solve_integrated

pytestmark = pytest.mark.slow

SPEC = IntegratedSpec(n_w=9, n_s=2, n_L=3, n_B=3, n_ops=1, n_strategy=1, n_btc=5, n_mixing=3,
                      n_crash=3, n_jump=2)
SHARED = {   # research name -> tool name, where they differ
    "dividend": "payout", "issue": "raised", "capital_floor": "capital_needed",
    "chance_below_floor": "chance_below_needed",
}
SAME = ("equity", "dollar_vault", "btc_vault", "equity_kept", "stakes", "own_btc", "hedged", "stablecoins",
        "grc_spend", "incident_chance", "borrower_rate", "ltv", "loans", "share_lent", "interest_income",
        "btc_vault_fees", "operating_costs", "compensation", "dollar_yield", "dollar_outside", "btc_yield",
        "btc_outside", "expected_bad_debt", "expected_shortfall")


@pytest.mark.parametrize("periods", [1, 4])
def test_the_tool_s_read_out_matches_the_research_one_at_a_solved_state(periods):
    # A few iterations are enough: any decisions will do, as long as both read the same ones.
    inputs = replace(INTEGRATED_DEFAULTS, periods_per_year=periods)
    sol = solve_integrated(inputs, REFERENCE, SPEC, max_iterations=2, polish_rounds=1)
    for at in ((0.3, 1.0, 2.0), (0.1, 0.5, 1.0)):
        research = readout(sol, *at)
        i = state_index(sol, *at)
        w, s, A_L, A_B = (float(x[i]) for x in sol.states())
        model = ModelFirm(params=inputs, state=FirmState(w=w, S=s, A_L=A_L, A_B=A_B), scale=1.0)
        tool = read_out(model, to_policy(sol.controls(), model, index=i), ScenarioSet(sol.rule, sol.jumps))
        for name in SAME:
            assert getattr(tool, name) == pytest.approx(getattr(research, name), rel=1e-9, abs=1e-12), name
        for theirs, ours in SHARED.items():
            assert getattr(tool, ours) == pytest.approx(getattr(research, theirs), rel=1e-9, abs=1e-12), theirs
        assert tool.allocation == pytest.approx(research.allocation, rel=1e-9)
