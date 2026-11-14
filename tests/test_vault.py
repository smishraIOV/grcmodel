"""The vault form, one period: fees, flows, compensation and the firm's books
(design.md section 2)."""

from dataclasses import replace

import pytest
import torch

from quant.model import jump_probability
from quant.numerics import REFERENCE
from quant.params import DEFAULTS
from quant.quadrature import ShockRule, pareto_jump_rule, student_t_rule
from quant.vault import (
    VaultControls,
    vault_capital_floor,
    vault_outcomes,
    vault_shock_correlation,
)

P = REFERENCE
BASE = replace(DEFAULTS, periods_per_year=1, jump_rate=0.0, tau=0.0)


def t(x):
    return P.tensor(x)


def calm_rule(shock=(0.0, 0.0, 0.0)):
    """One scenario, with the given shock and probability one."""
    return ShockRule(nodes=P.tensor([list(shock)]), weights=P.ones(1))


def controls(K=0.05, g=0.0, q=0.0, H=0.0, alloc=(0.5, 0.3, 0.2)):
    return VaultControls(K=t([K]), g=t([g]), q=t([q]), H=t([H]), alloc=t([list(alloc)]))


def by_hand(p, A, c, shock, jump_share=0.0, S=0.0):
    """The design's formulas, written out with plain floats."""
    a, (x1, x2, e) = c.alloc[0].tolist(), shock
    import math
    phi = lambda aa, rho: rho + (1 - rho) * math.exp(-aa * S)  # noqa: E731
    R = sum(a[i] * (p.vault_mu[i] - phi(p.a_A, p.rho_A) * p.vault_loss[i]) for i in range(3))
    R += phi(p.a_B, p.rho_B) * (a[0] * p.vault_vol[0] * x1 + a[1] * p.vault_vol[1] * x2)
    R += sum(a[i] * p.vault_btc[i] for i in range(3)) * p.s_BTC * e
    hit = jump_share * sum(a[i] * p.vault_jump[i] for i in range(3))
    R -= hit
    fees = p.fee_mgmt * A + p.fee_perf * max(R, 0.0) * A
    cover = p.compensation * hit
    if p.guarantee is not None:
        cover = max(cover, max(p.guarantee - p.r - R, 0.0))
    R_dep = R + cover
    aum = (A * (1 + p.r + R_dep) - fees + p.inflow_speed * (p.capacity - A)
           - p.outflow_perf * max(-R_dep, 0.0) * A - (p.outflow_incident * A if jump_share > 0 else 0.0))
    K, g, q, H = (float(c.K), float(c.g), float(c.q), float(c.H))
    wealth = ((1 + p.r) * (K - g) - p.tau * K + (q - H) * (p.mu_BTC + p.s_BTC * e)
              - 0.5 * p.c_H * H * H + fees - p.opex - cover * A + p.co_invest * A * R)
    return wealth, max(aum, 0.0), R


@pytest.mark.parametrize("shock", [(0.0, 0.0, 0.0), (1.5, -0.5, 2.0), (-3.0, -2.0, -1.0)])
def test_accounting_matches_the_formulas(shock):
    """Good and bad periods, with BTC, a treasury and a hedge."""
    p = replace(BASE, mu_BTC=0.1, c_H=0.5, tau=0.01)
    c = controls(K=0.08, g=0.004, q=0.03, H=0.01)
    out = vault_outcomes(p, t([0.7]), c, t([0.02]), calm_rule(shock), None)
    w, A, R = by_hand(p, 0.7, c, shock, S=0.02)
    assert float(out.wealth) == pytest.approx(w, abs=1e-14)
    assert float(out.aum) == pytest.approx(A, abs=1e-14)
    assert float(out.excess) == pytest.approx(R, abs=1e-14)


def test_no_performance_fee_after_a_loss():
    p = BASE
    c = controls()
    bad = vault_outcomes(p, t([1.0]), c, t([0.0]), calm_rule((-3.0, -3.0, -3.0)), None)
    good = vault_outcomes(p, t([1.0]), c, t([0.0]), calm_rule((0.0, 0.0, 0.0)), None)
    assert float(bad.excess) < 0 < float(good.excess)
    fee_bad = float(bad.wealth) - (float(good.wealth) - p.fee_perf * float(good.excess))
    assert fee_bad == pytest.approx(0.0, abs=1e-14)


def test_aum_grows_toward_capacity_when_nothing_happens():
    """No returns, no fees: AUM closes the inflow share of the gap each period."""
    p = replace(BASE, vault_mu=(0.0, 0.0, 0.0), vault_loss=(0.0, 0.0, 0.0), fee_mgmt=0.0, r=0.0, r_E=0.06)
    out = vault_outcomes(p, t([0.4]), controls(), t([0.0]), calm_rule(), None)
    assert float(out.aum) == pytest.approx(0.4 + p.inflow_speed * (1.0 - 0.4), abs=1e-14)


def test_an_incident_costs_aum_and_compensation():
    p = replace(BASE, jump_rate=0.1)
    jumps = pareto_jump_rule(p.jump_mean, p.jump_tail, P, n=4)
    c = controls(alloc=(1.0, 0.0, 0.0))
    out = vault_outcomes(p, t([1.0]), c, t([0.0]), calm_rule(), jumps)
    # outcome 0 is no jump; outcome 1 is the smallest jump
    sev = float(jumps.severity[0])
    w0, A0, _ = by_hand(p, 1.0, c, (0.0, 0.0, 0.0))
    w1, A1, _ = by_hand(p, 1.0, c, (0.0, 0.0, 0.0), jump_share=sev)
    assert float(out.wealth[0, 1]) == pytest.approx(w1, abs=1e-14)
    assert float(out.aum[0, 1]) == pytest.approx(A1, abs=1e-14)
    assert A1 < A0 - p.outflow_incident * 1.0 + 1e-12


def test_outcome_weights_sum_to_one_and_respond_to_grc():
    p = replace(BASE, jump_rate=0.1)
    rule = student_t_rule(vault_shock_correlation(p), p.nu, P, n_hermite=3, n_mixing=4)
    jumps = pareto_jump_rule(p.jump_mean, p.jump_tail, P, n=4)
    g = t([0.01]).requires_grad_(True)
    c = VaultControls(K=t([0.05]), g=g, q=t([0.0]), H=t([0.0]), alloc=t([[0.5, 0.3, 0.2]]))
    out = vault_outcomes(p, t([1.0]), c, g, rule, jumps)
    assert float(out.weights.sum().detach()) == pytest.approx(1.0, abs=1e-13)
    no_jump_mass = out.weights.reshape(-1, 1 + 4)[:, 0].sum()
    assert float(no_jump_mass.detach()) == pytest.approx(1 - float(jump_probability(p, t(0.01))), abs=1e-13)
    (grad,) = torch.autograd.grad(out.expect(out.wealth).sum(), g)
    assert float(grad) != 0.0


def test_capital_floor_takes_the_tightest_constraint():
    p = replace(BASE, k_A=0.02, k_BTC=0.1)
    assert float(vault_capital_floor(p, t(1.0), t(0.0), t(0.0))) == pytest.approx(0.02)
    assert float(vault_capital_floor(p, t(1.0), t(0.01), t(0.05))) == pytest.approx(0.06)


@pytest.mark.parametrize("co, floor", [(0.1, None), (0.0, 0.0), (0.1, 0.0)])
@pytest.mark.parametrize("shock", [(0.0, 0.0, 0.0), (-3.0, -2.0, -1.0)])
def test_co_investment_and_guarantee_accounting(co, floor, shock):
    """The firm's stake earns the vault's return; the guarantee tops depositors
    up to the floor and the firm pays it -- the larger of it and compensation."""
    p = replace(BASE, co_invest=co, guarantee=floor, jump_rate=0.1)
    jumps = pareto_jump_rule(p.jump_mean, p.jump_tail, P, n=4)
    c = controls(K=0.15, alloc=(0.6, 0.4, 0.0))
    out = vault_outcomes(p, t([0.8]), c, t([0.0]), calm_rule(shock), jumps)
    for j, share in enumerate([0.0] + jumps.severity.tolist()):
        w, A, _ = by_hand(p, 0.8, c, shock, jump_share=share)
        assert float(out.wealth[0, j]) == pytest.approx(w, abs=1e-14)
        assert float(out.aum[0, j]) == pytest.approx(A, abs=1e-14)


def test_a_guarantee_protects_depositors_principal():
    """With a zero floor, depositors' money never shrinks from returns alone."""
    p = replace(BASE, guarantee=0.0, r=0.0, r_E=0.06, inflow_speed=0.0,
                fee_mgmt=0.0, outflow_incident=0.0)
    out = vault_outcomes(p, t([1.0]), controls(), t([0.0]), calm_rule((-3.0, -3.0, -3.0)), None)
    assert float(out.excess) < 0
    assert float(out.aum) == pytest.approx(1.0, abs=1e-14)


def test_co_investment_uses_cash():
    p = replace(BASE, co_invest=0.1, k_A=0.02)
    assert float(vault_capital_floor(p, t(1.0), t(0.0), t(0.0))) == pytest.approx(0.1)
