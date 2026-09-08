from quant.static import run_convexity_sweep


def test_optimal_g_increases_with_convexity():
    """The core Froot-Stein qualitative result: as the convexity of external
    financing costs rises, the firm optimally invests more in GRC (to avoid
    the increasingly punishing cost of a shortfall)."""
    results = run_convexity_sweep()
    optimal_gs = [g for _convexity, g, _value in results]

    for earlier, later in zip(optimal_gs, optimal_gs[1:]):
        assert later >= earlier - 1e-6
    assert optimal_gs[-1] > optimal_gs[0]
