"""Monte Carlo stage (docs/quant-model.md section 5.2).

Samples batched risk draws for the families in docs/framework.md section 1 and
feeds them to the same FirmValueModel / optimize_policy harness quant/static.py
uses with its four explicit states. The distributions are illustrative, not
calibrated -- which is exactly why quant/threshold.py reports break-even
parameter values rather than treating any single answer as a point estimate.

Results are reported across several seeds with a confidence interval. The
severity distributions are heavy-tailed, so a single-seed answer quoted to four
decimals overstates what the sample supports (docs/critical-review.md F6).
"""

import statistics
from dataclasses import dataclass, field

import torch

from quant.device import get_device
from quant.model import (
    FirmValueModel,
    GrcAlphas,
    GrcBudgets,
    PolicyResult,
    RiskDraw,
    optimize_policy,
)
from quant.static import (
    DISTRESS_REFERENCE,
    FINANCING_CONVEXITY,
    INITIAL_EQUITY,
    PRODUCTION_CURVATURE,
    PRODUCTION_SCALE,
)

DEFAULT_SEEDS = (0, 1, 2, 3, 4)


@dataclass
class RiskSamplerConfig:
    """Illustrative distribution parameters, chosen to match the expected loss
    of quant/static.py's four-state shock (4.0 credit + 3.5 operational +
    1.5 compliance) so the two stages are comparable.

    Credit is a continuous loss. Operational incidents are rare but heavy-
    tailed when they land. A compliance breach is rarer still and carries a
    fixed severity -- losing a licence costs what it costs.
    """

    credit_loss_mean: float = 4.0
    op_probability: float = 0.25
    op_severity_mean: float = 14.0
    compliance_probability: float = 0.05
    compliance_severity: float = 30.0


def sample_risk_draw(
    n_paths: int, config: RiskSamplerConfig, generator: torch.Generator
) -> RiskDraw:
    def exponential(mean: float) -> torch.Tensor:
        uniform = torch.rand(n_paths, generator=generator, dtype=torch.float64)
        return -mean * torch.log1p(-uniform)

    def bernoulli(probability: float) -> torch.Tensor:
        return (
            torch.rand(n_paths, generator=generator, dtype=torch.float64) < probability
        ).to(torch.float64)

    return RiskDraw(
        credit_loss=exponential(config.credit_loss_mean),
        op_occurs=bernoulli(config.op_probability),
        op_severity=exponential(config.op_severity_mean),
        compliance_occurs=bernoulli(config.compliance_probability),
        compliance_severity=torch.full((n_paths,), config.compliance_severity, dtype=torch.float64),
        base_weight=torch.ones(n_paths, dtype=torch.float64),
        compliance_base_probability=config.compliance_probability,
    )


def build_model(**overrides) -> FirmValueModel:
    params = dict(
        initial_equity=INITIAL_EQUITY,
        financing_convexity=FINANCING_CONVEXITY,
        distress_reference=DISTRESS_REFERENCE,
        production_scale=PRODUCTION_SCALE,
        production_curvature=PRODUCTION_CURVATURE,
        alphas=GrcAlphas(),
    )
    params.update(overrides)
    return FirmValueModel(**params)


def run_policy_search(
    model: FirmValueModel,
    n_paths: int = 8192,
    config: RiskSamplerConfig | None = None,
    seed: int = 0,
    n_steps: int = 3000,
) -> PolicyResult:
    """One sample-average-approximation solve: draw a batch, optimize against it."""
    generator = torch.Generator().manual_seed(seed)
    draw = sample_risk_draw(n_paths, config or RiskSamplerConfig(), generator)
    return optimize_policy(model, draw, n_steps=n_steps, device=get_device())


@dataclass
class SeedStudy:
    """Policy results across independent samples, so the sampling error in the
    answer is visible rather than hidden behind a fixed default seed."""

    results: list[PolicyResult] = field(default_factory=list)

    def interval(self, attribute: str) -> tuple[float, float]:
        """Mean and 95% confidence half-width for one reported quantity."""
        values = [getattr(result, attribute) for result in self.results]
        mean = statistics.fmean(values)
        if len(values) < 2:
            return mean, 0.0
        return mean, 1.96 * statistics.stdev(values) / len(values) ** 0.5


def run_seed_study(
    model: FirmValueModel,
    n_paths: int = 8192,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    config: RiskSamplerConfig | None = None,
    n_steps: int = 3000,
) -> SeedStudy:
    return SeedStudy(
        [
            run_policy_search(model, n_paths=n_paths, config=config, seed=seed, n_steps=n_steps)
            for seed in seeds
        ]
    )


def loss_quantiles(
    draw: RiskDraw, result: PolicyResult, alphas: GrcAlphas, quantiles: tuple[float, ...]
) -> list[float]:
    """Distribution of mitigated loss at the chosen policy -- the tail that the
    convex financing premium actually prices."""
    budgets = GrcBudgets(
        credit=torch.tensor(result.credit, dtype=torch.float64),
        operational=torch.tensor(result.operational, dtype=torch.float64),
        compliance=torch.tensor(result.compliance, dtype=torch.float64),
    )
    losses = draw.mitigated_loss(budgets, alphas)
    return [torch.quantile(losses, q).item() for q in quantiles]


def main() -> None:
    device = get_device()
    print(f"device: {device}")
    n_paths = 8192
    model = build_model()
    study = run_seed_study(model, n_paths=n_paths)

    print(f"\nMonte Carlo policy search: {n_paths} paths x {len(DEFAULT_SEEDS)} seeds")
    print(f"{'quantity':>22} | {'mean':>9} | {'95% CI':>13}")
    for label, attribute in (
        ("credit GRC", "credit"),
        ("operational GRC", "operational"),
        ("compliance GRC", "compliance"),
        ("total GRC", "total"),
        ("firm value", "value"),
        ("constrained fraction", "constrained_fraction"),
    ):
        mean, half_width = study.interval(attribute)
        print(f"{label:>22} | {mean:>9.3f} | +/- {half_width:>9.3f}")

    generator = torch.Generator().manual_seed(DEFAULT_SEEDS[0])
    draw = sample_risk_draw(n_paths, RiskSamplerConfig(), generator)
    quantiles = (0.5, 0.9, 0.99)
    values = loss_quantiles(draw, study.results[0], model.alphas, quantiles)
    print("\nMitigated loss distribution at the chosen policy (seed 0)")
    print("  " + "  ".join(f"p{int(q * 100)}={v:.2f}" for q, v in zip(quantiles, values)))
    print("\nPrecision note: budgets are reported to 3 decimals because the")
    print("confidence interval above is wider than the 4th.")


if __name__ == "__main__":
    main()
