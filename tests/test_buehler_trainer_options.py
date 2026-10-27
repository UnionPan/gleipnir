"""The Buehler trainer's observation and tradability options.

``instrument_mask`` restricts what the policy may hold, and
``observe_variance`` / ``variance_proxy`` choose the variance feature: the
simulator's true variance, a RiskMetrics EWMA of squared returns, or none.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from gleipnir.deep_hedging.jax.env import GRID_COARSE, DeepHedgingEnvConfig  # noqa: E402
from gleipnir.deep_hedging.jax.pricing import HestonMarketParams, compile_padded_grid  # noqa: E402
from gleipnir.deep_hedging.training.torch_buehler import (  # noqa: E402
    BuehlerTrainer,
    TrainerConfig,
    differentiable_rollout,
    ewma_variance,
)
from gleipnir.deep_hedging.utils.contracts import LiabilitySpec  # noqa: E402

HORIZON = 12
N_INSTRUMENTS = GRID_COARSE.n_instruments


def _trainer(**options) -> BuehlerTrainer:
    env_config = DeepHedgingEnvConfig(horizon_steps=HORIZON, option_grid=GRID_COARSE, dt=1.0 / 252.0, scheme="qe")
    config = TrainerConfig(batch_size=32, n_epochs=1, policy_kind="lstm", lstm_hidden_size=8, lstm_n_blocks=1,
                           **options)
    return BuehlerTrainer.from_heston(
        config=config, env_config=env_config,
        market_params=HestonMarketParams(kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7, r=0.0, q=0.0),
        padded_grid=compile_padded_grid(GRID_COARSE, dt=env_config.dt, horizon_steps=HORIZON),
        liability=LiabilitySpec(kind="call", strike=100.0, maturity=HORIZON, quantity=1.0), seed=3)


def test_ewma_variance_follows_the_riskmetrics_recursion():
    spots = torch.tensor([[100.0, 101.0, 99.0, 99.5]])
    v0 = torch.tensor([0.04])
    dt, decay = 1.0 / 252.0, 0.94
    expected = [0.04]
    for a, b in zip(spots[0, :-1].tolist(), spots[0, 1:].tolist(), strict=True):
        expected.append(decay * expected[-1] + (1 - decay) * math.log(b / a) ** 2 / dt)
    np.testing.assert_allclose(ewma_variance(spots, v0, dt, decay)[0].numpy(), expected, rtol=1e-6)


def test_an_underlying_only_mask_never_holds_an_option():
    trainer = _trainer(instrument_mask=(1,) + (0,) * (N_INSTRUMENTS - 1))
    trajectory = trainer._generate_batch()
    assert not trajectory["action_masks"][..., 1:].bool().any()

    def buy_one_of_everything(obs, mask):
        return torch.ones(obs.shape[0], N_INSTRUMENTS)

    _, costs = differentiable_rollout(policy=buy_one_of_everything, trajectory=trajectory,
                                      transaction_cost_rates=trainer.transaction_cost_rates,
                                      horizon=HORIZON, n_instruments=N_INSTRUMENTS)
    # only the underlying trades (one share a step): no option is ever bought
    underlying = trainer.transaction_cost_rates[0] * trajectory["spots"][:, :HORIZON].sum(dim=1)
    assert torch.allclose(costs, underlying, rtol=1e-5)


@pytest.mark.parametrize(("options", "check"), [
    ({}, "oracle"),
    ({"variance_proxy": "ewma"}, "ewma"),
    ({"observe_variance": False}, "zero"),
])
def test_the_variance_feature_is_what_the_config_says(options, check):
    trainer = _trainer(**options)
    import jax

    from gleipnir.deep_hedging.utils.adapters import trajectory_batch_to_torch

    raw = trajectory_batch_to_torch(trainer._market_sampler(16, jax.random.PRNGKey(1)), device="cpu")
    seen = trainer._observed(raw)["variances"]
    if check == "oracle":
        assert torch.equal(seen, raw["variances"])
    elif check == "ewma":
        assert torch.allclose(seen, ewma_variance(raw["spots"], raw["variances"][:, 0], 1.0 / 252.0, 0.94))
        assert not torch.allclose(seen[:, 1:], raw["variances"][:, 1:])
    else:
        assert not seen.any()


def test_bad_options_are_refused():
    with pytest.raises(ValueError, match="instrument_mask has 2 entries"):
        _trainer(instrument_mask=(1, 0))
    with pytest.raises(ValueError, match="leaves no instrument"):
        _trainer(instrument_mask=(0,) * N_INSTRUMENTS)
    with pytest.raises(ValueError, match="variance_proxy"):
        _trainer(variance_proxy="garch")


def test_per_path_evaluation_matches_the_aggregate_metrics():
    trainer = _trainer()
    paths = trainer.evaluate_paths(n_paths=64, seed=5)
    metrics = trainer.evaluate(n_paths=64, seed=5)
    assert paths["positions"].shape == (64, HORIZON, N_INSTRUMENTS)
    assert float(np.std(paths["hedging_errors"])) == pytest.approx(metrics["std_hedging_error"], rel=1e-5)
    assert float(np.mean(paths["costs"])) == pytest.approx(metrics["mean_cost"], rel=1e-5)


@pytest.mark.parametrize("kind", ["delta", "delta_gamma", "delta_gamma_vega"])
def test_analytical_paths_match_the_analytical_metrics(kind):
    from gleipnir.deep_hedging.agents.analytical_delta_hedger import build_option_grid_spec
    from gleipnir.deep_hedging.utils.eval_baseline import analytical_hedge_paths, evaluate_analytical_hedger_batch

    trainer = _trainer()
    common = dict(market_sampler=trainer._market_sampler, horizon=HORIZON, n_instruments=N_INSTRUMENTS,
                  transaction_cost_rates=trainer.transaction_cost_rates.cpu().numpy(),
                  liability=LiabilitySpec(kind="call", strike=100.0, maturity=HORIZON, quantity=1.0),
                  hedge_kind=kind, option_grid_spec=build_option_grid_spec(GRID_COARSE), n_paths=64, eval_seed=5)
    paths = analytical_hedge_paths(**common)
    metrics = evaluate_analytical_hedger_batch(**common)
    assert paths["positions"].shape == (64, HORIZON, N_INSTRUMENTS)
    assert float(np.std(paths["hedging_errors"])) == pytest.approx(metrics["std_hedging_error"])
    held_options = np.abs(paths["positions"][..., 1:]).sum()
    assert (held_options == 0) == (kind == "delta")
