"""The learned band hedger: a no-trade band around the model's delta."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from mjollnir.backtest import hedge_market, simulate_hedge
from mjollnir.hedging import EuropeanOption, model_delta, model_price
from mjollnir.jax import configure_runtime
from mjollnir.params import ParamSet

from gleipnir.lab_hedger import artifact, band_strategy, init_params
from gleipnir.lab_hedger.policy import MAX_HALF_WIDTH

configure_runtime()
DT = 1.0 / 252
BOOK = ParamSet.create("heston", "P", {"v0": 0.06, "kappa": 2.0, "theta": 0.05, "sigma_v": 0.5, "rho": -0.7},
                       asset="TEST")
OPTION = EuropeanOption(strike=100.0, maturity=21 * DT, is_call=True)


def _run(net, cost_bps=10.0):
    base = model_delta(OPTION, BOOK, DT, 0.03, 0.0)
    premium = model_price(OPTION, BOOK, 100.0, rate=0.03, dividend=0.0, dt=DT)
    market = hedge_market(BOOK, dt=DT, mu=0.03, rate=0.03, spot0=100.0, initial_cash=premium, cost_bps=cost_bps)
    keys = jax.random.split(jax.random.PRNGKey(5), 512)
    delta = simulate_hedge(market, base.fn, base.params, keys, horizon_steps=21, payoff=OPTION.payoff)
    band = simulate_hedge(market, band_strategy(base.fn),
                          {"base": base.params, "net": net, "cost_bps": cost_bps, "risk_aversion": 5.0}, keys,
                          horizon_steps=21, payoff=OPTION.payoff)
    return delta, band


def _constant_band(half_width: float) -> dict:
    """A network whose band has the given half-width everywhere."""
    net = init_params()
    logit = float(np.log(half_width / (MAX_HALF_WIDTH - half_width)))
    return jax.tree_util.tree_map_with_path(
        lambda path, x: jnp.full_like(x, logit) if "Dense_2" in str(path) and "bias" in str(path) else x, net)


def test_a_closed_band_is_the_models_own_delta():
    (delta_pnl, delta_costs), (band_pnl, band_costs) = _run(_constant_band(1e-9))
    np.testing.assert_allclose(np.asarray(band_pnl), np.asarray(delta_pnl), atol=1e-6)
    np.testing.assert_allclose(np.asarray(band_costs), np.asarray(delta_costs), atol=1e-6)


def test_a_wider_band_trades_less():
    (_, delta_costs), (_, band_costs) = _run(_constant_band(0.1))
    assert float(band_costs.mean()) < 0.8 * float(delta_costs.mean())


def test_the_band_function_is_stable_per_model_hedger():
    base = model_delta(OPTION, BOOK, DT, 0.03, 0.0)
    assert band_strategy(base.fn) is band_strategy(base.fn)


def test_shipped_networks_load_and_tampered_ones_do_not(tmp_path):
    shipped = Path(__file__).resolve().parents[1] / "backend" / "data" / "lab" / "hedgers"
    trained = sorted(p for p in shipped.iterdir() if (p / "params.msgpack").exists()) if shipped.exists() else []
    if not trained:
        pytest.skip("no trained networks shipped")
    net, report = artifact.load(trained[0])
    assert report["model"] == trained[0].name and len(report["weights_sha256"]) == 64
    copy = artifact.save(net, {k: v for k, v in report.items() if k != "weights_sha256"}, tmp_path / "copy")
    weights = (copy / "params.msgpack").read_bytes()
    (copy / "params.msgpack").write_bytes(weights[:-1] + bytes([weights[-1] ^ 1]))
    with pytest.raises(artifact.HedgerIntegrityError):
        artifact.load(copy)
