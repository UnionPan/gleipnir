"""The deep-hedging market lists genuine option contracts.

Each grid slot holds one contract from issue to expiry (strike fixed, tenor
running down), so a position held across steps is a position in one
instrument, and its price is a martingale under the pricing measure. The
earlier grid re-priced every slot as a fresh constant-tenor option struck at
the current spot each step: today's variance then predicted the slot's next
move (a z-score near 40), which a learned hedger harvested as free P&L.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from mjollnir.jax import configure_runtime, fourier_price_batch

from gleipnir.deep_hedging.jax.env import GRID_COARSE, DeepHedgingEnvConfig, build_action_mask
from gleipnir.deep_hedging.jax.pricing import (
    HestonMarketParams,
    compile_padded_grid,
    contract_schedule,
    price_option_contracts,
)
from gleipnir.deep_hedging.jax.rollout import simulate_heston_market_batch

configure_runtime()
MARKET = HestonMarketParams(kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7, r=0.0, q=0.0)
DT = 1.0 / 252.0
HORIZON = 60
GRID = compile_padded_grid(GRID_COARSE, dt=DT, horizon_steps=HORIZON)
FIVE_DAY_ATM_CALL = 3  # instrument index: underlying, then (call, put) per strike, 5-day strikes first


@pytest.fixture(scope="module")
def market():
    config = DeepHedgingEnvConfig(horizon_steps=HORIZON, option_grid=GRID_COARSE, dt=DT, scheme="qe")
    return simulate_heston_market_batch(config=config, market=MARKET, padded_grid=GRID, initial_spot=100.0,
                                        initial_variance=0.04, keys=jax.random.split(jax.random.PRNGKey(7), 1024))


def test_a_contract_trades_for_its_maturity_then_expires_and_is_reissued():
    tau, issue = zip(*(contract_schedule(jnp.asarray([5]), t) for t in range(13)), strict=True)
    assert [int(x[0]) for x in tau] == [5, 4, 3, 2, 1, 0, 5, 4, 3, 2, 1, 0, 5]
    assert [bool(x[0]) for x in issue] == [t % 6 == 0 for t in range(13)]
    masks = [build_action_mask(GRID_COARSE, t, HORIZON)[FIVE_DAY_ATM_CALL] for t in range(13)]
    assert masks == [t % 6 != 5 for t in range(13)]  # not tradable on its expiry step


def test_strikes_stay_fixed_until_reissue_and_expiry_pays_intrinsic(market):
    strikes = market.option_strikes[:, :, FIVE_DAY_ATM_CALL - 1]
    spots = market.spots
    np.testing.assert_allclose(strikes[:, 0:6], spots[:, [0]] * np.ones((1, 6)))
    np.testing.assert_allclose(strikes[:, 6], spots[:, 6])  # reissued at the money
    np.testing.assert_allclose(market.instrument_prices[:, 5, FIVE_DAY_ATM_CALL],
                               np.maximum(spots[:, 5] - strikes[:, 5], 0.0), atol=1e-12)
    assert market.option_taus[5, FIVE_DAY_ATM_CALL - 1] == 0.0


@pytest.mark.parametrize(("spot", "variance"), [(90.0, 0.002), (100.0, 0.04), (112.0, 0.15)])
def test_contract_prices_match_the_reference_cos_pricer(spot, variance):
    strikes = jnp.asarray(100.0 * np.asarray(GRID.flat_moneyness))
    tau = jnp.asarray([2.0, 13.0]) * DT
    ours = np.asarray(price_option_contracts(jnp.asarray(spot), jnp.asarray(variance), strikes, tau, GRID, MARKET))
    for i in range(strikes.size):
        reference = fourier_price_batch(spot, strikes[i:i + 1], float(tau[int(GRID.flat_maturity_idx[i])]), 0.0, 0.0,
                                        variance, MARKET.kappa, MARKET.theta, MARKET.sigma_v, MARKET.rho,
                                        jnp.asarray([bool(GRID.flat_is_call[i])]))
        assert ours[i + 1] == pytest.approx(float(reference[0]), abs=1e-4)


@pytest.mark.parametrize("variance", [5e-5, 2e-4, 1e-3, 0.04, 0.25])
def test_128_terms_price_every_state_the_simulator_visits(variance):
    # far below its long-run level, today's variance once got a range sized for the long run, and
    # 128 terms mispriced short out-of-the-money contracts by more than it costs to trade them
    strikes = jnp.asarray(100.0 * np.asarray(GRID.flat_moneyness))
    for spot in (85.0, 100.0, 115.0):
        for days in (1, 3, 5, 12, 20):
            tau = jnp.asarray([min(days, 5), days]) * DT
            ours = price_option_contracts(jnp.asarray(spot), jnp.asarray(variance), strikes, tau, GRID, MARKET)
            fine = price_option_contracts(jnp.asarray(spot), jnp.asarray(variance), strikes, tau, GRID, MARKET,
                                          N_cos=4096)
            np.testing.assert_allclose(np.asarray(ours), np.asarray(fine), rtol=0, atol=1e-6)


@pytest.mark.parametrize("slot", [3, 4, 9, 10])  # 5- and 20-day at-the-money calls and puts
def test_todays_variance_does_not_predict_a_held_contracts_next_move(market, slot):
    held = market.action_masks[:-1, slot]
    gain = (market.instrument_prices[:, 1:, slot] - market.instrument_prices[:, :-1, slot])[:, held].ravel()
    gap = (MARKET.theta - market.variances[:, :-1])[:, held].ravel()
    slope, intercept = np.polyfit(gap, gain, 1)
    residual = gain - slope * gap - intercept
    assert abs(slope) / (residual.std() / (gap.std() * np.sqrt(gap.size))) < 4.0
    assert abs(gain.mean()) < 4.0 * gain.std() / np.sqrt(gain.size)
