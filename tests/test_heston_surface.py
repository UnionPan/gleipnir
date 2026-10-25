"""Heston fitted to an option chain: quote selection, recovery, bounded inputs."""

from datetime import date, timedelta

import jax.numpy as jnp
import numpy as np
import pytest
from mjollnir.calibration.marketdata import OptionChain
from mjollnir.jax import configure_runtime, fourier_price_batch
from mjollnir.synthetic_data.data_provider import OptionQuote

from backend.services import heston_surface

configure_runtime()
TRUE = {"v0": 0.06, "kappa": 2.0, "theta": 0.05, "sigma_v": 0.5, "rho": -0.6}
TODAY = date(2026, 3, 2)
S0, R, Q = 100.0, 0.04, 0.01


def _quote(strike, days, is_call, mid, volume=50, bid_fraction=0.98):
    return OptionQuote(strike=strike, expiry=TODAY + timedelta(days=days),
                       option_type="call" if is_call else "put", bid=mid * bid_fraction,
                       ask=mid * (2 - bid_fraction), mid=mid, last=mid, volume=volume, open_interest=100)


def _heston_chain(days=(21, 49, 91, 182, 364)):
    options = []
    for d in days:
        T = d / heston_surface.DAYS_PER_YEAR
        strikes = np.linspace(80, 120, 11)
        for is_call in (True, False):
            mids = np.asarray(fourier_price_batch(S0, jnp.asarray(strikes), T, R, Q, **TRUE,
                                                  is_call=jnp.full(len(strikes), is_call)))
            options += [_quote(float(k), d, is_call, float(m)) for k, m in zip(strikes, mids, strict=True)]
    return OptionChain(underlying="TEST", spot_price=S0, reference_date=TODAY,
                       risk_free_rate=R, dividend_yield=Q, options=options)


def test_quotes_keep_the_out_of_the_money_side_within_the_expiry_window():
    chain = _heston_chain(days=(3, 21, 91, 900))
    chain.options.append(_quote(100.0, 49, True, 3.0, bid_fraction=0.0))  # no bid
    quotes, weights, summary = heston_surface.chain_quotes(chain)
    for strike, T, is_call, _ in quotes:
        assert is_call == (strike >= S0 * np.exp((R - Q) * T))
        assert heston_surface.MIN_DAYS <= round(T * heston_surface.DAYS_PER_YEAR) <= heston_surface.MAX_DAYS
    assert summary["n_expiries"] == 2 and summary["days_range"] == [21, 91]
    assert len(weights) == len(quotes) and np.all(weights > 0)


def test_quotes_are_capped_at_the_bucket_most_liquid_first():
    chain = _heston_chain(days=tuple(range(14, 14 + 7 * 40, 7)))  # 40 expiries, 880 quotes
    # The last expiry falls past the expiry cap unless liquidity keeps it.
    liquid = next(o for o in reversed(chain.options) if not o.is_call and o.strike == 80.0)
    liquid.volume = 10_000
    quotes, _, summary = heston_surface.chain_quotes(chain)
    assert summary["n_quotes"] <= heston_surface.QUOTE_BUCKET
    assert summary["n_expiries"] <= heston_surface.EXPIRY_BUCKET
    last_T = (liquid.expiry - TODAY).days / heston_surface.DAYS_PER_YEAR
    assert any(k == 80.0 and t == pytest.approx(last_T) and not c for k, t, c, _ in quotes)


def test_fit_recovers_a_heston_chain_with_honest_errors():
    params, diagnostics = heston_surface.fit_chain(_heston_chain())
    assert diagnostics["converged"] and diagnostics["rmse_iv"] < 1e-3
    truth = {"v0": TRUE["v0"], "kappa": TRUE["kappa"], "theta": TRUE["theta"], "xi": TRUE["sigma_v"],
             "rho": TRUE["rho"]}
    for name, value in truth.items():
        assert params[name] == pytest.approx(value, rel=0.05), name
        assert diagnostics["std_errors"][name] >= 0
    assert len(diagnostics["params_hash"]) == 64


def test_a_chain_too_thin_to_fit_is_refused():
    chain = _heston_chain(days=(21,))
    chain.options = chain.options[:5]
    with pytest.raises(ValueError, match="usable out-of-the-money quotes"):
        heston_surface.fit_chain(chain)
