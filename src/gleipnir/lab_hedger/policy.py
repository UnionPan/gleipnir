"""A learned no-trade band around a model's own delta.

With proportional trading costs the optimal hedge is not to follow delta
but to keep the position inside a band around it, trading only to its edge
when the position leaves it (Whalley and Wilmott; Davis, Panas and
Zariphopoulou). The band's width depends on how fast delta moves (gamma),
the time left and the cost of trading. This policy keeps that shape and
learns the width: a small network maps

    (delta, gamma, sqrt of time to expiry, log moneyness, cost, call or put,
     risk aversion)

to a half-width w in shares, and the position becomes clip(position,
delta - w, delta + w). It inherits the model's delta (so it hedges
correctly when trading is free, w -> 0) and learns only when not to trade.
How wide the band should be is a preference, not a fact: the risk aversion
is an input, so one network serves cautious and cost-conscious desks.

Gamma is the derivative of the model's delta with respect to the spot,
taken by forward-mode differentiation through the model's own hedger.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import linen as nn

MAX_HALF_WIDTH = 0.5  # shares per option: the band never lets the hedge drift further
INITIAL_LOGIT = -3.0  # an untrained band is about 2% of a share wide: close to pure delta


class BandNetwork(nn.Module):
    hidden: tuple[int, ...] = (32, 32)

    @nn.compact
    def __call__(self, features):
        x = features
        for width in self.hidden:
            x = nn.gelu(nn.Dense(width)(x))
        logit = nn.Dense(1, kernel_init=nn.initializers.zeros,
                         bias_init=nn.initializers.constant(INITIAL_LOGIT))(x)
        return MAX_HALF_WIDTH * jax.nn.sigmoid(logit[..., 0])


NETWORK = BandNetwork()
N_FEATURES = 7


def init_params(seed: int = 0) -> dict:
    return NETWORK.init(jax.random.PRNGKey(seed), jnp.zeros((N_FEATURES,)))


def features(delta, gamma_s, tau, log_moneyness, cost_bps, is_call, risk_aversion):
    """The band's inputs, each of order one."""
    return jnp.stack([
        delta,
        jnp.log1p(jnp.abs(gamma_s)),          # dDelta/dlnS: large near expiry at the money
        jnp.sqrt(jnp.maximum(tau, 0.0)) * 4.0,  # sqrt(years): 1 at about a month
        log_moneyness * 10.0,
        jnp.log1p(cost_bps),
        jnp.where(is_call, 1.0, -1.0),
        jnp.log(risk_aversion),
    ])


_BAND_FNS: dict[Callable, Callable] = {}


def band_strategy(base_fn: Callable) -> Callable:
    """The band hedger around ``base_fn`` (a model hedger's function), as a stable,
    hashable strategy function ``fn(params, spot, state, position, step)``.

    ``params``: ``{"base": the model hedger's params, "net": the band network's
    weights, "cost_bps": the trading cost, "risk_aversion": the desk's}``.
    """
    if base_fn not in _BAND_FNS:
        def band(params, spot, state, position, step):
            base = params["base"]
            delta, ddelta = jax.jvp(lambda s: base_fn(base, s, state, position, step), (spot,), (jnp.ones_like(spot),))
            tau = base["maturity"] - step * base["dt"]
            x = features(delta, ddelta * spot, tau, jnp.log(spot / base["strike"]), params["cost_bps"],
                         base["is_call"], params["risk_aversion"])
            half_width = NETWORK.apply(params["net"], x)
            return jnp.clip(position, delta - half_width, delta + half_width)

        band.__name__ = f"band_around_{getattr(base_fn, '__name__', 'delta')}"
        _BAND_FNS[base_fn] = band
    return _BAND_FNS[base_fn]
