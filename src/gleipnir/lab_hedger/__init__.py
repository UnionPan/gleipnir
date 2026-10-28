"""The Hedging Lab's learned hedger: a no-trade band around each model's own delta,
trained offline through the lab's simulated worlds (see :mod:`.policy`, :mod:`.train`)."""

from .policy import band_strategy, init_params

__all__ = ["band_strategy", "init_params"]
