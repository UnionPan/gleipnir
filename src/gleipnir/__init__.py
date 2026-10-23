"""Gleipnir: a learning lab of proof-of-concept demos for market calibration and control.

Market simulation, calibration and pricing come from mjollnir. This package
holds what is specific to the demos:

- ``gleipnir.deep_hedging``: torch and JAX hedging agents, training loops
  (Buehler variance-plus-cost objective, POMARL), rollouts and baselines.
- ``gleipnir.utils``: plotting and environment-inspection helpers.
"""

__version__ = "0.1.0"
