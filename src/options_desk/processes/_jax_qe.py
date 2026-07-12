"""Back-compat shim: canonical implementation now lives in mjollnir.processes._jax_qe.

The market-simulation kernel was extracted to the standalone `mjollnir` library
(see docs/superpowers/specs/2026-07-12-mjollnir-extraction-design.md). This module
aliases itself to the mjollnir implementation so legacy `options_desk` import
paths keep working and share a single module state. New code should import from
`mjollnir.jax` (public API) instead.
"""
import sys as _sys
from mjollnir.processes import _jax_qe as _impl
_sys.modules[__name__] = _impl
