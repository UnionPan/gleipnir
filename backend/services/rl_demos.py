"""The Learning to Hedge page's artifacts: trained hedgers' evaluations, exported offline.

Both files are written by ``scripts/export_rl_demos.py`` from GPU training runs
and carry their own provenance (run ids, checkpoint and result-file SHA-256s).
Serving them costs nothing: nothing is trained or simulated per request.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

RL_DIR = Path(__file__).resolve().parents[1] / "data" / "rl"
ARTIFACTS = {"deep_hedging": "deep_hedging.json", "cliquet_ais": "cliquet_ais.json"}


@lru_cache(maxsize=8)
def _load(path: Path, modified_ns: int) -> dict:
    """One artifact, parsed once per version of the file on disk."""
    return json.loads(path.read_text())


def demos() -> dict[str, dict | None]:
    """Each artifact, or None where it has not been exported."""
    out: dict[str, dict | None] = {}
    for key, name in ARTIFACTS.items():
        path = RL_DIR / name
        out[key] = _load(path, path.stat().st_mtime_ns) if path.exists() else None
    return out
