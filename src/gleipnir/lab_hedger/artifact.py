"""Trained band networks on disk: weights plus a report, the weights' SHA-256 checked on load."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from flax import serialization

from .policy import init_params


class HedgerIntegrityError(RuntimeError):
    """The weights on disk are not the ones the report describes."""


def save(net: dict, report: dict, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    weights = serialization.to_bytes(net)
    (directory / "params.msgpack").write_bytes(weights)
    (directory / "report.json").write_text(
        json.dumps({**report, "weights_sha256": hashlib.sha256(weights).hexdigest()}, indent=1) + "\n")
    return directory


def load(directory: Path) -> tuple[dict, dict]:
    report = json.loads((directory / "report.json").read_text())
    weights = (directory / "params.msgpack").read_bytes()
    if hashlib.sha256(weights).hexdigest() != report["weights_sha256"]:
        raise HedgerIntegrityError(f"{directory}: weights do not match their recorded hash")
    return serialization.from_bytes(init_params(), weights), report
