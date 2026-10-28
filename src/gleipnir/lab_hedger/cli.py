"""Train a lab model's band hedger: ``python -m gleipnir.lab_hedger.cli --model heston``.

Reads the lab's shipped books of the model, trains on the configured JAX
device (MJOLLNIR_JAX_BACKEND=cuda for a GPU), validates against the model's
own delta and writes ``<out>/<model>/``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mjollnir.jax import configure_runtime

from .artifact import save
from .train import TrainConfig, load_shipped_books, train

ROOT = Path(__file__).resolve().parents[3]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="world name: gbm, heston, merton, regime_switching_gbm, garch, rough_bergomi")
    parser.add_argument("--steps", type=int, default=TrainConfig.steps)
    parser.add_argument("--lab-dir", type=Path, default=ROOT / "backend" / "data" / "lab")
    parser.add_argument("--out", type=Path, default=ROOT / "backend" / "data" / "lab" / "hedgers")
    args = parser.parse_args(argv)
    configure_runtime()
    books = load_shipped_books(args.lab_dir, args.model)
    print(f"{args.model}: {len(books)} books")
    net, report = train(args.model, books, TrainConfig(steps=args.steps))
    path = save(net, report, args.out / args.model)
    print(json.dumps(report["summary"], indent=1))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
