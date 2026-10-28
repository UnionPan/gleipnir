"""The Learning to Hedge artifacts are served as exported."""

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.services import rl_demos

client = TestClient(app)

ROUNDING = 2e-4  # the artifact's series are rounded to 4 decimals


@pytest.fixture(scope="module")
def deep_hedging() -> dict:
    artifact = rl_demos.demos()["deep_hedging"]
    assert artifact is not None
    return artifact


def test_the_demos_endpoint_serves_the_exported_artifacts():
    response = client.get("/api/v1/rl/demos")
    assert response.status_code == 200
    body = response.json()
    assert set(body) == set(rl_demos.ARTIFACTS)
    for key, name in rl_demos.ARTIFACTS.items():
        path = rl_demos.RL_DIR / name
        assert body[key] == (json.loads(path.read_text()) if path.exists() else None)


def test_the_cliquet_artifact_reproduces_aispgs_registered_table():
    cliquet = rl_demos.demos()["cliquet_ais"]
    assert cliquet is not None
    rmse = {k: round(m["terminal_pnl_rmse"]["mean"], 3) for k, m in cliquet["methods"].items()}
    assert rmse == {"reactive": 0.968, "sequential": 1.0, "concurrent": 0.949, "oracle_state": 0.961}
    assert round(cliquet["baselines"]["static_vanilla"]["terminal_pnl_rmse"], 3) == 1.328
    assert len(cliquet["provenance"]["results_sha256"]) == 64


def test_the_deep_hedging_artifact_has_the_chart_series(deep_hedging):
    setup, horizon = deep_hedging["setup"], deep_hedging["setup"]["horizon_days"]
    hedgers = {h["key"] for h in deep_hedging["hedgers"]}
    assert deep_hedging["schema_version"] == 2
    grid = setup["option_grid"]
    assert [(slot["tenor_days"], slot["moneyness"], slot["kind"]) for slot in grid] == [
        (tenor, moneyness, kind) for tenor in (5, 20) for moneyness in (0.95, 1.0, 1.05) for kind in ("call", "put")]
    assert isinstance(setup["initial_cash"], float)

    traders = set(deep_hedging["samples"][0]["option_holdings"])
    assert traders and traders < hedgers
    for sample in deep_hedging["samples"]:
        assert len(sample["spot"]) == len(sample["liability_value"]) == horizon + 1
        assert sample["liability_value"][0] == setup["fair_premium"]
        assert set(sample["portfolio"]) == set(sample["hedging_error"]) == hedgers
        assert all(len(account) == horizon + 1 for account in sample["portfolio"].values())
        assert set(sample["option_holdings"]) == traders
        for holdings in sample["option_holdings"].values():
            assert len(holdings) == len(grid) and all(len(slot) == horizon for slot in holdings)

    bands = deep_hedging["wealth_bands"]
    assert set(bands) == hedgers
    for band in bands.values():
        assert list(band) == ["p5", "p25", "p50", "p75", "p95"]
        quantiles = np.array(list(band.values()))
        assert quantiles.shape == (5, horizon + 1)
        assert (np.diff(quantiles, axis=0) >= 0).all()

    histogram = deep_hedging["terminal_histogram"]
    assert set(histogram) == hedgers | {"edges"}
    assert len(histogram["edges"]) == 61 and (np.diff(histogram["edges"]) > 0).all()
    for name in hedgers:
        assert len(histogram[name]["counts"]) == 60
        assert sum(histogram[name]["counts"]) == setup["eval_paths"]


def test_each_account_less_the_liability_ends_at_the_hedging_error(deep_hedging):
    setup = deep_hedging["setup"]
    strike = setup["initial_spot"]  # the liability is struck at the money
    for sample in deep_hedging["samples"]:
        value = sample["liability_value"]
        assert value[-1] == pytest.approx(max(sample["spot"][-1] - strike, 0.0), abs=1e-3)  # the payoff
        for name, account in sample["portfolio"].items():
            assert account[-1] - value[-1] == pytest.approx(sample["hedging_error"][name], abs=2 * ROUNDING)
    for name, band in deep_hedging["wealth_bands"].items():
        # every path starts from one state, so day 0's wealth is one number: the opening account less the premium
        opening = deep_hedging["samples"][0]["portfolio"][name][0] - setup["fair_premium"]
        assert all(q[0] == pytest.approx(opening, abs=2 * ROUNDING) for q in band.values())


def test_an_underlying_only_account_moves_with_its_position_less_its_costs(deep_hedging):
    """The account at each close: the previous one, plus the gain of the position
    held over the day, less the cost of that day's trade (none on the expiry day)."""
    setup = deep_hedging["setup"]
    rate = setup["costs"]["underlying"]
    traders = set(deep_hedging["samples"][0]["option_holdings"])
    for sample in deep_hedging["samples"]:
        spot = np.array(sample["spot"])
        for name in set(sample["portfolio"]) - traders:
            account, position = np.array(sample["portfolio"][name]), np.array(sample["positions"][name])
            trades = np.diff(position, prepend=0.0)
            costs = np.append(rate * np.abs(trades * spot[:-1]), 0.0)
            gains = np.concatenate([[setup["initial_cash"]], position * np.diff(spot)])
            # spots are rounded to 3 decimals, positions and accounts to 4
            np.testing.assert_allclose(np.diff(account, prepend=0.0), gains - costs, atol=1.5e-3)
