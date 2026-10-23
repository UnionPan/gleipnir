"""End-to-end tests for the Counterfactual Hedging Lab API."""

import pytest
from fastapi.testclient import TestClient

from backend.main import app

# No `with` block: the lifespan warm-up stays off, tests compile on demand.
client = TestClient(app)


def _hedge(**overrides):
    body = {"asset": "AAPL", "option": {"maturity_days": 21}, "seed": 3, **overrides}
    response = client.post("/api/v1/lab/hedge", json=body)
    assert response.status_code == 200, response.text
    return response.json()


def test_catalog_lists_shipped_calibrations():
    catalog = client.get("/api/v1/lab/assets").json()
    assert {a["ticker"] for a in catalog["assets"]} >= {"AAPL", "MSFT", "KO"}
    assert {s["name"] for s in catalog["scenarios"]} == {
        "vol_spike", "correlation_breakdown", "regime_shift"}
    assert catalog["maturities_days"] == [5, 21, 63]
    for asset in catalog["assets"]:
        assert set(asset["params"]) >= {"v0", "kappa", "theta", "sigma_v", "rho", "mu"}
        assert 0.05 < asset["vol_today"] < 1.5


def test_factual_world_hedging_reduces_risk():
    result = _hedge()
    assert result["counterfactual"] is None
    hedgers = result["factual"]["hedgers"]
    assert set(hedgers) == {"no_hedge", "bs_delta", "heston_delta"}
    for stats in hedgers.values():
        assert sum(stats["histogram"]) + stats["below"] + stats["above"] == result["n_paths"]
        assert stats["below"] + stats["above"] < 0.02 * result["n_paths"]
        assert stats["cvar_5"] < stats["mean"]
    assert hedgers["bs_delta"]["std"] < 0.35 * hedgers["no_hedge"]["std"]
    assert hedgers["no_hedge"]["mean_cost"] == 0.0 < hedgers["bs_delta"]["mean_cost"]
    for name, stats in hedgers.items():
        assert len(result["bin_edges"][name]) == len(stats["histogram"]) + 1
    no_hedge_span = result["bin_edges"]["no_hedge"][-1] - result["bin_edges"]["no_hedge"][0]
    bs_span = result["bin_edges"]["bs_delta"][-1] - result["bin_edges"]["bs_delta"][0]
    assert bs_span < 0.5 * no_hedge_span  # each hedger is binned at its own scale
    book = result["book"]
    assert book["premium"] > 0 and 0.05 < book["implied_vol"] < 1.5
    assert book["rho_source"] == "estimated" and book["parent_hash"] is None
    path = result["factual"]["sample_path"]
    assert len(path["spots"]) == 22 and len(path["positions"]["bs_delta"]) == 21


def test_counterfactual_world_keeps_the_book():
    calibration_hash = next(a["params_hash"] for a in client.get("/api/v1/lab/assets").json()["assets"]
                            if a["ticker"] == "AAPL")
    result = _hedge(rho=-0.7, scenario={"name": "vol_spike", "severity": 2.0})
    book, factual, shocked = result["book"], result["factual"], result["counterfactual"]
    assert book["rho_source"] == "assumed" and book["params"]["rho"] == -0.7
    assert book["parent_hash"] == calibration_hash
    assert factual["params_hash"] == book["params_hash"] != shocked["params_hash"]
    assert shocked["params"]["v0"] == pytest.approx(4.0 * factual["params"]["v0"])
    assert shocked["hedgers"]["bs_delta"]["std"] > factual["hedgers"]["bs_delta"]["std"]
    # Same random draw in both worlds: the sample paths start together and diverge.
    assert shocked["sample_path"]["spots"][0] == factual["sample_path"]["spots"][0]
    assert shocked["sample_path"]["spots"][-1] != factual["sample_path"]["spots"][-1]


@pytest.mark.parametrize(("body", "status"), [
    ({"asset": "NOPE"}, 404),
    ({"asset": "AAPL", "scenario": {"name": "vol_spike", "severity": 9.0}}, 400),
    ({"asset": "AAPL", "option": {"maturity_days": 10}}, 422),
    ({"asset": "AAPL", "rho": 0.5}, 422),
])
def test_rejects_bad_requests(body, status):
    assert client.post("/api/v1/lab/hedge", json=body).status_code == status
