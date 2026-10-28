"""End-to-end tests for the Counterfactual Hedging Lab API."""

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.services import learned_hedger
from backend.services.lab_models import LAB_MODELS

# No `with` block: the lifespan warm-up stays off, tests compile on demand.
client = TestClient(app)
MODELS = ["heston", "gbm", "merton_jump", "regime_switching_gbm", "garch", "rough_bergomi"]


def _hedge(**overrides):
    body = {"asset": "AAPL", "option": {"maturity_days": 21}, "seed": 3, **overrides}
    response = client.post("/api/v1/lab/hedge", json=body)
    assert response.status_code == 200, response.text
    return response.json()


def test_catalog_lists_every_model_for_every_shipped_asset():
    catalog = client.get("/api/v1/lab/assets").json()
    assert {a["ticker"] for a in catalog["assets"]} >= {"AAPL", "MSFT", "KO"}
    assert [m["name"] for m in catalog["models"]] == MODELS
    assert 0.0 < catalog["rate"] < 0.1 and "Treasury" in catalog["rate_source"]
    scenarios = {s["name"]: s["models"] for s in catalog["scenarios"]}
    assert scenarios["correlation_breakdown"] == ["heston", "rough_bergomi"]
    assert scenarios["jump_cascade"] == ["merton_jump"]
    for asset in catalog["assets"]:
        assert set(asset["models"]) == set(MODELS)
        assert set(asset["models"]["heston"]["params"]) == {"v0", "kappa", "theta", "sigma_v", "rho"}
        assert "mu" not in asset["models"]["gbm"]["params"]  # the drift policy: books drift at r - q
        assert 0.05 < asset["vol_today"] < 1.5 and 0.0 <= asset["dividend_yield"] < 0.1


@pytest.mark.parametrize("model", MODELS)
def test_every_model_runs_and_hedging_reduces_risk(model):
    result = _hedge(model=model)
    roles = {h["role"]: h["name"] for h in result["hedgers"]}
    learned = learned_hedger.available(LAB_MODELS[model].world)
    assert set(roles) == {"none", "bs", "model"} | ({"learned"} if learned else set()) and roles["none"] == "no_hedge"
    hedgers = result["factual"]["hedgers"]
    assert set(hedgers) == set(roles.values())
    for stats in hedgers.values():
        assert sum(stats["histogram"]) + stats["below"] + stats["above"] == result["n_paths"]
        assert stats["cvar_5"] < stats["mean"]
    assert hedgers[roles["model"]]["std"] < 0.6 * hedgers["no_hedge"]["std"], model
    assert hedgers["no_hedge"]["mean_cost"] == 0.0 < hedgers["bs_delta"]["mean_cost"]
    for name, stats in hedgers.items():
        assert len(result["bin_edges"][name]) == len(stats["histogram"]) + 1
    span = {n: e[-1] - e[0] for n, e in result["bin_edges"].items()}
    assert span["bs_delta"] < 0.5 * span["no_hedge"]  # each hedger is binned at its own scale
    book = result["book"]
    assert book["model"] == model and book["premium"] > 0 and 0.05 < book["implied_vol"] < 1.5
    assert book["rate"] > 0 and book["rho_source"] == ("estimated" if model in ("heston", "rough_bergomi") else None)
    path = result["factual"]["sample_path"]
    assert len(path["spots"]) == 22 and set(path["positions"]) == set(roles.values())


def test_books_follow_the_drift_policy():
    """At r - q, the premium is the world's own fair price: unhedged, the book's
    P&L discounted to today breaks even on average, up to Monte Carlo error."""
    catalog = client.get("/api/v1/lab/assets").json()
    nvda = next(a for a in catalog["assets"] if a["ticker"] == "NVDA")
    drift = nvda["historical_drift"]
    assert drift["ci_low"] < drift["estimate"] < drift["ci_high"]
    assert drift["ci_high"] - drift["ci_low"] > 0.4  # five years cannot pin a stock's drift down
    result = _hedge(asset="NVDA", option={"maturity_days": 63})
    no_hedge = result["factual"]["hedgers"]["no_hedge"]
    assert abs(no_hedge["mean"]) < 4 * no_hedge["std"] / result["n_paths"] ** 0.5


def test_counterfactual_world_keeps_the_book():
    calibration_hash = next(a for a in client.get("/api/v1/lab/assets").json()["assets"]
                            if a["ticker"] == "AAPL")["models"]["heston"]["params_hash"]
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


def test_jump_cascade_hurts_a_merton_book():
    result = _hedge(model="merton_jump", scenario={"name": "jump_cascade", "severity": 5.0})
    factual, shocked = result["factual"]["hedgers"], result["counterfactual"]["hedgers"]
    assert shocked["merton_delta"]["std"] > factual["merton_delta"]["std"]
    lam = result["factual"]["params"]["lambda_j"]
    assert result["counterfactual"]["params"]["lambda_j"] == pytest.approx(5.0 * lam)


@pytest.mark.parametrize(("body", "status"), [
    ({"asset": "NOPE"}, 404),
    ({"asset": "AAPL", "scenario": {"name": "vol_spike", "severity": 9.0}}, 400),
    ({"asset": "AAPL", "model": "gbm", "scenario": {"name": "correlation_breakdown", "severity": -0.9}}, 400),
    ({"asset": "AAPL", "model": "gbm", "rho": -0.7}, 400),
    ({"asset": "AAPL", "model": "ou"}, 422),
    ({"asset": "AAPL", "option": {"maturity_days": 10}}, 422),
    ({"asset": "AAPL", "rho": 0.5}, 422),
])
def test_rejects_bad_requests(body, status):
    assert client.post("/api/v1/lab/hedge", json=body).status_code == status


@pytest.mark.parametrize("model", [m for m in MODELS if learned_hedger.available(LAB_MODELS[m].world)])
def test_the_learned_band_trades_less_when_trading_costs_more(model):
    """Every trained band hedges (it removes most of the unhedged risk) and, at a high
    cost and a low risk aversion, spends less on trading than the model's own delta."""
    result = _hedge(model=model, cost_bps=30.0, risk_aversion=1.0)
    roles = {h["role"]: h["name"] for h in result["hedgers"]}
    hedgers = result["factual"]["hedgers"]
    band, delta = hedgers[roles["learned"]], hedgers[roles["model"]]
    assert band["std"] < 0.6 * hedgers["no_hedge"]["std"]
    assert band["mean_cost"] < delta["mean_cost"]
    note = next(h["note"] for h in result["hedgers"] if h["role"] == "learned")
    assert "30 bp" in note and "risk aversion 1" in note



def test_whole_paths_come_with_every_world_and_share_their_random_draws():
    result = _hedge(model="heston", scenario={"name": "vol_spike", "severity": 2.0})
    factual, shocked = result["factual"]["paths"], result["counterfactual"]["paths"]
    hedgers = {h["name"] for h in result["hedgers"]}
    assert len(factual["spots"]) == 40 and len(factual["spots"][0]) == 22
    assert set(factual["pnl"]) == set(factual["positions"]) == hedgers
    assert all(len(p) == 21 for p in factual["positions"]["bs_delta"])
    assert 0 <= factual["highlight"] < 40
    # the same draws: identical start, different worlds
    assert factual["spots"][0][0] == shocked["spots"][0][0] and factual["spots"][0][-1] != shocked["spots"][0][-1]


def test_responses_are_gzipped_when_asked():
    response = client.post("/api/v1/lab/hedge", json={"asset": "AAPL", "option": {"maturity_days": 21}, "seed": 3},
                           headers={"Accept-Encoding": "gzip"})
    assert response.status_code == 200 and response.headers.get("content-encoding") == "gzip"
