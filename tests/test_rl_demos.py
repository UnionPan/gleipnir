"""The Learning to Hedge artifacts are served as exported."""

import json

from fastapi.testclient import TestClient

from backend.main import app
from backend.services import rl_demos

client = TestClient(app)


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
