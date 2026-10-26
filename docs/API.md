# API

The backend is a FastAPI app; its generated documentation is the reference and is always current:

- Swagger UI: `/api/v1/docs` ([live](https://api.yunianpan.com/api/v1/docs))
- ReDoc: `/api/v1/redoc`
- OpenAPI schema: `/api/v1/openapi.json`

This page maps the endpoints to the pages that use them.

## Endpoints

| Group | Endpoints | Used by |
|-------|-----------|---------|
| Lab | `GET /lab/assets`, `POST /lab/hedge` | Hedging Lab |
| Calibration | `POST /calibration/fetch-data`, `/calibrate`, `/fetch-option-chain`, `/calibrate-qmeasure`, `/simulate`, `/vol-surface` | Calibration |
| Market | `GET /market/overview`, `/market/charts`, `/market/ohlc/{symbol}` | Market |
| Data | `POST /data/option-chain`, `GET /data/expirations/{symbol}` | API only |
| Surface | `POST /surface/build`, `/surface/smile` | API only |
| Smile | `POST /smile/compare` | API only |
| Pricing | `POST /pricing/price`, `/pricing/greeks` | API only |

All paths are under `/api/v1`; `GET /health` reports liveness.

## Errors

| Status | Meaning |
|--------|---------|
| 400 | The request is valid JSON but cannot be served (e.g. an unsupported model or an out-of-range severity). |
| 404 | Unknown asset or symbol data. |
| 422 | Request validation failed (field types and ranges). |
| 502 | Upstream market data (Yahoo Finance) was unavailable or malformed. |
| 500 | A bug; the traceback is in the server log. |

Model-implied volatilities that are undefined at a strike (for example a deep-wing option whose price carries no volatility information) are returned as `null`, never as a placeholder value.

## Example

```python
import requests

API = "https://api.yunianpan.com/api/v1"
run = requests.post(f"{API}/lab/hedge", json={
    "asset": "AAPL",
    "model": "merton_jump",  # heston, gbm, merton_jump, regime_switching_gbm, garch, rough_bergomi
    "scenario": {"name": "jump_cascade", "severity": 4.0},
    "option": {"is_call": True, "moneyness": 1.0, "maturity_days": 21},
}, timeout=60).json()

for hedger in run["hedgers"]:  # no hedge, Black-Scholes delta, the model's own delta
    for world in ("factual", "counterfactual"):
        stats = run[world]["hedgers"][hedger["name"]]
        print(hedger["label"], world, round(stats["std"], 2), round(stats["cvar_5"], 2))
```
