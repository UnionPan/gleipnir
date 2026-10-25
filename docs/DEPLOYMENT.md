# Deployment

Gleipnir runs as two Cloud Run services in `us-east1`, deployed by GitHub Actions on every push to `main`.

| Service | Image | Domain |
|---------|-------|--------|
| `options-desk-backend` | `deployment/docker/backend/Dockerfile` | [api.yunianpan.com](https://api.yunianpan.com/health) |
| `options-desk-frontend` | `deployment/docker/frontend/Dockerfile` | [app.yunianpan.com](https://app.yunianpan.com) |

The service names predate the rename to Gleipnir and are kept so the domain mappings stay intact.

## How a deploy happens

1. A push to `main` that touches the backend (`backend/`, `src/`, `pyproject.toml`, `uv.lock`, its Dockerfile or Cloud Build config) runs `.github/workflows/deploy-backend.yml`; frontend changes run `deploy-frontend.yml`.
2. The workflow authenticates with the `GCP_SA_KEY` secret (a service-account key) and submits `deployment/cloudbuild/<service>.yaml` to Cloud Build.
3. Cloud Build builds and pushes the image tagged with the commit SHA, then deploys it to Cloud Run.

Both workflows can also be started by hand from the Actions tab (`workflow_dispatch`).
CI (`backend-ci.yml`, `frontend-ci.yml`) runs lint and tests on pushes and pull requests and fails the build on any error.

Repository secrets: `GCP_PROJECT_ID` and `GCP_SA_KEY`.

## The backend image

Dependencies are installed exactly from `uv.lock`, so mjollnir is the commit pinned there; moving that pin redeploys the backend.
Only the `web` extra is installed (the API does not use torch).
The API compiles the Hedging Lab's simulations and the Heston chain fit before it opens its port.
That work has to happen during startup: Cloud Run gives an instance full CPU only while it starts or serves a request, and a compile left in a background thread took over 7 minutes there.
Cloud Run routes no traffic to an instance until its port is open, so during a deploy the previous revision keeps serving until the new one has compiled.

Cloud Run settings (`deployment/cloudbuild/backend.yaml`): 2 GiB memory, 2 CPUs with startup CPU boost, 300 s request timeout, port 8080, and one minimum instance so visitors never wait for a cold start.
The minimum instance is billed at Cloud Run's idle rate while no requests are in flight.

## Configuration

Backend settings are environment variables read by `backend/core/config.py`; production values live in `deployment/env/backend.env`.

| Variable | Meaning |
|----------|---------|
| `BACKEND_CORS_ORIGINS` | JSON list of allowed browser origins, e.g. `["https://app.yunianpan.com"]` |
| `DEFAULT_RISK_FREE_RATE` | Default rate for endpoints that take one |
| `WARM_UP` | When the compiled programs (lab simulations, Heston chain fit) compile: `startup` (default, before the port opens), `background` (in a thread; `make dev` uses it) or `off` (on first use) |
| `MJOLLNIR_JAX_BACKEND` | JAX device: `cpu` (default) or `cuda` |

The frontend reads `REACT_APP_API_URL` at build time from `frontend/.env.production`.

## Manual deploy

```bash
gcloud config set project "$PROJECT_ID"
gcloud builds submit --config=deployment/cloudbuild/backend.yaml \
  --substitutions=COMMIT_SHA="$(git rev-parse HEAD)" .
gcloud builds submit --config=deployment/cloudbuild/frontend.yaml \
  --substitutions=COMMIT_SHA="$(git rev-parse HEAD)" .
```

## Running the production images locally

```bash
docker compose up --build   # web app on http://localhost, API on http://localhost:8000
```

The web app bundle is built from `frontend/.env.production`, so in this setup it still calls the production API; use `make dev` for a fully local stack.

## Logs

```bash
gcloud logging read \
  'resource.type=cloud_run_revision AND resource.labels.service_name=options-desk-backend' \
  --limit 50
```
