# Local setup

## Requirements

- [uv](https://docs.astral.sh/uv/getting-started/installation/) for Python (it installs Python 3.12 from `.python-version` if needed).
- Node.js 24 and npm for the web app.
- Optional: an NVIDIA GPU. `make` detects `nvidia-smi` and installs CUDA JAX; the dev servers then compute on the GPU.

## Install

```bash
make install
```

This runs `uv sync` with the `web` and `rl` extras (plus `cuda` on a GPU machine) and `npm ci --legacy-peer-deps` in `frontend/`.
mjollnir comes from the commit pinned in `uv.lock`.

## Run the app

```bash
make dev
```

The API (port 8000) reloads on changes under `backend/` and `src/`, and the web app (port 3000) hot-reloads.
The React dev server proxies `/api` to the API, so a browser only needs port 3000.
A green `[dev] ready` line appears once both servers answer; Ctrl-C stops both.
The API compiles the lab's simulations in the background at startup, so the first few seconds of lab requests can be slower.

Override the ports with `GLEIPNIR_API_PORT` and `GLEIPNIR_WEB_PORT`.

### On a remote machine

When you are connected over SSH, `make dev` prints the address and the tunnel command.
On your own machine:

```bash
ssh -N -L 3000:localhost:3000 user@host     # then open http://localhost:3000
```

If port 3000 is already taken locally, any free local port works, e.g. `-L 3300:localhost:3000` and `http://localhost:3300`.

### Co-developing mjollnir

With a mjollnir checkout next to this repository (`../mjollnir`), every `make` target runs against it as an editable install, and the API reloads when mjollnir's source changes.
Set `MJOLLNIR_DIR` to use another location.
CI and the Docker images always use the pinned commit; to move it, change `rev` in `pyproject.toml` and run `uv lock`.

## Tests and lint

```bash
make test-backend              # pytest; deselects tests that need live market data
uv run pytest -m network       # only the live-market-data tests
make test-frontend             # Jest
make lint                      # ruff (rules pinned in pyproject.toml)
cd frontend && npm run lint    # ESLint, zero warnings
```

## Troubleshooting

**`port 8000 for the API server is already in use`.**
Another process is listening; stop it or set `GLEIPNIR_API_PORT` (likewise `GLEIPNIR_WEB_PORT`).

**`ModuleNotFoundError: No module named 'mjollnir...'`.**
The pinned mjollnir is older than the code expects.
Use a local checkout (see above) or move the pin forward.

**The page loads but shows no data.**
The API is not reachable: check the `[api]` lines in the `make dev` output.
