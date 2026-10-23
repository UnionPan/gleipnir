.PHONY: help install dev backend frontend docker-up docker-down test test-backend test-frontend lint format clean docs deploy-backend deploy-frontend

# Extras every local command runs with; keep one value so `uv sync` never prunes the env.
# CUDA JAX is added automatically on machines with an NVIDIA driver.
EXTRAS := --extra web --extra rl $(if $(shell command -v nvidia-smi 2>/dev/null),--extra cuda)
# Co-develop against a local mjollnir checkout when one exists next to this repo.
MJOLLNIR_DIR ?= ../mjollnir
MJOLLNIR_OVERLAY := $(if $(wildcard $(MJOLLNIR_DIR)/pyproject.toml),--with-editable $(MJOLLNIR_DIR))
UV_RUN := uv run $(EXTRAS) $(MJOLLNIR_OVERLAY)

help:
	@echo "Gleipnir - available commands"
	@echo "============================="
	@echo "install        - Install Python (uv) and frontend (npm) dependencies"
	@echo "dev            - Start API + web dev servers with live reload (Ctrl-C stops both)"
	@echo "backend        - Start only the API dev server"
	@echo "frontend       - Start only the web dev server"
	@echo "docker-up      - Start all services with Docker Compose"
	@echo "docker-down    - Stop all Docker services"
	@echo "test           - Run all tests"
	@echo "test-backend   - Run Python tests"
	@echo "test-frontend  - Run frontend tests"
	@echo "lint           - Lint Python code"
	@echo "format         - Format Python code"
	@echo "clean          - Clean build artifacts"
	@echo "docs           - Open API documentation"
	@echo ""
	@echo "Local mjollnir checkout: $(if $(MJOLLNIR_OVERLAY),$(MJOLLNIR_DIR) (editable),none found; using the pinned version)"

install:
	uv sync $(EXTRAS)
	cd frontend && npm ci --legacy-peer-deps

dev:
	$(UV_RUN) python scripts/dev.py

backend:
	$(UV_RUN) uvicorn backend.main:app --reload --reload-dir backend --reload-dir src --port 8000

frontend:
	cd frontend && npm start

docker-up:
	docker compose up --build

docker-down:
	docker compose down

test: test-backend test-frontend

test-backend:
	$(UV_RUN) pytest

test-frontend:
	cd frontend && CI=true npm test -- --watchAll=false

lint:
	uv run ruff check backend src scripts tests

format:
	uv run ruff format backend src scripts tests

clean:
	find . -type d -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .coverage htmlcov/ .pytest_cache/ src/*.egg-info
	cd frontend && rm -rf build/ node_modules/.cache/

docs:
	xdg-open http://localhost:8000/api/v1/docs 2>/dev/null || open http://localhost:8000/api/v1/docs

deploy-backend:
	gcloud builds submit --config=deployment/cloudbuild/backend.yaml

deploy-frontend:
	gcloud builds submit --config=deployment/cloudbuild/frontend.yaml
