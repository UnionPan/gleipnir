# Backend API image: dependencies installed exactly from uv.lock (mjollnir is a
# git dependency pinned by commit), web extra only; the API does not import torch.

FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.11.27 /uv /bin/uv
# git fetches the pinned mjollnir commit.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv

# Dependencies first, so source edits do not invalidate this layer.
COPY pyproject.toml uv.lock .python-version README.md ./
RUN uv sync --frozen --no-dev --extra web --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev --extra web --no-editable

FROM python:3.12-slim
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY backend ./backend
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1
EXPOSE 8080
CMD ["sh", "-c", "exec uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
