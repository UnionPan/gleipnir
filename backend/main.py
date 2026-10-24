"""FastAPI application entry point."""
import asyncio
import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from backend.core.config import settings
from backend.api.v1.api import api_router
from backend.services import lab_service
from backend.services.market_data import MarketDataError

_app_logger = logging.getLogger("backend")
_app_logger.setLevel(logging.INFO)
if not _app_logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s:     %(name)s - %(message)s"))
    _app_logger.addHandler(_handler)
    _app_logger.propagate = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Compile the lab's simulations as settings.LAB_WARM_UP says.

    Uvicorn opens its port only after this startup returns, so "startup" holds
    traffic until every program is compiled.
    """
    if settings.LAB_WARM_UP == "startup":
        await asyncio.to_thread(lab_service.warm_up)
    elif settings.LAB_WARM_UP == "background":
        threading.Thread(target=lab_service.warm_up, name="lab-warm-up", daemon=True).start()
    yield


app = FastAPI(
    lifespan=lifespan,
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=settings.DESCRIPTION,
    openapi_url=f"{settings.API_V1_PREFIX}/openapi.json",
    docs_url=f"{settings.API_V1_PREFIX}/docs",
    redoc_url=f"{settings.API_V1_PREFIX}/redoc",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router, prefix=settings.API_V1_PREFIX)


@app.exception_handler(MarketDataError)
async def market_data_unavailable(request: Request, exc: MarketDataError) -> JSONResponse:
    """Upstream market data failed: a gateway error, not a bug and not the client's fault."""
    return JSONResponse(status_code=502, content={"detail": str(exc)})


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Options Desk API",
        "version": settings.VERSION,
        "docs": f"{settings.API_V1_PREFIX}/docs",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
