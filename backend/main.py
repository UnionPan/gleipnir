"""FastAPI application entry point."""
import asyncio
import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from backend.core.config import settings
from backend.core.compute_guard import Overloaded, admit_request
from backend.api.v1.api import api_router
from backend.services import warm_up
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
    """Compile the API's programs as settings.WARM_UP says.

    Uvicorn opens its port only after this startup returns, so "startup" holds
    traffic until every program is compiled.
    """
    if settings.WARM_UP == "startup":
        await asyncio.to_thread(warm_up.warm_up)
    elif settings.WARM_UP == "background":
        threading.Thread(target=warm_up.warm_up, name="warm-up", daemon=True).start()
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

def _overloaded(exc: Overloaded) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail},
                        headers={"Retry-After": str(exc.retry_after)})


# Registered before CORS so CORS wraps it (the last middleware added is the
# outermost): a 429 must carry CORS headers, or the browser hides it.
@app.middleware("http")
async def admission(request: Request, call_next):
    """A generous per-client request limit in front of everything (floods get 429)."""
    try:
        admit_request(request)
    except Overloaded as exc:
        return _overloaded(exc)
    return await call_next(request)


app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Retry-After"],  # the compute guard's 429/503 say when to retry
)

# Outermost: lab results carry whole simulated paths (~100 kB of JSON, a fifth gzipped).
app.add_middleware(GZipMiddleware, minimum_size=2048)

app.include_router(api_router, prefix=settings.API_V1_PREFIX)


@app.exception_handler(Overloaded)
async def overloaded(request: Request, exc: Overloaded) -> JSONResponse:
    """A heavy endpoint refused the job: the client's budget is spent or the instance is busy."""
    return _overloaded(exc)


@app.exception_handler(MarketDataError)
async def market_data_unavailable(request: Request, exc: MarketDataError) -> JSONResponse:
    """Upstream market data failed: a gateway error, not a bug and not the client's fault."""
    return JSONResponse(status_code=502, content={"detail": str(exc)})


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "docs": f"{settings.API_V1_PREFIX}/docs",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
