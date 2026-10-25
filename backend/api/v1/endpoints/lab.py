"""Counterfactual Hedging Lab endpoints.

Handlers are plain ``def`` so FastAPI runs them in its threadpool: a backtest
never blocks the event loop that serves every other request.
"""

from fastapi import APIRouter, Depends, HTTPException

from backend.core.compute_guard import heavy_job
from backend.schemas.lab import HedgeRequest, HedgeResponse, LabCatalog
from backend.services import lab_service

router = APIRouter()


@router.get("/assets", response_model=LabCatalog)
def get_catalog() -> LabCatalog:
    """Shipped real-data calibrations, scenarios, and the lab's fixed menus."""
    return lab_service.catalog()


@router.post("/hedge", response_model=HedgeResponse, dependencies=[Depends(heavy_job)])
def run_hedge(request: HedgeRequest) -> HedgeResponse:
    """Run a short-option book through the factual world and, optionally, one counterfactual."""
    try:
        return lab_service.run_hedge(request)
    except lab_service.UnknownAssetError as exc:
        raise HTTPException(status_code=404, detail=f"unknown asset {exc.args[0]!r}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
