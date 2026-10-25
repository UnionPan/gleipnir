"""Volatility smile endpoints."""
from fastapi import APIRouter, Depends, HTTPException

from backend.core.compute_guard import heavy_job
from backend.services.smile_service import SmileService
from backend.schemas.smile import VolSmileRequest, VolSmileComparisonResponse

router = APIRouter()


@router.post("/compare", response_model=VolSmileComparisonResponse, dependencies=[Depends(heavy_job)])
def get_volatility_smile_comparison(request: VolSmileRequest):
    """Get volatility smile with market IV vs model-calculated IVs."""
    try:
        return SmileService.get_volatility_smile_comparison(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
