"""Learning to Hedge: evaluations of trained hedgers, precomputed offline."""

from fastapi import APIRouter

from backend.services import rl_demos

router = APIRouter()


@router.get("/demos")
def get_demos() -> dict[str, dict | None]:
    """The deep-hedging and AIS cliquet evaluations (null where not exported)."""
    return rl_demos.demos()
