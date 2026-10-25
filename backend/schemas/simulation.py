"""
Schemas for counterfactual simulation endpoints.

Sizes are capped (see backend/core/compute_guard.py); the horizon stops at 2
years, where the drift policy would need an equilibrium model the site does
not have yet.
"""
from pydantic import BaseModel, Field, model_validator

MAX_SIM_STEPS = 504
MAX_SIM_PATHS = 2000
MAX_SIM_HORIZON_YEARS = 2.0
MAX_PATHS_RETURNED = 50


class SimulationRequest(BaseModel):
    """Request to simulate paths from calibrated parameters."""
    model: str = Field(..., description="Model name (gbm, ou, heston, rough_bergomi, merton_jump, garch, regime_switching_gbm)")
    parameters: dict[str, float] = Field(..., description="Calibrated model parameters")
    s0: float = Field(..., gt=0, le=1e7, description="Initial spot price")
    n_steps: int = Field(default=252, ge=1, le=MAX_SIM_STEPS, description="Number of time steps")
    n_paths: int = Field(default=200, ge=1, le=MAX_SIM_PATHS, description="Number of simulated paths")
    dt: float = Field(default=1.0 / 252.0, gt=0, le=1.0 / 12, description="Time step size in years")
    max_paths_return: int = Field(default=20, ge=1, le=MAX_PATHS_RETURNED, description="Max sample paths to return")
    random_seed: int | None = Field(default=None, ge=0, le=2**32 - 1, description="Optional RNG seed")

    @model_validator(mode="after")
    def _horizon(self) -> "SimulationRequest":
        if self.n_steps * self.dt > MAX_SIM_HORIZON_YEARS + 1e-9:
            raise ValueError("the simulation horizon is limited to 2 years")
        return self


class SimulationStats(BaseModel):
    """Summary statistics for final spot distribution."""
    mean: float
    std: float
    p5: float
    p50: float
    p95: float
    min: float
    max: float


class SimulationResponse(BaseModel):
    """Simulation results and summary stats."""
    model: str
    n_steps: int
    n_paths: int
    dt: float
    time_grid: list[float]
    mean_path: list[float]
    sample_paths: list[list[float]]
    stats: SimulationStats
