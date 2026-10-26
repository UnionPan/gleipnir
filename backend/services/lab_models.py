"""The models the Hedging Lab runs, and how a Calibration-tab fit becomes a world.

Every real-world model on the Calibration tab can run in the lab except OU:
its mean reversion is a statement about drift, and under the drift policy a
book of 3 months or less drifts at r - q, which leaves OU a plain GBM.

``world_params`` maps the API's parameter names (what the Calibration tab
shows) onto mjollnir's world names, adding the state a world starts from:
today's regime, today's GARCH variance.
"""

from __future__ import annotations

from dataclasses import dataclass

DT = 1.0 / 252


@dataclass(frozen=True)
class LabModel:
    world: str          # mjollnir world / ParamSet model
    label: str
    summary: str        # what the model adds to the hedging story


LAB_MODELS: dict[str, LabModel] = {
    "heston": LabModel("heston", "Heston",
                       "Volatility mean-reverts and moves with the price; the hedger watches the variance."),
    "gbm": LabModel("gbm", "Black-Scholes (GBM)",
                    "One constant volatility: the textbook world where delta hedging is exact in the limit."),
    "merton_jump": LabModel("merton", "Merton jump-diffusion",
                            "Diffusion plus sudden jumps that no delta hedge can follow."),
    "regime_switching_gbm": LabModel("regime_switching_gbm", "Regime-switching GBM",
                                     "Volatility jumps between calm and turbulent regimes; the hedger knows today's."),
    "garch": LabModel("garch", "GARCH(1,1)",
                      "Volatility clusters day by day; the hedger uses the GARCH forecast to expiry."),
    "rough_bergomi": LabModel("rough_bergomi", "Rough Bergomi",
                              "Rough, spiky volatility; the hedger uses today's instantaneous variance."),
}
NOT_IN_LAB = {
    "ou": "OU's mean reversion is drift; at r - q with no excess drift it is a plain GBM, so run GBM instead.",
}


def world_params(model: str, parameters: dict[str, float], derived: dict[str, float | None] | None = None) -> dict:
    """A Calibration-tab fit's parameters in mjollnir's world names (drift dropped:
    the lab follows the drift policy)."""
    derived = derived or {}
    p = parameters
    if model == "heston":
        return {"v0": p["v0"], "kappa": p["kappa"], "theta": p["theta"], "sigma_v": p["xi"], "rho": p["rho"]}
    if model == "gbm":
        return {"sigma": p["sigma"]}
    if model == "merton_jump":
        return {"sigma": p["sigma"], "lambda_j": p["lambda"], "mu_j": p["mu_j"], "sigma_j": p["sigma_j"]}
    if model == "regime_switching_gbm":
        n = sum(1 for k in p if k.startswith("sigma_regime"))
        out = {f"sigma_{k}": p[f"sigma_regime{k}"] for k in range(n)}
        out.update({f"p_{i}{j}": p[f"p_{i}{j}"] for i in range(n) for j in range(n)})
        out["regime0"] = float(derived.get("regime_today") or 0.0)
        return out
    if model == "garch":
        vol_today = derived.get("vol_today")
        if vol_today is None:
            raise ValueError("a GARCH world needs today's conditional variance (vol_today)")
        return {"omega": p["omega"], "alpha": p["alpha"], "beta": p["beta"], "h0": vol_today**2 * DT}
    if model == "rough_bergomi":
        return {"xi0": p["xi0"], "eta": p["eta"], "rho": p["rho"], "H": p["H"]}
    raise ValueError(NOT_IN_LAB.get(model, f"{model} cannot run in the lab"))
