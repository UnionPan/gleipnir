"""
Calibration service for P-measure and Q-measure models
"""
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, date
from backend.services.market_data import FETCH_ERRORS, MarketDataError, fetch_dividend_yield
from backend.services import heston_surface, path_simulation, physical_fits
from backend.services.model_vols import finite_or_none, heston_ivs

# Import P-measure calibrators

# Import Q-measure calibrators and data providers
from mjollnir.calibration.risk_neutral import (
    SABRCalibrator,
    SABRCalibrationResult,
    DupireCalibrator,
    DupireResult,
)
from mjollnir.calibration.marketdata import (
    YFinanceFetcher,
    OptionChain,
)



class CalibrationService:
    """Service for fetching data and calibrating P-measure and Q-measure models"""

    @staticmethod
    def simulate_model(
        model: str,
        parameters: dict[str, float],
        s0: float,
        n_steps: int,
        n_paths: int,
        dt: float,
        max_paths_return: int = 20,
        random_seed: int | None = None,
        risk_free_rate: float = 0.04,
        dividend_yield: float = 0.0,
        scenario_excess_drift: float = 0.05,
    ) -> dict:
        """Spot paths under the drift policy (backend.services.path_simulation)."""
        return path_simulation.simulate(
            model, parameters, s0, n_steps, n_paths, dt, max_paths_return=max_paths_return,
            random_seed=random_seed, risk_free_rate=risk_free_rate, dividend_yield=dividend_yield,
            scenario_excess_drift=scenario_excess_drift,
        )

    @staticmethod
    def fetch_ohlcv_data(tickers: list[str], start_date: str, end_date: str) -> list[dict]:
        """
        Fetch OHLCV data from yfinance

        Args:
            tickers: List of ticker symbols
            start_date: Start date string (YYYY-MM-DD)
            end_date: End date string (YYYY-MM-DD)

        Returns:
            List of OHLCV data dictionaries
        """
        results = []

        for ticker in tickers:
            try:
                # Fetch data from yfinance
                data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)

                if data.empty:
                    raise ValueError(f"No data available for {ticker}")

                # Flatten multi-index columns if necessary
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.droplevel(1)

                # Convert to dict format
                ohlcv = {
                    "ticker": ticker,
                    "dates": [d.strftime('%Y-%m-%d') for d in data.index],
                    "open": data['Open'].values.flatten().tolist(),
                    "high": data['High'].values.flatten().tolist(),
                    "low": data['Low'].values.flatten().tolist(),
                    "close": data['Close'].values.flatten().tolist(),
                    "volume": data['Volume'].values.flatten().tolist(),
                }

                results.append(ohlcv)

            except FETCH_ERRORS as e:
                raise MarketDataError(f"Failed to fetch data for {ticker}: {e!s}") from e

        return results

    @staticmethod
    def calibrate_model(
        ticker: str,
        prices: np.ndarray,
        dates: list[str],
        model: str,
        method: str,
        include_drift: bool
    ) -> tuple[dict[str, float], dict]:
        """Fit a P-measure model to daily closes (backend.services.physical_fits)."""
        return physical_fits.fit(model, method, prices, dates, include_drift)

    # ========== Q-measure (Risk-Neutral) Calibration Methods ==========

    @staticmethod
    def fetch_option_chain(
        ticker: str,
        reference_date: str | None = None,
        risk_free_rate: float = 0.05,
        expiry: str | None = None,
    ) -> dict:
        """
        Fetch option chain from yfinance

        Args:
            ticker: Ticker symbol
            reference_date: Reference date (YYYY-MM-DD), defaults to today
            risk_free_rate: Risk-free rate for pricing
            expiry: Specific expiry date (YYYY-MM-DD) or None for all

        Returns:
            Dictionary with option chain data
        """
        # Parse reference date
        if reference_date:
            ref_date = datetime.strptime(reference_date, '%Y-%m-%d').date()
        else:
            ref_date = date.today()

        dividend_yield = fetch_dividend_yield(ticker)

        # Initialize fetcher
        fetcher = YFinanceFetcher(
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
        )

        # Fetch option chain
        try:
            chain = fetcher.get_option_chain(ticker, reference_date=ref_date, expiry=expiry)
        except FETCH_ERRORS as e:
            raise MarketDataError(f"Failed to fetch the option chain for {ticker}: {e!s}") from e

        # Convert to dict format
        options_list = []
        for opt in chain.options:
            options_list.append({
                'strike': opt.strike,
                'expiry': opt.expiry.strftime('%Y-%m-%d'),
                'option_type': opt.option_type,
                'bid': opt.bid,
                'ask': opt.ask,
                'mid': opt.mid,
                'last': opt.last,
                'volume': opt.volume,
                'open_interest': opt.open_interest,
                'implied_volatility': opt.implied_volatility,
                'moneyness': opt.strike / chain.spot_price,
            })

        # Get available expiries
        expiries = sorted(set(opt.expiry.strftime('%Y-%m-%d') for opt in chain.options))

        return {
            'ticker': ticker,
            'spot_price': chain.spot_price,
            'reference_date': chain.reference_date.strftime('%Y-%m-%d'),
            'risk_free_rate': chain.risk_free_rate,
            'dividend_yield': chain.dividend_yield,
            'options': options_list,
            'expiries': expiries,
            'n_options': len(options_list),
        }


    @staticmethod
    def calibrate_qmeasure_model(
        ticker: str,
        model: str,
        reference_date: str | None = None,
        risk_free_rate: float = 0.05,
        filter_params: dict | None = None,
        calibration_method: str = 'differential_evolution',
        maxiter: int = 1000,
    ) -> tuple[dict[str, float], dict]:
        """
        Calibrate a Q-measure model to option prices

        Args:
            ticker: Ticker symbol
            model: Model name ('heston', 'sabr', 'dupire')
            reference_date: Reference date (YYYY-MM-DD)
            risk_free_rate: Risk-free rate
            filter_params: Parameters for filtering options (min_volume, min_oi, etc.)
            calibration_method: Optimization method
            maxiter: Maximum iterations

        Returns:
            Tuple of (parameters, diagnostics)
        """
        # Parse reference date
        if reference_date:
            ref_date = datetime.strptime(reference_date, '%Y-%m-%d').date()
        else:
            ref_date = date.today()

        dividend_yield = fetch_dividend_yield(ticker)

        # Initialize fetcher
        fetcher = YFinanceFetcher(
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
        )

        # Fetch option chain
        try:
            chain = fetcher.get_option_chain(ticker, reference_date=ref_date)
        except FETCH_ERRORS as e:
            raise MarketDataError(f"Failed to fetch the option chain for {ticker}: {e!s}") from e

        # Apply filters if provided
        if filter_params:
            chain = chain.filter(
                min_volume=filter_params.get('min_volume', 0),
                min_open_interest=filter_params.get('min_open_interest', 0),
                max_spread_pct=filter_params.get('max_spread_pct', 1.0),
                moneyness_range=tuple(filter_params.get('moneyness_range', [0.8, 1.2])),
            )

        if len(chain.options) == 0:
            raise ValueError(f"No valid options found for {ticker} after filtering")

        # Calibrate based on model type
        if model == 'heston':
            return heston_surface.fit_chain(chain)
        elif model == 'sabr':
            return CalibrationService._calibrate_sabr_qmeasure(
                chain=chain,
                method=calibration_method,
                dividend_yield=dividend_yield,
            )
        elif model == 'dupire':
            return CalibrationService._calibrate_dupire_qmeasure(
                chain=chain,
                dividend_yield=dividend_yield,
            )
        else:
            raise ValueError(f"Unsupported Q-measure model: {model}. Supported: 'heston', 'sabr', 'dupire'")


    @staticmethod
    def _calibrate_sabr_qmeasure(
        chain: OptionChain,
        method: str = 'L-BFGS-B',
        dividend_yield: float = 0.0,
    ) -> tuple[dict[str, float], dict]:
        """Calibrate SABR model to option chain (Q-measure)"""

        # Initialize calibrator with beta=0.5 (popular choice)
        # User can also set beta=None to calibrate it
        calibrator = SABRCalibrator(
            beta=0.5,  # Fix beta for faster calibration
            weighting='vega',
        )

        # Run calibration
        result: SABRCalibrationResult = calibrator.calibrate(
            chain=chain,
            forward=None,  # Will compute from spot and rate
            maturity=None,  # Will use median maturity
            method=method,
        )

        # Extract parameters (convert to Python types)
        parameters = {
            'alpha': float(result.alpha),
            'beta': float(result.beta),
            'rho': float(result.rho),
            'nu': float(result.nu),
        }

        # Extract diagnostics (convert to Python types)
        diagnostics = {
            'rmse_iv': float(result.rmse_iv),
            'max_error_iv': float(result.max_error_iv),
            'mean_error_iv': float(result.mean_error_iv),
            'n_iterations': int(result.n_iterations),
            'computation_time': float(result.computation_time),
            'success': bool(result.success),
            'message': str(result.message),
            'model_type': 'SABR Stochastic Volatility',
            'note': f'Beta fixed at {float(result.beta):.2f}. Calibrated to IV surface.',
            'errorMetrics': {
                'rmse_iv': float(result.rmse_iv),
                'max_error_iv': float(result.max_error_iv),
            },
        }

        return parameters, diagnostics

    @staticmethod
    def _calibrate_dupire_qmeasure(
        chain: OptionChain,
        dividend_yield: float = 0.0,
    ) -> tuple[dict[str, float], dict]:
        """Extract Dupire local volatility surface from option chain (Q-measure)"""

        # Initialize calibrator
        calibrator = DupireCalibrator(
            smoothing=True,
            smoothing_sigma=1.0,
            interpolation='rbf',
        )

        # Run extraction
        result: DupireResult = calibrator.calibrate(
            chain=chain,
            spot=None,  # Will use chain.spot_price
            rate=None,  # Will use chain.risk_free_rate
            dividend_yield=dividend_yield,
            n_strikes=50,
            n_maturities=20,
        )

        # For Dupire, we return surface statistics rather than parameters
        # since it's a non-parametric model (convert to Python types)
        parameters = {
            'min_local_vol': float(np.min(result.local_vols)),
            'max_local_vol': float(np.max(result.local_vols)),
            'mean_local_vol': float(np.mean(result.local_vols)),
            'atm_local_vol': float(result.lv_function(chain.spot_price, result.maturities[len(result.maturities)//2])),
        }

        # Extract diagnostics (convert to Python types)
        diagnostics = {
            'n_options': int(result.n_options),
            'strike_range': [float(result.min_strike), float(result.max_strike)],
            'maturity_range': [float(result.min_maturity), float(result.max_maturity)],
            'grid_size': [len(result.maturities), len(result.strikes)],
            'computation_time': float(result.computation_time),
            'smoothing_applied': bool(result.smoothing_applied),
            'model_type': 'Dupire Local Volatility',
            'note': 'Non-parametric model. Returns local vol surface σ_LV(K,T).',
            'surface_data': {
                'strikes': [float(x) for x in result.strikes.tolist()],
                'maturities': [float(x) for x in result.maturities.tolist()],
                'local_vols': [[float(x) for x in row] for row in result.local_vols.tolist()],
            },
            'errorMetrics': {},
        }

        return parameters, diagnostics


    @staticmethod
    def generate_vol_surface(
        ticker: str,
        model: str,
        parameters: dict[str, float],
        spot_price: float,
        risk_free_rate: float,
        dividend_yield: float = 0.0,
        n_strikes: int = 30,
        n_maturities: int = 20,
        strike_range: tuple[float, float] = (0.7, 1.3),
        maturity_range: tuple[float, float] = (0.05, 2.0),
    ) -> dict:
        """
        Generate volatility surface from calibrated model parameters.

        Args:
            ticker: Ticker symbol
            model: Model name ('heston', 'sabr', 'dupire')
            parameters: Calibrated model parameters
            spot_price: Current spot price
            risk_free_rate: Risk-free rate
            dividend_yield: Dividend yield
            n_strikes: Number of strike points
            n_maturities: Number of maturity points
            strike_range: Strike range as (min_moneyness, max_moneyness)
            maturity_range: Maturity range in years (min, max)

        Returns:
            Dictionary with surface data (strikes, maturities, vols)
        """
        # Generate grid
        strikes = np.linspace(
            spot_price * strike_range[0],
            spot_price * strike_range[1],
            n_strikes
        )
        maturities = np.linspace(maturity_range[0], maturity_range[1], n_maturities)

        if model == 'heston':
            return CalibrationService._generate_heston_surface(
                parameters, strikes, maturities, spot_price, risk_free_rate, dividend_yield
            )
        elif model == 'sabr':
            return CalibrationService._generate_sabr_surface(
                parameters, strikes, maturities, spot_price, risk_free_rate, dividend_yield
            )
        elif model == 'dupire':
            # Dupire surface is already computed, just return it
            return {
                'strikes': parameters.get('strikes', strikes.tolist()),
                'maturities': parameters.get('maturities', maturities.tolist()),
                'vols': parameters.get('local_vols', []),
                'surface_type': 'local_vol',
            }
        else:
            raise ValueError(f"Unsupported model: {model}")

    @staticmethod
    def _generate_heston_surface(
        params: dict[str, float],
        strikes: np.ndarray,
        maturities: np.ndarray,
        S0: float,
        r: float,
        q: float,
    ) -> dict:
        """Implied-vol surface of a Heston model (backend.services.model_vols).

        Points whose price carries no vol information are None, never a placeholder vol.
        """
        surface = [
            finite_or_none(heston_ivs(S0, strikes, float(T), r, q, v0=params['v0'], kappa=params['kappa'],
                                      theta=params['theta'], sigma_v=params['xi'], rho=params['rho']))
            for T in maturities
        ]
        return {
            'strikes': strikes.tolist(),
            'maturities': maturities.tolist(),
            'vols': surface,
            'surface_type': 'implied_vol',
        }

    @staticmethod
    def _generate_sabr_surface(
        params: dict[str, float],
        strikes: np.ndarray,
        maturities: np.ndarray,
        S0: float,
        r: float,
        q: float,
    ) -> dict:
        """Implied-vol surface of a SABR model (Hagan's formula on the forward)."""
        from mjollnir.calibration.risk_neutral import SABRCalibrator

        calibrator = SABRCalibrator()
        surface = []
        for T in maturities:
            forward = S0 * np.exp((r - q) * T)
            ivs = np.array([
                calibrator._sabr_hagan_iv(forward, K, T, params['alpha'], params['beta'],
                                          params['rho'], params['nu'])
                for K in strikes
            ])
            surface.append(finite_or_none(ivs))
        return {
            'strikes': strikes.tolist(),
            'maturities': maturities.tolist(),
            'vols': surface,
            'surface_type': 'implied_vol',
        }


# Singleton instance
calibration_service = CalibrationService()
