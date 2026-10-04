"""Chaos, Volatility, and Contagion Feature Engineering.

Contains:
1. Rolling Shannon Entropy (Chaos Index): Quantifying shift unpredictability.
2. GARCH(1,1) Conditional Volatility: Heteroskedasticity and volatility clustering in labor hours.
3. Hawkes Process Intensity: Self-exciting point process modeling contagion waves
   in unscheduled absences and turnover.
4. fit_workforce_hmm_regimes: Hidden Markov Model latent regime detection.
5. Bradford Factor (Absenteeism Friction Score): S^2 * D formula measuring operational
   scheduling friction of short, frequent, unplanned absences over a configurable 52/53-week window.
"""
import json
import logging
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

try:
    import numba  # type: ignore
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False

logger = logging.getLogger(__name__)

__all__ = [
    "calculate_shannon_entropy",
    "compute_shannon_entropy",
    "rolling_shannon_entropy",
    "GARCHVolatilityModel",
    "GARCHVolatility",
    "GARCHResult",
    "GARCHModelResult",
    "fit_garch_volatility",
    "rolling_garch_volatility",
    "HawkesProcessContagion",
    "WorkforceHMMRegimes",
    "WorkforceHMM",
    "fit_workforce_hmm_regimes",
    "calculate_bradford_score",
    "compute_bradford_score",
    "classify_bradford_risk",
    "calculate_bradford_factor",
    "compute_bradford_factor",
]


def calculate_shannon_entropy(series: Union[pd.Series, np.ndarray, list], base: float = 2.0) -> float:
    """Calculate Shannon entropy of a discrete distribution: H(X) = -sum(p * log(p))."""
    if not isinstance(series, pd.Series):
        s = pd.Series(series)
    else:
        s = series
        
    counts = s.value_counts()
    if len(counts) <= 1:
        return 0.0
        
    probs = counts / counts.sum()
    log_func = np.log2 if base == 2.0 else (np.log if base == np.e else lambda x: np.log(x) / np.log(base))
    entropy = -float(np.sum(probs * log_func(probs)))
    return max(0.0, entropy)


def _rolling_entropy_core(
    codes: np.ndarray,
    n: int,
    k_uniques: int,
    window: int,
    min_p: int,
    f_table: np.ndarray,
    log_table: np.ndarray,
    normalize: bool,
    base: float,
) -> np.ndarray:
    out = np.full(n, np.nan, dtype=np.float64)
    counts = np.zeros(k_uniques, dtype=np.int64)
    n_valid = 0
    active_unique = 0
    sum_f = 0.0
    log_base = np.log(base)

    for i in range(n):
        c_in = codes[i]
        if c_in >= 0:
            old_cnt = counts[c_in]
            new_cnt = old_cnt + 1
            counts[c_in] = new_cnt
            sum_f += f_table[new_cnt] - f_table[old_cnt]
            if old_cnt == 0:
                active_unique += 1
            n_valid += 1

        if i >= window:
            c_out = codes[i - window]
            if c_out >= 0:
                old_cnt = counts[c_out]
                new_cnt = old_cnt - 1
                counts[c_out] = new_cnt
                sum_f += f_table[new_cnt] - f_table[old_cnt]
                if new_cnt == 0:
                    active_unique -= 1
                n_valid -= 1

        chunk_len = i + 1 if i + 1 < window else window
        if chunk_len >= min_p:
            if n_valid <= 1 or active_unique <= 1:
                out[i] = 0.0
            else:
                h = log_table[n_valid] - (sum_f / n_valid)
                if h < 0.0:
                    h = 0.0
                if normalize:
                    k = active_unique if active_unique > 2 else 2
                    max_h = np.log2(k) if base == 2.0 else np.log(k) / log_base
                    out[i] = h / max_h if max_h > 0 else 0.0
                else:
                    out[i] = h
    return out


if _NUMBA_AVAILABLE:
    _rolling_entropy_fast = numba.njit(fastmath=True)(_rolling_entropy_core)
else:
    _rolling_entropy_fast = _rolling_entropy_core


def rolling_shannon_entropy(
    df: pd.DataFrame,
    val_col: str,
    window: int = 14,
    group_col: Optional[str] = None,
    base: float = 2.0,
    normalize: bool = False,
) -> pd.Series:
    """Compute rolling Shannon entropy (Chaos Index) over a moving window of categorical/binned shifts.
    
    Optimized O(N) sliding window frequency tracker with Numba JIT acceleration
    when available, scaling effortlessly to 100K+ rows without re-slicing DataFrames.
    
    Handles strings, categories, and numerical shift types robustly.
    If normalize=True, scales H(X) by log(K) so result lies in [0, 1].
    """
    if val_col not in df.columns:
        raise KeyError(f"Column '{val_col}' not found in DataFrame.")
    if group_col is not None and group_col not in df.columns:
        raise KeyError(f"Column '{group_col}' not found in DataFrame.")
    if window < 1:
        raise ValueError(f"window must be a positive integer >= 1, got {window}")

    log_base = np.log(base)
    f_table = np.zeros(window + 1, dtype=np.float64)
    log_table = np.zeros(window + 1, dtype=np.float64)
    for x in range(1, window + 1):
        f_table[x] = x * (np.log(x) / log_base)
        log_table[x] = np.log(x) / log_base

    min_p = max(2, window // 2)

    def _rolling_series_entropy(s: pd.Series) -> pd.Series:
        codes, uniques = pd.factorize(s)
        n = len(codes)
        k_uniques = len(uniques)
        if k_uniques == 0 or n == 0:
            return pd.Series(np.full(n, np.nan, dtype=np.float64), index=s.index)

        arr = np.ascontiguousarray(codes, dtype=np.int64)
        out = _rolling_entropy_fast(
            arr, n, k_uniques, window, min_p, f_table, log_table, normalize, float(base)
        )
        return pd.Series(out, index=s.index)

    if group_col is not None:
        return df.groupby(group_col, group_keys=False)[val_col].apply(_rolling_series_entropy)
    else:
        return _rolling_series_entropy(df[val_col])


class GARCHVolatilityModel:
    """GARCH(p, q) estimator for dynamic conditional volatility and heteroskedasticity.

    Models time-varying volatility clustering in workforce metrics (e.g., weekly overtime
    hours, unscheduled call-outs, staffing fluctuations).

    Parameters
    ----------
    p : int, default=1
        GARCH autoregressive volatility lag order.
    q : int, default=1
        ARCH shock lag order.
    mean : str, default='Constant'
        Mean model specification ('Constant', 'Zero', or 'AR').
    rescale : bool, default=True
        Whether to automatically rescale the series for numerical stability during optimization.
    """

    def __init__(
        self,
        p: int = 1,
        q: int = 1,
        mean: str = "Constant",
        rescale: bool = True,
    ):
        self.p = p
        self.q = q
        self.mean = mean
        self.rescale = rescale
        self.model_ = None
        self.res_ = None
        self.scale_factor_ = 1.0
        self.mean_val_ = 0.0
        self.conditional_volatility_ = None
        self.aic_ = None
        self.bic_ = None
        self.params_ = None
        self.summary_ = None
        self.is_fitted_ = False

    def fit(self, series: Union[pd.Series, np.ndarray, list], y: Any = None) -> "GARCHVolatilityModel":
        """Fit GARCH(p, q) model to a time series."""
        try:
            from arch import arch_model
        except ImportError:
            raise ImportError(
                "GARCHVolatilityModel requires 'arch'. "
                "Install with: pip install 'people-analytics-toolkit[timeseries]'"
            )

        if not isinstance(series, pd.Series):
            s = pd.Series(series)
        else:
            s = series

        self.mean_val_ = float(s.mean())
        detrended = s - self.mean_val_

        model = arch_model(
            detrended,
            vol="Garch",
            p=self.p,
            q=self.q,
            mean=self.mean,
            rescale=self.rescale,
        )
        res = model.fit(disp="off")

        self.model_ = model
        self.res_ = res
        self.scale_factor_ = getattr(res, "scale", 1.0)
        cond_vol = res.conditional_volatility / self.scale_factor_
        self.conditional_volatility_ = pd.Series(
            cond_vol,
            index=s.index,
            name="garch_conditional_volatility",
        )
        self.aic_ = float(res.aic)
        self.bic_ = float(res.bic)
        self.params_ = res.params.to_dict()
        self.summary_ = str(res.summary())
        self.is_fitted_ = True
        return self

    def transform(self, series: Optional[Union[pd.Series, np.ndarray, list]] = None) -> pd.Series:
        """Return conditional volatility series."""
        if not self.is_fitted_:
            raise ValueError("GARCHVolatilityModel has not been fitted yet. Call .fit() first.")
        if series is None:
            return self.conditional_volatility_
        if isinstance(series, pd.Series) and series.equals(self.conditional_volatility_):
            return self.conditional_volatility_
        return GARCHVolatilityModel(p=self.p, q=self.q, mean=self.mean, rescale=self.rescale).fit(series).conditional_volatility_

    def fit_transform(self, series: Union[pd.Series, np.ndarray, list], y: Any = None) -> pd.Series:
        """Fit model and return conditional volatility."""
        return self.fit(series).transform()

    def predict(self, horizon: int = 1) -> pd.DataFrame:
        """Forecast conditional variance and volatility over a forward horizon."""
        if not self.is_fitted_:
            raise ValueError("GARCHVolatilityModel has not been fitted yet. Call .fit() first.")
        forecast_res = self.res_.forecast(horizon=horizon)
        var_forecast = forecast_res.variance.iloc[-1] / (self.scale_factor_ ** 2)
        vol_forecast = np.sqrt(var_forecast)
        return pd.DataFrame({
            "variance": var_forecast,
            "volatility": vol_forecast,
        })


GARCHVolatility = GARCHVolatilityModel


@dataclass
class GARCHResult:
    """Structured result object for GARCH volatility estimation."""
    summary: str
    aic: float
    bic: float
    params: Dict[str, float]
    arch_result: Any
    model: Any

    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def keys(self) -> List[str]:
        return [f.name for f in fields(self)]

    def values(self) -> List[Any]:
        return [getattr(self, k) for k in self.keys()]

    def items(self) -> List[Tuple[str, Any]]:
        return [(k, getattr(self, k)) for k in self.keys()]

    def to_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self.keys()}

    def __iter__(self):
        return iter(self.keys())

    def __len__(self) -> int:
        return len(self.keys())

    @property
    def log_likelihood(self) -> Optional[float]:
        """Log-likelihood of fitted GARCH model."""
        if hasattr(self.arch_result, "loglikelihood"):
            return float(self.arch_result.loglikelihood)
        return None

    @property
    def p(self) -> Optional[int]:
        """GARCH lag order p."""
        if self.model is not None and hasattr(self.model, "p"):
            return self.model.p
        return None

    @property
    def q(self) -> Optional[int]:
        """ARCH lag order q."""
        if self.model is not None and hasattr(self.model, "q"):
            return self.model.q
        return None

    @property
    def omega(self) -> Optional[float]:
        """Baseline variance constant omega."""
        return self.params.get("omega") if self.params else None

    @property
    def alpha(self) -> Optional[float]:
        """ARCH coefficient alpha[1]."""
        if not self.params:
            return None
        return self.params.get("alpha[1]", self.params.get("alpha"))

    @property
    def beta(self) -> Optional[float]:
        """GARCH persistence coefficient beta[1]."""
        if not self.params:
            return None
        return self.params.get("beta[1]", self.params.get("beta"))

    @property
    def converged(self) -> bool:
        """True if optimizer successfully converged."""
        if hasattr(self.arch_result, "convergence_flag"):
            return self.arch_result.convergence_flag == 0
        return True


GARCHModelResult = GARCHResult


def fit_garch_volatility(
    series: pd.Series,
    p: int = 1,
    q: int = 1,
    mean: str = "Constant",
) -> Tuple[pd.Series, GARCHResult]:
    """Fit GARCH(p, q) model to estimate dynamic conditional volatility (heteroskedasticity).
    
    Formula:
        sigma_t^2 = omega + alpha * epsilon_{t-1}^2 + beta * sigma_{t-1}^2
    """
    model = GARCHVolatilityModel(p=p, q=q, mean=mean).fit(series)
    meta = GARCHResult(
        summary=model.summary_,
        aic=model.aic_,
        bic=model.bic_,
        params=model.params_,
        arch_result=model.res_,
        model=model,
    )
    return model.conditional_volatility_, meta


def rolling_garch_volatility(
    data: Union[pd.DataFrame, pd.Series, np.ndarray, list],
    val_col: Optional[str] = None,
    window: Optional[int] = None,
    min_periods: Optional[int] = None,
    step: int = 1,
    group_col: Optional[str] = None,
    p: int = 1,
    q: int = 1,
    mean: str = "Constant",
    annualize: bool = False,
    periods_per_year: float = 52.0,
) -> pd.Series:
    r"""Compute rolling/windowed GARCH(p, q) conditional volatility across time series or group cohorts.

    Extends single-series `fit_garch_volatility` into a flexible rolling-window
    and DataFrame-level estimator. Evaluates dynamic time-varying heteroskedasticity
    and volatility clustering in workforce metrics (e.g. weekly overtime hours,
    unscheduled call-out rates, staffing volume fluctuations).

    Mathematical Formulation:
        For each rolling window :math:`[t - W + 1, t]`:
        .. math::
            \sigma_t^2 = \omega + \sum_{i=1}^p \alpha_i \epsilon_{t-i}^2 + \sum_{j=1}^q \beta_j \sigma_{t-j}^2

    Parameters
    ----------
    data : pd.DataFrame, pd.Series, np.ndarray, or list
        Input data containing the time-series values to model.
    val_col : str, optional
        Column name in `data` containing the numeric series when `data` is a DataFrame.
    window : int, optional
        Rolling window size (e.g. 30 or 52). If None, fits GARCH over each group's
        entire historical trajectory.
    min_periods : int, optional
        Minimum number of non-null observations required before estimating GARCH.
        Defaults to `max(15, window // 2)`. Points prior to `min_periods` are NaN.
    step : int, default=1
        Refitting step stride. When `step > 1`, GARCH parameters are re-estimated
        every `step` observations and forward-propagated to accelerate execution on long series.
    group_col : str, optional
        Grouping column name (e.g. 'store_id', 'department', 'role') for cohort-wise
        volatility estimation.
    p : int, default=1
        GARCH autoregressive volatility lag order.
    q : int, default=1
        ARCH shock lag order.
    mean : str, default='Constant'
        Mean model specification ('Constant', 'Zero', or 'AR').
    annualize : bool, default=False
        Whether to scale conditional volatility by :math:`\sqrt{\text{periods\_per\_year}}`.
    periods_per_year : float, default=52.0
        Annualization multiplier factor (e.g. 52.0 for weekly data, 252.0 for business daily).

    Returns
    -------
    pd.Series
        Rolling GARCH conditional volatility indexed identically to input data.
    """
    try:
        from arch import arch_model
    except ImportError:
        raise ImportError(
            "rolling_garch_volatility requires 'arch'. "
            "Install with: pip install 'people-analytics-toolkit[timeseries]'"
        )

    def _rolling_series_garch(s: pd.Series) -> pd.Series:
        s_clean = pd.to_numeric(s, errors="coerce")
        n = len(s_clean)
        out = np.full(n, np.nan)
        w = window if window is not None and window > 0 else n
        m_p = min_periods if min_periods is not None else (min(w, max(15, w // 2)) if w < n else max(15, p + q + 2))

        # Full-series GARCH fit when window is None or window >= n
        if window is None or w >= n:
            valid_mask = s_clean.notna()
            valid_sub = s_clean[valid_mask]
            if len(valid_sub) >= max(10, p + q + 2):
                try:
                    vol, _ = fit_garch_volatility(valid_sub, p=p, q=q, mean=mean)
                    res_vals = vol.to_numpy()
                    if annualize:
                        res_vals = res_vals * np.sqrt(periods_per_year)
                    out_series = pd.Series(np.nan, index=s.index, name="rolling_garch_volatility")
                    out_series.loc[valid_sub.index] = res_vals
                    return out_series
                except Exception as e:
                    logger.warning("GARCH optimization failed on full series: %s. Falling back to sample std.", e)
                    std_val = float(valid_sub.std(ddof=1)) if len(valid_sub) > 1 else 0.0
                    if annualize:
                        std_val *= np.sqrt(periods_per_year)
                    return pd.Series(std_val, index=s.index, name="rolling_garch_volatility")
            else:
                std_val = float(valid_sub.std(ddof=1)) if len(valid_sub) > 1 else np.nan
                if annualize and not np.isnan(std_val):
                    std_val *= np.sqrt(periods_per_year)
                return pd.Series(std_val, index=s.index, name="rolling_garch_volatility")

        # Rolling window fitting
        last_vol = np.nan
        for i in range(n):
            start = max(0, i - w + 1)
            chunk = s_clean.iloc[start : i + 1].dropna()

            if len(chunk) < m_p:
                continue

            should_refit = (step <= 1) or ((i - m_p + 1) % step == 0) or np.isnan(last_vol)
            if should_refit:
                chunk_std = float(chunk.std(ddof=1)) if len(chunk) > 1 else 0.0
                if chunk_std <= 1e-6:
                    last_vol = chunk_std
                else:
                    try:
                        vol_chunk, _ = fit_garch_volatility(chunk, p=p, q=q, mean=mean)
                        last_vol = float(vol_chunk.iloc[-1])
                    except Exception:
                        last_vol = chunk_std

            out[i] = last_vol * np.sqrt(periods_per_year) if annualize else last_vol

        return pd.Series(out, index=s.index, name="rolling_garch_volatility")

    if isinstance(data, pd.DataFrame):
        df = data
        if val_col is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                raise ValueError("DataFrame contains no numeric columns to estimate volatility.")
            col = numeric_cols[0]
        else:
            if val_col not in df.columns:
                raise ValueError(f"Column '{val_col}' not found in DataFrame.")
            col = val_col

        if group_col is not None:
            if group_col not in df.columns:
                raise ValueError(f"Group column '{group_col}' not found in DataFrame.")
            return df.groupby(group_col, group_keys=False)[col].apply(_rolling_series_garch)
        else:
            return _rolling_series_garch(df[col])
    else:
        if isinstance(data, pd.Series):
            s = data
        else:
            s = pd.Series(data)
        return _rolling_series_garch(s)


class HawkesProcessContagion:
    """Univariate Hawkes Self-Exciting Point Process with exponential decay kernel.
    
    Intensity equation:
        lambda(t) = mu + sum_{t_i < t} alpha * exp(-beta * (t - t_i))
        
    Where:
        mu = baseline exogenous event intensity (background rate)
        alpha = self-excitation jump size (contagion magnitude per event)
        beta = exponential recovery rate (speed of mean reversion)
        Branching ratio n = alpha / beta (must be < 1 for stationarity)
    """
    
    def __init__(self, mu: float = 0.2, alpha: float = 0.5, beta: float = 1.0):
        if mu <= 0:
            raise ValueError(f"Baseline intensity mu must be positive, got {mu}")
        if alpha < 0:
            raise ValueError(f"Self-excitation jump alpha must be non-negative, got {alpha}")
        if beta <= 0:
            raise ValueError(f"Recovery rate beta must be positive, got {beta}")
        self.mu = float(mu)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.events_: List[float] = []
        self.is_fitted_: bool = False
        self.converged_: bool = False
        
    def fit(self, events: List[float], max_t: Optional[float] = None):
        """Fit Hawkes process parameters (mu, alpha, beta) via Maximum Likelihood Estimation (MLE)."""
        ev = np.sort(np.asarray(events, dtype=float))
        self.events_ = ev.tolist()
        if max_t is not None and max_t <= 0:
            raise ValueError(f"max_t must be positive, got {max_t}")
        if max_t is not None and len(ev) > 0 and max_t < ev[-1]:
            raise ValueError(f"max_t ({max_t}) cannot be less than latest event timestamp ({ev[-1]}).")
        T = max_t if max_t is not None else (ev[-1] + 1.0 if len(ev) > 0 else 1.0)
        n = len(ev)
        
        if n < 3:
            msg = (
                f"HawkesProcessContagion requires at least 3 events for reliable MLE fitting, "
                f"got {n}. Retaining default/initial parameters (mu={self.mu}, alpha={self.alpha}, beta={self.beta})."
            )
            logger.warning(msg)
            warnings.warn(msg, UserWarning, stacklevel=2)
            self.is_fitted_ = False
            self.converged_ = False
            return self
            
        def neg_log_likelihood(params):
            mu_val, alpha_val, beta_val = params
            if mu_val <= 1e-4 or alpha_val < 0.0 or beta_val <= 1e-4 or alpha_val >= beta_val:
                return 1e10
                
            # Log-intensity sum
            log_lambda_sum = 0.0
            r_matrix = np.zeros(n)
            for i in range(1, n):
                r_matrix[i] = np.exp(-beta_val * (ev[i] - ev[i - 1])) * (1.0 + r_matrix[i - 1])
                
            intensities = mu_val + alpha_val * r_matrix
            if np.any(intensities <= 0):
                return 1e10
                
            log_lambda_sum = np.sum(np.log(intensities))
            
            # Integrated intensity compensation: mu * T + sum(alpha/beta * (1 - exp(-beta * (T - t_i))))
            integral = mu_val * T + (alpha_val / beta_val) * np.sum(1.0 - np.exp(-beta_val * (T - ev)))
            
            return -(log_lambda_sum - integral)
            
        init_guess = [max(0.1, n / T * 0.4), 0.4, 1.0]
        bounds = [(1e-3, 5.0), (1e-3, 5.0), (1e-2, 10.0)]
        res = minimize(neg_log_likelihood, init_guess, bounds=bounds, method="L-BFGS-B")
        
        self.is_fitted_ = True
        if res.success:
            self.mu, self.alpha, self.beta = float(res.x[0]), float(res.x[1]), float(res.x[2])
            self.converged_ = True
        else:
            self.converged_ = False
            msg = (
                f"HawkesProcessContagion MLE optimization failed to converge: {res.message}. "
                f"Retaining initial/current parameters (mu={self.mu}, alpha={self.alpha}, beta={self.beta})."
            )
            logger.warning(msg)
            warnings.warn(msg, UserWarning, stacklevel=2)
            
        return self
        
    def predict_intensity(self, time_points: Union[np.ndarray, List[float]]) -> np.ndarray:
        """Evaluate Hawkes intensity lambda(t) at arbitrary query timestamps."""
        t_arr = np.asarray(time_points, dtype=float)
        ev = np.asarray(self.events_, dtype=float)
        intensities = np.full_like(t_arr, self.mu)
        
        for idx, t in enumerate(t_arr):
            past = ev[ev < t]
            if len(past) > 0:
                jump_sum = np.sum(self.alpha * np.exp(-self.beta * (t - past)))
                intensities[idx] += jump_sum
                
        return intensities

    @property
    def branching_ratio(self) -> float:
        """Branching ratio n = alpha / beta.
        
        Represents the expected number of second-generation events directly triggered
        by a single event. For the Hawkes process to be stationary (subcritical/stable),
        the branching ratio must satisfy n < 1. If n >= 1, the contagion cascade is
        supercritical (explosive / runaway cascade).
        """
        if self.beta <= 0:
            return float("inf")
        return float(self.alpha / self.beta)

    @property
    def is_stationary(self) -> bool:
        """Check whether the Hawkes process is stationary (subcritical, branching_ratio < 1.0)."""
        return self.branching_ratio < 1.0

    def simulate(
        self,
        T: float,
        max_events: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Generate synthetic event sequences using Ogata's modified thinning algorithm.
        
        Parameters
        ----------
        T : float
            Time horizon length for the simulation (must be positive).
        max_events : Optional[int], default=None
            Maximum number of events to generate before terminating early.
        seed : Optional[int], default=None
            Random seed for reproducible point process simulation.
            
        Returns
        -------
        np.ndarray
            Array of simulated event timestamps strictly within [0, T].
        """
        if T <= 0:
            raise ValueError(f"Simulation horizon T must be positive, got {T}")
        if max_events is not None and max_events <= 0:
            raise ValueError(f"max_events must be positive, got {max_events}")
            
        rng = np.random.default_rng(seed)
        t = 0.0
        events: List[float] = []
        
        # Upper bound safeguard to prevent infinite loop if supercritical (alpha >= beta)
        safety_cap = max_events if max_events is not None else 100000
        
        while t < T and len(events) < safety_cap:
            if len(events) == 0:
                lam_star = self.mu
            else:
                ev_arr = np.asarray(events, dtype=float)
                lam_star = self.mu + float(np.sum(self.alpha * np.exp(-self.beta * (t - ev_arr))))
                
            if lam_star <= 0:
                break
                
            u1 = rng.uniform(0.0, 1.0)
            w = -np.log(u1) / lam_star
            s = t + w
            if s > T:
                break
                
            # Evaluate true intensity at candidate time s
            if len(events) == 0:
                lam_s = self.mu
            else:
                ev_arr = np.asarray(events, dtype=float)
                lam_s = self.mu + float(np.sum(self.alpha * np.exp(-self.beta * (s - ev_arr))))
                
            u2 = rng.uniform(0.0, 1.0)
            if u2 <= (lam_s / lam_star):
                events.append(s)
                t = s
            else:
                t = s
                
        return np.asarray(events, dtype=float)

    def goodness_of_fit(
        self,
        events: Optional[Union[List[float], np.ndarray]] = None,
        max_t: Optional[float] = None,
    ) -> dict:
        """Evaluate model goodness-of-fit via the Random Time Change Theorem.
        
        Transforms event times via the compensator (integrated intensity) Lambda(t_k).
        Under the true Hawkes model, the transformed inter-arrival durations:
            tau_k = Lambda(t_k) - Lambda(t_{k-1})
        are independent standard Exponential(1) random variables, and
            U_k = 1 - exp(-tau_k)
        are independent Uniform(0, 1) random variables.
        
        A Kolmogorov-Smirnov test is performed against Exponential(1).
        
        Parameters
        ----------
        events : Optional[Union[List[float], np.ndarray]], default=None
            Event sequence to evaluate. If None, uses the events from fit() (`self.events_`).
        max_t : Optional[float], default=None
            Observation window end time T. Defaults to latest event timestamp + 1.0.
            
        Returns
        -------
        dict
            Dictionary containing:
            - 'ks_stat': Kolmogorov-Smirnov test statistic against Exp(1)
            - 'ks_pvalue': p-value of the KS test (p > 0.05 indicates good fit)
            - 'is_good_fit': boolean (True if ks_pvalue > 0.05)
            - 'tau': numpy array of transformed inter-arrival durations
            - 'transformed_uniforms': numpy array of transformed U_k ~ Unif(0, 1)
            - 'log_likelihood': log-likelihood of the events under the model
            - 'aic': Akaike Information Criterion (2*k - 2*logL, k=3)
            - 'bic': Bayesian Information Criterion (k*ln(n) - 2*logL, k=3)
            - 'branching_ratio': alpha / beta
            - 'is_stationary': True if branching_ratio < 1.0
            - 'lag1_autocorr': lag-1 autocorrelation of tau (should be near 0 for i.i.d.)
        """
        if events is None:
            if not self.events_:
                raise ValueError("No events provided and HawkesProcessContagion has not been fitted yet.")
            ev = np.sort(np.asarray(self.events_, dtype=float))
        else:
            ev = np.sort(np.asarray(events, dtype=float))
            
        n = len(ev)
        if n == 0:
            raise ValueError("Event sequence cannot be empty.")
            
        T = max_t if max_t is not None else (ev[-1] + 1.0)
        if T <= 0:
            raise ValueError(f"max_t must be positive, got {T}")
            
        # Compute compensator Lambda(t_k)
        Lambda = np.zeros(n, dtype=float)
        for k in range(n):
            t_k = ev[k]
            past = ev[:k]
            Lambda[k] = self.mu * t_k + float(np.sum((self.alpha / self.beta) * (1.0 - np.exp(-self.beta * (t_k - past)))))
            
        tau = np.diff(np.insert(Lambda, 0, 0.0))
        u = 1.0 - np.exp(-tau)
        
        # KS test
        if n >= 3:
            ks_res = stats.kstest(tau, "expon")
            ks_stat = float(ks_res.statistic)
            ks_pvalue = float(ks_res.pvalue)
            is_good_fit = bool(ks_pvalue > 0.05)
        else:
            ks_stat = float("nan")
            ks_pvalue = float("nan")
            is_good_fit = False
            
        # Autocorrelation of tau
        if n >= 3:
            diff = tau - np.mean(tau)
            denom = np.sum(diff ** 2)
            lag1_autocorr = float(np.sum(diff[:-1] * diff[1:]) / denom) if denom > 0 else 0.0
        else:
            lag1_autocorr = float("nan")
            
        # Log likelihood
        r_matrix = np.zeros(n, dtype=float)
        for i in range(1, n):
            r_matrix[i] = np.exp(-self.beta * (ev[i] - ev[i - 1])) * (1.0 + r_matrix[i - 1])
        intensities = self.mu + self.alpha * r_matrix
        intensities = np.maximum(intensities, 1e-12)
        integral = self.mu * T + (self.alpha / self.beta) * float(np.sum(1.0 - np.exp(-self.beta * (T - ev))))
        log_l = float(np.sum(np.log(intensities)) - integral)
        
        k_params = 3
        aic = float(2 * k_params - 2 * log_l)
        bic = float(k_params * np.log(n) - 2 * log_l) if n > 0 else float("nan")
        
        return {
            "ks_stat": ks_stat,
            "ks_pvalue": ks_pvalue,
            "is_good_fit": is_good_fit,
            "tau": tau,
            "transformed_uniforms": u,
            "log_likelihood": log_l,
            "aic": aic,
            "bic": bic,
            "branching_ratio": self.branching_ratio,
            "is_stationary": self.is_stationary,
            "lag1_autocorr": lag1_autocorr,
        }

    def to_dict(self) -> dict:
        """Serialize model configuration and fitted state to a dictionary."""
        return {
            "params": {
                "mu": float(self.mu),
                "alpha": float(self.alpha),
                "beta": float(self.beta),
            },
            "fitted": {
                "is_fitted_": bool(self.is_fitted_),
                "converged_": bool(self.converged_),
                "events_": [float(x) for x in self.events_],
                "branching_ratio": float(self.branching_ratio),
                "is_stationary": bool(self.is_stationary),
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "HawkesProcessContagion":
        """Reconstruct a HawkesProcessContagion instance from a dictionary."""
        params = data.get("params", {})
        instance = cls(**params)
        fitted = data.get("fitted", {})
        instance.is_fitted_ = fitted.get("is_fitted_", False)
        instance.converged_ = fitted.get("converged_", False)
        instance.events_ = fitted.get("events_", [])
        return instance

    def to_json(
        self,
        filepath_or_buffer: Optional[Union[str, Any]] = None,
        indent: int = 2,
    ) -> Optional[str]:
        """Serialize model to JSON string or write to file/buffer."""
        data = self.to_dict()
        if filepath_or_buffer is None:
            return json.dumps(data, indent=indent)
        elif isinstance(filepath_or_buffer, str):
            with open(filepath_or_buffer, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=indent)
            return None
        else:
            json.dump(data, filepath_or_buffer, indent=indent)
            return None

    @classmethod
    def from_json(cls, json_str_or_filepath: Union[str, Any]) -> "HawkesProcessContagion":
        """Restore a HawkesProcessContagion instance from a JSON string or file path."""
        if isinstance(json_str_or_filepath, str):
            if json_str_or_filepath.strip().startswith("{"):
                data = json.loads(json_str_or_filepath)
            else:
                with open(json_str_or_filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
        else:
            data = json.load(json_str_or_filepath)
        return cls.from_dict(data)


class WorkforceHMMRegimes:
    """Gaussian Hidden Markov Model (HMM) estimator for workforce operational regimes.

    Infers latent operational states (e.g., standard baseline, seasonal surge,
    attrition crisis) across multivariate temporal workforce metrics.

    Parameters
    ----------
    n_components : int, default=3
        Number of hidden latent regimes to infer.
    covariance_type : str, default='diag'
        Covariance matrix structure ('diag', 'full', 'spherical', 'tied').
    n_iter : int, default=150
        Maximum EM iterations.
    seed : int, default=42
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_components: int = 3,
        covariance_type: str = "diag",
        n_iter: int = 150,
        seed: int = 42,
    ):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.seed = seed
        self.emission_cols_ = None
        self.model_ = None
        self.state_order_ = None
        self.means_ = None
        self.transmat_ = None
        self.regime_profile_ = None
        self.is_fitted_ = False

    def _extract_array(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        emission_cols: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        if isinstance(X, pd.DataFrame):
            cols = emission_cols if emission_cols is not None else (
                self.emission_cols_ if self.emission_cols_ is not None else X.select_dtypes(include=[np.number]).columns.tolist()
            )
            missing = [c for c in cols if c not in X.columns]
            if missing:
                raise KeyError(f"Emission columns not found in DataFrame: {missing}")
            return X[cols].to_numpy(dtype=float), cols
        else:
            arr = np.asarray(X, dtype=float)
            cols = emission_cols or [f"feature_{i}" for i in range(arr.shape[1] if arr.ndim > 1 else 1)]
            return (arr.reshape(-1, 1) if arr.ndim == 1 else arr), cols

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        emission_cols: Optional[List[str]] = None,
        y: Any = None,
    ) -> "WorkforceHMMRegimes":
        """Fit Gaussian HMM to multivariate temporal observations."""
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError:
            raise ImportError(
                "WorkforceHMMRegimes requires 'hmmlearn'. "
                "Install with: pip install 'people-analytics-toolkit[timeseries]'"
            )

        X_mat, cols = self._extract_array(X, emission_cols=emission_cols)
        self.emission_cols_ = cols

        model = GaussianHMM(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.seed,
        )
        model.fit(X_mat)

        # Sort states by first emission mean (e.g. weekly_overtime_hours) so State 0 is always lowest intensity
        state_order = np.argsort(model.means_[:, 0])
        self.model_ = model
        self.state_order_ = state_order
        self.means_ = model.means_[state_order]
        self.transmat_ = model.transmat_[state_order, :][:, state_order]

        # Posterior state probabilities
        posteriors = model.predict_proba(X_mat)[:, state_order]
        decoded_states = np.argmax(posteriors, axis=1)

        # Profile summary
        profile_records = []
        for state_idx in range(self.n_components):
            orig_idx = state_order[state_idx]
            rec = {
                "latent_state": state_idx,
                "prevalence_pct": float(np.mean(decoded_states == state_idx) * 100.0),
            }
            for f_idx, col in enumerate(self.emission_cols_):
                rec[f"mean_{col}"] = float(model.means_[orig_idx, f_idx])
                if model.covars_.ndim == 3:
                    var_val = model.covars_[orig_idx, f_idx, f_idx]
                elif model.covars_.ndim == 2:
                    var_val = model.covars_[orig_idx, f_idx]
                else:
                    var_val = model.covars_[orig_idx]
                rec[f"std_{col}"] = float(np.sqrt(np.maximum(0.0, float(var_val))))
            profile_records.append(rec)

        self.regime_profile_ = pd.DataFrame(profile_records)
        self.is_fitted_ = True
        return self

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Compute posterior state probabilities reordered by regime intensity."""
        if not self.is_fitted_:
            raise ValueError("WorkforceHMMRegimes has not been fitted yet. Call .fit() first.")
        X_mat, _ = self._extract_array(X, emission_cols=self.emission_cols_)
        raw_posteriors = self.model_.predict_proba(X_mat)
        return raw_posteriors[:, self.state_order_]

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Decode most likely latent regime sequence (Viterbi/posterior argmax)."""
        posteriors = self.predict_proba(X)
        return np.argmax(posteriors, axis=1)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Append latent state and posterior regime probabilities to DataFrame."""
        if not self.is_fitted_:
            raise ValueError("WorkforceHMMRegimes has not been fitted yet. Call .fit() first.")
        if not isinstance(df, pd.DataFrame):
            raise TypeError("transform requires a pd.DataFrame")
        posteriors = self.predict_proba(df)
        decoded_states = np.argmax(posteriors, axis=1)

        result_df = df.copy()
        result_df["hmm_latent_state"] = decoded_states
        for state_idx in range(self.n_components):
            result_df[f"prob_latent_state_{state_idx}"] = posteriors[:, state_idx]
        return result_df

    def fit_transform(
        self,
        df: pd.DataFrame,
        emission_cols: Optional[List[str]] = None,
        y: Any = None,
    ) -> pd.DataFrame:
        """Fit model and return DataFrame with regime annotations."""
        return self.fit(df, emission_cols=emission_cols).transform(df)


WorkforceHMM = WorkforceHMMRegimes


def fit_workforce_hmm_regimes(
    df: pd.DataFrame,
    emission_cols: List[str],
    n_components: int = 3,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Fit a Gaussian Hidden Markov Model (HMM) to extract latent operational regimes.
    
    Infers whether high overtime represents a benign planned seasonal surge or
    an acute understaffed attrition spiral.
    
    Returns:
        (df_with_posterior_regime_probs, regime_profile_summary, model_metadata)
    """
    model = WorkforceHMMRegimes(
        n_components=n_components,
        seed=seed,
    )
    model.fit(df, emission_cols=emission_cols)
    result_df = model.transform(df)
    meta = {
        "transition_matrix": model.transmat_,
        "means": model.means_,
        "model": model.model_,
        "estimator": model,
    }
    return result_df, model.regime_profile_, meta


# ---------------------------------------------------------------------------
# 5. Bradford Factor (Absenteeism Friction Score)
# ---------------------------------------------------------------------------

def calculate_bradford_score(
    spells: Union[int, float, pd.Series, np.ndarray],
    days_absent: Union[int, float, pd.Series, np.ndarray],
) -> Union[int, float, pd.Series, np.ndarray]:
    """Calculate the classic Bradford Factor (Absenteeism Friction Score): B = S^2 * D.

    Parameters
    ----------
    spells : Union[int, float, pd.Series, np.ndarray]
        Total count of distinct absence spells (instances) in the evaluation period.
    days_absent : Union[int, float, pd.Series, np.ndarray]
        Total cumulative days absent across all spells in the evaluation period.

    Returns
    -------
    Union[int, float, pd.Series, np.ndarray]
        The calculated Bradford Score B = S^2 * D.
    """
    s = np.asarray(spells) if isinstance(spells, (pd.Series, list)) else spells
    d = np.asarray(days_absent) if isinstance(days_absent, (pd.Series, list)) else days_absent

    score = (s ** 2) * d
    if isinstance(spells, pd.Series):
        return pd.Series(score, index=spells.index, name="bradford_score")
    return score


def classify_bradford_risk(
    score: Union[int, float, pd.Series, np.ndarray],
    thresholds: Optional[Dict[str, float]] = None,
) -> Union[str, pd.Series, np.ndarray]:
    """Classify Bradford scores into operational friction and HR intervention tiers.

    Standard benchmark thresholds:
    - 0 to 50: 'Low Friction' (Standard operational tolerance; acceptable attendance)
    - 51 to 200: 'Moderate Friction' (Noticeable friction; line-manager informal check-in)
    - 201 to 500: 'Substantial Friction' (Formal attendance review trigger; scheduling disruption)
    - > 500: 'Critical Friction' (Acute operational destabilization; disciplinary / schedule overhaul)
    """
    th = thresholds or {"low": 50.0, "moderate": 200.0, "substantial": 500.0}

    def _classify_single(val: float) -> str:
        if pd.isna(val) or val <= th["low"]:
            return "Low Friction"
        elif val <= th["moderate"]:
            return "Moderate Friction"
        elif val <= th["substantial"]:
            return "Substantial Friction"
        else:
            return "Critical Friction"

    if isinstance(score, (int, float, np.integer, np.floating)):
        return _classify_single(float(score))
    elif isinstance(score, pd.Series):
        return score.apply(_classify_single)
    else:
        arr = np.asarray(score)
        vfunc = np.vectorize(_classify_single)
        return vfunc(arr)


def calculate_bradford_factor(
    df: pd.DataFrame,
    employee_col: str = "employee_id",
    date_col: Optional[str] = None,
    start_date_col: Optional[str] = None,
    end_date_col: Optional[str] = None,
    duration_col: Optional[str] = None,
    is_absent_col: Optional[str] = None,
    window_weeks: int = 52,
    as_of_date: Optional[Union[str, pd.Timestamp]] = None,
    planned_col: Optional[str] = None,
    unplanned_only: bool = False,
) -> pd.DataFrame:
    """Compute Bradford Factor absenteeism friction metrics across workforce cohorts.

    Supports three primary data ingestion modes:
    1. Direct Summary Records: If `df` contains pre-aggregated 'absence_spells' (or 'spells')
       and 'total_days_absent' (or 'days_absent').
    2. Event Log / Spell Records: If `start_date_col` and `end_date_col` (or `duration_col`)
       are specified, each record represents an absence spell.
    3. Daily Attendance Records: If `date_col` and `is_absent_col` are specified, consecutive
       calendar absent days are automatically clustered into discrete spells.

    Parameters
    ----------
    df : pd.DataFrame
        Workforce attendance or absence dataframe.
    employee_col : str
        Column identifying unique associates.
    date_col : Optional[str]
        Date column for daily records or single-day events.
    start_date_col : Optional[str]
        Start date column for absence spells.
    end_date_col : Optional[str]
        End date column for absence spells.
    duration_col : Optional[str]
        Numeric duration (in days) column for spell records.
    is_absent_col : Optional[str]
        Boolean or integer flag (1=absent, 0=present) for daily attendance tracking.
    window_weeks : int, default=52
        Lookback evaluation window in weeks (e.g., 52 weeks = 364 days; 53 weeks = 371 days).
    as_of_date : Optional[Union[str, pd.Timestamp]]
        Evaluation snapshot date. Defaults to the maximum date in the dataset.
    planned_col : Optional[str]
        Column indicating whether an absence was planned (e.g. approved leave, FMLA, PTO)
        versus unplanned call-out.
    unplanned_only : bool, default=False
        If True, only counts unplanned absences in the primary Bradford score.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed or keyed by employee_id containing:
        - 'absence_spells' (S): Count of distinct absence instances in the window
        - 'total_days_absent' (D): Cumulative absence days in the window
        - 'bradford_score': S^2 * D
        - 'bradford_risk_band': 'Low Friction', 'Moderate Friction', 'Substantial Friction', 'Critical Friction'
        - 'disruption_multiplier': S^2 (ratio of friction to a single continuous absence block)
        - Additional planned vs unplanned decomposition fields if `planned_col` is provided.
    """
    records = df.copy()

    if records.empty:
        base_cols = [
            employee_col,
            "absence_spells",
            "total_days_absent",
            "bradford_score",
            "bradford_risk_band",
            "disruption_multiplier",
        ]
        return pd.DataFrame(columns=base_cols)

    # Mode 1: Pre-aggregated summary metrics
    summary_spell_cols = [c for c in ["absence_spells", "spells", "spell_count"] if c in records.columns]
    summary_day_cols = [c for c in ["total_days_absent", "days_absent", "days"] if c in records.columns]

    if summary_spell_cols and summary_day_cols and not (date_col or start_date_col or is_absent_col):
        s_col = summary_spell_cols[0]
        d_col = summary_day_cols[0]
        records["absence_spells"] = records[s_col]
        records["total_days_absent"] = records[d_col]
        records["bradford_score"] = calculate_bradford_score(records["absence_spells"], records["total_days_absent"])
        records["bradford_risk_band"] = classify_bradford_risk(records["bradford_score"])
        records["disruption_multiplier"] = np.where(
            records["total_days_absent"] > 0,
            records["absence_spells"] ** 2,
            1.0,
        )
        return records

    # Window cutoff calculation
    ref_date_series = None
    if start_date_col and start_date_col in records.columns:
        records[start_date_col] = pd.to_datetime(records[start_date_col])
        ref_date_series = records[start_date_col]
    elif date_col and date_col in records.columns:
        records[date_col] = pd.to_datetime(records[date_col])
        ref_date_series = records[date_col]

    if as_of_date is not None:
        target_date = pd.to_datetime(as_of_date)
    elif ref_date_series is not None and not ref_date_series.dropna().empty:
        target_date = ref_date_series.max()
    else:
        target_date = pd.Timestamp.now()

    window_days = window_weeks * 7
    window_start = target_date - pd.Timedelta(days=window_days)

    unique_employees = records[employee_col].unique()
    result_rows = []

    # Mode 2: Daily Attendance Records
    if is_absent_col and is_absent_col in records.columns and date_col and date_col in records.columns:
        for emp_id in unique_employees:
            emp_df = records[records[employee_col] == emp_id].sort_values(date_col)
            # Filter to window
            in_window = emp_df[(emp_df[date_col] >= window_start) & (emp_df[date_col] <= target_date)]
            absent_df = in_window[in_window[is_absent_col].astype(bool)].copy()

            if absent_df.empty:
                result_rows.append({
                    employee_col: emp_id,
                    "absence_spells": 0,
                    "total_days_absent": 0,
                    "bradford_score": 0,
                    "bradford_risk_band": "Low Friction",
                    "disruption_multiplier": 1.0,
                })
                continue

            # Identify spells by consecutive calendar days
            absent_dates = absent_df[date_col].dt.normalize()
            date_diff = absent_dates.diff().dt.days
            new_spell = date_diff != 1
            absent_df["spell_id"] = new_spell.cumsum()

            if unplanned_only and planned_col and planned_col in absent_df.columns:
                target_absent = absent_df[~absent_df[planned_col].astype(bool)]
            else:
                target_absent = absent_df

            spells = int(target_absent["spell_id"].nunique())
            days = len(target_absent)
            b_score = int(calculate_bradford_score(spells, days))
            risk_band = classify_bradford_risk(b_score)
            mult = float(spells ** 2) if days > 0 else 1.0

            row = {
                employee_col: emp_id,
                "absence_spells": spells,
                "total_days_absent": days,
                "bradford_score": b_score,
                "bradford_risk_band": risk_band,
                "disruption_multiplier": mult,
            }

            if planned_col and planned_col in absent_df.columns:
                unp = absent_df[~absent_df[planned_col].astype(bool)]
                pla = absent_df[absent_df[planned_col].astype(bool)]
                u_spells = int(unp["spell_id"].nunique())
                u_days = len(unp)
                u_score = int(calculate_bradford_score(u_spells, u_days))
                p_spells = int(pla["spell_id"].nunique())
                p_days = len(pla)
                p_score = int(calculate_bradford_score(p_spells, p_days))
                row["unplanned_spells"] = u_spells
                row["unplanned_days_absent"] = u_days
                row["unplanned_bradford_score"] = u_score
                row["planned_spells"] = p_spells
                row["planned_days_absent"] = p_days
                row["planned_bradford_score"] = p_score
                row["operational_friction_delta"] = u_score - p_score

            result_rows.append(row)

        res_df = pd.DataFrame(result_rows)
        return res_df

    # Mode 3: Spell-level event log
    for emp_id in unique_employees:
        emp_df = records[records[employee_col] == emp_id].copy()

        # Date filtering
        if start_date_col and start_date_col in emp_df.columns:
            emp_df = emp_df[(emp_df[start_date_col] >= window_start) & (emp_df[start_date_col] <= target_date)]
        elif date_col and date_col in emp_df.columns:
            emp_df = emp_df[(emp_df[date_col] >= window_start) & (emp_df[date_col] <= target_date)]

        if emp_df.empty:
            result_rows.append({
                employee_col: emp_id,
                "absence_spells": 0,
                "total_days_absent": 0,
                "bradford_score": 0,
                "bradford_risk_band": "Low Friction",
                "disruption_multiplier": 1.0,
            })
            continue

        # Compute durations
        if duration_col and duration_col in emp_df.columns:
            durations = emp_df[duration_col].values
        elif start_date_col and end_date_col and end_date_col in emp_df.columns:
            records_end = pd.to_datetime(emp_df[end_date_col])
            records_start = pd.to_datetime(emp_df[start_date_col])
            durations = (records_end - records_start).dt.days + 1
        else:
            durations = np.ones(len(emp_df), dtype=int)

        emp_df["computed_duration"] = durations

        if unplanned_only and planned_col and planned_col in emp_df.columns:
            target_df = emp_df[~emp_df[planned_col].astype(bool)]
        else:
            target_df = emp_df

        spells = len(target_df)
        days = int(target_df["computed_duration"].sum())
        b_score = int(calculate_bradford_score(spells, days))
        risk_band = classify_bradford_risk(b_score)
        mult = float(spells ** 2) if days > 0 else 1.0

        row = {
            employee_col: emp_id,
            "absence_spells": spells,
            "total_days_absent": days,
            "bradford_score": b_score,
            "bradford_risk_band": risk_band,
            "disruption_multiplier": mult,
        }

        if planned_col and planned_col in emp_df.columns:
            unp = emp_df[~emp_df[planned_col].astype(bool)]
            pla = emp_df[emp_df[planned_col].astype(bool)]
            u_spells = len(unp)
            u_days = int(unp["computed_duration"].sum())
            u_score = int(calculate_bradford_score(u_spells, u_days))
            p_spells = len(pla)
            p_days = int(pla["computed_duration"].sum())
            p_score = int(calculate_bradford_score(p_spells, p_days))
            row["unplanned_spells"] = u_spells
            row["unplanned_days_absent"] = u_days
            row["unplanned_bradford_score"] = u_score
            row["planned_spells"] = p_spells
            row["planned_days_absent"] = p_days
            row["planned_bradford_score"] = p_score
            row["operational_friction_delta"] = u_score - p_score

        result_rows.append(row)

    res_df = pd.DataFrame(result_rows)
    return res_df


# API Consistency Aliases: compute_* <=> calculate_*
compute_shannon_entropy = calculate_shannon_entropy
compute_bradford_score = calculate_bradford_score
compute_bradford_factor = calculate_bradford_factor
