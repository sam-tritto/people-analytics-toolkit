"""Memory and Temporal Lifecycles Feature Engineering.

This module provides high-performance, mathematically rigorous time-series feature engineering,
memory retention transforms, adaptive moving averages, volatility forecasting, and survival embeddings
tailored for longitudinal human capital analytics.

Key Capabilities & Algorithmic Modules
--------------------------------------
1. Fractional Differencing & Memory Retention:
   - :func:`get_fractional_weights`: Binomial expansion weights for (1 - L)^d.
   - :func:`fractional_difference`: Applies real-order differencing preserving long-range memory while achieving stationarity.
   - :func:`scan_fractional_differencing`: Automated ADF grid search to discover the minimum stationarity-inducing d.

2. Adaptive Moving Averages & Event Timing:
   - :func:`weighted_weeks_since`: Exponentially decaying recency metrics for recognition, training, and milestones.
   - :func:`calculate_ewma`: Exponentially weighted moving average with ghost-drop mitigation.
   - :func:`scan_ewma_spans`: Automated hyperparameter tuning and model selection across multi-timescale EWMA features.
   - :func:`calculate_ewms`: Exponentially weighted moving standard deviation (rolling volatility).
   - :func:`calculate_poisson_ewma`: Poisson-rate adjusted EWMA for intermittent or count-based event streams.

3. Exponential Smoothing & Forecasting:
   - :class:`ExponentialSmoothingResult`: Structured container for level, trend, and seasonal smoothing components.
   - :func:`double_exponential_smoothing`: Holt linear trend exponential smoothing.
   - :func:`triple_exponential_smoothing`: Holt-Winters additive/multiplicative seasonal forecasting.

4. Survival & Hazard Embeddings:
   - :class:`HazardEmbeddings`: Container for non-linear Kaplan-Meier survival and Nelson-Aalen hazard features.
   - :class:`HazardEmbeddingTransformer`: Scikit-learn compatible transformer for tenure-to-hazard embeddings.
   - :func:`fit_hazard_embeddings`: Convenience function to fit and transform event-tenure datasets.

5. Config-Driven Rolling Metrics:
   - :class:`RollingMetricConfig`: Configuration specification for multi-window rolling aggregations.
   - :func:`compute_rolling_metrics`: Config-driven aggregation pipeline supporting 10+ statistical operators.
   - :class:`RollingMetricsPipeline`: Reusable pipeline for rolling window feature generation.

6. Singular Spectrum Analysis (SSA) & SVD Smoothing:
   - :func:`singular_spectrum_analysis`: Non-parametric time-series decomposition into trend, oscillatory, and noise components.
   - :func:`compute_svd_smoothed_baseline`: High-level DataFrame pipeline extracting SSA smoothed baselines and residuals.
   - :class:`SingularSpectrumAnalysis`: Transformer class for embedding and extracting trajectory components.

7. Temporal Lag & Differencing Feature Engineering:
   - :func:`compute_lag_features`: Multi-period lag feature generation across entity-level groupings.
   - :func:`compute_differencing_features`: Discrete differencing and percentage rate-of-change pipelines.
   - :class:`TemporalFeaturesPipeline`: Fluent transformer for unified lag and differencing feature extraction.
"""

from dataclasses import dataclass, field, fields
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import hankel
from statsmodels.tsa.stattools import adfuller

from people_analytics_toolkit._deprecation import deprecated_alias


def get_fractional_weights(d: float, length: int, threshold: float = 1e-4) -> np.ndarray:
    """Compute binomial expansion weights for fractional differencing operator (1 - L)^d.
    
    Formula:
        w_0 = 1.0
        w_k = -w_{k-1} * (d - k + 1) / k

    Parameters
    ----------
    d : float
        Fractional differencing coefficient in [0.0, 1.0].
    length : int
        Maximum number of past lags to compute weights for.
    threshold : float, default=1e-4
        Cutoff threshold below which weights are truncated to 0.

    Returns
    -------
    np.ndarray
        Array of binomial expansion weights.

    Raises
    ------
    ValueError
        If `d` is not in [0, 1], `length` <= 0, or `threshold` <= 0.
    """
    if not (0.0 <= float(d) <= 1.0):
        raise ValueError(f"Differencing order d must be in [0, 1], got {d}")
    if length <= 0:
        raise ValueError(f"length must be a positive integer, got {length}")
    if threshold <= 0:
        raise ValueError(f"threshold must be positive, got {threshold}")
    weights = [1.0]
    for k in range(1, length):
        w_k = -weights[-1] * (d - k + 1) / k
        if abs(w_k) < threshold:
            break
        weights.append(w_k)
    return np.array(weights)


def fractional_difference(
    series: Union[pd.Series, np.ndarray, list],
    d: float,
    threshold: float = 1e-4,
) -> pd.Series:
    """Apply fractional differencing of order d to a time series.
    
    Preserves historical memory while eliminating unit roots and non-stationarity.

    Parameters
    ----------
    series : pd.Series, np.ndarray, or list
        Input sequence.
    d : float
        Fractional differencing order in [0.0, 1.0].
    threshold : float, default=1e-4
        Cutoff threshold for weight truncation.

    Returns
    -------
    pd.Series
        Fractionally differenced time series.

    Raises
    ------
    TypeError
        If `series` is not a pandas Series, numpy array, or list.
    ValueError
        If `d` is not in [0, 1], or if `series` is empty.
    """
    if not (0.0 <= float(d) <= 1.0):
        raise ValueError(f"Differencing order d must be in [0, 1], got {d}")
    if not isinstance(series, (pd.Series, np.ndarray, list)):
        raise TypeError(f"series must be a pandas Series, numpy ndarray, or list, got {type(series).__name__}")
    if len(series) == 0:
        raise ValueError("Series cannot be empty.")
        
    if isinstance(series, (np.ndarray, list)):
        s = pd.Series(series)
    else:
        s = series.copy()
        
    n = len(s)
    weights = get_fractional_weights(d, n, threshold=threshold)
    k = len(weights)
    
    # Convolution with memory weights
    diff_values = np.full(n, np.nan)
    s_vals = s.to_numpy(dtype=float)
    
    # Reverse weights for dot product with trailing window
    w_rev = weights[::-1]
    for i in range(k - 1, n):
        window = s_vals[i - k + 1 : i + 1]
        diff_values[i] = np.dot(w_rev, window)
        
    return pd.Series(diff_values, index=s.index, name=f"frac_diff_d_{d:.2f}")


def scan_fractional_differencing(
    series: pd.Series,
    d_values: Optional[np.ndarray] = None,
    threshold: float = 1e-4,
) -> pd.DataFrame:
    """Scan candidate d values in [0, 1] to identify the optimal d* for stationarity.
    
    Computes ADF test p-value and Pearson correlation with original raw series.
    Optimal d* is the minimum d where p-value < 0.05, maximizing memory retention.

    Parameters
    ----------
    series : pd.Series
        Input non-stationary time series.
    d_values : np.ndarray, optional
        Grid of candidate differencing coefficients. Defaults to 21 evenly spaced points in [0, 1].
    threshold : float, default=1e-4
        Memory weight truncation threshold.

    Returns
    -------
    pd.DataFrame
        Diagnostics DataFrame tracking ADF test statistic, p-value, and correlation for each d.

    Raises
    ------
    ValueError
        If any candidate value in `d_values` is outside [0.0, 1.0].
    """
    if d_values is None:
        d_values = np.linspace(0.0, 1.0, 21)
    else:
        d_arr = np.asarray(d_values, dtype=float)
        if len(d_arr) > 0 and (np.any(d_arr < 0.0) or np.any(d_arr > 1.0)):
            raise ValueError(f"All candidate d values must be in [0, 1], got min={d_arr.min()}, max={d_arr.max()}")
        
    records = []
    clean_series = series.dropna()
    
    for d in d_values:
        if d == 0.0:
            diff_s = clean_series.copy()
            corr = 1.0
        else:
            diff_s = fractional_difference(clean_series, d=d, threshold=threshold).dropna()
            # Align indices for correlation
            common_idx = clean_series.index.intersection(diff_s.index)
            corr = float(np.corrcoef(clean_series.loc[common_idx], diff_s.loc[common_idx])[0, 1])
            
        # ADF Test
        try:
            adf_res = adfuller(diff_s, maxlag=5, autolag="AIC")
            adf_stat = float(adf_res[0])
            p_val = float(adf_res[1])
        except Exception:
            adf_stat = np.nan
            p_val = np.nan
            
        records.append({
            "d": round(float(d), 2),
            "adf_stat": adf_stat,
            "p_value": p_val,
            "is_stationary": bool(p_val < 0.05) if not np.isnan(p_val) else False,
            "correlation_with_raw": corr,
        })
        
    return pd.DataFrame(records)


def weighted_weeks_since(
    events_df: pd.DataFrame,
    current_week: int,
    half_life_weeks: float = 8.0,
    employee_id_col: str = "employee_id",
    event_week_col: str = "award_week",
    tier_col: str = "award_tier",
    custom_tier_weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Compute continuous time-decayed recognition feature from tiered awards.
    
    Tier default weights: Gold = 1.0 ($100), Silver = 0.5 ($50), Bronze = 0.25 ($25).
    Decay formula:
        S_e(t) = w_e * exp( - ln(2) / half_life * (t - t_e) )

    Parameters
    ----------
    events_df : pd.DataFrame
        Event-level recognition log.
    current_week : int
        Current evaluation week integer.
    half_life_weeks : float, default=8.0
        Half-life in weeks for exponential decay.
    employee_id_col : str, default='employee_id'
        Column name identifying employees.
    event_week_col : str, default='award_week'
        Column name indicating week of the event.
    tier_col : str, default='award_tier'
        Column name indicating recognition tier.
    custom_tier_weights : dict, optional
        Custom mapping from tier string to float weight.

    Returns
    -------
    pd.DataFrame
        Aggregated associate recognition features.

    Raises
    ------
    ValueError
        If `half_life_weeks` <= 0.
    """
    if custom_tier_weights is None:
        weights_map = {"Gold": 1.0, "Silver": 0.5, "Bronze": 0.25}
    else:
        weights_map = custom_tier_weights
        
    decay_rate = np.log(2.0) / half_life_weeks
    
    # Filter to events that occurred on or before current_week
    past_events = events_df[events_df[event_week_col] <= current_week].copy()
    
    past_events["elapsed_weeks"] = current_week - past_events[event_week_col]
    past_events["tier_weight"] = past_events[tier_col].map(weights_map).fillna(0.25)
    past_events["decayed_signal"] = past_events["tier_weight"] * np.exp(-decay_rate * past_events["elapsed_weeks"])
    
    # Aggregate per employee
    agg = past_events.groupby(employee_id_col).agg(
        total_decayed_recognition_pulse=("decayed_signal", "sum"),
        weeks_since_most_recent_award=("elapsed_weeks", "min"),
        most_recent_award_tier=("award_tier", "last"),
        total_awards_count=(event_week_col, "count"),
    ).reset_index()
    
    return agg


def calculate_ewma(
    series: Union[pd.Series, np.ndarray, list],
    span: Optional[int] = None,
    alpha: Optional[float] = None,
    half_life: Optional[float] = None,
) -> pd.Series:
    """Calculate Exponentially Weighted Moving Average (EWMA).
    
    Avoids ghost drops caused by simple rolling windows and adapts smoothly to level shifts.
    """
    if span is None and alpha is None and half_life is None:
        span = 12
    s = series if isinstance(series, pd.Series) else pd.Series(series)
    return s.ewm(span=span, alpha=alpha, halflife=half_life, adjust=False).mean()


def scan_ewma_spans(
    series: Union[pd.Series, np.ndarray, list],
    spans: Optional[Sequence[int]] = None,
    half_lives: Optional[Sequence[float]] = None,
    target_series: Optional[Union[pd.Series, np.ndarray, list]] = None,
    criterion: str = "rmse_one_step",
) -> pd.DataFrame:
    r"""Scan candidate EWMA spans or half-lives to identify optimal smoothing parameters.

    Analogous to `scan_fractional_differencing()` for stationarity tuning,
    `scan_ewma_spans()` evaluates candidate EWMA memory decay parameters against
    unsupervised one-step-ahead forecasting error (RMSE, MAE, AIC),
    noise-reduction ratios, and (optionally) supervised predictive power against
    a downstream outcome target.

    Mathematical Formulations:
        - Exponential Smoothing:
          .. math::
              E_t = \alpha y_t + (1 - \alpha) E_{t-1}, \quad \alpha = \frac{2}{\text{span} + 1}
        - Equivalent Half-Life:
          .. math::
              h = -\frac{\ln(2)}{\ln(1 - \alpha)} \approx \frac{\text{span} \cdot \ln(2)}{2}
        - One-Step-Ahead Forecast Error:
          .. math::
              \hat{y}_t = E_{t-1}, \quad e_t = y_t - \hat{y}_t
              \text{RMSE}_{\text{one\_step}} = \sqrt{\frac{1}{T-1} \sum_{t=2}^T (y_t - E_{t-1})^2}
        - Noise Reduction Ratio (NRR):
          .. math::
              \text{NRR} = \frac{\text{Std}(\Delta E)}{\text{Std}(\Delta y)}
        - Akaike Information Criterion:
          .. math::
              \text{AIC} = (T - 1) \ln(\text{MSE}_{\text{one\_step}}) + 2k, \quad k = 1

    Parameters
    ----------
    series : pd.Series, np.ndarray, or list
        Input continuous time series to smooth and evaluate.
    spans : sequence of int, optional
        Candidate spans to scan (e.g. [3, 5, 8, 12, 14, 21, 28, 35, 42, 52, 60]).
        If None and half_lives is None, defaults to a standard multi-frequency grid.
    half_lives : sequence of float, optional
        Candidate half-lives to scan. If provided, converted into equivalent spans/alphas.
    target_series : pd.Series, np.ndarray, or list, optional
        Supervised target outcome (e.g. turnover risk, absenteeism, performance rating).
        When provided, computes correlation, R^2, and regression RMSE against the target.
    criterion : str, default='rmse_one_step'
        Evaluation metric used to flag `is_optimal` candidate.
        Options: 'rmse_one_step', 'mae_one_step', 'aic', 'bic', 'noise_reduction_ratio',
        'correlation_with_raw', or 'target_correlation' / 'target_r2' if target_series provided.

    Returns
    -------
    pd.DataFrame
        DataFrame of candidate configurations with columns.

    Raises
    ------
    ValueError
        If series length is less than 4, if any candidate span <= 1, or if half_life <= 0.
    """
    if isinstance(series, pd.Series):
        s = series.dropna().astype(float)
    else:
        s = pd.Series(series, dtype=float).dropna()

    if len(s) < 4:
        raise ValueError(f"Series must contain at least 4 observations to evaluate EWMA, got {len(s)}")

    # Determine candidates
    candidate_configs = []
    if spans is not None:
        for span in spans:
            if span <= 1:
                raise ValueError(f"Span must be > 1, got {span}")
            a = 2.0 / (span + 1.0)
            hl = -np.log(2.0) / np.log(1.0 - a) if (1.0 - a) > 0 else np.nan
            candidate_configs.append({"span": int(span), "alpha": float(a), "half_life": float(hl)})
    elif half_lives is not None:
        for hl in half_lives:
            if hl <= 0:
                raise ValueError(f"half_life must be positive, got {hl}")
            a = 1.0 - np.exp(-np.log(2.0) / hl)
            span = int(round(2.0 / a - 1.0)) if a > 0 else 1
            span = max(2, span)
            candidate_configs.append({"span": span, "alpha": float(a), "half_life": float(hl)})
    else:
        default_spans = [3, 5, 7, 10, 14, 21, 28, 35, 42, 52, 60]
        for span in default_spans:
            a = 2.0 / (span + 1.0)
            hl = -np.log(2.0) / np.log(1.0 - a) if (1.0 - a) > 0 else np.nan
            candidate_configs.append({"span": int(span), "alpha": float(a), "half_life": float(hl)})

    # Target series handling
    target_clean = None
    if target_series is not None:
        if isinstance(target_series, pd.Series):
            target_clean = target_series.loc[s.index]
        else:
            target_clean = pd.Series(target_series, index=s.index, dtype=float)

    y_vals = s.to_numpy()
    T = len(y_vals)
    delta_y_std = float(np.std(np.diff(y_vals), ddof=1)) if T > 2 else 1e-6
    delta_y_std = delta_y_std if delta_y_std > 1e-12 else 1e-6

    records = []
    for cfg in candidate_configs:
        span_val = cfg["span"]
        alpha_val = cfg["alpha"]
        hl_val = cfg["half_life"]

        ewma_s = calculate_ewma(s, span=span_val)
        ewma_vals = ewma_s.to_numpy()

        # One-step ahead forecast error: y_t vs ewma_{t-1}
        y_true = y_vals[1:]
        y_pred = ewma_vals[:-1]
        errors = y_true - y_pred
        mse_one_step = float(np.mean(errors ** 2))
        rmse_one_step = float(np.sqrt(mse_one_step))
        mae_one_step = float(np.mean(np.abs(errors)))

        # In-sample error
        rmse_in_sample = float(np.sqrt(np.mean((y_vals - ewma_vals) ** 2)))

        # Correlation with raw
        std_ewma = float(np.std(ewma_vals))
        std_raw = float(np.std(y_vals))
        if std_ewma > 1e-12 and std_raw > 1e-12:
            corr_raw = float(np.corrcoef(y_vals, ewma_vals)[0, 1])
        else:
            corr_raw = 1.0 if std_ewma <= 1e-12 and std_raw <= 1e-12 else 0.0

        # Noise reduction ratio
        delta_ewma_std = float(np.std(np.diff(ewma_vals), ddof=1)) if T > 2 else 0.0
        nrr = float(delta_ewma_std / delta_y_std)

        # AIC / BIC for 1-step ahead residuals
        n_obs = T - 1
        safe_mse = max(mse_one_step, 1e-12)
        aic = float(n_obs * np.log(safe_mse) + 2.0 * 1.0)
        bic = float(n_obs * np.log(safe_mse) + 1.0 * np.log(n_obs))

        rec = {
            "span": span_val,
            "alpha": round(alpha_val, 4),
            "half_life": round(hl_val, 2) if not np.isnan(hl_val) else np.nan,
            "rmse_one_step": round(rmse_one_step, 4),
            "mae_one_step": round(mae_one_step, 4),
            "rmse_in_sample": round(rmse_in_sample, 4),
            "correlation_with_raw": round(corr_raw, 4),
            "noise_reduction_ratio": round(nrr, 4),
            "aic": round(aic, 2),
            "bic": round(bic, 2),
        }

        # Target metrics if available
        if target_clean is not None:
            t_vals = target_clean.to_numpy()
            std_t = float(np.std(t_vals))
            if std_ewma > 1e-12 and std_t > 1e-12:
                t_corr = float(np.corrcoef(ewma_vals, t_vals)[0, 1])
            else:
                t_corr = 0.0

            slope, intercept, r_val, p_val, std_err = stats.linregress(ewma_vals, t_vals)
            t_r2 = float(r_val ** 2) if not np.isnan(r_val) else 0.0
            t_pred = slope * ewma_vals + intercept
            t_rmse = float(np.sqrt(np.mean((t_vals - t_pred) ** 2)))

            rec["target_correlation"] = round(t_corr, 4)
            rec["target_r2"] = round(t_r2, 4)
            rec["target_rmse"] = round(t_rmse, 4)

        records.append(rec)

    res_df = pd.DataFrame(records)

    # Flag optimal
    crit = criterion.lower()
    if crit in ("target_correlation", "correlation_with_raw", "target_r2"):
        if crit not in res_df.columns:
            crit = "rmse_one_step"
        best_idx = res_df[crit].abs().idxmax()
    elif crit in res_df.columns:
        best_idx = res_df[crit].idxmin()
    else:
        best_idx = res_df["rmse_one_step"].idxmin()

    res_df["is_optimal"] = False
    res_df.loc[best_idx, "is_optimal"] = True

    best_row = res_df.loc[best_idx]
    res_df.attrs["best_span"] = int(best_row["span"])
    res_df.attrs["best_half_life"] = float(best_row["half_life"])
    res_df.attrs["best_alpha"] = float(best_row["alpha"])

    return res_df


def calculate_ewms(
    series: Union[pd.Series, np.ndarray, list],
    half_life: Optional[float] = None,
    span: Optional[int] = None,
    alpha: Optional[float] = None,
    decay_rate: Optional[float] = None,
    group_col: Optional[Union[pd.Series, np.ndarray, list]] = None,
) -> pd.Series:
    """Calculate Exponentially Weighted Moving Sum (EWMS) for discrete count data.
    
    Acts as a leaky accumulator/integrator for event counts (safety incidents,
    attendance infractions, disciplinary warnings, peer recognitions).
    
    Formula:
        S_t = C_t + lambda * S_{t-1} = sum_{k=0}^t lambda^k * C_{t-k}
        
    Where decay parameter lambda in [0, 1):
        - If half_life is provided: lambda = exp(-ln(2) / half_life)
        - If span is provided: alpha = 2 / (span + 1), lambda = 1 - alpha
        - If alpha is provided: lambda = 1 - alpha
        - If decay_rate is provided: lambda = decay_rate
        - Default: half_life = 8.0 periods
        
    Unlike EWMA (which normalizes by sum of weights to estimate average intensity per period),
    EWMS computes the cumulative recency-decayed event burden or momentum.

    Parameters
    ----------
    series : pd.Series, np.ndarray, or list
        Input event counts.
    half_life : float, optional
        Half-life in periods for exponential decay.
    span : int, optional
        EWMA span equivalent for decay parameter.
    alpha : float, optional
        Direct smoothing alpha parameter.
    decay_rate : float, optional
        Direct decay parameter lambda in [0, 1).
    group_col : pd.Series, np.ndarray, or list, optional
        Optional grouping variable for partition-level filtering.

    Returns
    -------
    pd.Series
        Filtered cumulative moving sum series.

    Raises
    ------
    ValueError
        If computed decay lambda falls outside [0, 1).
    """
    from scipy.signal import lfilter

    # Determine decay lambda
    if decay_rate is not None:
        lam = float(decay_rate)
    elif half_life is not None:
        lam = float(np.exp(-np.log(2.0) / half_life))
    elif span is not None:
        a = 2.0 / (span + 1.0)
        lam = 1.0 - a
    elif alpha is not None:
        lam = 1.0 - float(alpha)
    else:
        lam = float(np.exp(-np.log(2.0) / 8.0))

    if not 0.0 <= lam < 1.0:
        raise ValueError(f"Decay parameter lambda must be in [0, 1), got {lam}")

    s = series if isinstance(series, pd.Series) else pd.Series(series)
    
    def _compute_1d(arr_like: pd.Series) -> pd.Series:
        vals = arr_like.to_numpy(dtype=float)
        clean_vals = np.nan_to_num(vals, nan=0.0)
        # Leaky integrator: S_t - lam * S_{t-1} = C_t
        filtered = lfilter([1.0], [1.0, -lam], clean_vals)
        filtered[np.isnan(vals)] = np.nan
        return pd.Series(filtered, index=arr_like.index, name=f"ewms_lambda_{lam:.3f}")

    if group_col is not None:
        g = group_col if isinstance(group_col, pd.Series) else pd.Series(group_col, index=s.index)
        try:
            return s.groupby(g, group_keys=False).apply(_compute_1d, include_groups=False)
        except TypeError:
            return s.groupby(g, group_keys=False).apply(_compute_1d)
    else:
        return _compute_1d(s)


def calculate_poisson_ewma(
    series: Union[pd.Series, np.ndarray, list],
    alpha: float = 0.2,
    baseline_mean: Optional[float] = None,
) -> Tuple[pd.Series, pd.Series]:
    """Calculate Poisson EWMA intensity and standardized Z-score for count streams.
    
    In a Poisson process where Var(C_t) = E(C_t) = mu_0, the asymptotic variance
    of the EWMA estimator hat{mu}_t = alpha * C_t + (1 - alpha) * hat{mu}_{t-1} is:
        sigma^2_{hat{mu}} = (alpha / (2 - alpha)) * mu_0
        
    Returns:
        (smoothed_intensity_series, standardized_z_score_series)
    """
    s = series if isinstance(series, pd.Series) else pd.Series(series)
    smoothed = s.ewm(alpha=alpha, adjust=False).mean()
    
    mu_0 = float(s.mean()) if baseline_mean is None else float(baseline_mean)
    var_ewma = (alpha / (2.0 - alpha)) * max(1e-6, mu_0)
    z_scores = (smoothed - mu_0) / np.sqrt(var_ewma)
    
    return smoothed, z_scores


@dataclass
class ExponentialSmoothingResult:
    """Structured result object for exponential smoothing models and diagnostics."""
    alpha: float
    beta: float
    aic: float
    bic: float
    forecast: Optional[pd.Series] = None
    model: Any = None
    gamma: Optional[float] = None
    seasonal_periods: Optional[int] = None
    model_type: Optional[str] = None

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
        return [f.name for f in fields(self) if getattr(self, f.name) is not None]

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


SmoothingResult = ExponentialSmoothingResult


def double_exponential_smoothing(
    series: Union[pd.Series, np.ndarray, list],
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
    damped_trend: bool = False,
    forecast_periods: int = 0,
) -> Tuple[pd.Series, pd.DataFrame, ExponentialSmoothingResult]:
    """Apply Double Exponential Smoothing (Holt's Linear Trend) to model level and velocity.
    
    Formula:
        Level:  l_t = alpha * y_t + (1 - alpha) * (l_{t-1} + b_{t-1})
        Trend:  b_t = beta * (l_t - l_{t-1}) + (1 - beta) * b_{t-1}
        Fitted: hat{y}_t = l_{t-1} + b_{t-1}
        
    Expands basic EWMA by incorporating an explicit linear trend parameter (beta)
    to eliminate persistent lag during steady headcount growth or ramp periods.
    
    Returns:
        (fitted_series, decomposition_df, model_metadata)
        where decomposition_df contains ['level', 'trend']
    """
    from statsmodels.tsa.holtwinters import Holt
    
    s = series if isinstance(series, pd.Series) else pd.Series(series)
    clean_s = s.astype(float)
    
    model = Holt(clean_s, damped_trend=damped_trend, initialization_method="estimated")
    res = model.fit(
        smoothing_level=alpha,
        smoothing_trend=beta,
        optimized=True if (alpha is None or beta is None) else False,
    )
    
    fitted = pd.Series(res.fittedvalues, index=s.index, name="double_exp_fitted")
    decomp = pd.DataFrame({
        "level": res.level,
        "trend": res.trend,
    }, index=s.index)
    
    forecast_vals = None
    if forecast_periods > 0:
        forecast_vals = res.forecast(forecast_periods)
        
    meta = ExponentialSmoothingResult(
        alpha=float(res.params.get("smoothing_level", np.nan)),
        beta=float(res.params.get("smoothing_trend", np.nan)),
        aic=float(res.aic) if hasattr(res, "aic") else np.nan,
        bic=float(res.bic) if hasattr(res, "bic") else np.nan,
        forecast=forecast_vals,
        model=res,
        model_type="double_exponential",
    )
    
    return fitted, decomp, meta


def triple_exponential_smoothing(
    series: Union[pd.Series, np.ndarray, list],
    seasonal_periods: int = 52,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
    gamma: Optional[float] = None,
    trend: str = "add",
    seasonal: str = "add",
    damped_trend: bool = False,
    forecast_periods: int = 0,
) -> Tuple[pd.Series, pd.DataFrame, ExponentialSmoothingResult]:
    """Apply Triple Exponential Smoothing (Holt-Winters) to model level, trend, and seasonality.
    
    Formula (Additive):
        Level:    l_t = alpha * (y_t - s_{t-m}) + (1 - alpha) * (l_{t-1} + b_{t-1})
        Trend:    b_t = beta * (l_t - l_{t-1}) + (1 - beta) * b_{t-1}
        Seasonal: s_t = gamma * (y_t - l_{t-1} - b_{t-1}) + (1 - gamma) * s_{t-m}
        Fitted:   hat{y}_t = l_{t-1} + b_{t-1} + s_{t-m}
        
    Captures multi-horizon recurring annual/quarterly hiring cycles, peak retail traffic,
    and cyclical seasonal turnover surges.
    
    Returns:
        (fitted_series, decomposition_df, model_metadata)
        where decomposition_df contains ['level', 'trend', 'seasonal']
    """
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    
    s = series if isinstance(series, pd.Series) else pd.Series(series)
    clean_s = s.astype(float)
    
    model = ExponentialSmoothing(
        clean_s,
        trend=trend,
        seasonal=seasonal,
        seasonal_periods=seasonal_periods,
        damped_trend=damped_trend,
        initialization_method="estimated",
    )
    res = model.fit(
        smoothing_level=alpha,
        smoothing_trend=beta,
        smoothing_seasonal=gamma,
        optimized=True if (alpha is None or beta is None or gamma is None) else False,
    )
    
    fitted = pd.Series(res.fittedvalues, index=s.index, name="triple_exp_fitted")
    decomp = pd.DataFrame({
        "level": res.level,
        "trend": res.trend,
        "seasonal": res.season,
    }, index=s.index)
    
    forecast_vals = None
    if forecast_periods > 0:
        forecast_vals = res.forecast(forecast_periods)
        
    meta = ExponentialSmoothingResult(
        alpha=float(res.params.get("smoothing_level", np.nan)),
        beta=float(res.params.get("smoothing_trend", np.nan)),
        gamma=float(res.params.get("smoothing_seasonal", np.nan)),
        seasonal_periods=seasonal_periods,
        aic=float(res.aic) if hasattr(res, "aic") else np.nan,
        bic=float(res.bic) if hasattr(res, "bic") else np.nan,
        forecast=forecast_vals,
        model=res,
        model_type="triple_exponential_additive",
    )
    
    return fitted, decomp, meta


class HazardEmbeddings:
    """Kaplan-Meier survival curves and Nelson-Aalen cumulative hazard estimator.

    Embeds non-linear behavioral turnover risk coordinates directly into workforce records.

    Parameters
    ----------
    tenure_col : str, default='tenure_weeks'
        Column name for duration / tenure in units of time (e.g. weeks or months).
    event_col : str, default='turnover_event'
        Column name for binary event indicator (1 if departed/occurred, 0 if censored/active).
    """

    def __init__(
        self,
        tenure_col: str = "tenure_weeks",
        event_col: str = "turnover_event",
    ):
        self.tenure_col = tenure_col
        self.event_col = event_col
        self.kmf_ = None
        self.naf_ = None
        self.surv_probs_ = None
        self.cum_hazard_ = None
        self.surv_map_ = None
        self.haz_map_ = None
        self.last_surv_ = 1.0
        self.last_haz_ = 0.0
        self.max_tenure_ = 0
        self.is_fitted_ = False

    def fit(self, roster_df: pd.DataFrame, y: Any = None) -> "HazardEmbeddings":
        """Fit Kaplan-Meier and Nelson-Aalen estimators on employee tenure history."""
        try:
            from lifelines import KaplanMeierFitter, NelsonAalenFitter
        except ImportError:
            raise ImportError(
                "HazardEmbeddings requires 'lifelines'. "
                "Install with: pip install 'people-analytics-toolkit[survival]'"
            )

        if not isinstance(roster_df, pd.DataFrame):
            raise TypeError("roster_df must be a pandas DataFrame")
        if self.tenure_col not in roster_df.columns:
            raise KeyError(f"tenure_col '{self.tenure_col}' not found in roster_df")
        if self.event_col not in roster_df.columns:
            raise KeyError(f"event_col '{self.event_col}' not found in roster_df")

        durations = roster_df[self.tenure_col].to_numpy(dtype=float)
        events = roster_df[self.event_col].to_numpy(dtype=float)

        kmf = KaplanMeierFitter()
        kmf.fit(durations, event_observed=events, label="Turnover_Survival")

        naf = NelsonAalenFitter()
        naf.fit(durations, event_observed=events, label="Turnover_Hazard")

        # Lookup tables
        max_tenure = max(1, int(np.nanmax(durations)))
        self.max_tenure_ = max_tenure
        tenure_grid = np.arange(1, max_tenure + 1)

        surv_probs = kmf.predict(tenure_grid)
        cum_hazard = naf.predict(tenure_grid)

        self.kmf_ = kmf
        self.naf_ = naf
        self.surv_probs_ = surv_probs
        self.cum_hazard_ = cum_hazard
        self.surv_map_ = dict(zip(tenure_grid, surv_probs))
        self.haz_map_ = dict(zip(tenure_grid, cum_hazard))
        self.last_surv_ = float(surv_probs.iloc[-1]) if len(surv_probs) > 0 else 1.0
        self.last_haz_ = float(cum_hazard.iloc[-1]) if len(cum_hazard) > 0 else 0.0
        self.is_fitted_ = True
        return self

    def transform(self, roster_df: pd.DataFrame) -> pd.DataFrame:
        """Map survival probabilities and cumulative hazard coordinates into DataFrame."""
        if not self.is_fitted_:
            raise ValueError("HazardEmbeddings has not been fitted yet. Call .fit() first.")
        if not isinstance(roster_df, pd.DataFrame):
            raise TypeError("roster_df must be a pandas DataFrame")
        if self.tenure_col not in roster_df.columns:
            raise KeyError(f"tenure_col '{self.tenure_col}' not found in roster_df")

        result_df = roster_df.copy()
        result_df["survival_prob_embedding"] = (
            result_df[self.tenure_col].map(self.surv_map_).fillna(self.last_surv_)
        )
        result_df["cumulative_hazard_embedding"] = (
            result_df[self.tenure_col].map(self.haz_map_).fillna(self.last_haz_)
        )
        return result_df

    def fit_transform(self, roster_df: pd.DataFrame, y: Any = None) -> pd.DataFrame:
        """Fit survival/hazard curves and map embeddings onto roster_df."""
        return self.fit(roster_df).transform(roster_df)

    def predict_survival(self, tenures: Union[int, float, Sequence[float], pd.Series, np.ndarray]) -> Union[float, pd.Series, np.ndarray]:
        """Predict survival probabilities for given tenure durations."""
        if not self.is_fitted_:
            raise ValueError("HazardEmbeddings has not been fitted yet. Call .fit() first.")
        return self.kmf_.predict(tenures)

    def predict_hazard(self, tenures: Union[int, float, Sequence[float], pd.Series, np.ndarray]) -> Union[float, pd.Series, np.ndarray]:
        """Predict cumulative hazard for given tenure durations."""
        if not self.is_fitted_:
            raise ValueError("HazardEmbeddings has not been fitted yet. Call .fit() first.")
        return self.naf_.predict(tenures)


HazardEmbeddingTransformer = HazardEmbeddings


def fit_hazard_embeddings(
    roster_df: pd.DataFrame,
    tenure_col: str = "tenure_weeks",
    event_col: str = "turnover_event",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Fit Kaplan-Meier survival curves and Nelson-Aalen cumulative hazard.
    
    Embeds non-linear behavioral turnover risk coordinates directly into each employee row.
    """
    model = HazardEmbeddings(tenure_col=tenure_col, event_col=event_col)
    model.fit(roster_df)
    result_df = model.transform(roster_df)
    models = {
        "kmf": model.kmf_,
        "naf": model.naf_,
        "survival_table": model.surv_probs_,
        "hazard_table": model.cum_hazard_,
        "model": model,
    }
    return result_df, models


# =============================================================================
# Feature 45: Configuration-Driven Rolling Metrics
# =============================================================================

@dataclass
class RollingMetricConfig:
    """Configuration specification for a rolling metric calculation.

    Attributes:
        target_column: Name of the numerical column to compute rolling statistics over.
        window: Integer window size in observation periods (e.g., 52 for 52-week trailing rate,
            13 for quarter, 4 for 4-week fatigue variance).
        aggregation: Aggregation function name ('mean', 'sum', 'var', 'variance', 'std',
            'min', 'max', 'median', 'skew', 'kurt', 'kurtosis', 'quantile') or custom callable.
        output_column: Optional custom name for the resulting feature column. If None,
            automatically defaults to f"{target_column}_roll_{aggregation}_{window}".
        min_periods: Minimum number of observations required in window to produce a value.
            Defaults to 1 to avoid unnecessary initial NaNs while maintaining temporal rigor.
        lag: Non-negative integer shift applied to the rolling window. For example, lag=1
            strictly uses periods t-W to t-1, preventing lookahead leakage in forecasting.
        closed: Rolling window boundary closure ('right', 'left', 'both', 'neither').
        center: Whether to set window labels at the center of the window (default False for causal trailing metrics).
        quantile: Quantile float between 0.0 and 1.0 when aggregation is 'quantile'.
    """

    target_column: str
    window: int
    aggregation: Union[str, Callable] = "mean"
    output_column: Optional[str] = None
    min_periods: Optional[int] = 1
    lag: int = 0
    closed: Optional[str] = None
    center: bool = False
    quantile: Optional[float] = None


def compute_rolling_metrics(
    df: pd.DataFrame,
    configs: Union[
        RollingMetricConfig,
        Dict[str, Any],
        List[Union[RollingMetricConfig, Dict[str, Any]]],
    ],
    partition_cols: Optional[Union[str, List[str]]] = None,
    temporal_col: Optional[Union[str, List[str]]] = None,
    ascending: bool = True,
    inplace: bool = False,
) -> pd.DataFrame:
    """Dynamically calculate trailing statistics across defined hierarchical partitions.

    Guarantees chronological integrity within specific cohorts (like Store and Department)
    without altering the original dataframe structure or row order.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing partition, temporal, and target metric columns.
    configs : RollingMetricConfig, dict, or list of configs/dicts
        Specification of rolling operations to compute. Can also be a mapping of
        {output_column_name: spec_dict}.
    partition_cols : str or list of str, optional
        Hierarchical partition column(s), e.g. ["store_id", "dept_id"]. If None,
        rolling statistics are computed across the entire dataframe globally.
    temporal_col : str or list of str, optional
        Column(s) specifying chronological order, e.g. "week" or ["year", "week"].
        If provided, rows are sorted internally by partition + temporal order before
        calculating rolling transforms, and results are mapped back to the original index.
    ascending : bool, default=True
        Sort direction when temporal_col is provided.
    inplace : bool, default=False
        If True, assigns features directly into `df`; otherwise returns a copy.

    Returns
    -------
    pd.DataFrame
        DataFrame with new rolling feature columns appended, preserving original row order.

    Raises
    ------
    TypeError
        If `df` is not a pandas DataFrame, or if `configs` has invalid types.
    KeyError
        If `partition_cols`, `temporal_col`, or specified target columns are missing from `df`.
    ValueError
        If window specifications are non-positive or unsupported aggregations requested.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected df to be a pandas DataFrame, got {type(df)}")

    # Normalize configs into a list of RollingMetricConfig
    parsed_configs: List[RollingMetricConfig] = []
    if isinstance(configs, dict):
        # Check if dict of dicts: {"col_name": {"target_column": "x", ...}}
        first_val = next(iter(configs.values())) if configs else None
        if isinstance(first_val, dict):
            for out_name, spec in configs.items():
                spec_copy = dict(spec)
                if "output_column" not in spec_copy:
                    spec_copy["output_column"] = out_name
                parsed_configs.append(RollingMetricConfig(**spec_copy))
        else:
            parsed_configs.append(RollingMetricConfig(**configs))
    elif isinstance(configs, RollingMetricConfig):
        parsed_configs.append(configs)
    elif isinstance(configs, (list, tuple)):
        for item in configs:
            if isinstance(item, dict):
                parsed_configs.append(RollingMetricConfig(**item))
            elif isinstance(item, RollingMetricConfig):
                parsed_configs.append(item)
            else:
                raise TypeError(f"Invalid config item type: {type(item)}")
    else:
        raise TypeError(f"Invalid configs parameter type: {type(configs)}")

    if not parsed_configs:
        return df if inplace else df.copy()

    # Normalize partition columns
    if partition_cols is None:
        p_cols: List[str] = []
    elif isinstance(partition_cols, str):
        p_cols = [partition_cols]
    else:
        p_cols = list(partition_cols)

    # Normalize temporal columns
    if temporal_col is None:
        t_cols: List[str] = []
    elif isinstance(temporal_col, str):
        t_cols = [temporal_col]
    else:
        t_cols = list(temporal_col)

    # Verify column existence
    missing_cols = [col for col in (p_cols + t_cols) if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Columns not found in DataFrame: {missing_cols}")

    for cfg in parsed_configs:
        if cfg.target_column not in df.columns:
            raise KeyError(f"Target column '{cfg.target_column}' not found in DataFrame")
        if cfg.window <= 0:
            raise ValueError(f"Window must be positive integer, got {cfg.window}")
        if cfg.lag < 0:
            raise ValueError(f"Lag must be non-negative integer, got {cfg.lag}")

    # Prepare working dataframe sorted by partitions + temporal order
    if t_cols:
        sort_keys = p_cols + t_cols
        work_df = df.sort_values(by=sort_keys, ascending=ascending)
    else:
        work_df = df

    out_df = df if inplace else df.copy()

    # Pre-group if partition columns specified
    if p_cols:
        grouped = work_df.groupby(p_cols, sort=False)
    else:
        grouped = None

    for cfg in parsed_configs:
        # Determine output column name
        if cfg.output_column:
            out_col = cfg.output_column
        else:
            agg_name = cfg.aggregation if isinstance(cfg.aggregation, str) else "custom"
            out_col = f"{cfg.target_column}_roll_{agg_name}_{cfg.window}"
            if cfg.lag > 0:
                out_col += f"_lag{cfg.lag}"

        # Define rolling calculation closure
        def _calc_series_rolling(s: pd.Series) -> pd.Series:
            r = s.rolling(
                window=cfg.window,
                min_periods=cfg.min_periods,
                center=cfg.center,
                closed=cfg.closed,
            )
            if callable(cfg.aggregation):
                res = r.apply(cfg.aggregation, raw=True)
            elif cfg.aggregation in ("var", "variance"):
                res = r.var()
            elif cfg.aggregation == "mean":
                res = r.mean()
            elif cfg.aggregation == "sum":
                res = r.sum()
            elif cfg.aggregation == "std":
                res = r.std()
            elif cfg.aggregation == "min":
                res = r.min()
            elif cfg.aggregation == "max":
                res = r.max()
            elif cfg.aggregation == "median":
                res = r.median()
            elif cfg.aggregation == "skew":
                res = r.skew()
            elif cfg.aggregation in ("kurt", "kurtosis"):
                res = r.kurt()
            elif cfg.aggregation == "quantile":
                q = 0.5 if cfg.quantile is None else cfg.quantile
                res = r.quantile(q)
            else:
                res = getattr(r, cfg.aggregation)()

            if cfg.lag > 0:
                res = res.shift(cfg.lag)
            return res

        if grouped is not None:
            computed = grouped[cfg.target_column].transform(_calc_series_rolling)
        else:
            computed = _calc_series_rolling(work_df[cfg.target_column])

        # Assign back aligned to original row index
        out_df[out_col] = computed.reindex(out_df.index)

    return out_df


class RollingMetricsPipeline:
    """Object-oriented pipeline for configuration-driven rolling feature engineering.

    Allows chaining metric definitions, fitting/transforming DataFrames, and inspecting
    engineered feature names.

    Example:
    --------
    >>> pipeline = (
    ...     RollingMetricsPipeline(partition_cols=["store_id", "dept_id"], temporal_col="week")
    ...     .add_metric("turnover_rate", window=52, aggregation="mean", output_column="turnover_52w_mean")
    ...     .add_metric("headcount", window=13, aggregation="mean", output_column="headcount_13w_baseline")
    ...     .add_metric("assigned_hours", window=4, aggregation="var", output_column="hours_4w_fatigue_var")
    ... )
    >>> result_df = pipeline.transform(store_df)
    """

    def __init__(
        self,
        partition_cols: Optional[Union[str, List[str]]] = None,
        temporal_col: Optional[Union[str, List[str]]] = None,
        ascending: bool = True,
    ):
        self.partition_cols = partition_cols
        self.temporal_col = temporal_col
        self.ascending = ascending
        self.configs: List[RollingMetricConfig] = []
        self.is_fitted_: bool = False
        self.feature_names_in_: Optional[List[str]] = None
        self.n_features_in_: Optional[int] = None
        self.feature_names_out_: Optional[List[str]] = None

    def add_metric(
        self,
        target_column: str,
        window: int,
        aggregation: Union[str, Callable] = "mean",
        output_column: Optional[str] = None,
        min_periods: Optional[int] = 1,
        lag: int = 0,
        closed: Optional[str] = None,
        center: bool = False,
        quantile: Optional[float] = None,
    ) -> "RollingMetricsPipeline":
        """Add a rolling metric configuration to the pipeline."""
        cfg = RollingMetricConfig(
            target_column=target_column,
            window=window,
            aggregation=aggregation,
            output_column=output_column,
            min_periods=min_periods,
            lag=lag,
            closed=closed,
            center=center,
            quantile=quantile,
        )
        self.configs.append(cfg)
        return self

    def fit(self, df: pd.DataFrame, y: Any = None) -> "RollingMetricsPipeline":
        """Validate input DataFrame schema and record fitted pipeline metadata.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame to validate against configured partition, temporal, and target columns.
        y : Any, optional
            Ignored. Present for scikit-learn API compatibility.

        Returns
        -------
        self : RollingMetricsPipeline
            The fitted pipeline instance.

        Raises
        ------
        TypeError
            If df is not a pandas DataFrame.
        KeyError
            If any partition, temporal, or configured metric target column is missing from df.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected df to be a pandas DataFrame, got {type(df)}")

        if self.partition_cols is not None:
            p_cols = [self.partition_cols] if isinstance(self.partition_cols, str) else list(self.partition_cols)
            for col in p_cols:
                if col not in df.columns:
                    raise KeyError(f"Partition column '{col}' not found in DataFrame")

        if self.temporal_col is not None:
            t_cols = [self.temporal_col] if isinstance(self.temporal_col, str) else list(self.temporal_col)
            for col in t_cols:
                if col not in df.columns:
                    raise KeyError(f"Temporal column '{col}' not found in DataFrame")

        for cfg in self.configs:
            if cfg.target_column not in df.columns:
                raise KeyError(
                    f"Target column '{cfg.target_column}' configured in rolling pipeline not found in DataFrame"
                )

        self.feature_names_in_ = list(df.columns)
        self.n_features_in_ = len(df.columns)
        self.feature_names_out_ = self.get_feature_names()
        self.is_fitted_ = True
        return self

    def transform(self, df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
        """Apply all configured rolling metrics to the DataFrame."""
        return compute_rolling_metrics(
            df=df,
            configs=self.configs,
            partition_cols=self.partition_cols,
            temporal_col=self.temporal_col,
            ascending=self.ascending,
            inplace=inplace,
        )

    def fit_transform(self, df: pd.DataFrame, y: Any = None, inplace: bool = False) -> pd.DataFrame:
        """Fit pipeline to DataFrame schema and generate rolling features."""
        return self.fit(df, y=y).transform(df, inplace=inplace)

    def get_feature_names(self) -> List[str]:
        """Return the list of engineered rolling feature column names."""
        names = []
        for cfg in self.configs:
            if cfg.output_column:
                names.append(cfg.output_column)
            else:
                agg_name = cfg.aggregation if isinstance(cfg.aggregation, str) else "custom"
                col = f"{cfg.target_column}_roll_{agg_name}_{cfg.window}"
                if cfg.lag > 0:
                    col += f"_lag{cfg.lag}"
                names.append(col)
        return names


# =============================================================================
# Feature 46: Singular Spectrum Analysis (SVD Smoothing)
# =============================================================================

def singular_spectrum_analysis(
    series: Union[pd.Series, np.ndarray],
    window_length: Optional[int] = None,
    top_k: Optional[int] = 3,
    variance_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Apply Singular Spectrum Analysis (SSA) to extract lag-free smoothed baselines.

    Embeds a 1D time series into a multidimensional trajectory Hankel matrix,
    applies Singular Value Decomposition (SVD), selects the dominant top k
    singular components (or components meeting a cumulative variance threshold),
    and reconstructs the signal via anti-diagonal averaging.

    Separates core structural signal (trend and dominant seasonal cycles) from
    high-frequency random noise without introducing temporal phase-lag.

    Parameters
    ----------
    series : pd.Series or np.ndarray
        1D temporal sequence (e.g., daily store foot traffic or labor demand).
    window_length : int, optional
        Embedding dimension (Hankel window length L). Must satisfy 2 <= L <= N/2.
        If None, defaults to min(max(4, N // 3), 30).
    top_k : int, optional, default=3
        Number of leading singular components to retain for signal reconstruction.
        Ignored if variance_threshold is specified.
    variance_threshold : float, optional
        Target cumulative singular variance ratio in (0.0, 1.0] (e.g., 0.90 for 90%).
        If provided, automatically determines k.

    Returns
    -------
    dict
        - 'smoothed': pd.Series of lag-free reconstructed baseline signal.
        - 'residual': pd.Series of isolated high-frequency noise.
        - 'singular_values': np.ndarray of singular values Sigma.
        - 'explained_variance_ratio': np.ndarray of singular energy proportion.
        - 'n_components_used': int number of components retained.
        - 'components_df': pd.DataFrame with individual elementary reconstructed series.
        - 'imputed_mask': pd.Series of bool indicating which observations were missing (NaN) and linearly interpolated.
        - 'n_imputed': int total count of missing observations that were interpolated.

    Warns
    -----
    UserWarning
        If series contains missing values (NaNs) that require linear interpolation prior to Hankel embedding.

    Raises
    ------
    ValueError
        If series has fewer than 4 observations, contains only NaNs, or variance_threshold not in (0, 1].
    """
    if isinstance(series, pd.Series):
        s = series.to_numpy(dtype=float)
        orig_index = series.index
        orig_name = series.name or "signal"
    else:
        s = np.asarray(series, dtype=float)
        orig_index = pd.RangeIndex(len(s))
        orig_name = "signal"

    N = len(s)
    if N < 4:
        raise ValueError(f"Series length N={N} is too short for Singular Spectrum Analysis (minimum 4 observations).")

    # Determine embedding window length L
    if window_length is None:
        L = min(max(4, N // 3), 30)
    else:
        L = int(window_length)
        if L < 2:
            L = 2
        elif L > N // 2:
            L = N // 2

    K = N - L + 1

    # Handle missing values if any
    nan_mask = np.isnan(s)
    n_imputed = int(np.sum(nan_mask))
    if n_imputed > 0:
        valid_idx = np.where(~nan_mask)[0]
        if len(valid_idx) == 0:
            raise ValueError("Series contains only NaNs.")
        warnings.warn(
            f"Input series contains {n_imputed} missing value(s) (NaNs). "
            "Linear interpolation was applied prior to Hankel embedding.",
            UserWarning,
            stacklevel=2,
        )
        s = np.interp(np.arange(N), valid_idx, s[valid_idx])

    # 1. Embedding: Construct trajectory Hankel matrix X in R^{L x K} via scipy.linalg.hankel
    X = hankel(s[:L], s[L - 1 :])

    # 2. Singular Value Decomposition
    U, Sigma, Vt = np.linalg.svd(X, full_matrices=False)
    var_explained = (Sigma ** 2) / np.sum(Sigma ** 2)
    cum_var = np.cumsum(var_explained)

    # 3. Component Selection
    if variance_threshold is not None:
        vt = float(variance_threshold)
        if not (0.0 < vt <= 1.0):
            raise ValueError(f"variance_threshold must be in (0, 1], got {vt}")
        k = int(np.searchsorted(cum_var, vt)) + 1
        k = min(max(1, k), len(Sigma))
    elif top_k is not None:
        k = min(max(1, int(top_k)), len(Sigma))
    else:
        k = min(3, len(Sigma))

    # 4. Trajectory Reconstruction & Diagonal Averaging (Hankelization)
    X_rec = np.dot(U[:, :k] * Sigma[:k], Vt[:k, :])

    y_rec = np.zeros(N, dtype=float)
    counts = np.zeros(N, dtype=float)
    for i in range(L):
        y_rec[i : i + K] += X_rec[i, :]
        counts[i : i + K] += 1.0
    y_rec /= counts

    # Decompose individual elementary components
    comp_dict: Dict[str, pd.Series] = {}
    max_decomp = min(k, 12)
    for c_i in range(max_decomp):
        Xi = np.outer(U[:, c_i] * Sigma[c_i], Vt[c_i, :])
        yi = np.zeros(N, dtype=float)
        for i in range(L):
            yi[i : i + K] += Xi[i, :]
        comp_dict[f"component_{c_i+1}"] = pd.Series(yi / counts, index=orig_index)

    smoothed_series = pd.Series(y_rec, index=orig_index, name=f"{orig_name}_svd_smoothed")
    residual_series = pd.Series(s - y_rec, index=orig_index, name=f"{orig_name}_svd_residual")
    components_df = pd.DataFrame(comp_dict, index=orig_index)
    imputed_mask_series = pd.Series(nan_mask, index=orig_index, name=f"{orig_name}_imputed")

    return {
        "smoothed": smoothed_series,
        "residual": residual_series,
        "singular_values": Sigma,
        "explained_variance_ratio": var_explained,
        "n_components_used": k,
        "components_df": components_df,
        "imputed_mask": imputed_mask_series,
        "n_imputed": n_imputed,
    }


def compute_svd_smoothed_baseline(
    df: pd.DataFrame,
    target_col: str,
    output_col: Optional[str] = None,
    residual_col: Optional[str] = None,
    imputed_col: Optional[str] = None,
    window_length: Optional[int] = None,
    top_k: Optional[int] = 3,
    variance_threshold: Optional[float] = None,
    partition_cols: Optional[Union[str, List[str]]] = None,
    temporal_col: Optional[Union[str, List[str]]] = None,
    ascending: bool = True,
    inplace: bool = False,
) -> pd.DataFrame:
    """Compute lag-free SVD-smoothed baseline features across DataFrame partitions.

    Applies Singular Spectrum Analysis (SSA) independently across hierarchical
    cohorts (e.g., per Store), appending smoothed baseline and noise residual columns
    without phase-lag or data leakage.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing temporal data.
    target_col : str
        Column name of the erratic numerical series to smooth.
    output_col : str, optional
        Name of the reconstructed baseline column (defaults to f"{target_col}_svd_smoothed").
    residual_col : str, optional
        Name of the noise residual column (defaults to f"{target_col}_svd_residual").
    imputed_col : str, optional
        Name of the boolean indicator column marking rows where missing values were linearly interpolated.
        If None, the column is omitted.
    window_length : int, optional
        Embedding dimension L. Must satisfy 2 <= L <= N/2.
    top_k : int, optional, default=3
        Number of leading singular components to retain.
    variance_threshold : float, optional
        Target cumulative singular variance ratio in (0, 1].
    partition_cols : str or list of str, optional
        Cohort grouping column(s) (e.g. "store_id"). If None, applies globally.
    temporal_col : str or list of str, optional
        Column(s) specifying chronological order.
    ascending : bool, default=True
        Sort order when temporal_col is specified.
    inplace : bool, default=False
        Whether to modify DataFrame in-place.

    Returns
    -------
    pd.DataFrame
        DataFrame with smoothed baseline and residual features appended, preserving original row order.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected df to be a pandas DataFrame, got {type(df)}")
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame")

    out_name = output_col or f"{target_col}_svd_smoothed"
    res_name = residual_col or f"{target_col}_svd_residual"

    out_df = df if inplace else df.copy()

    if df.empty:
        out_df[out_name] = pd.Series(dtype=float)
        out_df[res_name] = pd.Series(dtype=float)
        if imputed_col:
            out_df[imputed_col] = pd.Series(dtype=bool)
        return out_df

    # Normalize partition columns
    if partition_cols is None:
        p_cols: List[str] = []
    elif isinstance(partition_cols, str):
        p_cols = [partition_cols]
    else:
        p_cols = list(partition_cols)

    # Normalize temporal columns
    if temporal_col is None:
        t_cols: List[str] = []
    elif isinstance(temporal_col, str):
        t_cols = [temporal_col]
    else:
        t_cols = list(temporal_col)

    # Working sorted copy
    if t_cols:
        sort_keys = p_cols + t_cols
        work_df = df.sort_values(by=sort_keys, ascending=ascending)
    else:
        work_df = df

    if p_cols:
        smoothed_parts = []
        residual_parts = []
        imputed_parts = []
        for _, group in work_df.groupby(p_cols, sort=False):
            ssa_res = singular_spectrum_analysis(
                series=group[target_col],
                window_length=window_length,
                top_k=top_k,
                variance_threshold=variance_threshold,
            )
            smoothed_parts.append(ssa_res["smoothed"])
            residual_parts.append(ssa_res["residual"])
            if imputed_col:
                imputed_parts.append(ssa_res["imputed_mask"])
        combined_smoothed = pd.concat(smoothed_parts)
        combined_residual = pd.concat(residual_parts)
        combined_imputed = pd.concat(imputed_parts) if imputed_col else None
    else:
        ssa_res = singular_spectrum_analysis(
            series=work_df[target_col],
            window_length=window_length,
            top_k=top_k,
            variance_threshold=variance_threshold,
        )
        combined_smoothed = ssa_res["smoothed"]
        combined_residual = ssa_res["residual"]
        combined_imputed = ssa_res["imputed_mask"] if imputed_col else None

    # Align back to original row index
    out_df[out_name] = combined_smoothed.reindex(out_df.index)
    out_df[res_name] = combined_residual.reindex(out_df.index)
    if imputed_col:
        out_df[imputed_col] = combined_imputed.reindex(out_df.index).fillna(False).astype(bool)

    return out_df


class SingularSpectrumAnalysis:
    """Non-parametric time-series decomposition and smoothing via Singular Spectrum Analysis (SSA).

    Embeds a 1D time series into a multidimensional trajectory matrix, applies SVD,
    and reconstructs lag-free trend and seasonal baselines via anti-diagonal averaging.

    Parameters
    ----------
    window_length : int, optional
        Embedding window length L. Must satisfy 2 <= L <= N/2.
    top_k : int, default=3
        Number of leading singular components to retain.
    variance_threshold : float, optional
        Target cumulative singular variance ratio in (0, 1].

    Examples
    --------
    >>> ssa = SingularSpectrumAnalysis(window_length=14, top_k=3)
    >>> smoothed_traffic = ssa.fit_transform(daily_foot_traffic)
    """

    def __init__(
        self,
        window_length: Optional[int] = None,
        top_k: Optional[int] = 3,
        variance_threshold: Optional[float] = None,
    ):
        self.window_length = window_length
        self.top_k = top_k
        self.variance_threshold = variance_threshold
        self.decomposition_: Optional[Dict[str, Any]] = None

    def fit(self, y: Union[pd.Series, np.ndarray]) -> "SingularSpectrumAnalysis":
        """Fit the SSA model and decompose the input series."""
        self.decomposition_ = singular_spectrum_analysis(
            series=y,
            window_length=self.window_length,
            top_k=self.top_k,
            variance_threshold=self.variance_threshold,
        )
        return self

    def transform(self, y: Optional[Union[pd.Series, np.ndarray]] = None) -> pd.Series:
        """Return the reconstructed lag-free smoothed baseline series."""
        if y is not None:
            self.fit(y)
        if self.decomposition_ is None:
            raise ValueError("Model is not fitted. Call fit() before transform().")
        return self.decomposition_["smoothed"]

    def fit_transform(self, y: Union[pd.Series, np.ndarray]) -> pd.Series:
        """Fit the SSA model and return the smoothed baseline."""
        return self.fit(y).transform()

    def decompose(self, y: Optional[Union[pd.Series, np.ndarray]] = None) -> pd.DataFrame:
        """Return the individual elementary reconstructed components."""
        if y is not None:
            self.fit(y)
        if self.decomposition_ is None:
            raise ValueError("Model is not fitted. Call fit() before decompose().")
        return self.decomposition_["components_df"]

    @property
    def singular_values_(self) -> np.ndarray:
        if self.decomposition_ is None:
            raise ValueError("Model is not fitted.")
        return self.decomposition_["singular_values"]

    @property
    def explained_variance_ratio_(self) -> np.ndarray:
        if self.decomposition_ is None:
            raise ValueError("Model is not fitted.")
        return self.decomposition_["explained_variance_ratio"]

    @property
    def n_components_used_(self) -> int:
        if self.decomposition_ is None:
            raise ValueError("Model is not fitted.")
        return self.decomposition_["n_components_used"]

    @property
    def imputed_mask_(self) -> pd.Series:
        """Return boolean mask indicating which observations were missing and linearly interpolated."""
        if self.decomposition_ is None:
            raise ValueError("Model is not fitted. Call fit() before accessing imputed_mask_.")
        return self.decomposition_["imputed_mask"]

    @property
    def n_imputed_(self) -> int:
        """Return total count of missing observations that were interpolated."""
        if self.decomposition_ is None:
            raise ValueError("Model is not fitted. Call fit() before accessing n_imputed_.")
        return self.decomposition_["n_imputed"]

    @property
    def has_imputed_(self) -> bool:
        """Return True if any missing values were detected and interpolated during fitting."""
        if self.decomposition_ is None:
            raise ValueError("Model is not fitted. Call fit() before accessing has_imputed_.")
        return bool(self.decomposition_["n_imputed"] > 0)


# =============================================================================
# Feature 47: Hierarchical Lag & Autoregressive Features
# =============================================================================

def compute_lag_features(
    df: pd.DataFrame,
    target_cols: Union[str, List[str]],
    lags: Union[int, List[int], Dict[str, int]] = 1,
    partition_cols: Optional[Union[str, List[str]]] = None,
    temporal_col: Optional[Union[str, List[str]]] = None,
    ascending: bool = True,
    prefix: Optional[str] = None,
    inplace: bool = False,
) -> pd.DataFrame:
    """Compute autoregressive historical lag features across hierarchical partitions.

    Guarantees chronological integrity within specific cohorts (e.g., Store and Department)
    and eliminates target leakage without altering original DataFrame row ordering or indices.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing temporal and metric columns.
    target_cols : str or list of str
        Target column(s) to compute historical lags for.
    lags : int, list of int, or dict of {name: int_lag}, default=1
        Lag orders to generate (e.g., [1, 2, 4, 13, 52] or {"prior_wk": 1, "prior_yr": 52}).
    partition_cols : str or list of str, optional
        Hierarchical cohort partition column(s). Lags strictly respect partition boundaries.
    temporal_col : str or list of str, optional
        Column(s) defining chronological order.
    ascending : bool, default=True
        Sort direction when temporal_col is provided.
    prefix : str, optional
        Custom prefix for generated column names.
    inplace : bool, default=False
        Whether to assign columns into `df` or return a copy.

    Returns
    -------
    pd.DataFrame
        DataFrame with new lag feature columns appended, preserving original row order.

    Raises
    ------
    TypeError
        If `df` is not a pandas DataFrame or `lags` type is invalid.
    KeyError
        If target columns are missing from `df`.
    ValueError
        If any specified lag is not a positive integer.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected df to be a pandas DataFrame, got {type(df)}")

    # Normalize target columns
    if isinstance(target_cols, str):
        t_target_cols = [target_cols]
    else:
        t_target_cols = list(target_cols)

    missing = [c for c in t_target_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Target columns not found in DataFrame: {missing}")

    # Normalize lags into dict of {col_suffix: int_lag}
    if isinstance(lags, int):
        if lags <= 0:
            raise ValueError(f"Lag must be a positive integer, got {lags}")
        lag_dict = {f"lag_{lags}": lags}
    elif isinstance(lags, (list, tuple)):
        lag_dict = {}
        for l in lags:
            if not isinstance(l, int) or l <= 0:
                raise ValueError(f"Each lag must be a positive integer, got {l}")
            lag_dict[f"lag_{l}"] = l
    elif isinstance(lags, dict):
        lag_dict = {}
        for name, l in lags.items():
            if not isinstance(l, int) or l <= 0:
                raise ValueError(f"Each lag must be a positive integer, got {l}")
            lag_dict[str(name)] = l
    else:
        raise TypeError(f"Invalid type for lags: {type(lags)}")

    out_df = df if inplace else df.copy()

    # Normalize partition and temporal columns
    if partition_cols is None:
        p_cols: List[str] = []
    elif isinstance(partition_cols, str):
        p_cols = [partition_cols]
    else:
        p_cols = list(partition_cols)

    if temporal_col is None:
        t_cols: List[str] = []
    elif isinstance(temporal_col, str):
        t_cols = [temporal_col]
    else:
        t_cols = list(temporal_col)

    # Sort working view chronologically within partitions
    if t_cols:
        sort_keys = p_cols + t_cols
        work_df = df.sort_values(by=sort_keys, ascending=ascending)
    else:
        work_df = df

    if p_cols:
        grouped = work_df.groupby(p_cols, sort=False)
    else:
        grouped = None

    for target_col in t_target_cols:
        for suffix, lag_val in lag_dict.items():
            if prefix:
                out_col = f"{prefix}_{target_col}_{suffix}"
            else:
                out_col = f"{target_col}_{suffix}"

            if grouped is not None:
                computed = grouped[target_col].shift(lag_val)
            else:
                computed = work_df[target_col].shift(lag_val)

            out_df[out_col] = computed.reindex(out_df.index)

    return out_df


# =============================================================================
# Feature 48: Multi-Horizon Differencing & Growth Rates (YoY, MoM, WoW & % Change)
# =============================================================================

def compute_differencing_features(
    df: pd.DataFrame,
    target_cols: Union[str, List[str]],
    periods: Union[int, List[int], Dict[str, int], str] = 1,
    pct_change: bool = False,
    as_percent: bool = False,
    partition_cols: Optional[Union[str, List[str]]] = None,
    temporal_col: Optional[Union[str, List[str]]] = None,
    ascending: bool = True,
    prefix: Optional[str] = None,
    inplace: bool = False,
) -> pd.DataFrame:
    """Compute multi-horizon discrete differencing and percentage growth rates.

    Supports absolute deltas (X_t - X_{t-s}) and relative percentage changes
    ((X_t - X_{t-s}) / X_{t-s}) across standard temporal horizons (YoY, MoM, WoW, QoQ)
    within hierarchical cohorts without cross-partition leakage.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    target_cols : str or list of str
        Column(s) to compute differences or percent changes for.
    periods : int, list of int, dict of {name: int_period}, or str preset, default=1
        - Predefined string presets: 'weekly' (WoW=1, MoM=4, QoQ=13, YoY=52),
          'monthly' (MoM=1, QoQ=3, YoY=12), or 'daily' (DoD=1, WoW=7, MoM=30, YoY=365).
        - Dict: e.g. {"wow": 1, "mom": 4, "yoy": 52}.
        - List or int: e.g. [1, 4, 52] or 1.
    pct_change : bool, default=False
        If True, calculates relative percent change (X_t - X_{t-s}) / X_{t-s}.
        If False, calculates discrete difference (X_t - X_{t-s}).
    as_percent : bool, default=False
        If True and pct_change=True, scales relative change to percentage (x 100).
    partition_cols : str or list of str, optional
        Cohort grouping column(s). Differencing does not cross cohort boundaries.
    temporal_col : str or list of str, optional
        Chronological order column(s).
    ascending : bool, default=True
        Sort direction.
    prefix : str, optional
        Optional prefix for output column names.
    inplace : bool, default=False
        Whether to modify DataFrame in-place.

    Returns
    -------
    pd.DataFrame
        DataFrame with new differencing and growth rate features appended, preserving original row order.

    Raises
    ------
    TypeError
        If `df` is not a pandas DataFrame or periods configuration is invalid.
    KeyError
        If target columns are missing from `df`.
    ValueError
        If any specified period is not a positive integer.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected df to be a pandas DataFrame, got {type(df)}")

    # Normalize target columns
    if isinstance(target_cols, str):
        t_target_cols = [target_cols]
    else:
        t_target_cols = list(target_cols)

    missing = [c for c in t_target_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Target columns not found in DataFrame: {missing}")

    # Presets mapping
    presets = {
        "weekly": {"wow": 1, "mom": 4, "qoq": 13, "yoy": 52},
        "monthly": {"mom": 1, "qoq": 3, "yoy": 12},
        "daily": {"dod": 1, "wow": 7, "mom": 30, "yoy": 365},
    }

    # Normalize periods into dict of {name: int_period}
    if isinstance(periods, str):
        preset_key = periods.lower()
        if preset_key not in presets:
            raise ValueError(f"Unknown periods preset '{periods}'. Available: {list(presets.keys())}")
        period_dict = presets[preset_key]
    elif isinstance(periods, int):
        if periods <= 0:
            raise ValueError(f"Period must be a positive integer, got {periods}")
        period_dict = {f"diff_{periods}": periods}
    elif isinstance(periods, (list, tuple)):
        period_dict = {}
        for p in periods:
            if not isinstance(p, int) or p <= 0:
                raise ValueError(f"Each period must be a positive integer, got {p}")
            period_dict[f"diff_{p}"] = p
    elif isinstance(periods, dict):
        period_dict = {}
        for name, p in periods.items():
            if not isinstance(p, int) or p <= 0:
                raise ValueError(f"Each period must be a positive integer, got {p}")
            period_dict[str(name)] = p
    else:
        raise TypeError(f"Invalid type for periods: {type(periods)}")

    out_df = df if inplace else df.copy()

    # Normalize partition and temporal columns
    if partition_cols is None:
        p_cols: List[str] = []
    elif isinstance(partition_cols, str):
        p_cols = [partition_cols]
    else:
        p_cols = list(partition_cols)

    if temporal_col is None:
        t_cols: List[str] = []
    elif isinstance(temporal_col, str):
        t_cols = [temporal_col]
    else:
        t_cols = list(temporal_col)

    # Sort working copy chronologically within partitions
    if t_cols:
        sort_keys = p_cols + t_cols
        work_df = df.sort_values(by=sort_keys, ascending=ascending)
    else:
        work_df = df

    if p_cols:
        grouped = work_df.groupby(p_cols, sort=False)
    else:
        grouped = None

    for target_col in t_target_cols:
        for suffix, period_val in period_dict.items():
            # Determine output column name
            col_suffix = suffix
            if pct_change and not col_suffix.lower().endswith("pct"):
                col_suffix = f"{col_suffix}_pct"

            if prefix:
                out_col = f"{prefix}_{target_col}_{col_suffix}"
            else:
                out_col = f"{target_col}_{col_suffix}"

            if grouped is not None:
                if pct_change:
                    shifted = grouped[target_col].shift(period_val)
                    curr = work_df[target_col]
                    with np.errstate(divide="ignore", invalid="ignore"):
                        computed = (curr - shifted) / shifted
                        computed = computed.replace([np.inf, -np.inf], np.nan)
                    if as_percent:
                        computed = computed * 100.0
                else:
                    computed = grouped[target_col].diff(period_val)
            else:
                if pct_change:
                    shifted = work_df[target_col].shift(period_val)
                    curr = work_df[target_col]
                    with np.errstate(divide="ignore", invalid="ignore"):
                        computed = (curr - shifted) / shifted
                        computed = computed.replace([np.inf, -np.inf], np.nan)
                    if as_percent:
                        computed = computed * 100.0
                else:
                    computed = work_df[target_col].diff(period_val)

            out_df[out_col] = computed.reindex(out_df.index)

    return out_df


class TemporalFeaturesPipeline:
    """Fluent pipeline for chaining hierarchical lags, multi-horizon differencing, and growth rates.

    Examples
    --------
    >>> pipeline = (
    ...     TemporalFeaturesPipeline(partition_cols=["store_id", "dept_id"], temporal_col="week")
    ...     .add_lags("headcount", lags=[1, 4, 52])
    ...     .add_differencing("headcount", periods={"wow": 1, "yoy": 52}, pct_change=False)
    ...     .add_differencing("turnover_rate", periods={"mom": 4, "yoy": 52}, pct_change=True, as_percent=True)
    ... )
    >>> result_df = pipeline.transform(df)
    """

    def __init__(
        self,
        partition_cols: Optional[Union[str, List[str]]] = None,
        temporal_col: Optional[Union[str, List[str]]] = None,
        ascending: bool = True,
    ):
        self.partition_cols = partition_cols
        self.temporal_col = temporal_col
        self.ascending = ascending
        self.steps: List[Tuple[str, Dict[str, Any]]] = []
        self.is_fitted_: bool = False
        self.feature_names_in_: Optional[List[str]] = None
        self.n_features_in_: Optional[int] = None
        self.feature_names_out_: Optional[List[str]] = None

    def add_lags(
        self,
        target_cols: Union[str, List[str]],
        lags: Union[int, List[int], Dict[str, int]] = 1,
        prefix: Optional[str] = None,
    ) -> "TemporalFeaturesPipeline":
        """Add lag feature generation step."""
        self.steps.append((
            "lags",
            {
                "target_cols": target_cols,
                "lags": lags,
                "prefix": prefix,
            }
        ))
        return self

    def add_differencing(
        self,
        target_cols: Union[str, List[str]],
        periods: Union[int, List[int], Dict[str, int], str] = 1,
        pct_change: bool = False,
        as_percent: bool = False,
        prefix: Optional[str] = None,
    ) -> "TemporalFeaturesPipeline":
        """Add differencing or percent change generation step."""
        self.steps.append((
            "diff",
            {
                "target_cols": target_cols,
                "periods": periods,
                "pct_change": pct_change,
                "as_percent": as_percent,
                "prefix": prefix,
            }
        ))
        return self

    def transform(self, df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
        """Apply all configured temporal feature steps to the DataFrame."""
        out = df if inplace else df.copy()
        for step_type, kwargs in self.steps:
            if step_type == "lags":
                out = compute_lag_features(
                    df=out,
                    partition_cols=self.partition_cols,
                    temporal_col=self.temporal_col,
                    ascending=self.ascending,
                    inplace=True,
                    **kwargs,
                )
            elif step_type == "diff":
                out = compute_differencing_features(
                    df=out,
                    partition_cols=self.partition_cols,
                    temporal_col=self.temporal_col,
                    ascending=self.ascending,
                    inplace=True,
                    **kwargs,
                )
        return out

    def fit(self, df: pd.DataFrame, y: Any = None) -> "TemporalFeaturesPipeline":
        """Validate DataFrame schema and record fitted pipeline metadata."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected df to be a pandas DataFrame, got {type(df)}")

        if self.partition_cols is not None:
            p_cols = [self.partition_cols] if isinstance(self.partition_cols, str) else list(self.partition_cols)
            for col in p_cols:
                if col not in df.columns:
                    raise KeyError(f"Partition column '{col}' not found in DataFrame")

        if self.temporal_col is not None:
            t_cols = [self.temporal_col] if isinstance(self.temporal_col, str) else list(self.temporal_col)
            for col in t_cols:
                if col not in df.columns:
                    raise KeyError(f"Temporal column '{col}' not found in DataFrame")

        for step_type, kwargs in self.steps:
            target_cols = [kwargs["target_cols"]] if isinstance(kwargs["target_cols"], str) else list(kwargs["target_cols"])
            for col in target_cols:
                if col not in df.columns:
                    raise KeyError(f"Target column '{col}' in step '{step_type}' not found in DataFrame")

        self.feature_names_in_ = list(df.columns)
        self.n_features_in_ = len(df.columns)
        self.feature_names_out_ = self.get_feature_names()
        self.is_fitted_ = True
        return self

    def fit_transform(self, df: pd.DataFrame, y: Any = None, inplace: bool = False) -> pd.DataFrame:
        """Fit pipeline to DataFrame schema and generate temporal lag and difference features."""
        return self.fit(df, y=y).transform(df, inplace=inplace)

    def get_feature_names(self) -> List[str]:
        """Return the list of generated feature column names."""
        names: List[str] = []
        for step_type, kwargs in self.steps:
            targets = [kwargs["target_cols"]] if isinstance(kwargs["target_cols"], str) else list(kwargs["target_cols"])
            prefix = kwargs.get("prefix")
            if step_type == "lags":
                lags = kwargs["lags"]
                if isinstance(lags, int):
                    lag_dict = {f"lag_{lags}": lags}
                elif isinstance(lags, (list, tuple)):
                    lag_dict = {f"lag_{l}": l for l in lags}
                elif isinstance(lags, dict):
                    lag_dict = lags
                else:
                    lag_dict = {}
                for t in targets:
                    for sfx in lag_dict.keys():
                        col = f"{prefix}_{t}_{sfx}" if prefix else f"{t}_{sfx}"
                        names.append(col)
            elif step_type == "diff":
                periods = kwargs["periods"]
                presets = {
                    "weekly": {"wow": 1, "mom": 4, "qoq": 13, "yoy": 52},
                    "monthly": {"mom": 1, "qoq": 3, "yoy": 12},
                    "daily": {"dod": 1, "wow": 7, "mom": 30, "yoy": 365},
                }
                if isinstance(periods, str) and periods.lower() in presets:
                    period_dict = presets[periods.lower()]
                elif isinstance(periods, dict):
                    period_dict = periods
                elif isinstance(periods, (list, tuple)):
                    period_dict = {f"diff_{p}": p for p in periods}
                elif isinstance(periods, int):
                    period_dict = {f"diff_{periods}": periods}
                else:
                    period_dict = {}
                is_pct = kwargs.get("pct_change", False)
                for t in targets:
                    for sfx in period_dict.keys():
                        col = f"{prefix}_{t}_{sfx}" if prefix else f"{t}_{sfx}"
                        if is_pct:
                            col += "_pct"
                        names.append(col)
        return names


# API Consistency Aliases: compute_* <=> calculate_* (Deprecated in favor of canonical names)
compute_ewma = deprecated_alias(calculate_ewma, "compute_ewma", "calculate_ewma")
compute_ewms = deprecated_alias(calculate_ewms, "compute_ewms", "calculate_ewms")
compute_poisson_ewma = deprecated_alias(calculate_poisson_ewma, "compute_poisson_ewma", "calculate_poisson_ewma")
calculate_rolling_metrics = deprecated_alias(compute_rolling_metrics, "calculate_rolling_metrics", "compute_rolling_metrics")
calculate_svd_smoothed_baseline = deprecated_alias(compute_svd_smoothed_baseline, "calculate_svd_smoothed_baseline", "compute_svd_smoothed_baseline")
calculate_lag_features = deprecated_alias(compute_lag_features, "calculate_lag_features", "compute_lag_features")
calculate_differencing_features = deprecated_alias(compute_differencing_features, "calculate_differencing_features", "compute_differencing_features")

__all__ = [
    "fractional_difference",
    "get_fractional_weights",
    "scan_fractional_differencing",
    "weighted_weeks_since",
    "calculate_ewma",
    "compute_ewma",
    "scan_ewma_spans",
    "calculate_ewms",
    "compute_ewms",
    "calculate_poisson_ewma",
    "compute_poisson_ewma",
    "ExponentialSmoothingResult",
    "SmoothingResult",
    "double_exponential_smoothing",
    "triple_exponential_smoothing",
    "HazardEmbeddings",
    "HazardEmbeddingTransformer",
    "fit_hazard_embeddings",
    "RollingMetricConfig",
    "compute_rolling_metrics",
    "calculate_rolling_metrics",
    "RollingMetricsPipeline",
    "singular_spectrum_analysis",
    "compute_svd_smoothed_baseline",
    "calculate_svd_smoothed_baseline",
    "SingularSpectrumAnalysis",
    "compute_lag_features",
    "calculate_lag_features",
    "compute_differencing_features",
    "calculate_differencing_features",
    "TemporalFeaturesPipeline",
]



