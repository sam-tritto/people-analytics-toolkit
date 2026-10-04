"""Attribution and Variance Accounting Feature Engineering.

Contains:
1. Log-Mean Divisia Index (LMDI-I): Structural decomposition method that breaks down
   year-over-year changes in company-wide turnover into Rate Effect vs Structural Mix Effect
   with mathematical zero residuals.
2. Leave-One-Out Expected Differential (LOO-ED): Isolates marginal manager value.
3. SHAP Interaction Values & Inflection Thresholds: 2D interaction decomposition and zero-crossing detection.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from people_analytics_toolkit._deprecation import deprecated_alias


def logarithmic_mean(a: float, b: float, epsilon: float = 1e-12) -> float:
    """Calculate the logarithmic mean L(a, b) = (a - b) / (ln(a) - ln(b)).
    
    Handles boundary cases when a == b or values are close to zero.
    """
    if a <= 0.0 or b <= 0.0:
        return 0.0
    if abs(a - b) < epsilon:
        return float(a)
    return float((a - b) / (np.log(a) - np.log(b)))


def lmdi_rate_mix_decomposition(
    df: pd.DataFrame,
    dept_col: str = "department",
    h0_col: str = "headcount_y0",
    r0_col: str = "turnover_rate_y0",
    h1_col: str = "headcount_y1",
    r1_col: str = "turnover_rate_y1",
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Perform additive LMDI-I decomposition of total enterprise turnover rate change.
    
    Enterprise turnover rate:
        R = sum_i s_i * r_i
    Where:
        s_i = headcount share of department i (H_i / H_total)
        r_i = turnover rate of department i
        
    Change:
        Delta R = R^1 - R^0 = Delta R_rate + Delta R_mix
        
    LMDI Weights:
        w_i = s_i * r_i
        L_i = L(w_i^1, w_i^0)
        
    Decomposed components:
        Delta R_rate = sum_i L_i * ln(r_i^1 / r_i^0)
        Delta R_mix  = sum_i L_i * ln(s_i^1 / s_i^0)
        
    Leaves zero residuals: Delta R_rate + Delta R_mix - Delta R == 0.000000.
    """
    data = df.copy()
    
    total_h0 = data[h0_col].sum()
    total_h1 = data[h1_col].sum()
    
    data["share_y0"] = data[h0_col] / total_h0
    data["share_y1"] = data[h1_col] / total_h1
    
    data["weight_y0"] = data["share_y0"] * data[r0_col]
    data["weight_y1"] = data["share_y1"] * data[r1_col]
    
    # Calculate logarithmic mean weight L(w1, w0)
    data["log_mean_weight"] = [
        logarithmic_mean(w1, w0) for w1, w0 in zip(data["weight_y1"], data["weight_y0"])
    ]
    
    # Rate and Mix effects per department
    data["rate_effect"] = data["log_mean_weight"] * np.log(data[r1_col] / data[r0_col])
    data["mix_effect"] = data["log_mean_weight"] * np.log(data["share_y1"] / data["share_y0"])
    data["total_department_effect"] = data["rate_effect"] + data["mix_effect"]
    
    # Enterprise aggregates
    r_y0 = data["weight_y0"].sum()
    r_y1 = data["weight_y1"].sum()
    delta_r_actual = r_y1 - r_y0
    
    total_rate_effect = data["rate_effect"].sum()
    total_mix_effect = data["mix_effect"].sum()
    total_decomposed = total_rate_effect + total_mix_effect
    residual = total_decomposed - delta_r_actual
    
    summary = {
        "turnover_rate_y0": float(r_y0),
        "turnover_rate_y1": float(r_y1),
        "delta_turnover_rate_actual": float(delta_r_actual),
        "rate_effect": float(total_rate_effect),
        "mix_effect": float(total_mix_effect),
        "total_decomposed": float(total_decomposed),
        "residual": float(residual),
        "rate_effect_pct_of_change": float(total_rate_effect / delta_r_actual * 100.0) if delta_r_actual != 0 else 0.0,
        "mix_effect_pct_of_change": float(total_mix_effect / delta_r_actual * 100.0) if delta_r_actual != 0 else 0.0,
    }
    
    return data, summary


class LeaveOneOutExpectedDifferential:
    """Leave-One-Out Expected Differential (LOO-ED).
    
    Adapted from sports analytics (Expected Goal Differential / plus-minus).
    Isolates the marginal value (alpha) of a specific manager or supervisor by comparing
    actual shift output against counterfactual expected output modeled without that manager's identity.
    
    Strips away shift advantages (e.g. flagship store, prime Saturday traffic) to expose true on-floor impact.
    """
    
    def __init__(self, ridge_alpha: float = 1.0):
        self.ridge_alpha = ridge_alpha
        self.counterfactual_model_ = None
        self.feature_columns_: List[str] = []
        
    def fit_transform(
        self,
        df: pd.DataFrame,
        entity_col: str,
        target_col: str,
        context_cols: List[str],
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Compute LOO-ED per shift and aggregate marginal alpha per entity (manager).
        
        Returns:
            (shift_level_df_with_loo_ed, entity_summary_df)
        """
        from sklearn.linear_model import Ridge
        
        # One-hot encode context features (store, day-of-week, traffic, staffing)
        X_context = pd.get_dummies(df[context_cols], drop_first=True)
        self.feature_columns_ = list(X_context.columns)
        y = df[target_col].to_numpy(dtype=float)
        
        # Fit baseline model explaining output solely from external context
        self.counterfactual_model_ = Ridge(alpha=self.ridge_alpha)
        self.counterfactual_model_.fit(X_context, y)
        
        expected_output = self.counterfactual_model_.predict(X_context)
        
        result_df = df.copy()
        result_df["expected_output_counterfactual"] = expected_output
        result_df["loo_expected_differential"] = y - expected_output
        
        # Aggregate per manager
        summary = result_df.groupby(entity_col).agg(
            shifts_managed=(target_col, "count"),
            raw_mean_output=(target_col, "mean"),
            expected_mean_output=("expected_output_counterfactual", "mean"),
            loo_expected_differential=("loo_expected_differential", "mean"),
            loo_ed_std=("loo_expected_differential", "std"),
        ).reset_index()
        
        # Sort by marginal value (LOO-ED)
        summary = summary.sort_values("loo_expected_differential", ascending=False).reset_index(drop=True)
        
        return result_df, summary


def compute_shap_interaction_matrix(
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: Optional[List[str]] = None,
    is_classification: bool = True,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Train LightGBM model and compute exact SHAP 2D interaction value matrices.
    
    Decomposes model predictions into main effects (matrix diagonal) and
    pairwise coupled interaction effects (off-diagonal).
    
    Returns:
        (shap_interaction_values_3d, mean_abs_interaction_matrix_2d, interaction_pairs_df)
    """
    try:
        import lightgbm as lgb
        import shap
    except (ImportError, OSError) as err:
        raise ImportError(
            "compute_shap_interaction_matrix requires 'shap' and 'lightgbm'. "
            "Install with: pip install 'people-analytics-toolkit[explainability]' "
            "(note: on macOS, lightgbm also requires 'brew install libomp')."
        ) from err
    
    cols = list(X.columns) if feature_names is None else feature_names
    X_train = X[cols].copy()
    
    if is_classification:
        model = lgb.LGBMClassifier(n_estimators=60, max_depth=4, random_state=seed, verbose=-1, n_jobs=1)
    else:
        model = lgb.LGBMRegressor(n_estimators=60, max_depth=4, random_state=seed, verbose=-1, n_jobs=1)
        
    model.fit(X_train, y)
    
    explainer = shap.TreeExplainer(model)
    # 3D array: (n_samples, n_features, n_features)
    interaction_vals = explainer.shap_interaction_values(X_train)
    
    # If binary classification returns list for each class, select positive class
    if isinstance(interaction_vals, list):
        interaction_vals = interaction_vals[1]
        
    # Mean absolute interaction matrix across all samples
    mean_abs_matrix = np.mean(np.abs(interaction_vals), axis=0)
    
    # Extract ranked pairwise interactions (excluding diagonal main effects)
    n_feat = len(cols)
    pair_records = []
    for i in range(n_feat):
        for j in range(i + 1, n_feat):
            pair_records.append({
                "feature_1": cols[i],
                "feature_2": cols[j],
                "interaction_strength": float(mean_abs_matrix[i, j] * 2),  # symmetric pair sum
            })
            
    pairs_df = pd.DataFrame(pair_records).sort_values("interaction_strength", ascending=False).reset_index(drop=True)
    
    return interaction_vals, mean_abs_matrix, pairs_df


def extract_shap_inflection_thresholds(
    X: pd.DataFrame,
    shap_values: np.ndarray,
    n_grid_points: int = 200,
    seed: int = 42,
) -> Tuple[Dict[str, dict], pd.DataFrame]:
    """Extract critical behavioral inflection thresholds where SHAP values cross zero.
    
    Fits a 1D surrogate regression model on (feature_value, shap_value),
    evaluates on a continuous grid, detects sign changes, and interpolates exact
    zero-crossing thresholds where a metric shifts from protective (SHAP < 0)
    to hazardous (SHAP > 0) for turnover/termination rates.
    
    Returns:
        (thresholds_dict, summary_results_df)
    """
    from sklearn.ensemble import GradientBoostingRegressor
    
    feature_names = list(X.columns)
    # If shap_values is a list for binary classes, pick the positive outcome class (Terminated)
    if isinstance(shap_values, list):
        s_vals = shap_values[1]
    elif shap_values.ndim == 3:
        s_vals = shap_values[:, :, 1]
    else:
        s_vals = shap_values
        
    thresholds = {}
    
    for i, feat_name in enumerate(feature_names):
        feat_vals = X[feat_name].to_numpy(dtype=float)
        feat_shap = s_vals[:, i]
        
        # Sort values for monotonic grid evaluation
        sort_idx = np.argsort(feat_vals)
        sorted_x = feat_vals[sort_idx]
        sorted_shap = feat_shap[sort_idx]
        
        # Fit 1D surrogate regressor (GradientBoostingRegressor ensures stable, smooth interpolation)
        surrogate_reg = GradientBoostingRegressor(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            random_state=seed,
        )
        surrogate_reg.fit(sorted_x.reshape(-1, 1), sorted_shap)
        
        # Evaluate over fine grid across observed range
        x_min, x_max = float(sorted_x.min()), float(sorted_x.max())
        grid_x = np.linspace(x_min, x_max, n_grid_points)
        pred_shap = surrogate_reg.predict(grid_x.reshape(-1, 1))
        
        # Find exact zero-crossings via sign changes with linear interpolation
        zero_crossings = []
        sign_changes = np.where(np.diff(np.sign(pred_shap)))[0]
        
        for idx in sign_changes:
            x1, y1 = grid_x[idx], pred_shap[idx]
            x2, y2 = grid_x[idx + 1], pred_shap[idx + 1]
            if y1 * y2 < 0:  # Confirmed sign change
                zero_crossing = x1 - y1 * (x2 - x1) / (y2 - y1)
                zero_crossings.append(round(float(zero_crossing), 2))
                
        # Direction of hazard at upper tail: is high value of feature driving positive SHAP (hazard)?
        upper_hazard = bool(pred_shap[-1] > 0)
        
        # Robust primary threshold selection:
        # Filter out boundary crossings in extreme tails (< 5th or > 95th percentile)
        q05, q95 = np.percentile(feat_vals, [5, 95])
        support_crossings = [zc for zc in zero_crossings if q05 <= zc <= q95]
        candidate_crossings = support_crossings if support_crossings else zero_crossings
        
        primary_threshold = None
        if candidate_crossings:
            # Pick crossing with largest local rate of change |dy/dx| (steepest inflection)
            best_zc = candidate_crossings[0]
            max_slope = -1.0
            for zc in candidate_crossings:
                c_idx = int(np.argmin(np.abs(grid_x - zc)))
                i_lo = max(0, c_idx - 2)
                i_hi = min(len(grid_x) - 1, c_idx + 2)
                dx = grid_x[i_hi] - grid_x[i_lo]
                dy = pred_shap[i_hi] - pred_shap[i_lo]
                slope = abs(dy / dx) if dx > 0 else 0.0
                if slope > max_slope:
                    max_slope = slope
                    best_zc = zc
            primary_threshold = best_zc
        
        thresholds[feat_name] = {
            "predicted_thresholds": zero_crossings,
            "primary_threshold": primary_threshold,
            "mean": round(float(np.mean(feat_vals)), 2),
            "median": round(float(np.median(feat_vals)), 2),
            "high_value_is_hazard": upper_hazard,
            "grid_x": grid_x,
            "pred_shap": pred_shap,
            "sorted_x": sorted_x,
            "sorted_shap": sorted_shap,
        }
        
    summary_rows = []
    for k, v in thresholds.items():
        summary_rows.append({
            "feature": k,
            "primary_inflection_threshold": v["primary_threshold"],
            "all_thresholds": str(v["predicted_thresholds"]),
            "cohort_mean": v["mean"],
            "cohort_median": v["median"],
            "high_value_increases_turnover_risk": v["high_value_is_hazard"],
        })
        
    results_df = pd.DataFrame(summary_rows).set_index("feature")
    return thresholds, results_df


def calculate_retained_vs_terminated_shap_contributions(
    X: pd.DataFrame,
    shap_values: np.ndarray,
    y: pd.Series,
) -> pd.DataFrame:
    """Compute average SHAP value contributions for Retained vs Terminated cohorts.
    
    Computes directional attribution contrasting retained versus voluntary termination cohorts.
    """
    feature_names = list(X.columns)
    if isinstance(shap_values, list):
        s_vals = shap_values[1]
    elif shap_values.ndim == 3:
        s_vals = shap_values[:, :, 1]
    else:
        s_vals = shap_values
        
    shap_df = pd.DataFrame(s_vals, columns=feature_names, index=X.index)
    shap_df["is_terminated"] = y.to_numpy()
    
    retained_mean = shap_df[shap_df["is_terminated"] == 0][feature_names].mean()
    terminated_mean = shap_df[shap_df["is_terminated"] == 1][feature_names].mean()
    
    combined = pd.DataFrame({
        "Retained Cohort Mean SHAP": retained_mean,
        "Terminated Cohort Mean SHAP": terminated_mean,
    })
    
    combined["total_absolute_shap"] = combined["Retained Cohort Mean SHAP"].abs() + combined["Terminated Cohort Mean SHAP"].abs()
    combined = combined.sort_values("total_absolute_shap", ascending=True)
    
    return combined


# =============================================================================
# Feature 49: Correlation Heatmap Table (Pearson, Spearman, Kendall)
# =============================================================================

def correlation_heatmap_df(
    df: pd.DataFrame,
    target: str,
    method: str = "spearman",
    p_threshold: float = 0.05,
    padjust: str = "none",
    drop_insignificant: bool = False,
    drop_ci_contain_0: bool = False,
    viz: bool = True,
    return_styler: bool = False,
) -> Union[pd.DataFrame, Tuple[Any, pd.DataFrame]]:
    """Compute correlation metrics and generate a styled heatmap table of predictors to a target.

    Calculates pairwise correlation coefficients (Pearson, Spearman, Kendall, bicor, etc.)
    with p-values, 95% confidence intervals, and statistical power using pingouin.
    Applies conditional styling:
      - Dark red if the p-value is greater than p_threshold (insignificant).
      - Grey if the confidence interval spans 0.
      - Tiered orange gradient based on correlation magnitude if significant.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing predictors and the target variable.
    target : str
        Target column name to correlate predictors against.
    method : str, default='spearman'
        Correlation method: 'spearman', 'pearson', 'kendall', 'bicor', 'percbend', 'shepherd', 'skipped'.
    p_threshold : float, default=0.05
        P-value threshold for statistical significance.
    padjust : str, default='none'
        Multiple testing correction method: 'none', 'bonf', 'holm', 'sidak', 'fdr_bh', 'fdr_by'.
    drop_insignificant : bool, default=False
        Whether to drop rows where P >= p_threshold.
    drop_ci_contain_0 : bool, default=False
        Whether to drop rows where the 95% confidence interval spans 0.
    viz : bool, default=True
        Whether to display the styled DataFrame in interactive notebook environments.
    return_styler : bool, default=False
        If True, returns tuple of (styled_styler, target_corr_df).
        If False, returns target_corr_df (with styler attached to attrs['styler']).

    Returns
    -------
    pd.DataFrame or Tuple[pandas.io.formats.style.Styler, pd.DataFrame]
        DataFrame of correlation metrics sorted by correlation strength.

    Raises
    ------
    ImportError
        If `pingouin` is not installed.
    TypeError
        If `df` is not a pandas DataFrame.
    KeyError
        If `target` column is not found in `df`.
    ValueError
        If `target` is not numeric, or if no numeric predictor columns exist.
    """
    try:
        import pingouin as pg
    except ImportError as e:
        raise ImportError(
            "correlation_heatmap_df requires 'pingouin'. "
            "Install with: pip install 'people-analytics-toolkit' or pip install pingouin."
        ) from e

    try:
        from IPython.display import display
    except ImportError:
        display = None

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected df to be a pandas DataFrame, got {type(df)}")
    if target not in df.columns:
        raise KeyError(f"Target variable '{target}' not found in DataFrame columns")

    # Select numeric columns
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if target not in numeric_cols:
        raise ValueError(f"Target variable '{target}' must be numeric, got dtype {df[target].dtype}")

    work_df = df[numeric_cols]
    if len(work_df.columns) < 2:
        raise ValueError(f"DataFrame must contain at least one numeric predictor column besides '{target}'")

    if df.empty:
        empty_cols = [
            "METRIC",
            f"CORRELATION TO {target}",
            "P",
            "LB",
            "UB",
            "POWER",
            "METHOD",
            "CONTAINS_0",
            "SIGNIFICANT",
        ]
        empty_res = pd.DataFrame(columns=empty_cols)
        empty_res["SIGNIFICANT"] = empty_res["SIGNIFICANT"].astype(int)
        empty_res["CONTAINS_0"] = empty_res["CONTAINS_0"].astype(int)
        empty_res.attrs["styler"] = empty_res.style
        if return_styler:
            return empty_res.style, empty_res
        return empty_res

    valid_n = len(work_df.dropna())
    if valid_n < 3:
        raise ValueError(
            f"Sample size (N={valid_n}) is too small to calculate correlation "
            f"confidence intervals and p-values (minimum 3 observations required)."
        )

    # Compute pairwise correlation via pingouin
    raw_corr = pg.pairwise_corr(
        work_df,
        columns=[target],
        method=method,
        alternative="two-sided",
        padjust=padjust,
    )

    if raw_corr.empty:
        raise ValueError(f"No pairwise correlations could be computed for target '{target}'")

    raw_corr = raw_corr.round(4)

    # Detect CI column name across pingouin versions ('CI95' vs 'CI95%')
    ci_col = "CI95" if "CI95" in raw_corr.columns else ("CI95%" if "CI95%" in raw_corr.columns else None)

    # Detect p-value column name across versions ('p_corr'/'p-corr' vs 'p_unc'/'p-unc')
    if padjust != "none":
        p_col = "p_corr" if "p_corr" in raw_corr.columns else ("p-corr" if "p-corr" in raw_corr.columns else "p_unc")
    else:
        p_col = "p_unc" if "p_unc" in raw_corr.columns else ("p-unc" if "p-unc" in raw_corr.columns else "p_corr")

    # Ensure power column exists
    if "power" not in raw_corr.columns:
        raw_corr["power"] = np.nan

    corr_col_name = f"CORRELATION TO {target}"

    keep_cols = ["Y", "r", p_col, "power"]
    if ci_col is not None:
        keep_cols.insert(2, ci_col)

    target_corr_df = raw_corr[keep_cols].copy()
    target_corr_df = target_corr_df.sort_values(by="r", ascending=False).reset_index(drop=True)

    rename_map = {
        "Y": "METRIC",
        p_col: "P",
        "r": corr_col_name,
        "power": "POWER",
    }
    if ci_col is not None:
        rename_map[ci_col] = "CI"

    target_corr_df = target_corr_df.rename(columns=rename_map)

    # Extract LB and UB from Confidence Interval
    if "CI" in target_corr_df.columns:
        target_corr_df["LB"] = target_corr_df["CI"].apply(
            lambda x: float(x[0]) if (isinstance(x, (list, tuple, np.ndarray)) and len(x) >= 2) else np.nan
        )
        target_corr_df["UB"] = target_corr_df["CI"].apply(
            lambda x: float(x[1]) if (isinstance(x, (list, tuple, np.ndarray)) and len(x) >= 2) else np.nan
        )
        target_corr_df.drop("CI", axis=1, inplace=True)
    else:
        target_corr_df["LB"] = np.nan
        target_corr_df["UB"] = np.nan

    target_corr_df["METHOD"] = method
    target_corr_df["CONTAINS_0"] = np.where(
        (target_corr_df["LB"].isna()) | (target_corr_df["UB"].isna()),
        1,
        np.where((np.sign(target_corr_df["LB"]) != np.sign(target_corr_df["UB"])), 1, 0),
    )
    target_corr_df["SIGNIFICANT"] = np.where((target_corr_df["P"] < p_threshold), 1, 0)

    # Color palette
    lilac = "#8b7cbf"
    royal_blue = "#2b6cb0"
    light_blue = "#7db8f8"
    sage_green = "#81a554"
    dark_slate = "#2f4f4f"  # CSS darkslategray

    def highlight_sign_diff(row):
        p_val = row.get("P", np.nan)
        lb_val = row.get("LB", np.nan)
        ub_val = row.get("UB", np.nan)
        r_val = abs(row.get(corr_col_name, 0.0))

        # Check if p-value is greater than threshold (insignificant)
        if pd.isna(p_val) or p_val > p_threshold:
            return ["background-color: salmon; color:white"] * len(row)
        # Check if LB and UB have different signs (spans 0)
        elif (not pd.isna(lb_val)) and (not pd.isna(ub_val)) and (np.sign(lb_val) != np.sign(ub_val)) and (lb_val != 0) and (ub_val != 0):
            return ["background-color: darkslategray; color:white"] * len(row)
        # Apply color gradient based on correlation strength
        elif r_val < 0.1:
            return ["background-color: white; color:black"] * len(row)
        elif r_val < 0.25:
            return [f"background-color: {light_blue}; color:black"] * len(row)
        elif r_val < 0.5:
            return [f"background-color: {royal_blue}; color:black"] * len(row)
        elif r_val < 0.75:
            return [f"background-color: {lilac}; color:black"] * len(row)
        elif r_val <= 1.0:
            return [f"background-color: {sage_green}; color:black"] * len(row)
        else:
            return [""] * len(row)

    styled_target_corr_df = target_corr_df.copy()
    styled_target_corr_df.drop(["SIGNIFICANT", "CONTAINS_0"], axis=1, inplace=True, errors="ignore")
    styled_target_corr_df = styled_target_corr_df.style.apply(highlight_sign_diff, axis=1)

    if viz and display is not None:
        display(styled_target_corr_df)

    # Reorder columns
    ordered_cols = ["METRIC", corr_col_name, "P", "LB", "UB", "POWER", "METHOD", "CONTAINS_0", "SIGNIFICANT"]
    target_corr_df = target_corr_df[[c for c in ordered_cols if c in target_corr_df.columns]]

    # Attach styler as metadata attribute
    target_corr_df.attrs["styler"] = styled_target_corr_df

    # Filter insignificant correlations if requested
    if drop_insignificant:
        target_corr_df = target_corr_df[target_corr_df["SIGNIFICANT"] == 1]
        target_corr_df.drop(["SIGNIFICANT"], axis=1, inplace=True, errors="ignore")

    # Filter correlations spanning 0 if requested
    if drop_ci_contain_0:
        target_corr_df = target_corr_df[target_corr_df["CONTAINS_0"] == 0]
        target_corr_df.drop(["CONTAINS_0"], axis=1, inplace=True, errors="ignore")

    target_corr_df = target_corr_df.reset_index(drop=True)

    if return_styler:
        return styled_target_corr_df, target_corr_df
    return target_corr_df


# API Consistency Aliases: compute_* <=> calculate_* (Deprecated in favor of canonical names)
calculate_shap_interaction_matrix = deprecated_alias(
    compute_shap_interaction_matrix,
    "calculate_shap_interaction_matrix",
    "compute_shap_interaction_matrix",
)
compute_retained_vs_terminated_shap_contributions = deprecated_alias(
    calculate_retained_vs_terminated_shap_contributions,
    "compute_retained_vs_terminated_shap_contributions",
    "calculate_retained_vs_terminated_shap_contributions",
)

__all__ = [
    "logarithmic_mean",
    "lmdi_rate_mix_decomposition",
    "LeaveOneOutExpectedDifferential",
    "compute_shap_interaction_matrix",
    "calculate_shap_interaction_matrix",
    "extract_shap_inflection_thresholds",
    "calculate_retained_vs_terminated_shap_contributions",
    "compute_retained_vs_terminated_shap_contributions",
    "correlation_heatmap_df",
]

