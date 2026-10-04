"""Compensation and Pay Equity Feature Engineering.

Contains:
1. Compa-Ratio: Base salary relative to salary band midpoint evaluating market competitiveness and parity.
2. Range Penetration: Position within salary band between minimum and maximum bounds.
3. Pay Band Classification: Automatic quartile and red/green circle categorizations.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


def calculate_salary_band_midpoint(
    band_min: Union[pd.Series, np.ndarray, list, float],
    band_max: Optional[Union[pd.Series, np.ndarray, list, float]] = None,
    spread: Optional[Union[pd.Series, np.ndarray, list, float]] = None,
) -> Union[pd.Series, np.ndarray, float]:
    r"""Calculate the official salary band midpoint from minimum and maximum bounds or spread.

    Formulas:
        Midpoint = (Band_Min + Band_Max) / 2.0

    Alternatively, if range spread :math:`S = \frac{\text{Max} - \text{Min}}{\text{Min}}` is provided:
        Midpoint = Band_Min \times \left(1 + \frac{S}{2}\right)

    Parameters
    ----------
    band_min : pd.Series, np.ndarray, list, or float
        Minimum salary bound for the salary grade / band.
    band_max : pd.Series, np.ndarray, list, or float, optional
        Maximum salary bound for the salary grade / band.
    spread : pd.Series, np.ndarray, list, or float, optional
        Range spread expressed as a decimal (e.g. 0.50 for a 50% band spread).
        Used if band_max is not provided.

    Returns
    -------
    pd.Series, np.ndarray, or float
        Calculated midpoint(s). Returns a pd.Series if band_min or band_max is a Series,
        a scalar float if inputs are scalars, or a numpy array otherwise.

    Raises
    ------
    ValueError
        If `band_min` > `band_max`, `spread` < 0, or neither `band_max` nor `spread` is provided.
    """
    if band_max is not None:
        b_min = np.asarray(band_min, dtype=float)
        b_max = np.asarray(band_max, dtype=float)
        if np.any(b_min > b_max):
            raise ValueError("band_min cannot be greater than band_max.")
        mid = (b_min + b_max) / 2.0
    elif spread is not None:
        b_min = np.asarray(band_min, dtype=float)
        s = np.asarray(spread, dtype=float)
        if np.any(s < 0):
            raise ValueError("Salary band spread must be non-negative.")
        mid = b_min * (1.0 + s / 2.0)
    else:
        raise ValueError("Must provide either 'band_max' or 'spread' alongside 'band_min'.")

    if isinstance(band_min, pd.Series):
        return pd.Series(mid, index=band_min.index, name="band_midpoint")
    elif isinstance(band_max, pd.Series):
        return pd.Series(mid, index=band_max.index, name="band_midpoint")
    elif np.isscalar(band_min) and (band_max is None or np.isscalar(band_max)):
        return float(np.asarray(mid).item())
    return mid


def calculate_compa_ratio(
    salary: Union[pd.Series, np.ndarray, list, float],
    midpoint: Optional[Union[pd.Series, np.ndarray, list, float]] = None,
    band_min: Optional[Union[pd.Series, np.ndarray, list, float]] = None,
    band_max: Optional[Union[pd.Series, np.ndarray, list, float]] = None,
) -> Union[pd.Series, np.ndarray, float]:
    """Calculate Compa-Ratio (Comparative Ratio) for evaluating internal pay equity.
    
    Formula:
        Compa-Ratio = Base_Salary / Midpoint
        
    If midpoint is not explicitly provided, it is derived via calculate_salary_band_midpoint():
        Midpoint = (band_min + band_max) / 2.0
        
    Interpretation:
        - 1.00 (100%): Exact market/band parity.
        - < 0.80: Significantly below midpoint (often early in tenure or green-circled).
        - 0.80 - 1.20: Standard competitive target range (typically 80%-120%).
        - > 1.20: Significantly above midpoint (often senior subject-matter experts or red-circled).

    Parameters
    ----------
    salary : pd.Series, np.ndarray, list, or float
        Employee actual base salary.
    midpoint : pd.Series, np.ndarray, list, or float, optional
        Salary band midpoint.
    band_min : pd.Series, np.ndarray, list, or float, optional
        Band minimum.
    band_max : pd.Series, np.ndarray, list, or float, optional
        Band maximum.

    Returns
    -------
    pd.Series, np.ndarray, or float
        Calculated Compa-Ratio.

    Raises
    ------
    ValueError
        If neither `midpoint` nor both `band_min` and `band_max` are provided.
    """
    s = np.asarray(salary, dtype=float)
    
    if midpoint is not None:
        mid = np.asarray(midpoint, dtype=float)
    elif band_min is not None and band_max is not None:
        mid_val = calculate_salary_band_midpoint(band_min, band_max)
        mid = np.asarray(mid_val, dtype=float)
    else:
        raise ValueError("Must provide either 'midpoint' or both 'band_min' and 'band_max'.")
        
    with np.errstate(divide="ignore", invalid="ignore"):
        compa = np.where(mid > 0, s / mid, np.nan)
        
    if isinstance(salary, pd.Series):
        return pd.Series(compa, index=salary.index, name="compa_ratio")
    elif np.isscalar(salary):
        return float(compa.item())
    return compa


def calculate_range_penetration(
    salary: Union[pd.Series, np.ndarray, list, float],
    band_min: Union[pd.Series, np.ndarray, list, float],
    band_max: Union[pd.Series, np.ndarray, list, float],
) -> Union[pd.Series, np.ndarray, float]:
    """Calculate Range Penetration (progress through the established salary grade).
    
    Formula:
        Range Penetration = (Base_Salary - Band_Min) / (Band_Max - Band_Min)
        
    Unlike Compa-Ratio (which is relative to a single center point), Range Penetration
    scales salary directly across the total width of the salary band [Min, Max].
    
    Interpretation:
        - 0.00 (0%):   Exactly at band minimum (entry-level hire).
        - 0.25 (25%):  First quartile boundary.
        - 0.50 (50%):  Exact midpoint of salary band (fully competent/performing).
        - 0.75 (75%):  Third quartile boundary.
        - 1.00 (100%): Top of salary band (maximum progression).
        - < 0.00:      Green-circled (salary falls below official band minimum).
        - > 1.00:      Red-circled (salary exceeds official band maximum).

    Parameters
    ----------
    salary : pd.Series, np.ndarray, list, or float
        Employee actual base salary.
    band_min : pd.Series, np.ndarray, list, or float
        Salary band minimum.
    band_max : pd.Series, np.ndarray, list, or float
        Salary band maximum.

    Returns
    -------
    pd.Series, np.ndarray, or float
        Calculated Range Penetration.

    Raises
    ------
    ValueError
        If `band_min` > `band_max`.
    """
    s = np.asarray(salary, dtype=float)
    b_min = np.asarray(band_min, dtype=float)
    b_max = np.asarray(band_max, dtype=float)

    if np.any(b_min > b_max):
        raise ValueError("band_min cannot be greater than band_max.")

    spread = b_max - b_min
    with np.errstate(divide="ignore", invalid="ignore"):
        rp = np.where(spread > 0, (s - b_min) / spread, np.nan)

    series_index = None
    if isinstance(salary, pd.Series):
        series_index = salary.index
    elif isinstance(band_min, pd.Series):
        series_index = band_min.index
    elif isinstance(band_max, pd.Series):
        series_index = band_max.index

    if series_index is not None:
        return pd.Series(rp, index=series_index, name="range_penetration")
    elif np.isscalar(salary) and np.isscalar(band_min) and np.isscalar(band_max):
        return float(np.asarray(rp).item())
    elif isinstance(salary, list):
        return list(rp)
    return rp


def classify_pay_band_status(
    range_penetration_or_salary: Union[pd.Series, np.ndarray, list, float],
    band_min: Optional[Union[pd.Series, np.ndarray, list, float]] = None,
    band_max: Optional[Union[pd.Series, np.ndarray, list, float]] = None,
) -> Union[pd.Series, np.ndarray, list, str]:
    """Categorize pay band position into standard human capital management tiers.

    Accepts either pre-calculated ``range_penetration`` or (``salary``, ``band_min``, ``band_max``).

    Categories:
        - 'Green-Circled (<Min)'
        - 'Q1: Developing (0-25%)'
        - 'Q2: Core Proficient (25-50%)'
        - 'Q3: Advanced (50-75%)'
        - 'Q4: Senior Cap (75-100%)'
        - 'Red-Circled (>Max)'

    Parameters
    ----------
    range_penetration_or_salary : pd.Series, np.ndarray, list, or float
        Pre-calculated range penetration or employee actual base salary.
    band_min : pd.Series, np.ndarray, list, or float, optional
        Salary band minimum bound.
    band_max : pd.Series, np.ndarray, list, or float, optional
        Salary band maximum bound.

    Returns
    -------
    pd.Series, np.ndarray, list, or str
        Categorized pay band status matching input type and preserving index.

    Raises
    ------
    ValueError
        If `band_min` and `band_max` are provided and `band_min` > `band_max`.
    """
    if band_min is not None and band_max is not None:
        rp = calculate_range_penetration(range_penetration_or_salary, band_min, band_max)
    else:
        rp = range_penetration_or_salary

    rp_arr = np.asarray(rp, dtype=float)

    conditions = [
        rp_arr < 0.0,
        (rp_arr >= 0.0) & (rp_arr < 0.25),
        (rp_arr >= 0.25) & (rp_arr < 0.50),
        (rp_arr >= 0.50) & (rp_arr < 0.75),
        (rp_arr >= 0.75) & (rp_arr <= 1.0),
        rp_arr > 1.0,
    ]
    choices = [
        "Green-Circled (<Min)",
        "Q1: Developing (0-25%)",
        "Q2: Core Proficient (25-50%)",
        "Q3: Advanced (50-75%)",
        "Q4: Senior Cap (75-100%)",
        "Red-Circled (>Max)",
    ]

    categorized = np.select(conditions, choices, default="Unknown")

    # Inherit pandas Series index if any input was a Series
    series_index = None
    if isinstance(range_penetration_or_salary, pd.Series):
        series_index = range_penetration_or_salary.index
    elif isinstance(rp, pd.Series):
        series_index = rp.index
    elif isinstance(band_min, pd.Series):
        series_index = band_min.index
    elif isinstance(band_max, pd.Series):
        series_index = band_max.index

    if series_index is not None:
        return pd.Series(categorized, index=series_index, name="pay_band_status")
    elif np.isscalar(range_penetration_or_salary) or (isinstance(rp_arr, np.ndarray) and rp_arr.ndim == 0):
        return str(categorized.item())
    elif isinstance(range_penetration_or_salary, list):
        return list(categorized)
    return categorized


def is_red_circled(
    salary_or_rp: Union[pd.Series, np.ndarray, list, float],
    band_max: Optional[Union[pd.Series, np.ndarray, list, float]] = None,
) -> Union[pd.Series, np.ndarray, list, bool]:
    """Identify employees whose salary exceeds their official salary band maximum (Range Penetration > 1.0).

    Parameters
    ----------
    salary_or_rp : pd.Series, np.ndarray, list, or float
        Either employee base salary (if band_max is provided) or pre-calculated range penetration.
    band_max : pd.Series, np.ndarray, list, or float, optional
        Maximum salary bound for the employee's band.

    Returns
    -------
    pd.Series, np.ndarray, list, or bool
        Boolean mask or indicator of red-circled status matching the input container type.
    """
    if band_max is not None:
        s = np.asarray(salary_or_rp, dtype=float)
        b_max = np.asarray(band_max, dtype=float)
        res = s > b_max
    else:
        rp = np.asarray(salary_or_rp, dtype=float)
        res = rp > 1.0

    series_index = None
    if isinstance(salary_or_rp, pd.Series):
        series_index = salary_or_rp.index
    elif isinstance(band_max, pd.Series):
        series_index = band_max.index

    if series_index is not None:
        return pd.Series(res, index=series_index, name="is_red_circled")
    elif np.isscalar(salary_or_rp) and (band_max is None or np.isscalar(band_max)):
        return bool(np.asarray(res).item())
    elif isinstance(salary_or_rp, list):
        return list(res)
    return res


def is_green_circled(
    salary_or_rp: Union[pd.Series, np.ndarray, list, float],
    band_min: Optional[Union[pd.Series, np.ndarray, list, float]] = None,
) -> Union[pd.Series, np.ndarray, list, bool]:
    """Identify employees whose salary falls below their official salary band minimum (Range Penetration < 0.0).

    Parameters
    ----------
    salary_or_rp : pd.Series, np.ndarray, list, or float
        Either employee base salary (if band_min is provided) or pre-calculated range penetration.
    band_min : pd.Series, np.ndarray, list, or float, optional
        Minimum salary bound for the employee's band.

    Returns
    -------
    pd.Series, np.ndarray, list, or bool
        Boolean mask or indicator of green-circled status matching the input container type.
    """
    if band_min is not None:
        s = np.asarray(salary_or_rp, dtype=float)
        b_min = np.asarray(band_min, dtype=float)
        res = s < b_min
    else:
        rp = np.asarray(salary_or_rp, dtype=float)
        res = rp < 0.0

    series_index = None
    if isinstance(salary_or_rp, pd.Series):
        series_index = salary_or_rp.index
    elif isinstance(band_min, pd.Series):
        series_index = band_min.index

    if series_index is not None:
        return pd.Series(res, index=series_index, name="is_green_circled")
    elif np.isscalar(salary_or_rp) and (band_min is None or np.isscalar(band_min)):
        return bool(np.asarray(res).item())
    elif isinstance(salary_or_rp, list):
        return list(res)
    return res


def calculate_pay_equity_gap(
    data: Optional[Union[pd.DataFrame, pd.Series, np.ndarray, list]] = None,
    salary: Optional[Union[pd.Series, np.ndarray, list, str]] = None,
    group: Optional[Union[pd.Series, np.ndarray, list, str]] = None,
    reference_group: Optional[Any] = None,
    comparison_group: Optional[Any] = None,
    control_cols: Optional[Union[str, List[str]]] = None,
    salary_col: Optional[str] = None,
    group_col: Optional[str] = None,
) -> Dict[str, Any]:
    r"""Calculate group-level pay equity gaps, summary distributions, and controlled cohort benchmarks.

    Provides unadjusted (raw) group-level wage gap analysis, distribution
    across pay quartiles (following UK Gender Pay Gap and EEO reporting standards),
    and controlled (adjusted) within-cohort pay gap calculations.

    Mathematical Formulations:
        - Median Pay Gap:
          .. math::
              \text{Gap}_{\text{median}} = \frac{\text{Median}_{\text{ref}} - \text{Median}_{\text{comp}}}{\text{Median}_{\text{ref}}}
        - Mean Pay Gap:
          .. math::
              \text{Gap}_{\text{mean}} = \frac{\text{Mean}_{\text{ref}} - \text{Mean}_{\text{comp}}}{\text{Mean}_{\text{ref}}}
        - Dollar / Absolute Gap:
          .. math::
              \Delta_{\text{median}} = \text{Median}_{\text{ref}} - \text{Median}_{\text{comp}}
        - Pay Ratio (e.g. cents on the dollar):
          .. math::
              \text{Ratio}_{\text{median}} = \frac{\text{Median}_{\text{comp}}}{\text{Median}_{\text{ref}}}
        - Controlled / Adjusted Pay Gap:
          .. math::
              \text{Gap}_{\text{controlled}} = \sum_{c} w_c \cdot \text{Gap}_{c}
          where :math:`w_c` is the cohort weight (total cohort headcount proportion).

    Parameters
    ----------
    data : pd.DataFrame, pd.Series, np.ndarray, or list, optional
        Input DataFrame containing employee compensation and demographic records,
        or an array/Series of salary values if called positionally as `(salary, group)`.
    salary : pd.Series, np.ndarray, list, or str, optional
        Salary values or name of the salary column in `data`.
    group : pd.Series, np.ndarray, list, or str, optional
        Demographic group labels or name of the group column in `data`.
    reference_group : Any, optional
        Label of the baseline/reference group (e.g., 'Male', 'Majority', 1).
        If None, automatically inferred from standard conventions ('Male', 'M', 1)
        or the group with the largest headcount.
    comparison_group : Any, optional
        Label of the comparison group (e.g., 'Female', 'Minority', 0).
        If None, automatically inferred from standard conventions ('Female', 'F', 0)
        or the remaining group in binary comparisons.
    control_cols : str or list of str, optional
        Column name(s) in `data` to control for (e.g. 'job_level', 'department', 'grade').
        When specified, computes within-cohort gaps and cohort-weighted adjusted gaps.
    salary_col : str, optional
        Explicit column name for salary in `data`.
    group_col : str, optional
        Explicit column name for demographic group in `data`.

    Returns
    -------
    dict
        Dictionary containing:
        - 'median_gap': Unadjusted median pay gap as a decimal (e.g. 0.15 for 15%).
        - 'pct_median_gap': Unadjusted median pay gap as a percentage (e.g. 15.0%).
        - 'mean_gap': Unadjusted mean pay gap as a decimal.
        - 'pct_mean_gap': Unadjusted mean pay gap as a percentage.
        - 'dollar_median_gap': Absolute difference in median salary (ref - comp).
        - 'dollar_mean_gap': Absolute difference in mean salary (ref - comp).
        - 'pay_ratio_median': Ratio of comparison to reference median pay (e.g. 0.85).
        - 'pay_ratio_mean': Ratio of comparison to reference mean pay.
        - 'reference_group': Label of the reference group.
        - 'comparison_group': Label of the comparison group.
        - 'ref_median': Median salary of reference group.
        - 'comp_median': Median salary of comparison group.
        - 'ref_mean': Mean salary of reference group.
        - 'comp_mean': Mean salary of comparison group.
        - 'group_stats': pd.DataFrame of descriptive statistics (count, mean, median, std, p25, p75, iqr, min, max) per group.
        - 'quartile_distribution': pd.DataFrame with employee distribution across salary quartiles.
        - (If `control_cols` provided):
            - 'controlled_median_gap': Weighted average median gap across cohorts.
            - 'controlled_pct_median_gap': Percentage controlled median gap.
            - 'controlled_mean_gap': Weighted average mean gap across cohorts.
            - 'controlled_pct_mean_gap': Percentage controlled mean gap.
            - 'cohort_gaps': pd.DataFrame detailing per-cohort gaps and headcounts.

    Raises
    ------
    ValueError
        If required salary or group columns are missing, if length mismatch occurs,
        if data has fewer than 2 distinct demographic groups, or if reference/comparison
        group specifications are invalid or identical.
    """
    # 1. Parse inputs into a DataFrame with clean salary and group columns
    if isinstance(data, pd.DataFrame):
        df = data.copy()
        s_col = salary_col or (salary if isinstance(salary, str) else None)
        g_col = group_col or (group if isinstance(group, str) else None)

        if s_col is None:
            if isinstance(salary, (pd.Series, np.ndarray, list)):
                df["_salary_temp"] = np.asarray(salary, dtype=float)
                s_col = "_salary_temp"
            else:
                raise ValueError("Must provide 'salary_col' or a salary Series/array when 'data' is a DataFrame.")
        if g_col is None:
            if isinstance(group, (pd.Series, np.ndarray, list)):
                df["_group_temp"] = np.asarray(group)
                g_col = "_group_temp"
            else:
                raise ValueError("Must provide 'group_col' or a group Series/array when 'data' is a DataFrame.")
    else:
        if data is not None:
            sal_vals = data
            grp_vals = salary if group is None else group
        else:
            if salary is None or group is None:
                raise ValueError("Must provide both salary and group data.")
            sal_vals = salary
            grp_vals = group

        sal_arr = np.asarray(sal_vals, dtype=float)
        grp_arr = np.asarray(grp_vals)
        if len(sal_arr) != len(grp_arr):
            raise ValueError(f"Length mismatch: salary ({len(sal_arr)}) and group ({len(grp_arr)}) must have the same length.")

        s_col = "salary"
        g_col = "group"
        df = pd.DataFrame({s_col: sal_arr, g_col: grp_arr})

    df = df.dropna(subset=[s_col, g_col]).copy()
    if len(df) == 0:
        raise ValueError("Input data contains no valid non-null rows.")

    unique_groups = df[g_col].unique().tolist()
    if len(unique_groups) < 2:
        raise ValueError(f"Group column must contain at least 2 distinct groups, found: {unique_groups}")

    # 2. Resolve reference and comparison groups
    if reference_group is None:
        common_refs = ["male", "m", "majority", "white", "reference", "ref", "1", 1]
        ref_match = None
        for g in unique_groups:
            if any(str(g).strip().lower() == str(r).lower() for r in common_refs):
                ref_match = g
                break
        if ref_match is not None:
            reference_group = ref_match
        else:
            reference_group = df[g_col].value_counts().index[0]

    if reference_group not in unique_groups:
        raise ValueError(f"Reference group '{reference_group}' not found in groups: {unique_groups}")

    if comparison_group is None:
        remaining = [g for g in unique_groups if g != reference_group]
        common_comps = ["female", "f", "minority", "0", 0]
        comp_match = None
        for g in remaining:
            if any(str(g).strip().lower() == str(c).lower() for c in common_comps):
                comp_match = g
                break
        if comp_match is not None:
            comparison_group = comp_match
        else:
            comparison_group = df[df[g_col] != reference_group][g_col].value_counts().index[0]

    if comparison_group not in unique_groups:
        raise ValueError(f"Comparison group '{comparison_group}' not found in groups: {unique_groups}")
    if reference_group == comparison_group:
        raise ValueError("Reference group and comparison group cannot be identical.")

    # 3. Descriptive Group Statistics
    group_stats_records = []
    for g in unique_groups:
        g_data = df[df[g_col] == g][s_col]
        cnt = int(len(g_data))
        m = float(g_data.mean()) if cnt > 0 else np.nan
        med = float(g_data.median()) if cnt > 0 else np.nan
        std = float(g_data.std(ddof=1)) if cnt > 1 else 0.0
        p25 = float(g_data.quantile(0.25)) if cnt > 0 else np.nan
        p75 = float(g_data.quantile(0.75)) if cnt > 0 else np.nan
        iqr = float(p75 - p25) if cnt > 0 else np.nan
        min_v = float(g_data.min()) if cnt > 0 else np.nan
        max_v = float(g_data.max()) if cnt > 0 else np.nan
        group_stats_records.append({
            "group": g,
            "count": cnt,
            "mean": m,
            "median": med,
            "std": std,
            "p25": p25,
            "p75": p75,
            "iqr": iqr,
            "min": min_v,
            "max": max_v,
        })
    group_stats = pd.DataFrame(group_stats_records).set_index("group")

    # 4. Unadjusted Gaps (Reference vs Comparison)
    ref_salaries = df[df[g_col] == reference_group][s_col]
    comp_salaries = df[df[g_col] == comparison_group][s_col]

    ref_med = float(ref_salaries.median())
    comp_med = float(comp_salaries.median())
    ref_mean = float(ref_salaries.mean())
    comp_mean = float(comp_salaries.mean())

    if ref_med > 0:
        median_gap_val = float((ref_med - comp_med) / ref_med)
        pct_median_gap_val = float(median_gap_val * 100.0)
        pay_ratio_median = float(comp_med / ref_med)
    else:
        median_gap_val = np.nan
        pct_median_gap_val = np.nan
        pay_ratio_median = np.nan

    if ref_mean > 0:
        mean_gap_val = float((ref_mean - comp_mean) / ref_mean)
        pct_mean_gap_val = float(mean_gap_val * 100.0)
        pay_ratio_mean = float(comp_mean / ref_mean)
    else:
        mean_gap_val = np.nan
        pct_mean_gap_val = np.nan
        pay_ratio_mean = np.nan

    dollar_median_gap = float(ref_med - comp_med)
    dollar_mean_gap = float(ref_mean - comp_mean)

    # 5. Pay Quartiles Distribution (UK Gender Pay Gap / Regulatory format)
    try:
        df["_pay_quartile"] = pd.qcut(
            df[s_col],
            q=4,
            labels=["Q1_Lower", "Q2_Lower_Middle", "Q3_Upper_Middle", "Q4_Upper"],
            duplicates="drop",
        )
        ct_counts = pd.crosstab(df["_pay_quartile"], df[g_col])
        ct_pct = pd.crosstab(df["_pay_quartile"], df[g_col], normalize="index") * 100.0

        quartile_rows = []
        for q_label in ct_counts.index:
            total_in_q = int(ct_counts.loc[q_label].sum())
            row = {"quartile": q_label, "total_employees": total_in_q}
            for g in unique_groups:
                g_cnt = int(ct_counts.loc[q_label, g]) if g in ct_counts.columns else 0
                pct = float(ct_pct.loc[q_label, g]) if g in ct_pct.columns else 0.0
                row[f"{g}_count"] = g_cnt
                row[f"{g}_pct_of_quartile"] = pct
            quartile_rows.append(row)
        quartile_dist = pd.DataFrame(quartile_rows).set_index("quartile")
    except Exception:
        quartile_dist = pd.DataFrame()

    # 6. Controlled / Adjusted Cohort Gaps (if control_cols specified)
    cohort_gaps_df = None
    controlled_median_gap = None
    controlled_pct_median_gap = None
    controlled_mean_gap = None
    controlled_pct_mean_gap = None

    if control_cols is not None:
        if isinstance(control_cols, str):
            controls = [control_cols]
        else:
            controls = list(control_cols)

        for c in controls:
            if c not in df.columns:
                raise ValueError(f"Control column '{c}' not found in data.")

        cohort_records = []
        grouped = df.groupby(controls, observed=True)
        for cohort_name, cohort_sub in grouped:
            c_ref = cohort_sub[cohort_sub[g_col] == reference_group][s_col]
            c_comp = cohort_sub[cohort_sub[g_col] == comparison_group][s_col]

            n_ref = len(c_ref)
            n_comp = len(c_comp)
            n_total = len(cohort_sub)

            if n_ref > 0 and n_comp > 0:
                c_ref_med = float(c_ref.median())
                c_comp_med = float(c_comp.median())
                c_ref_mean = float(c_ref.mean())
                c_comp_mean = float(c_comp.mean())

                c_med_gap = (c_ref_med - c_comp_med) / c_ref_med if c_ref_med > 0 else np.nan
                c_mean_gap = (c_ref_mean - c_comp_mean) / c_ref_mean if c_ref_mean > 0 else np.nan
            else:
                c_ref_med = float(c_ref.median()) if n_ref > 0 else np.nan
                c_comp_med = float(c_comp.median()) if n_comp > 0 else np.nan
                c_ref_mean = float(c_ref.mean()) if n_ref > 0 else np.nan
                c_comp_mean = float(c_comp.mean()) if n_comp > 0 else np.nan
                c_med_gap = np.nan
                c_mean_gap = np.nan

            cohort_records.append({
                "cohort": cohort_name,
                "n_ref": n_ref,
                "n_comp": n_comp,
                "n_total": n_total,
                "ref_median": c_ref_med,
                "comp_median": c_comp_med,
                "ref_mean": c_ref_mean,
                "comp_mean": c_comp_mean,
                "median_gap": c_med_gap,
                "pct_median_gap": c_med_gap * 100.0 if not np.isnan(c_med_gap) else np.nan,
                "mean_gap": c_mean_gap,
                "pct_mean_gap": c_mean_gap * 100.0 if not np.isnan(c_mean_gap) else np.nan,
            })

        cohort_gaps_df = pd.DataFrame(cohort_records).set_index("cohort")
        valid_cohorts = cohort_gaps_df.dropna(subset=["median_gap", "mean_gap"])
        if len(valid_cohorts) > 0:
            total_weight = valid_cohorts["n_total"].sum()
            if total_weight > 0:
                weights = valid_cohorts["n_total"] / total_weight
                controlled_median_gap = float((valid_cohorts["median_gap"] * weights).sum())
                controlled_pct_median_gap = float(controlled_median_gap * 100.0)
                controlled_mean_gap = float((valid_cohorts["mean_gap"] * weights).sum())
                controlled_pct_mean_gap = float(controlled_mean_gap * 100.0)

    res = {
        "median_gap": median_gap_val,
        "pct_median_gap": pct_median_gap_val,
        "mean_gap": mean_gap_val,
        "pct_mean_gap": pct_mean_gap_val,
        "dollar_median_gap": dollar_median_gap,
        "dollar_mean_gap": dollar_mean_gap,
        "pay_ratio_median": pay_ratio_median,
        "pay_ratio_mean": pay_ratio_mean,
        "reference_group": reference_group,
        "comparison_group": comparison_group,
        "ref_median": ref_med,
        "comp_median": comp_med,
        "ref_mean": ref_mean,
        "comp_mean": comp_mean,
        "group_stats": group_stats,
        "quartile_distribution": quartile_dist,
    }

    if control_cols is not None:
        res["controlled_median_gap"] = controlled_median_gap
        res["controlled_pct_median_gap"] = controlled_pct_median_gap
        res["controlled_mean_gap"] = controlled_mean_gap
        res["controlled_pct_mean_gap"] = controlled_pct_mean_gap
        res["cohort_gaps"] = cohort_gaps_df

    return res


def median_gap(
    salary: Union[pd.DataFrame, pd.Series, np.ndarray, list],
    group: Optional[Union[pd.Series, np.ndarray, list, str]] = None,
    reference_group: Optional[Any] = None,
    comparison_group: Optional[Any] = None,
    as_percentage: bool = False,
    salary_col: Optional[str] = None,
    group_col: Optional[str] = None,
) -> float:
    r"""Convenience method to calculate the unadjusted median pay equity gap.

    Formula:
        .. math::
            \text{Gap} = \frac{\text{Median}_{\text{ref}} - \text{Median}_{\text{comp}}}{\text{Median}_{\text{ref}}}

    Parameters
    ----------
    salary : pd.DataFrame, pd.Series, np.ndarray, or list
        Either salary values (Series, array, list) or a DataFrame.
    group : pd.Series, np.ndarray, list, or str, optional
        Demographic group labels, or group column name in DataFrame if `salary` is a DataFrame.
    reference_group : Any, optional
        Reference group label (e.g. 'Male', 'Majority').
    comparison_group : Any, optional
        Comparison group label (e.g. 'Female', 'Minority').
    as_percentage : bool, default False
        If True, returns the gap as a percentage (e.g. 15.0 for 15%),
        otherwise as a proportion (0.15).
    salary_col : str, optional
        Salary column name if DataFrame is provided.
    group_col : str, optional
        Group column name if DataFrame is provided.

    Returns
    -------
    float
        Median pay gap.
    """
    res = calculate_pay_equity_gap(
        data=salary if isinstance(salary, pd.DataFrame) else None,
        salary=salary if not isinstance(salary, pd.DataFrame) else salary_col,
        group=group if not isinstance(salary, pd.DataFrame) else (group_col or (group if isinstance(group, str) else None)),
        reference_group=reference_group,
        comparison_group=comparison_group,
        salary_col=salary_col,
        group_col=group_col,
    )
    return float(res["pct_median_gap"] if as_percentage else res["median_gap"])


calculate_median_pay_gap = median_gap

# API Consistency Aliases: compute_* <=> calculate_*
compute_salary_band_midpoint = calculate_salary_band_midpoint
compute_compa_ratio = calculate_compa_ratio
compute_range_penetration = calculate_range_penetration
compute_pay_equity_gap = calculate_pay_equity_gap
compute_median_pay_gap = calculate_median_pay_gap

__all__ = [
    "calculate_salary_band_midpoint",
    "compute_salary_band_midpoint",
    "calculate_compa_ratio",
    "compute_compa_ratio",
    "calculate_range_penetration",
    "compute_range_penetration",
    "classify_pay_band_status",
    "is_red_circled",
    "is_green_circled",
    "calculate_pay_equity_gap",
    "compute_pay_equity_gap",
    "median_gap",
    "calculate_median_pay_gap",
    "compute_median_pay_gap",
]


