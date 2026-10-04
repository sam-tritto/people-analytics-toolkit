"""Prescriptive Interventions: Causal Inference, Uplift Modeling, and Observational Counterfactuals.

This module implements mathematically rigorous causal inference, uplift modeling,
synthetic control evaluation, and directed acyclic graph (DAG) structure learning designed
to bridge predictive risk modeling and prescriptive human capital interventions:

Key Capabilities & Algorithmic Modules
--------------------------------------
1. Augmented Inverse Propensity Weighting (AIPW):
   - :class:`AIPWResult`: Container for doubly robust treatment effect estimation results.
   - :func:`aipw_estimator`: Doubly robust Average Treatment Effect (ATE) estimator synthesizing
     propensity score weighting and cross-fitted outcome regression.

2. Synthetic Difference-in-Differences (SDiD):
   - :func:`synthetic_difference_in_differences`: Regularized unit and time weighting panel estimator
     for macro-policy evaluations that resolves violations of the parallel trends assumption.

3. Heterogeneous Uplift Modeling & Causal Forests:
   - :class:`HonestCausalTree`: Single honest decision tree maximizing treatment effect heterogeneity
     variance with independent structure-split and leaf-estimation samples.
   - :class:`CausalForestDML`: Heterogeneous treatment effect estimation (CATE) using Robinson
     orthogonalization and an ensemble of out-of-fold honest causal trees.
   - :func:`qini_curve`: Cumulative incremental uplift curve across targeting percentiles.
   - :func:`area_under_uplift_curve`: Area Under the Uplift Curve (AUUC) metric for model ranking.
   - :func:`doubly_robust_uplift_eval`: Model evaluation scoring via doubly robust expected MSE (MSE_W).

4. Synthetic Controls for Policy & Pilot Evaluations:
   - :func:`compute_synthetic_control_weights`: Convex quadratic optimization yielding optimal donor
     unit weights that minimize pre-treatment root mean square prediction error (RMSPE).
   - :func:`build_synthetic_control_twin`: Constructs the counterfactual synthetic twin trajectory
     and computes post-treatment causal lift and cumulative percentage impact.
   - :func:`build_multi_pilot_synthetic_controls`: Batch estimation across multiple treated pilot
     business units or retail stores with placebo-gap inference.

5. Directed Causal Structure Learning (DAG Discovery):
   - :func:`notears_linear`: Continuous structure learning for linear structural equation models
     (SEMs) via smooth acyclicity optimization (tr(e^{W ∘ W}) - d = 0).
   - :func:`compute_directed_causal_edge_weights`: Estimates total directed causal effects via
     matrix inversion ((I - W)^{-1} - I) to untangle direct effects from indirect mediator pathways.
   - :func:`diagnose_causal_confounding`: Distinguishes genuine directed causal links from spurious
     associations caused by shared organizational confounders.
"""

from dataclasses import dataclass, fields
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import warnings
import numpy as np
import pandas as pd
from scipy import optimize, stats
import scipy.linalg as sla
import networkx as nx
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold

from people_analytics_toolkit._deprecation import deprecated_alias

try:
    import numba  # type: ignore
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False

__all__ = [
    "AIPWResult",
    "aipw_estimator",
    "synthetic_difference_in_differences",
    "HonestCausalTree",
    "CausalForestDML",
    "qini_curve",
    "area_under_uplift_curve",
    "doubly_robust_uplift_eval",
    "compute_synthetic_control_weights",
    "calculate_synthetic_control_weights",
    "build_synthetic_control_twin",
    "build_multi_pilot_synthetic_controls",
    "notears_linear",
    "compute_directed_causal_edge_weights",
    "calculate_directed_causal_edge_weights",
    "diagnose_causal_confounding",
]


@dataclass
class AIPWResult:
    """Structured result object for Augmented Inverse Propensity Weighting (AIPW) estimation."""
    ate: float
    se: float
    z_stat: float
    p_value: float
    ci_lower: float
    ci_upper: float
    naive_ate: float
    propensity_scores: np.ndarray
    mu0_pred: np.ndarray
    mu1_pred: np.ndarray
    influence_function: np.ndarray

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
    def mu0_hat(self) -> np.ndarray:
        """Alias for counterfactual control prediction mu0_pred."""
        return self.mu0_pred

    @property
    def mu1_hat(self) -> np.ndarray:
        """Alias for counterfactual treated prediction mu1_pred."""
        return self.mu1_pred

    @property
    def n_samples(self) -> int:
        """Number of observation units in dataset."""
        return len(self.propensity_scores)


def aipw_estimator(
    y: Union[np.ndarray, pd.Series],
    treatment: Union[np.ndarray, pd.Series],
    X: Union[np.ndarray, pd.DataFrame],
    propensity_model: Optional[BaseEstimator] = None,
    outcome_model: Optional[BaseEstimator] = None,
    clip_range: Tuple[float, float] = (0.01, 0.99),
    n_splits: int = 5,
    random_state: int = 42,
) -> AIPWResult:
    r"""Compute the Augmented Inverse Propensity Weighting (AIPW) doubly robust ATE estimator.

    AIPW synthesizes Inverse Propensity Weighting (IPW) with Outcome Regression:
    .. math::
        \text{ATE}_{AIPW} = \frac{1}{N} \sum_{i=1}^N \left(
            \hat{\mu}_1(X_i) - \hat{\mu}_0(X_i)
            + \frac{T_i (Y_i - \hat{\mu}_1(X_i))}{\hat{\pi}(X_i)}
            - \frac{(1 - T_i)(Y_i - \hat{\mu}_0(X_i))}{1 - \hat{\pi}(X_i)}
        \right)

    The estimator remains consistent and unbiased if either the propensity score model
    or the outcome regression model is correctly specified.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        Observed continuous or binary outcome variable.
    treatment : array-like of shape (n_samples,)
        Binary treatment indicator (1 for treated, 0 for control).
    X : array-like of shape (n_samples, n_features)
        Covariate feature matrix.
    propensity_model : BaseEstimator, optional
        Scikit-learn classifier with `predict_proba`. Defaults to LogisticRegression.
    outcome_model : BaseEstimator, optional
        Scikit-learn regressor with `predict`. Defaults to Ridge(alpha=1.0).
    clip_range : tuple of (float, float), default=(0.01, 0.99)
        Lower and upper bounds for clipping propensity scores to prevent extreme weights.
    n_splits : int, default=5
        Number of cross-fitting folds to eliminate in-sample overfitting bias.
    random_state : int, default=42
        Random seed for cross-fitting fold splits.

    Returns
    -------
    AIPWResult
        Structured result container with:
        - 'ate': Estimated Average Treatment Effect.
        - 'se': Asymptotic standard error derived from the influence function.
        - 'z_stat': Wald test z-statistic.
        - 'p_value': Two-sided p-value.
        - 'ci_lower': Lower bound of the 95% confidence interval.
        - 'ci_upper': Upper bound of the 95% confidence interval.
        - 'naive_ate': Unadjusted difference in means (Y_treated - Y_control).
        - 'propensity_scores': Cross-fitted propensity scores.
        - 'mu0_pred': Counterfactual predictions for control state.
        - 'mu1_pred': Counterfactual predictions for treated state.
        - 'influence_function': Vector of individual influence function values psi_i.

    Raises
    ------
    ValueError
        If `y`, `treatment`, or `X` have differing sample lengths;
        if `treatment` is not binary with values {0, 1};
        if the minority treatment group has fewer than 2 samples with `n_splits > 1`;
        or if any CV training fold contains only treated or only control units.
    """
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    t_arr = np.asarray(treatment, dtype=np.float64).ravel()
    X_mat = np.asarray(X, dtype=np.float64)

    n_samples = len(y_arr)
    if len(t_arr) != n_samples or len(X_mat) != n_samples:
        raise ValueError("y, treatment, and X must have identical sample lengths.")

    unique_t = np.unique(t_arr)
    if not np.array_equal(np.sort(unique_t), [0.0, 1.0]):
        raise ValueError("treatment must be binary with values {0, 1}.")

    if propensity_model is None:
        propensity_model = LogisticRegression(C=1.0, max_iter=1000, random_state=random_state)
    if outcome_model is None:
        outcome_model = Ridge(alpha=1.0, random_state=random_state)

    pi_hat = np.zeros(n_samples, dtype=np.float64)
    mu0_hat = np.zeros(n_samples, dtype=np.float64)
    mu1_hat = np.zeros(n_samples, dtype=np.float64)

    min_class_count = int(min(np.sum(t_arr == 0), np.sum(t_arr == 1)))
    if min_class_count < 2 and n_splits > 1:
        raise ValueError(
            f"Minority treatment group has only {min_class_count} sample(s). "
            "At least 2 samples per treatment arm are required for cross-validation."
        )

    if n_splits > 1 and n_samples >= 2 * n_splits and min_class_count >= 2:
        effective_splits = min(n_splits, min_class_count)
        if effective_splits < n_splits:
            warnings.warn(
                f"n_splits={n_splits} exceeds minority treatment count ({min_class_count}). "
                f"Reducing n_splits to {effective_splits}.",
                UserWarning,
                stacklevel=2,
            )
        skf = StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=random_state)
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_mat, t_arr)):
            X_tr, X_te = X_mat[train_idx], X_mat[test_idx]
            y_tr = y_arr[train_idx]
            t_tr = t_arr[train_idx]

            idx_c = np.where(t_tr == 0)[0]
            idx_t = np.where(t_tr == 1)[0]

            if len(idx_c) == 0 or len(idx_t) == 0:
                raise ValueError(
                    f"CV fold {fold_idx} has constant treatment (treated={len(idx_t)}, control={len(idx_c)}). "
                    "Each cross-validation training fold must contain both treated and control units."
                )

            # Fit propensity model
            p_clf = clone(propensity_model)
            p_clf.fit(X_tr, t_tr)
            pi_prob = p_clf.predict_proba(X_te)[:, 1]
            pi_hat[test_idx] = np.clip(pi_prob, clip_range[0], clip_range[1])

            # Fit outcome model for control (T=0)
            m0_reg = clone(outcome_model)
            m0_reg.fit(X_tr[idx_c], y_tr[idx_c])
            mu0_hat[test_idx] = m0_reg.predict(X_te)

            # Fit outcome model for treated (T=1)
            m1_reg = clone(outcome_model)
            m1_reg.fit(X_tr[idx_t], y_tr[idx_t])
            mu1_hat[test_idx] = m1_reg.predict(X_te)
    else:
        idx_c = np.where(t_arr == 0)[0]
        idx_t = np.where(t_arr == 1)[0]
        if len(idx_c) == 0 or len(idx_t) == 0:
            raise ValueError(
                f"Constant treatment detected (treated={len(idx_t)}, control={len(idx_c)}). "
                "Both treated and control units are required."
            )

        p_clf = clone(propensity_model)
        p_clf.fit(X_mat, t_arr)
        pi_hat = np.clip(p_clf.predict_proba(X_mat)[:, 1], clip_range[0], clip_range[1])

        m0_reg = clone(outcome_model)
        m0_reg.fit(X_mat[idx_c], y_arr[idx_c])
        mu0_hat = m0_reg.predict(X_mat)

        m1_reg = clone(outcome_model)
        m1_reg.fit(X_mat[idx_t], y_arr[idx_t])
        mu1_hat = m1_reg.predict(X_mat)

    # AIPW Doubly Robust Score
    dr_score = (
        (mu1_hat - mu0_hat)
        + (t_arr * (y_arr - mu1_hat)) / pi_hat
        - ((1.0 - t_arr) * (y_arr - mu0_hat)) / (1.0 - pi_hat)
    )

    ate = float(np.mean(dr_score))
    psi = dr_score - ate
    se = float(np.sqrt(np.mean(psi**2) / n_samples))
    z_stat = float(ate / se) if se > 0 else 0.0
    p_val = float(2.0 * (1.0 - stats.norm.cdf(abs(z_stat))))
    ci_lower = float(ate - 1.96 * se)
    ci_upper = float(ate + 1.96 * se)

    naive_ate = float(np.mean(y_arr[t_arr == 1]) - np.mean(y_arr[t_arr == 0]))

    return AIPWResult(
        ate=ate,
        se=se,
        z_stat=z_stat,
        p_value=p_val,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        naive_ate=naive_ate,
        propensity_scores=pi_hat,
        mu0_pred=mu0_hat,
        mu1_pred=mu1_hat,
        influence_function=psi,
    )


def synthetic_difference_in_differences(
    df: pd.DataFrame,
    unit_col: str,
    time_col: str,
    outcome_col: str,
    treated_units: List[Any],
    post_period_start: Any,
    l2_regularization: float = 1e-4,
) -> Dict[str, Any]:
    r"""Compute the Synthetic Difference-in-Differences (SDiD) policy effect estimator.

    SDiD relaxes the parallel trends assumption of standard DiD by applying double regularized
    weighting: unit weights :math:`\omega` that match the pre-treatment trajectory of treated units,
    and time weights :math:`\lambda` that match the post-treatment level of control units:
    .. math::
        \hat{\tau}^{SDiD} = \left( \sum_{i \in \text{Tr}} \frac{Y_{i, \text{post}}}{N_{tr}}
        - \sum_{i \in \text{Co}} \omega_i Y_{i, \text{post}} \right)
        - \sum_{t \in \text{Pre}} \lambda_t \left( \sum_{i \in \text{Tr}} \frac{Y_{i, t}}{N_{tr}}
        - \sum_{i \in \text{Co}} \omega_i Y_{i, t} \right)

    Parameters
    ----------
    df : pd.DataFrame
        Balanced or semi-balanced panel dataset.
    unit_col : str
        Column designating individual units (e.g. office location, department, business unit).
    time_col : str
        Column designating time periods (e.g. quarter, month, year).
    outcome_col : str
        Column designating the continuous dependent outcome metric.
    treated_units : list
        List of unit identifiers belonging to the treated group.
    post_period_start : any
        Value indicating the first time period where the treatment policy took effect.
    l2_regularization : float, default=1e-4
        L2 regularization parameter (zeta^2) to ensure well-behaved weight dispersion.

    Returns
    -------
    dict
        Dictionary containing:
        - 'att': SDiD Average Treatment Effect on the Treated.
        - 'se': Jackknife standard error across units.
        - 't_stat': Empirical t-statistic.
        - 'p_value': Two-sided p-value.
        - 'ci_lower': 95% confidence interval lower bound.
        - 'ci_upper': 95% confidence interval upper bound.
        - 'unit_weights': Pandas Series of control unit weights omega.
        - 'time_weights': Pandas Series of pre-treatment time weights lambda.
        - 'treated_mean_pre': Mean outcome of treated units in pre-treatment.
        - 'treated_mean_post': Mean outcome of treated units in post-treatment.
        - 'synthetic_mean_pre': Synthetic control pre-treatment outcome.
        - 'synthetic_mean_post': Synthetic control post-treatment outcome.

    Raises
    ------
    ValueError
        If no treated units are found in the dataset;
        if no control units are available for counterfactual estimation;
        or if `post_period_start` is not found in the time column.
    """
    pivot = df.pivot(index=unit_col, columns=time_col, values=outcome_col)
    pivot = pivot.dropna(axis=0)

    all_units = list(pivot.index)
    all_times = list(pivot.columns)

    tr_set = set(treated_units)
    control_units = [u for u in all_units if u not in tr_set]
    treated_units_found = [u for u in all_units if u in tr_set]

    if not treated_units_found:
        raise ValueError("None of the specified treated units were found in the dataset.")
    if not control_units:
        raise ValueError("No control units available for synthetic counterfactual estimation.")

    try:
        post_idx = all_times.index(post_period_start)
    except ValueError as e:
        raise ValueError(f"post_period_start {post_period_start} not found in time column.") from e

    pre_times = all_times[:post_idx]
    post_times = all_times[post_idx:]

    if not pre_times or not post_times:
        raise ValueError("Both pre-treatment and post-treatment periods must contain at least one time step.")

    # Sub-matrices
    Y_co_pre = pivot.loc[control_units, pre_times].values  # shape: (N_co, T_pre)
    Y_co_post = pivot.loc[control_units, post_times].values  # shape: (N_co, T_post)
    Y_tr_pre = pivot.loc[treated_units_found, pre_times].values  # shape: (N_tr, T_pre)
    Y_tr_post = pivot.loc[treated_units_found, post_times].values  # shape: (N_tr, T_post)

    y_tr_pre_mean = np.mean(Y_tr_pre, axis=0)  # shape: (T_pre,)
    y_tr_post_mean = np.mean(Y_tr_post, axis=0)  # shape: (T_post,)

    n_co = len(control_units)
    t_pre = len(pre_times)

    # 1. Optimize unit weights omega: min || y_tr_pre_mean - c0 - Y_co_pre^T omega ||^2 + zeta^2 ||omega||^2
    def unit_objective(params):
        c0 = params[0]
        omega = params[1:]
        diff = y_tr_pre_mean - c0 - np.dot(omega, Y_co_pre)
        return float(np.sum(diff**2) + l2_regularization * np.sum(omega**2))

    init_omega = np.ones(n_co) / n_co
    init_unit_params = np.concatenate([[np.mean(y_tr_pre_mean) - np.mean(Y_co_pre)], init_omega])
    unit_bounds = [(None, None)] + [(0.0, 1.0) for _ in range(n_co)]
    unit_constraints = [{"type": "eq", "fun": lambda p: np.sum(p[1:]) - 1.0}]

    res_unit = optimize.minimize(
        unit_objective,
        init_unit_params,
        method="SLSQP",
        bounds=unit_bounds,
        constraints=unit_constraints,
        options={"maxiter": 500, "ftol": 1e-7},
    )
    omega_opt = np.maximum(res_unit.x[1:], 0.0)
    omega_opt = omega_opt / np.sum(omega_opt)

    # 2. Optimize time weights lambda: min || y_co_post_mean - c1 - Y_co_pre lambda ||^2 + zeta^2 ||lambda||^2
    y_co_post_mean = np.mean(Y_co_post, axis=1)  # shape: (N_co,)

    def time_objective(params):
        c1 = params[0]
        lambd = params[1:]
        diff = y_co_post_mean - c1 - np.dot(Y_co_pre, lambd)
        return float(np.sum(diff**2) + l2_regularization * np.sum(lambd**2))

    init_lambda = np.ones(t_pre) / t_pre
    init_time_params = np.concatenate([[np.mean(y_co_post_mean) - np.mean(Y_co_pre)], init_lambda])
    time_bounds = [(None, None)] + [(0.0, 1.0) for _ in range(t_pre)]
    time_constraints = [{"type": "eq", "fun": lambda p: np.sum(p[1:]) - 1.0}]

    res_time = optimize.minimize(
        time_objective,
        init_time_params,
        method="SLSQP",
        bounds=time_bounds,
        constraints=time_constraints,
        options={"maxiter": 500, "ftol": 1e-7},
    )
    lambda_opt = np.maximum(res_time.x[1:], 0.0)
    lambda_opt = lambda_opt / np.sum(lambda_opt)

    # 3. Calculate SDiD Point Estimate
    # Post treatment difference: Mean(Y_tr_post) - Omega-weighted Mean(Y_co_post)
    treated_post_scalar = float(np.mean(y_tr_post_mean))
    synthetic_post_scalar = float(np.mean(np.dot(omega_opt, Y_co_post)))

    # Pre treatment difference: Lambda-weighted Mean(Y_tr_pre) - Omega & Lambda weighted Mean(Y_co_pre)
    treated_pre_scalar = float(np.dot(lambda_opt, y_tr_pre_mean))
    synthetic_pre_scalar = float(np.dot(omega_opt, np.dot(Y_co_pre, lambda_opt)))

    att_sdid = (treated_post_scalar - synthetic_post_scalar) - (treated_pre_scalar - synthetic_pre_scalar)

    # 4. Standard Error via Jackknife across control units
    jackknife_estimates = []
    if n_co >= 3:
        for j in range(n_co):
            om_sub = np.delete(omega_opt, j)
            if np.sum(om_sub) > 0:
                om_sub = om_sub / np.sum(om_sub)
                Y_co_post_sub = np.delete(Y_co_post, j, axis=0)
                Y_co_pre_sub = np.delete(Y_co_pre, j, axis=0)
                syn_post_j = float(np.mean(np.dot(om_sub, Y_co_post_sub)))
                syn_pre_j = float(np.dot(om_sub, np.dot(Y_co_pre_sub, lambda_opt)))
                att_j = (treated_post_scalar - syn_post_j) - (treated_pre_scalar - syn_pre_j)
                jackknife_estimates.append(att_j)

        if jackknife_estimates:
            jk_arr = np.array(jackknife_estimates)
            n_jk = len(jk_arr)
            jk_se = float(np.sqrt(((n_jk - 1) / n_jk) * np.sum((jk_arr - np.mean(jk_arr)) ** 2)))
        else:
            jk_se = 0.01
    else:
        jk_se = 0.01

    se = float(jk_se) if jk_se > 0 else 0.01
    t_stat = float(att_sdid / se)
    p_val = float(2.0 * (1.0 - stats.norm.cdf(abs(t_stat))))
    ci_lower = float(att_sdid - 1.96 * se)
    ci_upper = float(att_sdid + 1.96 * se)

    return {
        "att": float(att_sdid),
        "se": se,
        "t_stat": t_stat,
        "p_value": p_val,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "unit_weights": pd.Series(omega_opt, index=control_units, name="unit_weight"),
        "time_weights": pd.Series(lambda_opt, index=pre_times, name="time_weight"),
        "treated_mean_pre": float(np.mean(y_tr_pre_mean)),
        "treated_mean_post": treated_post_scalar,
        "synthetic_mean_pre": float(np.mean(np.dot(omega_opt, Y_co_pre))),
        "synthetic_mean_post": synthetic_post_scalar,
    }


def compute_synthetic_control_weights(
    donor_matrix_pre: np.ndarray,
    target_series_pre: np.ndarray,
    donor_names: Optional[List[Any]] = None,
    intercept: bool = True,
    l2_regularization: float = 1e-4,
) -> Tuple[pd.Series, float, float]:
    r"""Compute optimal convex synthetic control weights W* using SLSQP quadratic optimization.

    Solves the constrained convex optimization problem:
    .. math::
        \min_{\mathbf{w}, c_0} \sum_{t=1}^{T_0} \left( y_{1, t}^{\text{pre}} - c_0 - \sum_{j=1}^J w_j Y_{j, t}^{\text{pre}} \right)^2 + \zeta^2 \sum_{j=1}^J w_j^2
    subject to:
        w_j \ge 0, \quad \sum_{j=1}^J w_j = 1

    Parameters
    ----------
    donor_matrix_pre : np.ndarray
        Array of shape (J, T_pre) containing pre-treatment observations for J donor units across T_pre time steps.
    target_series_pre : np.ndarray
        Array of shape (T_pre,) containing pre-treatment observations for the target pilot unit.
    donor_names : Optional[List[Any]]
        Names or identifiers for the donor units. Defaults to [0, 1, ..., J-1].
    intercept : bool, default=True
        If True, includes level shift constant c0 (Synth-DiD style relaxation).
        If False, enforces c0 = 0 (classical Abadie Synthetic Control).
    l2_regularization : float, default=1e-4
        L2 regularization parameter (zeta^2) on weights to ensure numerical stability and well-conditioned solution.

    Returns
    -------
    Tuple[pd.Series, float, float]
        - weights: pd.Series of optimal donor weights summing to 1.0.
        - intercept_c0: Optimal baseline level shift constant.
        - pre_rmspe: Pre-treatment root mean squared prediction error between target and synthetic twin.

    Raises
    ------
    ValueError
        If `donor_matrix_pre` or `target_series_pre` has zero time steps or length mismatch.
    """
    n_donors, t_pre = donor_matrix_pre.shape
    names = donor_names if donor_names is not None else [f"Donor_{j}" for j in range(n_donors)]

    if intercept:
        def objective(params):
            c0 = params[0]
            w = params[1:]
            synth = c0 + np.dot(w, donor_matrix_pre)
            diff = target_series_pre - synth
            return float(np.sum(diff**2) + l2_regularization * np.sum(w**2))

        init_w = np.ones(n_donors) / n_donors
        c0_init = float(np.mean(target_series_pre) - np.mean(donor_matrix_pre))
        init_params = np.concatenate([[c0_init], init_w])
        bounds = [(None, None)] + [(0.0, 1.0) for _ in range(n_donors)]
        constraints = [{"type": "eq", "fun": lambda p: np.sum(p[1:]) - 1.0}]
    else:
        def objective(params):
            w = params
            synth = np.dot(w, donor_matrix_pre)
            diff = target_series_pre - synth
            return float(np.sum(diff**2) + l2_regularization * np.sum(w**2))

        init_params = np.ones(n_donors) / n_donors
        bounds = [(0.0, 1.0) for _ in range(n_donors)]
        constraints = [{"type": "eq", "fun": lambda p: np.sum(p) - 1.0}]

    res = optimize.minimize(
        objective,
        init_params,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 600, "ftol": 1e-8},
    )

    if intercept:
        c0_opt = float(res.x[0])
        w_opt = np.maximum(res.x[1:], 0.0)
    else:
        c0_opt = 0.0
        w_opt = np.maximum(res.x, 0.0)

    if np.sum(w_opt) > 0:
        w_opt = w_opt / np.sum(w_opt)
    else:
        w_opt = np.ones(n_donors) / n_donors

    synth_pre = c0_opt + np.dot(w_opt, donor_matrix_pre)
    pre_rmspe = float(np.sqrt(np.mean((target_series_pre - synth_pre)**2)))

    weights_series = pd.Series(w_opt, index=names, name="synthetic_control_weight")
    return weights_series, c0_opt, pre_rmspe


def build_synthetic_control_twin(
    df: pd.DataFrame,
    unit_col: str,
    time_col: str,
    outcome_col: str,
    pilot_unit: Any,
    post_period_start: Any,
    donor_pool: Optional[List[Any]] = None,
    intercept: bool = True,
    l2_regularization: float = 1e-4,
) -> Dict[str, Any]:
    """Build a synthetic control twin for a single pilot unit and evaluate causal lift trajectory.

    Parameters
    ----------
    df : pd.DataFrame
        Panel dataframe containing time-series observations across units.
    unit_col : str
        Column identifying individual units (e.g. store_id, location_id).
    time_col : str
        Column designating time periods (e.g. week, month, quarter).
    outcome_col : str
        Column containing the continuous outcome metric (e.g. turnover_rate).
    pilot_unit : Any
        Identifier of the pilot unit receiving the policy intervention.
    post_period_start : Any
        First time period where the treatment took effect.
    donor_pool : Optional[List[Any]]
        List of untreated unit IDs eligible as donors. If None, uses all other units in df.
    intercept : bool, default=True
        Whether to include a level shift intercept c0 (Synth-DiD style).
    l2_regularization : float, default=1e-4
        L2 regularization penalty on weights.

    Returns
    -------
    Dict[str, Any]
        - 'pilot_unit': Pilot unit identifier
        - 'weights': pd.Series of optimal donor weights summing to 1.0
        - 'active_weights': pd.Series of active donor weights (> 0.001)
        - 'intercept': Optimal level shift c0
        - 'pre_rmspe': Pre-treatment Root Mean Squared Prediction Error
        - 'post_actual_mean': Mean outcome of pilot unit in post-treatment
        - 'post_synthetic_mean': Mean outcome of synthetic twin in post-treatment
        - 'post_causal_lift_mean': Mean delta (Y_actual - Y_synth) in post-treatment period
        - 'post_causal_lift_pct': Percentage lift relative to synthetic baseline in post period
        - 'trajectory_df': pd.DataFrame with [time_col, 'actual_outcome', 'synthetic_twin', 'causal_lift', 'is_post_treatment']

    Raises
    ------
    ValueError
        If `pilot_unit` is not in `df[unit_col]`, `post_period_start` is not in `df[time_col]`,
        pre- or post-treatment period is empty, or donor pool contains no valid untreated units.
    """
    pivot = df.pivot(index=unit_col, columns=time_col, values=outcome_col).dropna(axis=0)

    if pilot_unit not in pivot.index:
        raise ValueError(f"Pilot unit '{pilot_unit}' not found in '{unit_col}' column.")

    all_times = list(pivot.columns)
    if post_period_start not in all_times:
        raise ValueError(f"post_period_start '{post_period_start}' not found in '{time_col}' values.")

    post_idx = all_times.index(post_period_start)
    pre_times = all_times[:post_idx]
    post_times = all_times[post_idx:]

    if not pre_times or not post_times:
        raise ValueError("Both pre-treatment and post-treatment periods must contain at least one time step.")

    if donor_pool is not None:
        donors = [u for u in donor_pool if u in pivot.index and u != pilot_unit]
    else:
        donors = [u for u in pivot.index if u != pilot_unit]

    if not donors:
        raise ValueError("Donor pool must contain at least one valid untreated unit.")

    Y_target_pre = pivot.loc[pilot_unit, pre_times].values.astype(float)
    Y_donors_pre = pivot.loc[donors, pre_times].values.astype(float)

    weights, c0, pre_rmspe = compute_synthetic_control_weights(
        donor_matrix_pre=Y_donors_pre,
        target_series_pre=Y_target_pre,
        donor_names=donors,
        intercept=intercept,
        l2_regularization=l2_regularization,
    )

    # Reconstruct trajectory across all times
    Y_target_all = pivot.loc[pilot_unit, all_times].values.astype(float)
    Y_donors_all = pivot.loc[donors, all_times].values.astype(float)
    synth_all = c0 + np.dot(weights.values, Y_donors_all)
    causal_lift_all = Y_target_all - synth_all

    traj_df = pd.DataFrame({
        time_col: all_times,
        "actual_outcome": np.round(Y_target_all, 3),
        "synthetic_twin": np.round(synth_all, 3),
        "causal_lift": np.round(causal_lift_all, 3),
        "is_post_treatment": [t in post_times for t in all_times],
    })

    post_mask = traj_df["is_post_treatment"]
    mean_post_actual = float(traj_df.loc[post_mask, "actual_outcome"].mean())
    mean_post_synth = float(traj_df.loc[post_mask, "synthetic_twin"].mean())
    mean_post_lift = float(traj_df.loc[post_mask, "causal_lift"].mean())
    lift_pct = float((mean_post_lift / max(abs(mean_post_synth), 1e-6)) * 100.0)

    active_w = weights[weights > 0.001].sort_values(ascending=False)

    return {
        "pilot_unit": pilot_unit,
        "weights": weights,
        "active_weights": active_w,
        "intercept": round(c0, 4),
        "pre_rmspe": round(pre_rmspe, 4),
        "post_actual_mean": round(mean_post_actual, 3),
        "post_synthetic_mean": round(mean_post_synth, 3),
        "post_causal_lift_mean": round(mean_post_lift, 3),
        "post_causal_lift_pct": round(lift_pct, 2),
        "trajectory_df": traj_df,
    }


def build_multi_pilot_synthetic_controls(
    df: pd.DataFrame,
    unit_col: str,
    time_col: str,
    outcome_col: str,
    pilot_units: List[Any],
    post_period_start: Any,
    donor_pool: Optional[List[Any]] = None,
    intercept: bool = True,
    l2_regularization: float = 1e-4,
) -> Dict[str, Any]:
    """Build synthetic control twins across multiple pilot locations (e.g. 5 pilot stores).

    Parameters
    ----------
    df : pd.DataFrame
        Panel dataframe containing time-series observations across units.
    unit_col : str
        Column identifying units (e.g. store_id).
    time_col : str
        Time period column.
    outcome_col : str
        Outcome metric.
    pilot_units : List[Any]
        List of pilot locations (e.g. ['Store_01', 'Store_02', ...]).
    post_period_start : Any
        Start of the pilot rollout.
    donor_pool : Optional[List[Any]]
        Donor pool of untreated locations. If None, excludes all pilot_units.
    intercept : bool, default=True
        Whether to allow intercept shift.
    l2_regularization : float, default=1e-4
        L2 penalty on weights.

    Returns
    -------
    Dict[str, Any]
        - 'summary_df': pd.DataFrame with performance metrics per pilot store
        - 'weights_matrix': pd.DataFrame of shape (N_pilots, N_donors)
        - 'trajectories_df': Combined DataFrame of all pilot units and their synthetic twins
        - 'aggregate_causal_lift': Mean causal lift across all pilot locations
        - 'aggregate_causal_lift_pct': Mean percentage lift across all pilot locations

    Raises
    ------
    ValueError
        If `pilot_units` list is empty, or if any pilot location fails validation.
    """
    summary_rows = []
    weight_rows = {}
    traj_dfs = []

    pilot_set = set(pilot_units)
    if donor_pool is None:
        eligible_donors = [u for u in df[unit_col].unique() if u not in pilot_set]
    else:
        eligible_donors = [u for u in donor_pool if u not in pilot_set]

    for p_unit in pilot_units:
        single_res = build_synthetic_control_twin(
            df=df,
            unit_col=unit_col,
            time_col=time_col,
            outcome_col=outcome_col,
            pilot_unit=p_unit,
            post_period_start=post_period_start,
            donor_pool=eligible_donors,
            intercept=intercept,
            l2_regularization=l2_regularization,
        )

        top_donors_str = ", ".join([f"{k} ({v:.1%})" for k, v in single_res["active_weights"].head(3).items()])

        summary_rows.append({
            "pilot_unit": p_unit,
            "pre_rmspe": single_res["pre_rmspe"],
            "post_actual_mean": single_res["post_actual_mean"],
            "post_synthetic_mean": single_res["post_synthetic_mean"],
            "post_causal_lift": single_res["post_causal_lift_mean"],
            "post_causal_lift_pct": single_res["post_causal_lift_pct"],
            "top_donors": top_donors_str,
        })

        weight_rows[p_unit] = single_res["weights"]

        p_traj = single_res["trajectory_df"].copy()
        p_traj[unit_col] = p_unit
        traj_dfs.append(p_traj)

    summary_df = pd.DataFrame(summary_rows)
    weights_matrix = pd.DataFrame(weight_rows).T.fillna(0.0)
    all_trajectories = pd.concat(traj_dfs, ignore_index=True)

    agg_lift = float(summary_df["post_causal_lift"].mean())
    agg_lift_pct = float(summary_df["post_causal_lift_pct"].mean())

    return {
        "summary_df": summary_df,
        "weights_matrix": weights_matrix,
        "trajectories_df": all_trajectories,
        "aggregate_causal_lift": round(agg_lift, 3),
        "aggregate_causal_lift_pct": round(agg_lift_pct, 2),
    }


class _HonestCausalNode:
    """Internal node for honest causal tree representation."""

    def __init__(
        self,
        depth: int,
        feature_idx: Optional[int] = None,
        threshold: Optional[float] = None,
        left: Optional["_HonestCausalNode"] = None,
        right: Optional["_HonestCausalNode"] = None,
        tau_val: float = 0.0,
        tau_var: float = 0.0,
        n_estimation_samples: int = 0,
        gain: float = 0.0,
    ):
        self.depth = depth
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.tau_val = tau_val
        self.tau_var = tau_var
        self.n_estimation_samples = n_estimation_samples
        self.gain = gain

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


def _forest_predict_core(
    X: np.ndarray,
    lefts: np.ndarray,
    rights: np.ndarray,
    feats: np.ndarray,
    threshs: np.ndarray,
    taus: np.ndarray,
    tree_offsets: np.ndarray,
) -> np.ndarray:
    n_trees = len(tree_offsets)
    n_samples = X.shape[0]
    all_preds = np.empty((n_trees, n_samples), dtype=np.float64)

    for b in range(n_trees):
        offset = tree_offsets[b]
        for i in range(n_samples):
            node = 0
            while lefts[offset + node] != -1:
                f = feats[offset + node]
                th = threshs[offset + node]
                if X[i, f] <= th:
                    node = lefts[offset + node]
                else:
                    node = rights[offset + node]
            all_preds[b, i] = taus[offset + node]

    return all_preds


if _NUMBA_AVAILABLE:
    _forest_predict_fast = numba.njit(fastmath=True)(_forest_predict_core)
else:
    _forest_predict_fast = _forest_predict_core


class HonestCausalTree:
    r"""Single Honest Causal Tree implementing strict structural vs estimation sample segregation.

    Splits are chosen on the Structure Sample :math:`S_{tr}` to maximize heterogeneity
    in treatment effects:
    .. math::
        \Delta = \frac{N_L N_R}{N_L + N_R} (\hat{\tau}_L - \hat{\tau}_R)^2

    Leaf treatment effects :math:`\hat{\tau}_{\text{leaf}}` and variances are calculated
    strictly on the disjoint Estimation Sample :math:`S_{est}`.
    """

    def __init__(
        self,
        max_depth: int = 4,
        min_samples_leaf: int = 10,
        max_features: Optional[Union[int, float, str]] = "sqrt",
        random_state: Optional[int] = None,
    ):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.root: Optional[_HonestCausalNode] = None
        self.n_features_in_: Optional[int] = None

    def fit(
        self,
        X_tr: np.ndarray,
        T_res_tr: np.ndarray,
        Y_res_tr: np.ndarray,
        X_est: np.ndarray,
        T_res_est: np.ndarray,
        Y_res_est: np.ndarray,
    ) -> "HonestCausalTree":
        rng = np.random.RandomState(self.random_state)
        n_features = X_tr.shape[1]
        self.n_features_in_ = n_features

        def _calc_leaf_effect(t_res: np.ndarray, y_res: np.ndarray) -> Tuple[float, float, int]:
            n = len(t_res)
            denom = float(np.sum(t_res**2)) + 1e-6
            numer = float(np.sum(t_res * y_res))
            tau = numer / denom
            residuals = y_res - tau * t_res
            var = float(np.sum(residuals**2)) / (denom**2 + 1e-6)
            return float(tau), float(var), int(n)

        def _build_tree(
            idx_tr: np.ndarray,
            idx_est: np.ndarray,
            depth: int,
        ) -> _HonestCausalNode:
            tau_leaf, var_leaf, n_est = _calc_leaf_effect(T_res_est[idx_est], Y_res_est[idx_est])

            if depth >= self.max_depth or len(idx_tr) < 2 * self.min_samples_leaf or len(idx_est) < 2:
                return _HonestCausalNode(depth=depth, tau_val=tau_leaf, tau_var=var_leaf, n_estimation_samples=n_est)

            # Feature subsampling
            if self.max_features == "sqrt":
                n_feats_to_try = max(1, int(np.sqrt(n_features)))
            elif isinstance(self.max_features, float):
                n_feats_to_try = max(1, int(self.max_features * n_features))
            elif isinstance(self.max_features, int):
                n_feats_to_try = min(n_features, self.max_features)
            else:
                n_feats_to_try = n_features

            candidate_features = rng.choice(n_features, size=n_feats_to_try, replace=False)

            best_gain = -1.0
            best_feat = None
            best_thresh = None
            best_left_tr = None
            best_right_tr = None

            for feat in candidate_features:
                vals = X_tr[idx_tr, feat]
                thresholds = np.percentile(vals, np.linspace(10, 90, 9))
                for th in np.unique(thresholds):
                    left_mask = vals <= th
                    right_mask = ~left_mask

                    n_l = np.sum(left_mask)
                    n_r = np.sum(right_mask)
                    if n_l < self.min_samples_leaf or n_r < self.min_samples_leaf:
                        continue

                    # Evaluate split criterion on structure sample
                    t_l, y_l = T_res_tr[idx_tr[left_mask]], Y_res_tr[idx_tr[left_mask]]
                    t_r, y_r = T_res_tr[idx_tr[right_mask]], Y_res_tr[idx_tr[right_mask]]

                    tau_l = np.sum(t_l * y_l) / (np.sum(t_l**2) + 1e-6)
                    tau_r = np.sum(t_r * y_r) / (np.sum(t_r**2) + 1e-6)

                    gain = (n_l * n_r / (n_l + n_r)) * ((tau_l - tau_r) ** 2)
                    if gain > best_gain:
                        best_gain = gain
                        best_feat = feat
                        best_thresh = th
                        best_left_tr = idx_tr[left_mask]
                        best_right_tr = idx_tr[right_mask]

            if best_gain <= 0.0 or best_feat is None:
                return _HonestCausalNode(depth=depth, tau_val=tau_leaf, tau_var=var_leaf, n_estimation_samples=n_est)

            # Partition estimation sample using chosen structure split
            est_vals = X_est[idx_est, best_feat]
            left_est = idx_est[est_vals <= best_thresh]
            right_est = idx_est[est_vals > best_thresh]

            if len(left_est) == 0 or len(right_est) == 0:
                return _HonestCausalNode(depth=depth, tau_val=tau_leaf, tau_var=var_leaf, n_estimation_samples=n_est)

            left_child = _build_tree(best_left_tr, left_est, depth + 1)
            right_child = _build_tree(best_right_tr, right_est, depth + 1)

            return _HonestCausalNode(
                depth=depth,
                feature_idx=best_feat,
                threshold=best_thresh,
                left=left_child,
                right=right_child,
                tau_val=tau_leaf,
                tau_var=var_leaf,
                n_estimation_samples=n_est,
                gain=best_gain,
            )

        self.root = _build_tree(
            np.arange(len(X_tr)),
            np.arange(len(X_est)),
            depth=0,
        )
        return self

    def compute_feature_importances(self, importance_type: str = "gain") -> np.ndarray:
        """Compute feature importances for this tree.

        Parameters
        ----------
        importance_type : {'gain', 'split', 'frequency'}, default='gain'
            - 'gain': Weighted by treatment effect heterogeneity gain.
            - 'split' or 'frequency': Count of splits on each feature.

        Returns
        -------
        importances : np.ndarray of shape (n_features_in_,)
        """
        if self.root is None or self.n_features_in_ is None:
            raise ValueError("HonestCausalTree is not fitted yet.")

        importances = np.zeros(self.n_features_in_, dtype=np.float64)

        def _traverse(node: Optional[_HonestCausalNode]):
            if node is None or node.is_leaf:
                return
            if node.feature_idx is not None:
                if importance_type in ("split", "frequency"):
                    importances[node.feature_idx] += 1.0
                else:
                    importances[node.feature_idx] += node.gain
            _traverse(node.left)
            _traverse(node.right)

        _traverse(self.root)
        total = np.sum(importances)
        if total > 0:
            importances /= total
        return importances

    @property
    def feature_importances_(self) -> np.ndarray:
        """Normalized feature importances based on treatment effect heterogeneity gain."""
        return self.compute_feature_importances(importance_type="gain")

    def flatten(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Flatten tree into contiguous arrays for vectorized evaluation."""
        if self.root is None:
            return (
                np.array([-1], dtype=np.int32),
                np.array([-1], dtype=np.int32),
                np.array([-1], dtype=np.int32),
                np.array([0.0], dtype=np.float64),
                np.array([0.0], dtype=np.float64),
            )
        lefts: List[int] = []
        rights: List[int] = []
        feats: List[int] = []
        threshs: List[float] = []
        taus: List[float] = []

        queue = [self.root]
        i = 0
        while i < len(queue):
            curr = queue[i]
            taus.append(float(curr.tau_val))
            if curr.is_leaf:
                lefts.append(-1)
                rights.append(-1)
                feats.append(-1)
                threshs.append(0.0)
            else:
                feats.append(int(curr.feature_idx) if curr.feature_idx is not None else -1)
                threshs.append(float(curr.threshold) if curr.threshold is not None else 0.0)
                left_idx = len(queue)
                queue.append(curr.left)
                lefts.append(left_idx)
                right_idx = len(queue)
                queue.append(curr.right)
                rights.append(right_idx)
            i += 1

        return (
            np.array(lefts, dtype=np.int32),
            np.array(rights, dtype=np.int32),
            np.array(feats, dtype=np.int32),
            np.array(threshs, dtype=np.float64),
            np.array(taus, dtype=np.float64),
        )

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Vectorized prediction for all samples in X."""
        X_mat = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        lefts, rights, feats, threshs, taus = self.flatten()
        tree_offsets = np.array([0], dtype=np.int32)
        preds = _forest_predict_fast(X_mat, lefts, rights, feats, threshs, taus, tree_offsets)
        return preds[0]

    def predict_instance(self, x: np.ndarray) -> Tuple[float, float]:
        curr = self.root
        while not curr.is_leaf:
            if x[curr.feature_idx] <= curr.threshold:
                curr = curr.left
            else:
                curr = curr.right
        return curr.tau_val, curr.tau_var


class CausalForestDML:
    r"""Causal Forest with Double Machine Learning (DML) and Honest Splitting.

    Estimates heterogeneous treatment effects :math:`\tau(X) = \mathbb{E}[Y(1) - Y(0) \mid X]`
    by orthogonalizing the causal signal (Robinson transformation):
    .. math::
        \tilde{Y} = Y - \hat{m}(X), \quad \tilde{T} = T - \hat{e}(X)
        \tilde{Y} = \tau(X) \tilde{T} + \varepsilon

    Honest trees partition data into disjoint structure and estimation sets to ensure valid
    asymptotic confidence intervals and prevent leaf-level overfitting.

    Parameters
    ----------
    n_estimators : int, default=50
        Number of honest causal trees in the forest ensemble.
    max_depth : int, default=4
        Maximum depth of each causal tree.
    min_samples_leaf : int, default=10
        Minimum number of samples required in each leaf node.
    subsample_ratio : float, default=0.7
        Fraction of data subsampled for each tree before 50/50 honest splitting.
    propensity_model : BaseEstimator, optional
        Nuisance estimator for treatment propensity e(X).
    outcome_model : BaseEstimator, optional
        Nuisance estimator for outcome regression m(X).
    random_state : int, default=42
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_estimators: int = 50,
        max_depth: int = 4,
        min_samples_leaf: int = 10,
        subsample_ratio: float = 0.7,
        propensity_model: Optional[BaseEstimator] = None,
        outcome_model: Optional[BaseEstimator] = None,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.subsample_ratio = subsample_ratio
        self.propensity_model = propensity_model
        self.outcome_model = outcome_model
        self.random_state = random_state
        self.trees: List[HonestCausalTree] = []
        self.feature_names: Optional[List[str]] = None
        self.n_features_in_: Optional[int] = None

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        treatment: Union[np.ndarray, pd.Series],
        y: Union[np.ndarray, pd.Series],
    ) -> "CausalForestDML":
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X_mat = X.values.astype(np.float64)
        else:
            X_mat = np.asarray(X, dtype=np.float64)

        self.n_features_in_ = X_mat.shape[1]

        t_arr = np.asarray(treatment, dtype=np.float64).ravel()
        y_arr = np.asarray(y, dtype=np.float64).ravel()
        n_samples = len(y_arr)

        rng = np.random.RandomState(self.random_state)

        # 1. Double Machine Learning Orthogonalization
        p_model = (
            clone(self.propensity_model)
            if self.propensity_model is not None
            else LogisticRegression(max_iter=1000, random_state=self.random_state)
        )
        o_model = (
            clone(self.outcome_model)
            if self.outcome_model is not None
            else Ridge(alpha=1.0, random_state=self.random_state)
        )

        p_model.fit(X_mat, t_arr)
        pi_hat = np.clip(p_model.predict_proba(X_mat)[:, 1], 0.01, 0.99)
        T_res = t_arr - pi_hat

        o_model.fit(X_mat, y_arr)
        m_hat = o_model.predict(X_mat)
        Y_res = y_arr - m_hat

        # 2. Honest Forest Training
        self.trees = []
        for b in range(self.n_estimators):
            tree_rng = np.random.RandomState(rng.randint(0, 1000000))
            sub_size = max(20, int(self.subsample_ratio * n_samples))
            sub_indices = tree_rng.choice(n_samples, size=sub_size, replace=False)

            # 50/50 Honest Split: Structure Sample (str) and Estimation Sample (est)
            tree_rng.shuffle(sub_indices)
            half = sub_size // 2
            idx_tr = sub_indices[:half]
            idx_est = sub_indices[half:]

            tree = HonestCausalTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=tree_rng.randint(0, 1000000),
            )
            tree.fit(
                X_tr=X_mat[idx_tr],
                T_res_tr=T_res[idx_tr],
                Y_res_tr=Y_res[idx_tr],
                X_est=X_mat[idx_est],
                T_res_est=T_res[idx_est],
                Y_res_est=Y_res[idx_est],
            )
            self.trees.append(tree)

        self._build_flat_forest()
        return self

    @property
    def feature_importances_(self) -> np.ndarray:
        r"""Normalized feature importances across all honest causal trees in the forest.

        Measures the relative contribution of each feature to explaining heterogeneity
        in the Conditional Average Treatment Effect (CATE) :math:`\tau(X)`.
        Features that are chosen more frequently and provide larger heterogeneity gains
        receive higher importance scores.

        Returns
        -------
        importances : np.ndarray of shape (n_features,)
            The normalized feature importances summing to 1.0 (or all zeros if no splits occurred).
        """
        if not self.trees or getattr(self, "n_features_in_", None) is None:
            raise ValueError("CausalForestDML is not fitted yet. Call fit() before accessing feature_importances_.")

        all_tree_importances = np.array([tree.feature_importances_ for tree in self.trees])
        mean_importances = np.mean(all_tree_importances, axis=0)
        total = np.sum(mean_importances)
        if total > 0:
            return mean_importances / total
        return mean_importances

    def get_feature_importances(
        self,
        as_pandas: bool = True,
        importance_type: str = "gain",
    ) -> Union[pd.Series, np.ndarray]:
        """Compute and return feature importances for treatment effect heterogeneity.

        Parameters
        ----------
        as_pandas : bool, default=True
            If True, returns a pd.Series indexed by feature names (if available or generated)
            sorted in descending order of importance.
            If False, returns a 1D numpy array.
        importance_type : {'gain', 'split', 'frequency'}, default='gain'
            Criterion for importance calculation:
            - 'gain': Weighted by variance reduction / heterogeneity improvement in CATE.
            - 'split' or 'frequency': Fraction of total splits utilizing each feature.

        Returns
        -------
        pd.Series or np.ndarray
            Feature importance scores summing to 1.0.
        """
        if not self.trees or getattr(self, "n_features_in_", None) is None:
            raise ValueError("CausalForestDML is not fitted yet. Call fit() before accessing feature importances.")

        all_tree_importances = np.array([
            tree.compute_feature_importances(importance_type=importance_type)
            for tree in self.trees
        ])
        mean_importances = np.mean(all_tree_importances, axis=0)
        total = np.sum(mean_importances)
        if total > 0:
            norm_importances = mean_importances / total
        else:
            norm_importances = mean_importances

        if as_pandas:
            idx = self.feature_names if self.feature_names is not None else [f"x{i}" for i in range(len(norm_importances))]
            return pd.Series(
                norm_importances,
                index=idx,
                name="heterogeneity_feature_importance",
            ).sort_values(ascending=False)

        return norm_importances

    def _build_flat_forest(self):
        """Compile all trees into flat contiguous arrays for vectorized/JIT prediction."""
        all_lefts: List[int] = []
        all_rights: List[int] = []
        all_feats: List[int] = []
        all_threshs: List[float] = []
        all_taus: List[float] = []
        tree_offsets: List[int] = []

        curr_offset = 0
        for tree in self.trees:
            l, r, f, th, tau = tree.flatten()
            tree_offsets.append(curr_offset)
            all_lefts.extend(l)
            all_rights.extend(r)
            all_feats.extend(f)
            all_threshs.extend(th)
            all_taus.extend(tau)
            curr_offset += len(l)

        self._lefts = np.array(all_lefts, dtype=np.int32)
        self._rights = np.array(all_rights, dtype=np.int32)
        self._feats = np.array(all_feats, dtype=np.int32)
        self._threshs = np.array(all_threshs, dtype=np.float64)
        self._taus = np.array(all_taus, dtype=np.float64)
        self._tree_offsets = np.array(tree_offsets, dtype=np.int32)

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict the Conditional Average Treatment Effect (CATE) tau(x) for each sample."""
        if not self.trees:
            raise ValueError("CausalForestDML is not fitted yet. Call fit() before predicting.")
        X_mat = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        if getattr(self, "_lefts", None) is None:
            self._build_flat_forest()

        all_preds = _forest_predict_fast(
            X_mat,
            self._lefts,
            self._rights,
            self._feats,
            self._threshs,
            self._taus,
            self._tree_offsets,
        )
        return np.mean(all_preds, axis=0)

    def predict_interval(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        alpha: float = 0.05,
    ) -> Dict[str, np.ndarray]:
        """Predict CATE along with asymptotic standard error and Wald confidence intervals."""
        if not self.trees:
            raise ValueError("CausalForestDML is not fitted yet. Call fit() before predicting.")
        X_mat = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        if getattr(self, "_lefts", None) is None:
            self._build_flat_forest()

        all_preds = _forest_predict_fast(
            X_mat,
            self._lefts,
            self._rights,
            self._feats,
            self._threshs,
            self._taus,
            self._tree_offsets,
        )

        cate = np.mean(all_preds, axis=0)
        forest_var = np.var(all_preds, axis=0, ddof=1) if len(self.trees) > 1 else np.zeros_like(cate)
        se = np.sqrt(forest_var / len(self.trees) + 1e-8)

        z = stats.norm.ppf(1.0 - alpha / 2.0)
        ci_lower = cate - z * se
        ci_upper = cate + z * se

        return {
            "cate": cate,
            "se": se,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        }

    def identify_persuadables(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        min_effect: float = 0.0,
        alpha: float = 0.05,
    ) -> pd.DataFrame:
        """Identify 'persuadable' candidates whose predicted CATE significantly exceeds min_effect."""
        res = self.predict_interval(X, alpha=alpha)
        cate = res["cate"]
        ci_lower = res["ci_lower"]
        ci_upper = res["ci_upper"]

        is_persuadable = (cate > min_effect) & (ci_lower > 0)
        is_sleeping_dog = (cate < -min_effect) & (ci_upper < 0)

        archetype = np.full(len(cate), "Lost Cause / Sure Thing", dtype=object)
        archetype[is_persuadable] = "Persuadable (High Priority)"
        archetype[is_sleeping_dog] = "Do Not Disturb (Adverse Effect)"

        return pd.DataFrame(
            {
                "predicted_cate": cate,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "is_persuadable": is_persuadable,
                "archetype": archetype,
            }
        )


def qini_curve(
    y_true: Union[np.ndarray, pd.Series],
    treatment: Union[np.ndarray, pd.Series],
    uplift_scores: Union[np.ndarray, pd.Series],
    n_bins: int = 10,
) -> pd.DataFrame:
    r"""Compute cumulative Qini incremental gains across population percentiles.

    For any targeted fraction :math:`s`, the incremental gain :math:`u_s` is calculated
    by normalizing control outcomes to the volume of the treatment cohort:
    .. math::
        u_s = \sum_{i=1}^{N_t(s)} Y_i^T - \frac{N_t(s)}{N_c(s)} \sum_{i=1}^{N_c(s)} Y_i^C

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Observed true target outcome.
    treatment : array-like of shape (n_samples,)
        Binary treatment indicator (1 for treated, 0 for control).
    uplift_scores : array-like of shape (n_samples,)
        Predicted individual treatment effect or uplift score (higher = higher predicted impact).
    n_bins : int, default=10
        Number of percentile evaluation bins (e.g. deciles).

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by fraction with columns:
        - 'fraction': Targeted proportion of population (0.0 to 1.0).
        - 'n_treated': Cumulative treated observations targeted.
        - 'n_control': Cumulative control observations targeted.
        - 'treated_outcomes': Sum of treated outcomes.
        - 'control_outcomes': Sum of control outcomes.
        - 'incremental_gain': Qini cumulative incremental gain.
        - 'random_gain': Expected baseline gain under random targeting.

    Raises
    ------
    ValueError
        If `y_true`, `treatment`, or `uplift_scores` have differing sample lengths,
        or if `n_bins < 1`.
    """
    y_arr = np.asarray(y_true, dtype=np.float64).ravel()
    t_arr = np.asarray(treatment, dtype=np.float64).ravel()
    u_arr = np.asarray(uplift_scores, dtype=np.float64).ravel()

    order = np.argsort(-u_arr)
    y_sorted = y_arr[order]
    t_sorted = t_arr[order]

    n_samples = len(y_sorted)
    fractions = np.linspace(1.0 / n_bins, 1.0, n_bins)

    rows = [{
        "fraction": 0.0,
        "n_treated": 0,
        "n_control": 0,
        "treated_outcomes": 0.0,
        "control_outcomes": 0.0,
        "incremental_gain": 0.0,
        "random_gain": 0.0,
    }]

    total_t = np.sum(t_sorted == 1)
    total_c = np.sum(t_sorted == 0)
    total_y_t = np.sum(y_sorted[t_sorted == 1])
    total_y_c = np.sum(y_sorted[t_sorted == 0])
    total_gain = total_y_t - (total_t / max(1, total_c)) * total_y_c

    for frac in fractions:
        k = int(np.ceil(frac * n_samples))
        y_sub = y_sorted[:k]
        t_sub = t_sorted[:k]

        n_t = int(np.sum(t_sub == 1))
        n_c = int(np.sum(t_sub == 0))
        y_t = float(np.sum(y_sub[t_sub == 1]))
        y_c = float(np.sum(y_sub[t_sub == 0]))

        if n_c > 0 and n_t > 0:
            inc_gain = y_t - (n_t / n_c) * y_c
        else:
            inc_gain = 0.0

        random_gain = frac * total_gain

        rows.append({
            "fraction": float(frac),
            "n_treated": n_t,
            "n_control": n_c,
            "treated_outcomes": y_t,
            "control_outcomes": y_c,
            "incremental_gain": float(inc_gain),
            "random_gain": float(random_gain),
        })

    return pd.DataFrame(rows)


def area_under_uplift_curve(
    y_true: Union[np.ndarray, pd.Series],
    treatment: Union[np.ndarray, pd.Series],
    uplift_scores: Union[np.ndarray, pd.Series],
    n_bins: int = 20,
) -> Dict[str, float]:
    r"""Compute the Area Under the Uplift Curve (AUUC) and normalized Qini coefficient.

    Parameters
    ----------
    y_true : array-like
        Observed true target outcome.
    treatment : array-like
        Binary treatment indicator (1 for treated, 0 for control).
    uplift_scores : array-like
        Predicted CATE or uplift score.
    n_bins : int, default=20
        Number of bins for curve integration.

    Returns
    -------
    dict
        Dictionary containing:
        - 'auuc': Area under the model's Qini uplift curve.
        - 'auuc_random': Area under the random targeting baseline curve.
        - 'qini_score': Net uplift score (AUUC - AUUC_random).

    Raises
    ------
    ValueError
        If `y_true`, `treatment`, or `uplift_scores` have differing sample lengths.
    """
    q_df = qini_curve(y_true, treatment, uplift_scores, n_bins=n_bins)
    fracs = q_df["fraction"].values
    gains = q_df["incremental_gain"].values
    random_gains = q_df["random_gain"].values

    auuc = float(np.trapezoid(gains, fracs))
    auuc_random = float(np.trapezoid(random_gains, fracs))
    qini_score = float(auuc - auuc_random)

    return {
        "auuc": auuc,
        "auuc_random": auuc_random,
        "qini_score": qini_score,
    }


def doubly_robust_uplift_eval(
    y_true: Union[np.ndarray, pd.Series],
    treatment: Union[np.ndarray, pd.Series],
    uplift_scores: Union[np.ndarray, pd.Series],
    X: Union[np.ndarray, pd.DataFrame],
    clip_range: Tuple[float, float] = (0.01, 0.99),
) -> Dict[str, Any]:
    r"""Evaluate individual treatment effect predictions using Doubly Robust Expected MSE (MSE_W).

    Constructs unobservable counterfactual ground truth proxies via doubly robust scores:
    .. math::
        \Gamma_i = \hat{\mu}_1(X_i) - \hat{\mu}_0(X_i)
        + \frac{T_i (Y_i - \hat{\mu}_1(X_i))}{\hat{\pi}(X_i)}
        - \frac{(1 - T_i)(Y_i - \hat{\mu}_0(X_i))}{1 - \hat{\pi}(X_i)}
    .. math::
        MSE_W = \frac{1}{N} \sum_{i=1}^N (\hat{\tau}_i - \Gamma_i)^2

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Observed true target outcome.
    treatment : array-like of shape (n_samples,)
        Binary treatment indicator (1 for treated, 0 for control).
    uplift_scores : array-like of shape (n_samples,)
        Predicted CATE or uplift scores tau_hat(x).
    X : array-like of shape (n_samples, n_features)
        Baseline feature covariates.
    clip_range : tuple of (float, float), default=(0.01, 0.99)
        Propensity score clipping limits.

    Returns
    -------
    dict
        Dictionary containing:
        - 'mse_w': Expected Mean Squared Error of the treatment effect.
        - 'mean_predicted_cate': Mean of predicted uplift scores.
        - 'mean_doubly_robust_score': Mean of doubly robust Gamma scores (AIPW ATE).
        - 'gamma_scores': Array of individual doubly robust ground truth proxies.

    Raises
    ------
    ValueError
        If input arrays have differing sample lengths, or treatment is not binary.
    """
    y_arr = np.asarray(y_true, dtype=np.float64).ravel()
    t_arr = np.asarray(treatment, dtype=np.float64).ravel()
    tau_hat = np.asarray(uplift_scores, dtype=np.float64).ravel()
    X_mat = np.asarray(X, dtype=np.float64)

    p_clf = LogisticRegression(max_iter=1000, random_state=42)
    p_clf.fit(X_mat, t_arr)
    pi_hat = np.clip(p_clf.predict_proba(X_mat)[:, 1], clip_range[0], clip_range[1])

    idx_c = np.where(t_arr == 0)[0]
    idx_t = np.where(t_arr == 1)[0]

    m0_reg = Ridge(alpha=1.0, random_state=42)
    m0_reg.fit(X_mat[idx_c], y_arr[idx_c])
    mu0_hat = m0_reg.predict(X_mat)

    m1_reg = Ridge(alpha=1.0, random_state=42)
    m1_reg.fit(X_mat[idx_t], y_arr[idx_t])
    mu1_hat = m1_reg.predict(X_mat)

    gamma = (
        (mu1_hat - mu0_hat)
        + (t_arr * (y_arr - mu1_hat)) / pi_hat
        - ((1.0 - t_arr) * (y_arr - mu0_hat)) / (1.0 - pi_hat)
    )

    mse_w = float(np.mean((tau_hat - gamma) ** 2))

    return {
        "mse_w": mse_w,
        "mean_predicted_cate": float(np.mean(tau_hat)),
        "mean_doubly_robust_score": float(np.mean(gamma)),
        "gamma_scores": gamma,
    }


def notears_linear(
    X: np.ndarray,
    lambda1: float = 0.05,
    max_iter: int = 100,
    h_tol: float = 1e-8,
    rho_max: float = 1e16,
    w_threshold: float = 0.25,
) -> np.ndarray:
    r"""Solve the NOTEARS continuous optimization problem for linear Structural Equation Models.

    Finds a sparse directed acyclic graph (DAG) adjacency weight matrix :math:`W` minimizing:
    .. math::
        \min_{W} \frac{1}{2n} \|X - X W\|_F^2 + \lambda_1 \|W\|_1
        \quad \text{subject to } h(W) = \text{Tr}(\exp(W \circ W)) - d = 0

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Centered and standardized observational data matrix.
    lambda1 : float, default=0.05
        L1 regularization penalty parameter controlling graph sparsity.
    max_iter : int, default=100
        Maximum outer Augmented Lagrangian iterations.
    h_tol : float, default=1e-8
        Convergence tolerance on the smooth acyclicity constraint :math:`h(W)`.
    rho_max : float, default=1e16
        Maximum quadratic penalty parameter :math:`\rho`.
    w_threshold : float, default=0.25
        Threshold below which absolute edge weights are set to 0.

    Returns
    -------
    np.ndarray of shape (n_features, n_features)
        Directed causal edge weights matrix :math:`W` (:math:`W_{ij}` is effect of :math:`i \to j`).

    Raises
    ------
    ValueError
        If `X` has fewer than 2 features or contains NaN or infinite values.
    """
    n, d = X.shape
    C = (X.T @ X) / n

    def _h(W_mat: np.ndarray) -> Tuple[float, np.ndarray]:
        M = W_mat * W_mat
        E = sla.expm(M)
        h_val = float(np.trace(E) - d)
        grad_h = 2.0 * (E.T * W_mat)
        return h_val, grad_h

    # Split W = W_plus - W_minus with W_plus, W_minus >= 0 to handle L1 penalty smoothly
    bounds = []
    for i in range(2 * d * d):
        idx = i % (d * d)
        r, c = divmod(idx, d)
        if r == c:
            bounds.append((0.0, 0.0))  # Disallow self-loops
        else:
            bounds.append((0.0, None))

    w_est = np.zeros(2 * d * d)
    rho = 1.0
    alpha = 0.0
    h_curr = np.inf

    def _func(w_vec: np.ndarray) -> Tuple[float, np.ndarray]:
        W_p = w_vec[: d * d].reshape((d, d))
        W_m = w_vec[d * d :].reshape((d, d))
        W_mat = W_p - W_m

        diff = W_mat - np.eye(d)
        loss = 0.5 * np.trace(diff.T @ C @ diff)
        grad_W = C @ W_mat - C

        h_val, grad_h = _h(W_mat)

        obj = loss + 0.5 * rho * (h_val**2) + alpha * h_val + lambda1 * np.sum(w_vec)
        grad_W_augmented = grad_W + (rho * h_val + alpha) * grad_h
        grad_p = grad_W_augmented + lambda1
        grad_m = -grad_W_augmented + lambda1

        return obj, np.concatenate([grad_p.ravel(), grad_m.ravel()])

    for _ in range(max_iter):
        res = optimize.minimize(
            _func,
            w_est,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"maxiter": 200, "ftol": 1e-10},
        )
        w_est = res.x
        W_p = w_est[: d * d].reshape((d, d))
        W_m = w_est[d * d :].reshape((d, d))
        W_mat = W_p - W_m
        h_new, _ = _h(W_mat)

        if h_new > 0.25 * h_curr:
            rho = min(10.0 * rho, rho_max)
        alpha += rho * h_new
        h_curr = h_new

        if h_curr <= h_tol or rho >= rho_max:
            break

    # Apply thresholding
    W_mat[np.abs(W_mat) < w_threshold] = 0.0
    np.fill_diagonal(W_mat, 0.0)

    # Ensure graph is strictly a Directed Acyclic Graph (DAG)
    G = nx.DiGraph(W_mat)
    if not nx.is_directed_acyclic_graph(G):
        cycles = list(nx.simple_cycles(G))
        for cycle in cycles:
            min_weight = np.inf
            min_edge = None
            for idx in range(len(cycle)):
                u, v = cycle[idx], cycle[(idx + 1) % len(cycle)]
                wt = abs(W_mat[u, v])
                if wt < min_weight:
                    min_weight = wt
                    min_edge = (u, v)
            if min_edge is not None:
                W_mat[min_edge[0], min_edge[1]] = 0.0
                G.remove_edge(min_edge[0], min_edge[1])

    return W_mat


def compute_directed_causal_edge_weights(
    df: pd.DataFrame,
    variables: Optional[List[str]] = None,
    lambda1: float = 0.05,
    threshold: float = 0.25,
    standardize: bool = False,
) -> Dict[str, Any]:
    r"""Learn the underlying Directed Acyclic Graph (DAG) and directed causal edge weights.

    Untangles observational correlations by solving the NOTEARS smooth acyclicity
    optimization problem, extracting direct causal edge weights :math:`W`, computing total
    causal path effects :math:`T = (I - W)^{-1} - I`, and identifying mutual confounders.

    Parameters
    ----------
    df : pd.DataFrame
        Workforce observations containing numerical features.
    variables : Optional[List[str]], default=None
        Subset of columns to evaluate. If None, uses all numeric columns.
    lambda1 : float, default=0.05
        L1 sparsity penalty for edge discovery.
    threshold : float, default=0.25
        Threshold for filtering spurious or negligible direct edge weights.
    standardize : bool, default=False
        Whether to z-score standardize features prior to causal discovery.
        Default is False, preserving equal error variance identifiability (Peters & Buhlmann, 2014).

    Returns
    -------
    Dict[str, Any]
        - 'directed_weights_df': pd.DataFrame of direct causal edge weights W (row -> col).
        - 'adjacency_matrix': pd.DataFrame binary DAG adjacency matrix.
        - 'total_causal_effects_df': pd.DataFrame of cumulative path effects (I - W)^(-1) - I.
        - 'root_causes': List of exogenous parent variables (in-degree 0, out-degree > 0).
        - 'sink_outcomes': List of target downstream outcomes (out-degree 0, in-degree > 0).
        - 'topological_order': List of variables ordered by causal precedence.
        - 'correlation_matrix': pd.DataFrame Pearson correlation matrix for contrast.
        - 'confounders_map': Dict mapping variable pairs to common causal ancestors.

    Raises
    ------
    ValueError
        If fewer than 2 numerical variables are available in `df` for DAG causal discovery.
    """
    if variables is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        variables = numeric_cols
    else:
        variables = list(variables)

    if len(variables) < 2:
        raise ValueError("At least 2 numerical variables are required for DAG causal discovery.")

    X = df[variables].values.astype(float)
    if standardize:
        means = np.mean(X, axis=0)
        stds = np.std(X, axis=0)
        stds[stds == 0.0] = 1.0
        X = (X - means) / stds

    d = len(variables)
    W = notears_linear(X, lambda1=lambda1, w_threshold=threshold)

    directed_weights_df = pd.DataFrame(W, index=variables, columns=variables)
    adj_matrix = pd.DataFrame((np.abs(W) > 0).astype(int), index=variables, columns=variables)

    # Total causal influence matrix: T = (I - W)^(-1) - I
    try:
        total_effects = sla.inv(np.eye(d) - W) - np.eye(d)
    except sla.LinAlgError:
        total_effects = W.copy()
    total_causal_effects_df = pd.DataFrame(total_effects, index=variables, columns=variables)

    G = nx.DiGraph()
    for v in variables:
        G.add_node(v)
    for i, u in enumerate(variables):
        for j, v in enumerate(variables):
            if W[i, j] != 0.0:
                G.add_edge(u, v, weight=float(W[i, j]))

    if nx.is_directed_acyclic_graph(G):
        topological_order = list(nx.topological_sort(G))
    else:
        topological_order = variables

    root_causes = [v for v in variables if G.in_degree(v) == 0 and G.out_degree(v) > 0]
    sink_outcomes = [v for v in variables if G.out_degree(v) == 0 and G.in_degree(v) > 0]

    # Map confounders for each pair
    confounders_map = {}
    for i, u in enumerate(variables):
        for j, v in enumerate(variables):
            if i < j:
                u_ancestors = nx.ancestors(G, u) if u in G else set()
                v_ancestors = nx.ancestors(G, v) if v in G else set()
                common = list(u_ancestors.intersection(v_ancestors))
                if common:
                    confounders_map[f"{u} <-> {v}"] = common

    corr_df = df[variables].corr()

    return {
        "directed_weights_df": directed_weights_df,
        "adjacency_matrix": adj_matrix,
        "total_causal_effects_df": total_causal_effects_df,
        "root_causes": root_causes,
        "sink_outcomes": sink_outcomes,
        "topological_order": topological_order,
        "correlation_matrix": corr_df,
        "confounders_map": confounders_map,
        "graph": G,
    }


def diagnose_causal_confounding(
    weights_df: pd.DataFrame,
    cause_var: str,
    effect_var: str,
    corr_matrix: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    r"""Diagnose whether an observed association is direct causation, reverse causation, or spurious confounding.

    Parameters
    ----------
    weights_df : pd.DataFrame
        Directed causal edge weights matrix from `compute_directed_causal_edge_weights`.
    cause_var : str
        Hypothesized cause variable (e.g. 'overtime_hours').
    effect_var : str
        Hypothesized effect variable (e.g. 'turnover_risk').
    corr_matrix : Optional[pd.DataFrame], default=None
        Optional correlation matrix for comparison.

    Returns
    -------
    Dict[str, Any]
        - 'cause_var': Hypothesized cause
        - 'effect_var': Hypothesized effect
        - 'direct_causal_weight': Weight of directed edge cause -> effect
        - 'reverse_causal_weight': Weight of directed edge effect -> cause
        - 'has_directed_path': bool indicating if cause -> ... -> effect
        - 'has_reverse_path': bool indicating if effect -> ... -> cause
        - 'common_confounders': List of mutual causal ancestors
        - 'observed_correlation': Pearson r if corr_matrix provided, else None
        - 'verdict': Diagnostic summary ('Direct Root Cause', 'Confounded Association', 'Spurious Correlation')

    Raises
    ------
    ValueError
        If `cause_var` or `effect_var` does not exist in `weights_df`.
    """
    if cause_var not in weights_df.index or effect_var not in weights_df.columns:
        raise ValueError(f"Variables '{cause_var}' and '{effect_var}' must exist in weights_df.")

    direct_w = float(weights_df.loc[cause_var, effect_var])
    reverse_w = float(weights_df.loc[effect_var, cause_var])

    G = nx.DiGraph()
    for col in weights_df.columns:
        G.add_node(col)
    for u in weights_df.index:
        for v in weights_df.columns:
            if weights_df.loc[u, v] != 0.0:
                G.add_edge(u, v, weight=float(weights_df.loc[u, v]))

    has_directed = nx.has_path(G, cause_var, effect_var) if (cause_var in G and effect_var in G) else False
    has_reverse = nx.has_path(G, effect_var, cause_var) if (cause_var in G and effect_var in G) else False

    c_anc = nx.ancestors(G, cause_var) if cause_var in G else set()
    e_anc = nx.ancestors(G, effect_var) if effect_var in G else set()
    common_confounders = sorted(list(c_anc.intersection(e_anc)))

    obs_corr = None
    if corr_matrix is not None and cause_var in corr_matrix.index and effect_var in corr_matrix.columns:
        obs_corr = float(corr_matrix.loc[cause_var, effect_var])

    if direct_w != 0.0 and not common_confounders:
        verdict = f"Direct Root Cause: '{cause_var}' directly drives '{effect_var}' (edge weight: {direct_w:+.3f}) without common confounding."
    elif direct_w != 0.0 and common_confounders:
        verdict = f"Confounded Direct Causation: '{cause_var}' has a direct effect ({direct_w:+.3f}) on '{effect_var}', but is mutually confounded by {common_confounders}."
    elif has_directed and common_confounders:
        verdict = f"Mediated & Confounded: '{cause_var}' impacts '{effect_var}' through indirect pathway, but shared ancestors {common_confounders} inflate correlation."
    elif has_directed and not common_confounders:
        verdict = f"Indirect Causal Path: '{cause_var}' affects '{effect_var}' via downstream mediators without confounding."
    elif not has_directed and common_confounders:
        verdict = f"Spurious Association: No directed causal path from '{cause_var}' to '{effect_var}'. High correlation is driven entirely by common confounders: {common_confounders}."
    elif has_reverse:
        verdict = f"Reverse Causation: '{effect_var}' actually causes '{cause_var}' (reverse edge weight: {reverse_w:+.3f})."
    else:
        verdict = f"Structurally Independent: No causal path or common confounder discovered between '{cause_var}' and '{effect_var}'."

    return {
        "cause_var": cause_var,
        "effect_var": effect_var,
        "direct_causal_weight": direct_w,
        "reverse_causal_weight": reverse_w,
        "has_directed_path": has_directed,
        "has_reverse_path": has_reverse,
        "common_confounders": common_confounders,
        "observed_correlation": obs_corr,
        "verdict": verdict,
    }


# API Consistency Aliases: compute_* <=> calculate_* (Deprecated in favor of compute_*)
calculate_synthetic_control_weights = deprecated_alias(
    compute_synthetic_control_weights,
    "calculate_synthetic_control_weights",
    "compute_synthetic_control_weights",
)
calculate_directed_causal_edge_weights = deprecated_alias(
    compute_directed_causal_edge_weights,
    "calculate_directed_causal_edge_weights",
    "compute_directed_causal_edge_weights",
)

