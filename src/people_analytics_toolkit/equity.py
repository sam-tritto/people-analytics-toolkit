"""Algorithmic Equity, Pay Decomposition, and Individual Fairness.

This module provides econometric and machine learning tools for evaluating and
enforcing fairness in high-stakes Human Capital Management (HCM) decisions:

1. Oaxaca-Blinder Decomposition (Econometric Pay Equity):
   Partitions demographic wage/compensation gaps into an "Explained" endowment
   effect (observable qualifications like tenure, certifications, job grade)
   and an "Unexplained" structural effect (potential bias or differential returns).
   Supports the Neumark (1988) pooled reference model as well as group reference models.

2. Group Fairness Constraints via Exponentiated Gradient Reduction:
   Minimax optimization framework for fair classification (Demographic Parity
   and Equalized Odds), iteratively building an ensemble of cost-sensitive
   classifiers to guarantee algorithmic equity.

3. Individual Fairness & Local Predictive Consistency:
   Operationalizes the Lipschitz individual fairness condition by evaluating whether
   similar employees in human capital space receive consistent algorithmic predictions
   relative to their k-nearest neighbors.
"""

from dataclasses import dataclass, fields
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from people_analytics_toolkit.compensation import (
    calculate_pay_equity_gap,
    compute_pay_equity_gap,
    median_gap,
    calculate_median_pay_gap,
    compute_median_pay_gap,
)

__all__ = [
    "calculate_pay_equity_gap",
    "compute_pay_equity_gap",
    "median_gap",
    "calculate_median_pay_gap",
    "compute_median_pay_gap",
    "OaxacaBlinderResult",
    "DecompositionResult",
    "oaxaca_blinder_decomposition",
    "demographic_parity_difference",
    "demographic_parity_ratio",
    "equalized_odds_difference",
    "ExponentiatedGradientFairness",
    "calculate_individual_fairness_consistency",
    "compute_individual_fairness_consistency",
]



# ===========================================================================
# 1. OAXACA-BLINDER DECOMPOSITION (PAY & EVALUATION EQUITY)
# ===========================================================================

@dataclass
class OaxacaBlinderResult:
    """Structured result object for Oaxaca-Blinder pay equity gap decomposition."""
    raw_gap: float
    pct_raw_gap: float
    mean_majority: float
    mean_minority: float
    explained_effect: float
    unexplained_effect: float
    explained_pct: float
    unexplained_pct: float
    feature_breakdown: pd.DataFrame
    models: Dict[str, Any]
    metadata: Dict[str, Any]

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
    def majority_model(self) -> Any:
        """Fitted OLS regression model for the majority group."""
        return self.models.get("model_majority", self.models.get("majority"))

    @property
    def minority_model(self) -> Any:
        """Fitted OLS regression model for the minority group."""
        return self.models.get("model_minority", self.models.get("minority"))

    @property
    def pooled_model(self) -> Any:
        """Fitted pooled OLS regression model (if reference_type='pooled')."""
        return self.models.get("model_reference", self.models.get("pooled"))

    @property
    def reference_type(self) -> Optional[str]:
        """Weighting reference structure ('pooled', 'majority', or 'minority')."""
        return self.metadata.get("reference_type")

    @property
    def n_majority(self) -> Optional[int]:
        """Sample size of majority group."""
        return self.metadata.get("n_majority")

    @property
    def n_minority(self) -> Optional[int]:
        """Sample size of minority group."""
        return self.metadata.get("n_minority")


DecompositionResult = OaxacaBlinderResult


def oaxaca_blinder_decomposition(
    df: pd.DataFrame,
    outcome_col: str,
    group_col: str,
    feature_cols: List[str],
    group_majority: Any = 1,
    group_minority: Any = 0,
    reference_type: str = "pooled",
) -> OaxacaBlinderResult:
    """Perform Oaxaca-Blinder wage/outcome gap decomposition between two demographic groups.

    Decomposes the mean outcome difference (bar{Y}_A - bar{Y}_B) into:
    1. Endowment (Explained) Effect:
       Differences in observable qualifications/characteristics (tenure, education, role).
    2. Coefficient (Unexplained / Structural Bias) Effect:
       Differences in returns to characteristics, reflecting potential systemic disparities.

    Mathematical Formulation (Neumark / Oaxaca-Ransom Pooled Reference):
        bar{Y}_A - bar{Y}_B = (bar{X}_A - bar{X}_B) beta* + [bar{X}_A (beta_A - beta*) + bar{X}_B (beta* - beta_B)]
        - Explained = (bar{X}_A - bar{X}_B) beta*
        - Unexplained = bar{X}_A (beta_A - beta*) + bar{X}_B (beta* - beta_B)

    Parameters:
        df: Input DataFrame containing outcomes, demographics, and characteristics.
        outcome_col: Continuous outcome column (e.g. base salary, total comp, performance score).
        group_col: Binary demographic group column (e.g. gender, ethnicity indicator).
        feature_cols: Observable human capital and job characteristic features.
        group_majority: Value in group_col designating majority/reference group A (default 1).
        group_minority: Value in group_col designating minority/target group B (default 0).
        reference_type: 'pooled' (Neumark 1988 pooled model, recommended),
                        'majority' (Oaxaca 1973 group A weights), or
                        'minority' (group B weights).

    Returns:
        Dictionary containing:
        - raw_gap: bar{Y}_A - bar{Y}_B
        - pct_raw_gap: percentage gap relative to minority mean
        - mean_majority: bar{Y}_A
        - mean_minority: bar{Y}_B
        - explained_effect: portion of gap explained by qualifications
        - unexplained_effect: portion of gap attributable to structural differences/bias
        - explained_pct: percentage of total gap explained
        - unexplained_pct: percentage of total gap unexplained
        - feature_breakdown: detailed DataFrame attributing gap to individual features
        - models: dictionary of fitted statsmodels OLS regressions
    """
    df_clean = df.dropna(subset=[outcome_col, group_col] + feature_cols).copy()
    
    group_a_mask = df_clean[group_col] == group_majority
    group_b_mask = df_clean[group_col] == group_minority
    
    df_a = df_clean[group_a_mask]
    df_b = df_clean[group_b_mask]
    
    if len(df_a) == 0 or len(df_b) == 0:
        raise ValueError(
            f"Insufficient observations: majority group has {len(df_a)}, "
            f"minority group has {len(df_b)} rows."
        )
    
    # Outcomes and Means
    y_a = df_a[outcome_col].to_numpy(dtype=float)
    y_b = df_b[outcome_col].to_numpy(dtype=float)
    y_pooled = df_clean[outcome_col].to_numpy(dtype=float)
    
    mean_y_a = float(np.mean(y_a))
    mean_y_b = float(np.mean(y_b))
    raw_gap = mean_y_a - mean_y_b
    pct_raw_gap = (raw_gap / mean_y_b) * 100 if mean_y_b != 0 else np.nan
    
    # Design matrices with intercept
    X_a_raw = df_a[feature_cols].to_numpy(dtype=float)
    X_b_raw = df_b[feature_cols].to_numpy(dtype=float)
    X_pooled_raw = df_clean[feature_cols].to_numpy(dtype=float)
    
    X_a = sm.add_constant(X_a_raw, has_constant="add")
    X_b = sm.add_constant(X_b_raw, has_constant="add")
    X_pooled = sm.add_constant(X_pooled_raw, has_constant="add")
    
    all_cols = ["const"] + list(feature_cols)
    
    # Fit OLS regressions
    model_a = sm.OLS(y_a, X_a).fit()
    model_b = sm.OLS(y_b, X_b).fit()
    model_pooled = sm.OLS(y_pooled, X_pooled).fit()
    
    beta_a = model_a.params
    beta_b = model_b.params
    beta_pooled = model_pooled.params
    
    # Select reference coefficients beta*
    if reference_type == "pooled":
        beta_star = beta_pooled
    elif reference_type == "majority":
        beta_star = beta_a
    elif reference_type == "minority":
        beta_star = beta_b
    else:
        raise ValueError(f"Unknown reference_type '{reference_type}'. Choose 'pooled', 'majority', or 'minority'.")
        
    mean_x_a = np.mean(X_a, axis=0)
    mean_x_b = np.mean(X_b, axis=0)
    diff_x = mean_x_a - mean_x_b
    
    # Decompositions:
    # Explained (Endowments): (bar{X}_A - bar{X}_B) * beta*
    # Note: for 'const', diff_x is 1 - 1 = 0, so intercept does not explain qualification differences.
    explained_by_feature = diff_x * beta_star
    total_explained = float(np.sum(explained_by_feature))
    
    # Unexplained (Coefficients / Bias):
    # bar{X}_A * (beta_A - beta*) + bar{X}_B * (beta* - beta_B)
    unexplained_a_part = mean_x_a * (beta_a - beta_star)
    unexplained_b_part = mean_x_b * (beta_star - beta_b)
    unexplained_by_feature = unexplained_a_part + unexplained_b_part
    total_unexplained = float(np.sum(unexplained_by_feature))
    
    # Construct feature breakdown table
    breakdown_rows = []
    for idx, col in enumerate(all_cols):
        breakdown_rows.append({
            "feature": col,
            "mean_majority": float(mean_x_a[idx]),
            "mean_minority": float(mean_x_b[idx]),
            "mean_difference": float(diff_x[idx]),
            "beta_majority": float(beta_a[idx]),
            "beta_minority": float(beta_b[idx]),
            "beta_reference": float(beta_star[idx]),
            "explained_effect": float(explained_by_feature[idx]),
            "unexplained_effect": float(unexplained_by_feature[idx]),
            "total_feature_contribution": float(explained_by_feature[idx] + unexplained_by_feature[idx]),
        })
    breakdown_df = pd.DataFrame(breakdown_rows)
    
    explained_pct = (total_explained / raw_gap) * 100 if raw_gap != 0 else 0.0
    unexplained_pct = (total_unexplained / raw_gap) * 100 if raw_gap != 0 else 0.0
    
    return OaxacaBlinderResult(
        raw_gap=raw_gap,
        pct_raw_gap=pct_raw_gap,
        mean_majority=mean_y_a,
        mean_minority=mean_y_b,
        explained_effect=total_explained,
        unexplained_effect=total_unexplained,
        explained_pct=explained_pct,
        unexplained_pct=unexplained_pct,
        feature_breakdown=breakdown_df,
        models={
            "model_majority": model_a,
            "model_minority": model_b,
            "model_reference": model_pooled,
        },
        metadata={
            "reference_type": reference_type,
            "n_majority": len(df_a),
            "n_minority": len(df_b),
        },
    )


# ===========================================================================
# 2. GROUP FAIRNESS METRICS
# ===========================================================================

def demographic_parity_difference(
    y_pred: Union[pd.Series, np.ndarray],
    sensitive_features: Union[pd.Series, np.ndarray],
) -> float:
    """Calculate Demographic Parity Difference (Statistical Parity Gap).

    Defined as: max_a P(hat{Y}=1 | A=a) - min_a P(hat{Y}=1 | A=a).
    A value of 0.0 indicates perfect demographic parity.
    """
    y_p = np.asarray(y_pred).ravel()
    sens = np.asarray(sensitive_features).ravel()
    groups = np.unique(sens)
    
    rates = []
    for g in groups:
        mask = sens == g
        if np.sum(mask) > 0:
            rates.append(np.mean(y_p[mask]))
            
    if len(rates) < 2:
        return 0.0
    return float(np.max(rates) - np.min(rates))


def demographic_parity_ratio(
    y_pred: Union[pd.Series, np.ndarray],
    sensitive_features: Union[pd.Series, np.ndarray],
) -> float:
    """Calculate Demographic Parity Ratio (Four-Fifths / 80% Rule metric).

    Defined as: min_a P(hat{Y}=1 | A=a) / max_a P(hat{Y}=1 | A=a).
    Values >= 0.80 satisfy the EEOC 80% selection rate standard.
    """
    y_p = np.asarray(y_pred).ravel()
    sens = np.asarray(sensitive_features).ravel()
    groups = np.unique(sens)
    
    rates = []
    for g in groups:
        mask = sens == g
        if np.sum(mask) > 0:
            rates.append(np.mean(y_p[mask]))
            
    if len(rates) < 2 or np.max(rates) == 0:
        return 1.0
    return float(np.min(rates) / np.max(rates))


def equalized_odds_difference(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    sensitive_features: Union[pd.Series, np.ndarray],
) -> float:
    """Calculate Equalized Odds Difference.

    Defined as the maximum absolute difference across groups between True Positive
    Rates (TPR) and False Positive Rates (FPR):
        max( |TPR_{A=1} - TPR_{A=0}|, |FPR_{A=1} - FPR_{A=0}| )
    A value of 0.0 indicates perfect equalized odds.
    """
    y_t = np.asarray(y_true).ravel()
    y_p = np.asarray(y_pred).ravel()
    sens = np.asarray(sensitive_features).ravel()
    groups = np.unique(sens)
    
    tprs = []
    fprs = []
    for g in groups:
        mask = sens == g
        y_t_g = y_t[mask]
        y_p_g = y_p[mask]
        
        pos_mask = y_t_g == 1
        neg_mask = y_t_g == 0
        
        tpr = np.mean(y_p_g[pos_mask]) if np.sum(pos_mask) > 0 else 0.0
        fpr = np.mean(y_p_g[neg_mask]) if np.sum(neg_mask) > 0 else 0.0
        
        tprs.append(tpr)
        fprs.append(fpr)
        
    if len(tprs) < 2:
        return 0.0
        
    tpr_diff = np.max(tprs) - np.min(tprs)
    fpr_diff = np.max(fprs) - np.min(fprs)
    return float(max(tpr_diff, fpr_diff))


# ===========================================================================
# 3. EXPONENTIATED GRADIENT FAIRNESS REDUCTION
# ===========================================================================

class ExponentiatedGradientFairness(BaseEstimator, ClassifierMixin):
    """Enforces Group Fairness Constraints via Exponentiated Gradient Reduction.

    Treats fair classification as a zero-sum game between:
    - Learner: Minimizes cost-sensitive classification error.
    - Auditor: Maximizes fairness constraint violations across demographic groups.

    Supports:
    - 'demographic_parity': Enforces P(hat{Y}=1 | A=a) approx P(hat{Y}=1 | A=b) across all group pairs.
    - 'equalized_odds': Enforces equal TPR and FPR across all demographic group pairs.
    - Binary and multi-group sensitive attributes (e.g. race/ethnicity, age brackets).

    Parameters:
        estimator: Base scikit-learn classifier supporting sample_weight
                   (defaults to LogisticRegression).
        constraint: 'demographic_parity' or 'equalized_odds'.
        eps: Maximum allowed fairness violation tolerance (epsilon).
        max_iter: Number of game-theoretic reduction iterations (default 20).
        learning_rate: Exponentiated gradient step size eta (default 0.5).
        random_state: Random seed for reproducibility.
    """

    def __init__(
        self,
        estimator: Optional[BaseEstimator] = None,
        constraint: str = "demographic_parity",
        eps: float = 0.01,
        max_iter: int = 20,
        learning_rate: float = 0.5,
        random_state: int = 42,
    ):
        self.estimator = estimator
        self.constraint = constraint
        self.eps = eps
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.random_state = random_state
        
        self.classifiers_: List[BaseEstimator] = []
        self.weights_: List[float] = []
        self.dual_history_: List[np.ndarray] = []
        self.violation_history_: List[float] = []

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        sensitive_features: Union[pd.Series, np.ndarray],
    ) -> "ExponentiatedGradientFairness":
        """Fit fair classifier ensemble using Exponentiated Gradient updates.
        
        Supports binary and multi-group sensitive attributes.
        """
        rng = np.random.default_rng(self.random_state)
        X_mat = np.asarray(X)
        y_arr = np.asarray(y).ravel()
        sens = np.asarray(sensitive_features).ravel()
        
        n_samples = len(y_arr)
        groups = np.unique(sens)
        self.groups_ = groups
        self.classes_ = np.unique(y_arr)
        
        if len(groups) < 2:
            raise ValueError(
                f"ExponentiatedGradientFairness requires at least 2 sensitive groups, "
                f"found {len(groups)}: {groups}"
            )

        group_masks = {g: (sens == g) for g in groups}
        group_probs = {g: max(float(np.mean(group_masks[g])), 0.01) for g in groups}

        # Pairwise combinations for all groups: (g_a, g_b) with a < b
        pairs = []
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                pairs.append((groups[i], groups[j]))
        n_pairs = len(pairs)

        # Base estimator factory
        base_est = self.estimator if self.estimator is not None else LogisticRegression(
            solver="liblinear", random_state=self.random_state
        )

        # Setup constraints
        if self.constraint == "demographic_parity":
            n_constraints = 2 * n_pairs
        elif self.constraint == "equalized_odds":
            n_constraints = 4 * n_pairs
        else:
            raise ValueError(
                f"Unknown constraint '{self.constraint}'. Choose 'demographic_parity' or 'equalized_odds'."
            )

        # Initialize Auditor dual weights lambda uniformly
        lambdas = np.ones(n_constraints) / n_constraints
        
        self.classifiers_ = []
        self.weights_ = []
        self.dual_history_ = []
        self.violation_history_ = []

        for it in range(self.max_iter):
            # Compute sample-level Lagrangian gradient weights
            # Objective: min Error + sum_j lambda_j * violation_j(hat{Y})
            if self.constraint == "demographic_parity":
                group_modifiers = {g: 0.0 for g in groups}
                for p_idx, (g_a, g_b) in enumerate(pairs):
                    delta = lambdas[2 * p_idx] - lambdas[2 * p_idx + 1]
                    group_modifiers[g_a] += delta / group_probs[g_a]
                    group_modifiers[g_b] -= delta / group_probs[g_b]

                modifiers = np.array([group_modifiers[s] for s in sens])
                sample_weights = np.where(y_arr == 1, 1.0 - 0.5 * modifiers, 1.0 + 0.5 * modifiers)
                sample_weights = np.maximum(sample_weights, 0.01)

            elif self.constraint == "equalized_odds":
                group_mod_tpr = {g: 0.0 for g in groups}
                group_mod_fpr = {g: 0.0 for g in groups}
                for p_idx, (g_a, g_b) in enumerate(pairs):
                    delta_tpr = lambdas[4 * p_idx] - lambdas[4 * p_idx + 1]
                    delta_fpr = lambdas[4 * p_idx + 2] - lambdas[4 * p_idx + 3]
                    group_mod_tpr[g_a] += delta_tpr / group_probs[g_a]
                    group_mod_tpr[g_b] -= delta_tpr / group_probs[g_b]
                    group_mod_fpr[g_a] += delta_fpr / group_probs[g_a]
                    group_mod_fpr[g_b] -= delta_fpr / group_probs[g_b]

                mod_tpr = np.array([group_mod_tpr[s] for s in sens])
                mod_fpr = np.array([group_mod_fpr[s] for s in sens])
                sample_weights = np.where(y_arr == 1, 1.0 - 0.5 * mod_tpr, 1.0 + 0.5 * mod_fpr)
                sample_weights = np.maximum(sample_weights, 0.01)

            # Normalize sample weights to sum to n_samples
            sample_weights = (sample_weights / np.mean(sample_weights))

            # Train Learner classifier
            clf = clone(base_est)
            try:
                clf.fit(X_mat, y_arr, sample_weight=sample_weights)
            except TypeError:
                # If base estimator doesn't take sample_weight, bootstrap resample
                probs = sample_weights / np.sum(sample_weights)
                idx = rng.choice(n_samples, size=n_samples, replace=True, p=probs)
                clf.fit(X_mat[idx], y_arr[idx])
                
            y_pred_iter = clf.predict(X_mat)
            violations = np.zeros(n_constraints, dtype=float)

            # Evaluate Auditor constraint violations across all group pairs
            if self.constraint == "demographic_parity":
                group_rates = {}
                for g in groups:
                    mask = group_masks[g]
                    group_rates[g] = float(np.mean(y_pred_iter[mask])) if np.sum(mask) > 0 else 0.0

                diffs = []
                for p_idx, (g_a, g_b) in enumerate(pairs):
                    diff = group_rates[g_a] - group_rates[g_b]
                    diffs.append(abs(diff))
                    violations[2 * p_idx] = diff - self.eps
                    violations[2 * p_idx + 1] = -diff - self.eps

                curr_violation = max(diffs) if diffs else 0.0
            else:
                group_tpr = {}
                group_fpr = {}
                for g in groups:
                    mask = group_masks[g]
                    pos = mask & (y_arr == 1)
                    neg = mask & (y_arr == 0)
                    group_tpr[g] = float(np.mean(y_pred_iter[pos])) if np.sum(pos) > 0 else 0.0
                    group_fpr[g] = float(np.mean(y_pred_iter[neg])) if np.sum(neg) > 0 else 0.0

                max_diff_tpr = 0.0
                max_diff_fpr = 0.0
                for p_idx, (g_a, g_b) in enumerate(pairs):
                    diff_tpr = group_tpr[g_a] - group_tpr[g_b]
                    diff_fpr = group_fpr[g_a] - group_fpr[g_b]
                    max_diff_tpr = max(max_diff_tpr, abs(diff_tpr))
                    max_diff_fpr = max(max_diff_fpr, abs(diff_fpr))
                    violations[4 * p_idx] = diff_tpr - self.eps
                    violations[4 * p_idx + 1] = -diff_tpr - self.eps
                    violations[4 * p_idx + 2] = diff_fpr - self.eps
                    violations[4 * p_idx + 3] = -diff_fpr - self.eps

                curr_violation = max(max_diff_tpr, max_diff_fpr)

            # Store iteration results
            self.classifiers_.append(clf)
            self.weights_.append(1.0)
            self.dual_history_.append(lambdas.copy())
            self.violation_history_.append(curr_violation)

            # Exponentiated Gradient update for Auditor dual variables
            lambdas = lambdas * np.exp(self.learning_rate * violations)
            lambdas_sum = np.sum(lambdas)
            if lambdas_sum > 0:
                lambdas = lambdas / lambdas_sum
            else:
                lambdas = np.ones(n_constraints) / n_constraints

        # Normalize mixture weights across ensemble
        self.weights_ = [w / len(self.weights_) for w in self.weights_]
        return self

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict class probabilities as the ensemble expectation over fitted classifiers."""
        X_mat = np.asarray(X)
        if not self.classifiers_:
            raise ValueError("Model is not fitted. Call fit() first.")
            
        probas = np.zeros((len(X_mat), 2), dtype=float)
        for w, clf in zip(self.weights_, self.classifiers_):
            if hasattr(clf, "predict_proba"):
                probas += w * clf.predict_proba(X_mat)
            else:
                preds = clf.predict(X_mat)
                p1 = preds.astype(float)
                p0 = 1.0 - p1
                probas += w * np.column_stack([p0, p1])
                
        return probas

    def predict(self, X: Union[pd.DataFrame, np.ndarray], threshold: float = 0.5) -> np.ndarray:
        """Predict binary class labels using probability threshold."""
        probas = self.predict_proba(X)
        return (probas[:, 1] >= threshold).astype(int)


# ===========================================================================
# 4. INDIVIDUAL FAIRNESS & LOCAL PREDICTIVE CONSISTENCY
# ===========================================================================

def calculate_individual_fairness_consistency(
    X: Union[pd.DataFrame, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    n_neighbors: int = 5,
    distance_metric: str = "euclidean",
) -> Tuple[pd.Series, float, Dict[str, Any]]:
    """Calculate Individual Fairness and Local Predictive Consistency.

    Individual fairness is formalized via the Lipschitz continuity condition:
        d_Y(hat{y}_i, hat{y}_j) <= c * d_X(x_i, x_j)
    Axiom: Similar individuals in human capital space must receive similar predictions.

    Local Predictive Consistency measures instance-level agreement with nearest peers:
        Consistency_i = 1 - |hat{y}_i - bar{y}_{N_k(i)}|
        where bar{y}_{N_k(i)} is the average prediction of individual i's k-nearest neighbors.

    Parameters:
        X: Feature matrix of employee qualifications/characteristics.
        y_pred: Model predictions (binary labels or continuous probabilities in [0, 1]).
        n_neighbors: Number of nearest neighbors to define the local peer cluster (default 5).
        distance_metric: Distance metric for nearest neighbor graph (default 'euclidean').

    Returns:
        Tuple of:
        - consistency_series: Row-level consistency score in [0, 1] (1.0 = identical to peers).
        - mean_global_consistency: Overall enterprise Individual Fairness Consistency score in [0, 1].
        - metadata: Dictionary containing:
            - local_discrepancy: Instance-level absolute difference |hat{y}_i - bar{y}_{N_k(i)}|.
            - peer_neighborhood_means: Average prediction of nearest neighbors for each node.
            - empirical_lipschitz_ratio: Upper-bound local Lipschitz constant estimate.
            - high_discrepancy_indices: Outlier indices exceeding 2 standard deviations.
    """
    if isinstance(X, pd.DataFrame):
        X_mat = X.to_numpy(dtype=float)
        index = X.index
    else:
        X_mat = np.asarray(X, dtype=float)
        index = pd.RangeIndex(len(X_mat))
        
    y_p = np.asarray(y_pred, dtype=float).ravel()
    n_samples = len(X_mat)
    
    if n_samples <= n_neighbors:
        raise ValueError(
            f"Sample size ({n_samples}) must exceed n_neighbors ({n_neighbors})."
        )
        
    # Standardize feature coordinates to ensure scale-invariant metric space
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_mat)
    
    # Query k+1 neighbors (neighbor 0 is the query point itself)
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric=distance_metric)
    nn.fit(X_scaled)
    distances, indices = nn.kneighbors(X_scaled)
    
    # Exclude self (index 0)
    peer_indices = indices[:, 1:]
    peer_distances = distances[:, 1:]
    
    # Compute peer average prediction
    peer_predictions = y_p[peer_indices]
    peer_means = np.mean(peer_predictions, axis=1)
    
    # Local discrepancy Delta_i = |hat{y}_i - bar{y}_{N_k(i)}|
    local_discrepancy = np.abs(y_p - peer_means)
    consistency_scores = 1.0 - local_discrepancy
    
    # Compute empirical local Lipschitz constant estimate:
    # max_{j in N_k(i)} |hat{y}_i - hat{y}_j| / (d_X(i, j) + eps)
    discrepancy_matrix = np.abs(y_p[:, np.newaxis] - peer_predictions)
    lipschitz_ratios = discrepancy_matrix / (peer_distances + 1e-6)
    max_lipschitz_per_node = np.max(lipschitz_ratios, axis=1)
    
    mean_global_consistency = float(np.mean(consistency_scores))
    discrepancy_mean = np.mean(local_discrepancy)
    discrepancy_std = np.std(local_discrepancy)
    
    outlier_threshold = discrepancy_mean + 2.0 * max(discrepancy_std, 1e-4)
    high_discrepancy_mask = local_discrepancy > outlier_threshold
    high_discrepancy_indices = index[high_discrepancy_mask].tolist()
    
    consistency_series = pd.Series(consistency_scores, index=index, name="local_predictive_consistency")
    
    metadata = {
        "local_discrepancy": pd.Series(local_discrepancy, index=index, name="local_fairness_discrepancy"),
        "peer_neighborhood_means": pd.Series(peer_means, index=index, name="peer_neighborhood_mean_prediction"),
        "empirical_lipschitz_ratios": pd.Series(max_lipschitz_per_node, index=index, name="local_lipschitz_ratio"),
        "mean_global_consistency": mean_global_consistency,
        "outlier_threshold": float(outlier_threshold),
        "high_discrepancy_indices": high_discrepancy_indices,
        "n_neighbors": n_neighbors,
        "distance_metric": distance_metric,
    }
    
    return consistency_series, mean_global_consistency, metadata


# API Consistency Aliases: compute_* <=> calculate_*
compute_individual_fairness_consistency = calculate_individual_fairness_consistency
