"""High Cardinality and Hierarchical Cohort Feature Engineering.

Contains:
1. BayesianTargetEncoder: Empirical Bayes shrinkage for high-cardinality categories
   (e.g., STORE_ID x DEPT_CODE) with out-of-fold target encoding.
2. GroupedLOOZScore: Leave-One-Out cohort Z-Scores preventing severe outliers from
   masking themselves by inflating group mean and standard deviation.
"""

from typing import Any, List, Optional, Union
import json
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError

__all__ = ["BayesianTargetEncoder", "GroupedLOOZScore"]


class BayesianTargetEncoder(BaseEstimator, TransformerMixin):
    """Empirical Bayes shrinkage target encoder with m-estimate smoothing.
    
    Compatible with Scikit-Learn's BaseEstimator, TransformerMixin, Pipeline,
    and GridSearchCV, as well as providing full JSON, dictionary, and pickle
    serialization support.

    Formula:
        S_i = (n_i * y_bar_i + m * y_bar_global) / (n_i + m)
        
    Where:
        n_i = sample size for category i
        y_bar_i = sample mean of the target in category i
        m = smoothing parameter (effective prior pseudo-observations)
        y_bar_global = global target mean across the entire dataset
    """
    
    def __init__(
        self,
        m: float = 10.0,
        cv_folds: Optional[int] = 5,
        seed: int = 42,
        group_col: Optional[Union[str, List[str]]] = None,
        target_col: Optional[str] = None,
        return_2d: bool = False,
    ):
        if m <= 0:
            raise ValueError(f"Smoothing parameter m must be positive (m > 0), got {m}")
        if cv_folds is not None and cv_folds < 1:
            raise ValueError(f"cv_folds must be a positive integer or None, got {cv_folds}")
        self.m = float(m)
        self.cv_folds = cv_folds
        self.seed = seed
        self.group_col = group_col
        self.target_col = target_col
        self.return_2d = return_2d
        self.global_mean_: float = 0.0
        self.encoding_map_: dict = {}
        self.counts_map_: dict = {}
        self.is_fitted_: bool = False

    def _validate_inputs(
        self,
        df: pd.DataFrame,
        group_col: Optional[Union[str, List[str]]] = None,
        target_col: Optional[str] = None,
        target_series: Optional[pd.Series] = None,
    ):
        if self.m <= 0:
            raise ValueError(f"Smoothing parameter m must be positive (m > 0), got {self.m}")
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"df must be a pandas DataFrame, got {type(df).__name__}")
        if df.empty:
            raise ValueError("DataFrame cannot be empty.")
        if target_col is not None:
            if target_col not in df.columns:
                raise ValueError(f"target_col '{target_col}' not found in DataFrame.")
            if not pd.api.types.is_numeric_dtype(df[target_col]):
                raise ValueError(
                    f"target_col '{target_col}' must be numeric, got dtype {df[target_col].dtype}"
                )
        if target_series is not None:
            if not pd.api.types.is_numeric_dtype(target_series):
                raise ValueError(
                    f"target data must be numeric, got dtype {target_series.dtype}"
                )
        if group_col is not None:
            if isinstance(group_col, list):
                if not group_col:
                    raise ValueError("group_col list cannot be empty.")
                missing = [c for c in group_col if c not in df.columns]
                if missing:
                    raise ValueError(f"group_col columns {missing} not found in DataFrame.")
            elif isinstance(group_col, str):
                if group_col not in df.columns:
                    raise ValueError(f"group_col '{group_col}' not found in DataFrame.")
            else:
                raise TypeError(
                    f"group_col must be a string or list of strings, got {type(group_col).__name__}"
                )
        
    def fit(
        self,
        X: Union[pd.DataFrame, pd.Series, np.ndarray],
        y: Optional[Union[str, List[str], pd.Series, np.ndarray]] = None,
        target_col: Optional[str] = None,
        group_col: Optional[Union[str, List[str]]] = None,
    ):
        """Fit global mean and shrinkage mapping on full dataset.
        
        Supports both legacy people-analytics syntax:
            fit(df, group_col, target_col)
        and Scikit-Learn standard syntax:
            fit(X, y)
        """
        # Resolve group_col, target_col, and target_data based on invocation pattern
        if isinstance(y, (str, list)):
            # Legacy positional call: fit(df, group_col, target_col)
            actual_group_col = y
            actual_target_col = target_col
            target_data = None
        elif y is not None:
            # Standard Scikit-Learn call: fit(X, y)
            actual_group_col = group_col or self.group_col
            actual_target_col = None
            target_data = y
        else:
            # Keyword arguments or fitted from self attributes
            actual_group_col = group_col or self.group_col
            actual_target_col = target_col or self.target_col
            target_data = None

        if isinstance(X, pd.Series):
            df = X.to_frame()
        elif isinstance(X, np.ndarray):
            df = pd.DataFrame(
                X,
                columns=[f"col_{i}" for i in range(X.shape[1])] if X.ndim > 1 else ["col_0"],
            )
        elif isinstance(X, pd.DataFrame):
            df = X
        else:
            raise TypeError(f"X must be DataFrame, Series, or ndarray, got {type(X).__name__}")

        if actual_group_col is None:
            if df.shape[1] == 1:
                actual_group_col = df.columns[0]
            else:
                raise ValueError("group_col must be specified when DataFrame has multiple columns.")

        if target_data is not None:
            target_s = pd.Series(target_data, index=df.index)
            self._validate_inputs(df, actual_group_col, target_series=target_s)
            targets = target_s
        else:
            if actual_target_col is None:
                raise ValueError("target_col or y must be provided to fit.")
            self._validate_inputs(df, actual_group_col, target_col=actual_target_col)
            targets = df[actual_target_col]

        self.global_mean_ = float(targets.mean())
        
        # If group_col is a list, create a composite key
        if isinstance(actual_group_col, list):
            keys = df[actual_group_col].astype(str).agg("_".join, axis=1)
        else:
            keys = df[actual_group_col].astype(str)
            
        grouped = pd.DataFrame({"key": keys, "target": targets}).groupby("key")["target"].agg(["count", "mean"])
        
        counts = grouped["count"].to_numpy()
        means = grouped["mean"].to_numpy()
        smoothed = (counts * means + self.m * self.global_mean_) / (counts + self.m)
        
        self.encoding_map_ = dict(zip(grouped.index, smoothed))
        self.counts_map_ = dict(zip(grouped.index, counts))
        self.is_fitted_ = True
        return self
        
    def transform(
        self,
        X: Union[pd.DataFrame, pd.Series, np.ndarray],
        group_col: Optional[Union[str, List[str]]] = None,
    ) -> Union[pd.Series, np.ndarray]:
        """Transform test/out-of-sample data using learned bayesian shrinkage values.
        
        If return_2d is True, returns a 2D numpy array of shape (n_samples, 1).
        Otherwise, returns a pandas Series.
        """
        if not getattr(self, "is_fitted_", False):
            raise NotFittedError(
                "This BayesianTargetEncoder instance is not fitted yet. Call 'fit' before using 'transform'."
            )

        if isinstance(X, pd.Series):
            df = X.to_frame()
        elif isinstance(X, np.ndarray):
            df = pd.DataFrame(
                X,
                columns=[f"col_{i}" for i in range(X.shape[1])] if X.ndim > 1 else ["col_0"],
            )
        elif isinstance(X, pd.DataFrame):
            df = X
        else:
            raise TypeError(f"X must be DataFrame, Series, or ndarray, got {type(X).__name__}")

        actual_group_col = group_col or self.group_col
        if actual_group_col is None:
            if df.shape[1] == 1:
                actual_group_col = df.columns[0]
            else:
                raise ValueError("group_col must be specified when DataFrame has multiple columns.")

        self._validate_inputs(df, actual_group_col)
        if isinstance(actual_group_col, list):
            keys = df[actual_group_col].astype(str).agg("_".join, axis=1)
        else:
            keys = df[actual_group_col].astype(str)
            
        encoded = keys.map(self.encoding_map_).fillna(self.global_mean_)
        if self.return_2d:
            return encoded.to_numpy(dtype=float).reshape(-1, 1)
        return encoded

    def _build_inverse_mapping(self):
        """Construct sorted arrays for inverse transform lookup with frequency tie-breaking."""
        val_to_cat = {}
        for cat, val in self.encoding_map_.items():
            count = self.counts_map_.get(cat, 0)
            if val not in val_to_cat:
                val_to_cat[val] = (cat, count)
            else:
                existing_cat, existing_count = val_to_cat[val]
                # Prioritize category with higher sample count; if tied, alphabetical order
                if count > existing_count or (count == existing_count and str(cat) < str(existing_cat)):
                    val_to_cat[val] = (cat, count)

        sorted_items = sorted(val_to_cat.items(), key=lambda item: item[0])
        vals = np.array([item[0] for item in sorted_items], dtype=float)
        cats = np.array([item[1][0] for item in sorted_items], dtype=object)
        return vals, cats

    def inverse_transform(
        self,
        X: Union[pd.Series, pd.DataFrame, np.ndarray, List[float]],
        strategy: str = "nearest",
        tolerance: float = 1e-6,
    ) -> Union[pd.Series, pd.DataFrame, np.ndarray]:
        """Decode encoded numerical values back to their original categories.
        
        Parameters:
            X: Encoded numeric values as a pandas Series, single-column DataFrame,
               1D/2D NumPy array, or list of floats.
            strategy: Matching strategy:
                - 'nearest': matches each float to the closest category in target space.
                  Ties are broken by category frequency in training data.
                - 'exact': requires matches to be within `tolerance` of an encoded value.
                  Unmatched or out-of-range values become None.
            tolerance: Maximum absolute difference allowed when strategy='exact'.
            
        Returns:
            Decoded categories formatted to match the input container structure.
        """
        if not getattr(self, "is_fitted_", False):
            raise NotFittedError(
                "This BayesianTargetEncoder instance is not fitted yet. Call 'fit' before using 'inverse_transform'."
            )
        if not self.encoding_map_:
            raise ValueError("Encoding map is empty; cannot perform inverse_transform.")
        if strategy not in ("nearest", "exact"):
            raise ValueError(f"strategy must be 'nearest' or 'exact', got '{strategy}'")
        if tolerance < 0:
            raise ValueError(f"tolerance must be non-negative, got {tolerance}")

        is_series = False
        is_df = False
        is_2d = False
        orig_index = None
        orig_name = None

        if isinstance(X, pd.Series):
            is_series = True
            orig_index = X.index
            orig_name = X.name
            queries = X.to_numpy(dtype=float)
        elif isinstance(X, pd.DataFrame):
            if X.shape[1] != 1:
                raise ValueError(
                    f"inverse_transform requires a single-column DataFrame, got shape {X.shape}"
                )
            is_df = True
            orig_index = X.index
            orig_name = X.columns[0]
            queries = X.iloc[:, 0].to_numpy(dtype=float)
        elif isinstance(X, np.ndarray):
            if X.ndim == 2:
                if X.shape[1] != 1:
                    raise ValueError(
                        f"inverse_transform requires a 2D array with 1 column, got shape {X.shape}"
                    )
                is_2d = True
                queries = X.ravel().astype(float)
            elif X.ndim == 1:
                queries = X.astype(float)
            else:
                raise ValueError(f"X must be 1D or 2D array, got ndim={X.ndim}")
        elif isinstance(X, (list, tuple)):
            queries = np.array(X, dtype=float)
        else:
            raise TypeError(f"Unsupported type for X: {type(X).__name__}")

        if len(queries) == 0:
            decoded = np.array([], dtype=object)
        else:
            vals, cats = self._build_inverse_mapping()
            nan_mask = np.isnan(queries)
            safe_queries = np.where(nan_mask, vals[0], queries)

            if len(vals) == 1:
                chosen_idx = np.zeros(len(queries), dtype=int)
            else:
                idx = np.searchsorted(vals, safe_queries)
                idx = np.clip(idx, 1, len(vals) - 1)
                left_dist = np.abs(safe_queries - vals[idx - 1])
                right_dist = np.abs(safe_queries - vals[idx])
                chosen_idx = np.where(left_dist <= right_dist, idx - 1, idx)

            min_dist = np.abs(safe_queries - vals[chosen_idx])
            decoded = cats[chosen_idx].copy()

            if strategy == "exact":
                unmatched = (min_dist > tolerance) | nan_mask
                decoded[unmatched] = None
            else:
                decoded[nan_mask] = None

        if is_series:
            return pd.Series(decoded, index=orig_index, name=orig_name)
        elif is_df:
            if self.return_2d:
                return pd.DataFrame({orig_name: decoded}, index=orig_index)
            return pd.Series(decoded, index=orig_index, name=orig_name)
        elif is_2d or (self.return_2d and isinstance(X, np.ndarray)):
            return decoded.reshape(-1, 1)
        return decoded
        
    def fit_transform_oof(
        self,
        df: pd.DataFrame,
        group_col: Optional[Union[str, List[str]]] = None,
        target_col: Optional[str] = None,
    ) -> Union[pd.Series, np.ndarray]:
        """Fit-transform with K-Fold out-of-fold target encoding to eliminate target leakage."""
        actual_group_col = group_col or self.group_col
        actual_target_col = target_col or self.target_col
        if actual_group_col is None:
            if isinstance(df, pd.DataFrame) and df.shape[1] == 1:
                actual_group_col = df.columns[0]
            else:
                raise ValueError("group_col must be specified when DataFrame has multiple columns.")
        if actual_target_col is None:
            raise ValueError("target_col must be provided to fit_transform_oof.")

        self._validate_inputs(df, actual_group_col, actual_target_col)
        if self.cv_folds is None or self.cv_folds <= 1:
            self.fit(df, actual_group_col, actual_target_col)
            return self.transform(df, actual_group_col)
            
        self.global_mean_ = float(df[actual_target_col].mean())
        oof_encoded = pd.Series(index=df.index, dtype=float)
        
        rng = np.random.default_rng(self.seed)
        indices = np.arange(len(df))
        rng.shuffle(indices)
        folds = np.array_split(indices, self.cv_folds)
        
        if isinstance(actual_group_col, list):
            keys = df[actual_group_col].astype(str).agg("_".join, axis=1)
        else:
            keys = df[actual_group_col].astype(str)
            
        for fold_idx, val_idx in enumerate(folds):
            train_idx = np.setdiff1d(indices, val_idx)
            train_df = df.iloc[train_idx]
            train_keys = keys.iloc[train_idx]
            
            fold_global_mean = float(train_df[actual_target_col].mean())
            grouped = train_df.groupby(train_keys)[actual_target_col].agg(["count", "mean"])
            
            counts = grouped["count"].to_numpy()
            means = grouped["mean"].to_numpy()
            smoothed = (counts * means + self.m * fold_global_mean) / (counts + self.m)
            fold_map = dict(zip(grouped.index, smoothed))
            
            oof_encoded.iloc[val_idx] = keys.iloc[val_idx].map(fold_map).fillna(fold_global_mean)
            
        # Also fit full model for future transform calls
        self.fit(df, actual_group_col, actual_target_col)
        if self.return_2d:
            return oof_encoded.to_numpy(dtype=float).reshape(-1, 1)
        return oof_encoded

    def to_dict(self) -> dict:
        """Serialize encoder configuration and learned parameters to a dictionary."""
        return {
            "params": {
                "m": float(self.m),
                "cv_folds": self.cv_folds,
                "seed": self.seed,
                "group_col": self.group_col,
                "target_col": self.target_col,
                "return_2d": bool(self.return_2d),
            },
            "fitted": {
                "is_fitted_": bool(getattr(self, "is_fitted_", False)),
                "global_mean_": float(getattr(self, "global_mean_", 0.0)),
                "encoding_map_": {str(k): float(v) for k, v in getattr(self, "encoding_map_", {}).items()},
                "counts_map_": {str(k): int(v) for k, v in getattr(self, "counts_map_", {}).items()},
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BayesianTargetEncoder":
        """Construct and restore a BayesianTargetEncoder instance from a dictionary."""
        params = data.get("params", {})
        encoder = cls(**params)
        fitted = data.get("fitted", {})
        encoder.is_fitted_ = fitted.get("is_fitted_", False)
        encoder.global_mean_ = float(fitted.get("global_mean_", 0.0))
        encoder.encoding_map_ = fitted.get("encoding_map_", {})
        encoder.counts_map_ = fitted.get("counts_map_", {})
        return encoder

    def to_json(
        self,
        filepath_or_buffer: Optional[Union[str, Any]] = None,
        indent: int = 2,
    ) -> Optional[str]:
        """Serialize encoder to JSON string or write to a file/buffer."""
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
    def from_json(cls, json_str_or_filepath: Union[str, Any]) -> "BayesianTargetEncoder":
        """Restore a BayesianTargetEncoder instance from a JSON string or file path."""
        if isinstance(json_str_or_filepath, str):
            if json_str_or_filepath.strip().startswith("{"):
                data = json.loads(json_str_or_filepath)
            else:
                with open(json_str_or_filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
        else:
            data = json.load(json_str_or_filepath)
        return cls.from_dict(data)


class GroupedLOOZScore(BaseEstimator, TransformerMixin):
    """Grouped Leave-One-Out (LOO) Z-Score calculation and cohort anomaly scoring.
    
    Standard Z-score:
        Z_i = (x_i - mu) / sigma
        Where mu and sigma include x_i. If x_i is a massive outlier, it inflates mu
        and sigma, masking its own anomaly (especially in small cohorts).
        
    Leave-One-Out Z-score:
        mu_{-i} = (sum_{j != i} x_j) / (N - 1)
        sigma_{-i} = sqrt( sum_{j != i} (x_j - mu_{-i})^2 / (N - 2) )
        Z_LOO_i = (x_i - mu_{-i}) / sigma_{-i}

    Supports Scikit-Learn's BaseEstimator and TransformerMixin interfaces,
    allowing integration in Pipelines and tracking historical cohort benchmarks
    for inference on new out-of-sample data.
    """
    
    def __init__(
        self,
        group_col: Optional[Union[str, List[str]]] = None,
        val_col: Optional[str] = None,
        min_cohort_size: int = 3,
        epsilon: float = 1e-6,
        return_scores_only: bool = False,
        return_2d: bool = False,
    ):
        if min_cohort_size < 1:
            raise ValueError(f"min_cohort_size must be >= 1, got {min_cohort_size}")
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        self.group_col = group_col
        self.val_col = val_col
        self.min_cohort_size = min_cohort_size
        self.epsilon = epsilon
        self.return_scores_only = return_scores_only
        self.return_2d = return_2d
        self.cohort_stats_: dict = {}
        self.global_stats_: dict = {}
        self.is_fitted_: bool = False

    def _validate_inputs(
        self,
        df: pd.DataFrame,
        group_col: Optional[Union[str, List[str]]] = None,
        val_col: Optional[str] = None,
    ):
        if self.min_cohort_size < 1:
            raise ValueError(f"min_cohort_size must be >= 1, got {self.min_cohort_size}")
        if self.epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {self.epsilon}")
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"df must be a pandas DataFrame, got {type(df).__name__}")
        if df.empty:
            raise ValueError("DataFrame cannot be empty.")
        if val_col is not None:
            if val_col not in df.columns:
                raise ValueError(f"val_col '{val_col}' not found in DataFrame.")
            if not pd.api.types.is_numeric_dtype(df[val_col]):
                raise ValueError(
                    f"val_col '{val_col}' must be numeric, got dtype {df[val_col].dtype}"
                )
        if group_col is not None:
            if isinstance(group_col, list):
                if not group_col:
                    raise ValueError("group_col list cannot be empty.")
                missing = [c for c in group_col if c not in df.columns]
                if missing:
                    raise ValueError(f"group_col columns {missing} not found in DataFrame.")
            elif isinstance(group_col, str):
                if group_col not in df.columns:
                    raise ValueError(f"group_col '{group_col}' not found in DataFrame.")
            else:
                raise TypeError(
                    f"group_col must be a string or list of strings, got {type(group_col).__name__}"
                )

    @staticmethod
    def _extract_group_keys(df: pd.DataFrame, group_col: Union[str, List[str]]) -> pd.Series:
        if isinstance(group_col, list):
            return df[group_col].astype(str).agg("_".join, axis=1)
        return df[group_col].astype(str)

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[Any] = None,
        group_col: Optional[Union[str, List[str]]] = None,
        val_col: Optional[str] = None,
    ):
        """Fit cohort benchmark statistics on dataset."""
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        actual_group_col = group_col or self.group_col
        actual_val_col = val_col or self.val_col
        if actual_group_col is None:
            raise ValueError("group_col must be specified.")
        if actual_val_col is None:
            raise ValueError("val_col must be specified.")
        self._validate_inputs(df, actual_group_col, actual_val_col)

        keys = self._extract_group_keys(df, actual_group_col)
        vals = df[actual_val_col].to_numpy(dtype=float)

        n_global = len(vals)
        mu_global = float(np.mean(vals)) if n_global > 0 else 0.0
        std_global = float(np.std(vals, ddof=1)) if n_global > 1 else 0.0
        self.global_stats_ = {
            "count": n_global,
            "mean": mu_global,
            "std": std_global,
        }

        cohort_stats = {}
        grouped = df.groupby(keys)[actual_val_col]
        for key, group in grouped:
            g_vals = group.to_numpy(dtype=float)
            n_g = len(g_vals)
            mu_g = float(np.mean(g_vals)) if n_g > 0 else 0.0
            std_g = float(np.std(g_vals, ddof=1)) if n_g > 1 else 0.0
            sum_g = float(np.sum(g_vals))
            sum_sq_g = float(np.sum(g_vals ** 2))
            cohort_stats[str(key)] = {
                "count": n_g,
                "mean": mu_g,
                "std": std_g,
                "sum": sum_g,
                "sum_sq": sum_sq_g,
            }

        self.cohort_stats_ = cohort_stats
        self.is_fitted_ = True
        return self

    def transform(
        self,
        X: pd.DataFrame,
        in_sample: bool = False,
    ) -> Union[pd.DataFrame, pd.Series, np.ndarray]:
        """Transform data by computing cohort Z-scores and LOO Z-scores.
        
        Parameters:
            X: Input pandas DataFrame containing group_col and val_col.
            in_sample: If True, applies leave-one-out adjustments for training cohort members.
                       If False (default for inference), evaluates new out-of-sample data
                       against the fitted cohort benchmarks.
        """
        if not getattr(self, "is_fitted_", False):
            raise NotFittedError(
                "This GroupedLOOZScore instance is not fitted yet. Call 'fit' before using 'transform'."
            )
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        actual_group_col = self.group_col
        actual_val_col = self.val_col
        if actual_group_col is None or actual_val_col is None:
            raise ValueError("group_col and val_col must be configured on estimator.")
        self._validate_inputs(df, actual_group_col, actual_val_col)

        keys = self._extract_group_keys(df, actual_group_col)
        vals = df[actual_val_col].to_numpy(dtype=float)
        n = len(vals)

        std_z = np.zeros(n, dtype=float)
        loo_z = np.zeros(n, dtype=float)
        cohort_size = np.zeros(n, dtype=int)
        loo_mu = np.zeros(n, dtype=float)
        loo_sigma = np.zeros(n, dtype=float)

        for i, (k, x_i) in enumerate(zip(keys, vals)):
            stats = self.cohort_stats_.get(str(k))
            if stats is None:
                # Unseen cohort at inference time -> fallback to global cohort stats
                g_mu = self.global_stats_.get("mean", 0.0)
                g_std = self.global_stats_.get("std", 0.0)
                diff = x_i - g_mu
                z = 0.0 if np.abs(diff) < self.epsilon else diff / max(g_std, self.epsilon)
                std_z[i] = z
                loo_z[i] = z
                cohort_size[i] = 0
                loo_mu[i] = g_mu
                loo_sigma[i] = g_std
            else:
                n_c = stats["count"]
                cohort_size[i] = n_c
                if n_c < self.min_cohort_size:
                    std_z[i] = 0.0
                    loo_z[i] = 0.0
                    loo_mu[i] = stats["mean"]
                    loo_sigma[i] = 0.0
                else:
                    c_mu = stats["mean"]
                    c_std = stats["std"]
                    diff = x_i - c_mu
                    std_z[i] = 0.0 if np.abs(diff) < self.epsilon else diff / max(c_std, self.epsilon)

                    if in_sample and n_c > 2:
                        s1 = stats["sum"]
                        s2 = stats["sum_sq"]
                        mu_minus = (s1 - x_i) / (n_c - 1)
                        s2_minus = s2 - (x_i ** 2)
                        var_minus = max(0.0, (s2_minus - (n_c - 1) * (mu_minus ** 2)) / (n_c - 2))
                        sigma_minus = np.sqrt(var_minus)
                        l_diff = x_i - mu_minus
                        loo_z[i] = 0.0 if np.abs(l_diff) < self.epsilon else l_diff / max(sigma_minus, self.epsilon)
                        loo_mu[i] = mu_minus
                        loo_sigma[i] = sigma_minus
                    else:
                        loo_z[i] = std_z[i]
                        loo_mu[i] = c_mu
                        loo_sigma[i] = c_std

        if self.return_scores_only:
            res_series = pd.Series(loo_z, index=df.index, name="loo_z_score")
            if self.return_2d:
                return res_series.to_numpy().reshape(-1, 1)
            return res_series

        result = df.copy()
        result["cohort_size"] = cohort_size
        result["standard_z_score"] = std_z
        result["loo_cohort_mean"] = loo_mu
        result["loo_cohort_std"] = loo_sigma
        result["loo_z_score"] = loo_z
        return result

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: Optional[Any] = None,
        **fit_params,
    ) -> Union[pd.DataFrame, pd.Series, np.ndarray]:
        """Fit estimator on training cohorts and compute in-sample Leave-One-Out scores."""
        return self.fit(X, y=y, **fit_params).transform(X, in_sample=True)

    @classmethod
    def compute(
        cls,
        df: pd.DataFrame,
        group_col: Union[str, List[str]],
        val_col: str,
        min_cohort_size: int = 3,
        epsilon: float = 1e-6,
    ) -> pd.DataFrame:
        """Compute standard Z-score and LOO Z-score for comparison.
        
        Returns a DataFrame with columns:
        - cohort_size
        - standard_z_score
        - loo_cohort_mean
        - loo_cohort_std
        - loo_z_score
        """
        instance = cls(
            group_col=group_col,
            val_col=val_col,
            min_cohort_size=min_cohort_size,
            epsilon=epsilon,
        )
        return instance.fit_transform(df)

    def to_dict(self) -> dict:
        """Serialize configuration and fitted cohort benchmarks to dictionary."""
        return {
            "params": {
                "group_col": self.group_col,
                "val_col": self.val_col,
                "min_cohort_size": self.min_cohort_size,
                "epsilon": float(self.epsilon),
                "return_scores_only": bool(self.return_scores_only),
                "return_2d": bool(self.return_2d),
            },
            "fitted": {
                "is_fitted_": bool(getattr(self, "is_fitted_", False)),
                "cohort_stats_": getattr(self, "cohort_stats_", {}),
                "global_stats_": getattr(self, "global_stats_", {}),
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GroupedLOOZScore":
        """Reconstruct GroupedLOOZScore instance from dictionary."""
        params = data.get("params", {})
        instance = cls(**params)
        fitted = data.get("fitted", {})
        instance.is_fitted_ = fitted.get("is_fitted_", False)
        instance.cohort_stats_ = fitted.get("cohort_stats_", {})
        instance.global_stats_ = fitted.get("global_stats_", {})
        return instance

    def to_json(
        self,
        filepath_or_buffer: Optional[Union[str, Any]] = None,
        indent: int = 2,
    ) -> Optional[str]:
        """Serialize to JSON string or write to file/buffer."""
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
    def from_json(cls, json_str_or_filepath: Union[str, Any]) -> "GroupedLOOZScore":
        """Restore GroupedLOOZScore from JSON string or file path."""
        if isinstance(json_str_or_filepath, str):
            if json_str_or_filepath.strip().startswith("{"):
                data = json.loads(json_str_or_filepath)
            else:
                with open(json_str_or_filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
        else:
            data = json.load(json_str_or_filepath)
        return cls.from_dict(data)
