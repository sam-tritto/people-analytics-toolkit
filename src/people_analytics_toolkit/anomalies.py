"""Structural Anomalies, Distributional Drift, and Shape Similarity Feature Engineering.

Contains:
1. Autoencoder Reconstruction Error (The "Weirdness" Index): PyTorch bottleneck autoencoder
   for multivariate anomaly scoring across high-dimensional store operations.
2. Kullback-Leibler (KL) Divergence (The "Drift" Index): Distributional drift between local
   departmental staffing allocations and the corporate target benchmark.
3. Dynamic Time Warping (DTW) Distance: Phase-invariant temporal similarity against ideal
   seasonal labor/traffic profiles.
4. Local Outlier Factor (LOF via PyOD): Density-based peer outlier detection.
5. Longest Common Subsequence (LCSS) & Edit Distance on Real Sequences (EDR).
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from pathlib import Path
import json
import pickle
import numpy as np
import pandas as pd
from scipy.stats import chi2

from people_analytics_toolkit._deprecation import deprecated_alias

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None
    nn = None
    optim = None


if _TORCH_AVAILABLE:
    class _AutoencoderModule(nn.Module):
        """Bottleneck Autoencoder PyTorch Module for anomaly reconstruction."""
        def __init__(self, in_d: int, h_d: int, l_d: int):
            super().__init__()
            self.in_d = in_d
            self.h_d = h_d
            self.l_d = l_d
            self.encoder = nn.Sequential(
                nn.Linear(in_d, h_d),
                nn.BatchNorm1d(h_d),
                nn.LeakyReLU(0.1),
                nn.Linear(h_d, l_d),
            )
            self.decoder = nn.Sequential(
                nn.Linear(l_d, h_d),
                nn.BatchNorm1d(h_d),
                nn.LeakyReLU(0.1),
                nn.Linear(h_d, in_d),
            )
            
        def forward(self, x):
            return self.decoder(self.encoder(x))
else:
    _AutoencoderModule = None


def _configure_torch_reproducibility(seed: int) -> None:
    """Configure PyTorch random seed and deterministic backend flags for reproducibility."""
    if not _TORCH_AVAILABLE or torch is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "backends") and hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if hasattr(torch, "use_deterministic_algorithms"):
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass

try:
    import numba  # type: ignore
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False


def _dtw_dp_core(s1: np.ndarray, s2: np.ndarray, w: int) -> np.ndarray:
    n = len(s1)
    m = len(s2)
    dp = np.full((n + 1, m + 1), np.inf)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        j_start = max(1, i - w) if w > 0 else 1
        j_end = min(m, i + w) if w > 0 else m
        for j in range(j_start, j_end + 1):
            cost = abs(s1[i - 1] - s2[j - 1])
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return dp


def _lcss_dp_core(s1: np.ndarray, s2: np.ndarray, epsilon: float, delta: int) -> int:
    n = len(s1)
    m = len(s2)
    dp = np.zeros((n + 1, m + 1), dtype=np.int64)
    w = delta if delta > 0 else 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if (w == 0 or abs(i - j) <= w) and abs(s1[i - 1] - s2[j - 1]) <= epsilon:
                dp[i, j] = 1 + dp[i - 1, j - 1]
            else:
                dp[i, j] = max(dp[i - 1, j], dp[i, j - 1])
    return int(dp[n, m])


def _edr_dp_core(s1: np.ndarray, s2: np.ndarray, epsilon: float, window: int) -> float:
    n = len(s1)
    m = len(s2)
    dp = np.full((n + 1, m + 1), np.inf)
    dp[0, 0] = 0.0
    w = window if window > 0 else 0
    for i in range(1, n + 1):
        if w == 0 or i <= w:
            dp[i, 0] = float(i)
    for j in range(1, m + 1):
        if w == 0 or j <= w:
            dp[0, j] = float(j)
    for i in range(1, n + 1):
        j_start = max(1, i - w) if w > 0 else 1
        j_end = min(m, i + w) if w > 0 else m
        for j in range(j_start, j_end + 1):
            subcost = 0.0 if abs(s1[i - 1] - s2[j - 1]) <= epsilon else 1.0
            dp[i, j] = min(
                dp[i - 1, j - 1] + subcost,
                dp[i - 1, j] + 1.0,
                dp[i, j - 1] + 1.0,
            )
    return float(dp[n, m])


if _NUMBA_AVAILABLE:
    _dtw_dp = numba.njit(fastmath=True)(_dtw_dp_core)
    _lcss_dp = numba.njit(fastmath=True)(_lcss_dp_core)
    _edr_dp = numba.njit(fastmath=True)(_edr_dp_core)
else:
    _dtw_dp = _dtw_dp_core
    _lcss_dp = _lcss_dp_core
    _edr_dp = _edr_dp_core


def calculate_kl_divergence(
    p: np.ndarray,
    q: np.ndarray,
    epsilon: float = 1e-9,
) -> float:
    """Compute Kullback-Leibler (KL) Divergence D_KL(P || Q).
    
    Formula:
        D_KL(P || Q) = sum_x P(x) * ln( P(x) / (Q(x) + epsilon) )
        
    Where:
        P = observed local distribution (e.g. store role mix)
        Q = reference benchmark distribution (corporate ideal)
    """
    p_arr = np.asarray(p, dtype=float) + epsilon
    q_arr = np.asarray(q, dtype=float) + epsilon
    
    p_norm = p_arr / np.sum(p_arr)
    q_norm = q_arr / np.sum(q_arr)
    
    kl = np.sum(p_norm * np.log(p_norm / q_norm))
    return max(0.0, float(kl))


def calculate_jensen_shannon_divergence(
    p: np.ndarray,
    q: np.ndarray,
    epsilon: float = 1e-9,
) -> float:
    """Compute symmetric Jensen-Shannon Divergence JSD(P || Q) in [0, 1]."""
    p_arr = np.asarray(p, dtype=float) + epsilon
    q_arr = np.asarray(q, dtype=float) + epsilon
    
    p_norm = p_arr / np.sum(p_arr)
    q_norm = q_arr / np.sum(q_arr)
    m = 0.5 * (p_norm + q_norm)
    
    jsd = 0.5 * calculate_kl_divergence(p_norm, m) + 0.5 * calculate_kl_divergence(q_norm, m)
    return float(np.sqrt(max(0.0, jsd)))


def compute_dtw_distance(
    series1: Union[np.ndarray, pd.Series, List[float]],
    series2: Union[np.ndarray, pd.Series, List[float]],
    window: Optional[int] = None,
) -> Tuple[float, List[Tuple[int, int]], np.ndarray]:
    """Calculate Dynamic Time Warping (DTW) distance and optimal warping alignment path.
    
    Supports Sakoe-Chiba band constraints (via ``window``) and JIT-accelerated DP execution
    reducing computational complexity from O(N*M) to O(N*W).
    
    Parameters
    ----------
    series1 : array-like
        First temporal sequence.
    series2 : array-like
        Second temporal sequence.
    window : int, optional
        Sakoe-Chiba warping window radius. If specified, restricts the path search to
        |i - j| <= window. If None, computes unconstrained DTW.

    Returns
    -------
    total_distance : float
        Cumulative warping distance.
    warping_path : list of tuple of (int, int)
        Optimal alignment path through the cost matrix.
    cost_matrix : np.ndarray
        Full (N, M) accumulated cost matrix.
    """
    s1 = np.asarray(series1, dtype=float)
    s2 = np.asarray(series2, dtype=float)
    
    n = len(s1)
    m = len(s2)
    if n == 0 or m == 0:
        return 0.0, [], np.zeros((n, m))
        
    w = int(window) if window is not None and window > 0 else 0
    dtw_matrix = _dtw_dp(s1, s2, w)
    
    # Traceback optimal warping path
    i, j = n, m
    path = []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        diag = dtw_matrix[i - 1, j - 1]
        up = dtw_matrix[i - 1, j]
        left = dtw_matrix[i, j - 1]
        min_val = min(diag, up, left)
        if min_val == diag:
            i -= 1
            j -= 1
        elif min_val == up:
            i -= 1
        else:
            j -= 1
    while i > 0:
        path.append((i - 1, 0))
        i -= 1
    while j > 0:
        path.append((0, j - 1))
        j -= 1
        
    path.reverse()
    total_distance = float(dtw_matrix[n, m])
    
    return total_distance, path, dtw_matrix[1:, 1:]


class AutoencoderWeirdnessDetector:
    """PyTorch bottleneck autoencoder for operational anomaly scoring across multivariate metrics.
    
    Compresses high-dimensional store operational features into a lower-dimensional latent space.
    Data points that structurally decouple from peer correlations incur high Reconstruction Error.
    """
    
    def __init__(
        self,
        latent_dim: int = 4,
        hidden_dim: int = 12,
        lr: float = 0.005,
        epochs: int = 150,
        seed: int = 42,
        contamination: float = 0.05,
        threshold: Optional[float] = None,
    ):
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.seed = seed
        self.contamination = contamination
        self.threshold = threshold
        self.threshold_: Optional[float] = None
        self.train_weirdness_: Optional[np.ndarray] = None
        self.model_ = None
        self.mean_ = None
        self.std_ = None
        self.feature_names_: List[str] = []

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this estimator."""
        return {
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
            "lr": self.lr,
            "epochs": self.epochs,
            "seed": self.seed,
            "contamination": self.contamination,
            "threshold": self.threshold,
        }

    def set_params(self, **params) -> "AutoencoderWeirdnessDetector":
        """Set parameters for this estimator."""
        for key, val in params.items():
            if hasattr(self, key):
                setattr(self, key, val)
            else:
                raise ValueError(f"Invalid parameter {key} for AutoencoderWeirdnessDetector")
        return self

    def fit(self, X: Union[pd.DataFrame, np.ndarray]):
        """Fit autoencoder on healthy reference dataset and learn anomaly threshold."""
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "AutoencoderWeirdnessDetector requires 'torch'. "
                "Install with: pip install 'people-analytics-toolkit[deeplearning]'"
            )
        
        _configure_torch_reproducibility(self.seed)
        
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            x_vals = X.to_numpy(dtype=np.float32)
        else:
            self.feature_names_ = [f"feat_{i}" for i in range(X.shape[1])]
            x_vals = np.asarray(X, dtype=np.float32)
            
        # Z-score standardization
        self.mean_ = np.mean(x_vals, axis=0, keepdims=True)
        self.std_ = np.std(x_vals, axis=0, keepdims=True) + 1e-6
        x_norm = (x_vals - self.mean_) / self.std_
        
        input_dim = x_vals.shape[1]
        self.model_ = _AutoencoderModule(input_dim, self.hidden_dim, self.latent_dim)
        optimizer = optim.Adam(self.model_.parameters(), lr=self.lr, weight_decay=1e-5)
        criterion = nn.MSELoss()
        
        tensor_x = torch.from_numpy(x_norm)
        
        self.model_.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            reconstructed = self.model_(tensor_x)
            loss = criterion(reconstructed, tensor_x)
            loss.backward()
            optimizer.step()
            
        # Calculate training errors for automated thresholding
        self.model_.eval()
        with torch.no_grad():
            reconstructed_train = self.model_(tensor_x).cpu().numpy()
        sq_errors_train = (x_norm - reconstructed_train) ** 2
        self.train_weirdness_ = np.mean(sq_errors_train, axis=1)
        
        # Select threshold based on configuration
        if self.threshold is not None:
            self.threshold_ = float(self.threshold)
        else:
            self.select_threshold(method="percentile", contamination=self.contamination)
            
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> Tuple[pd.Series, pd.DataFrame]:
        """Compute row-level Weirdness Index (MSE) and feature-level error breakdown."""
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "AutoencoderWeirdnessDetector requires 'torch'. "
                "Install with: pip install 'people-analytics-toolkit[deeplearning]'"
            )
        if self.model_ is None or self.mean_ is None or self.std_ is None:
            raise ValueError("AutoencoderWeirdnessDetector is not fitted. Call fit() first.")
        
        if isinstance(X, pd.DataFrame):
            x_vals = X[self.feature_names_].to_numpy(dtype=np.float32)
            index = X.index
        else:
            x_vals = np.asarray(X, dtype=np.float32)
            index = pd.RangeIndex(len(x_vals))
            
        if x_vals.shape[1] != len(self.feature_names_):
            raise ValueError(
                f"Feature dimension mismatch: expected {len(self.feature_names_)} features, got {x_vals.shape[1]}."
            )
            
        x_norm = (x_vals - self.mean_) / self.std_
        tensor_x = torch.from_numpy(x_norm)
        
        self.model_.eval()
        with torch.no_grad():
            reconstructed = self.model_(tensor_x).cpu().numpy()
            
        # Squared error per feature
        sq_errors = (x_norm - reconstructed) ** 2
        weirdness_index = np.mean(sq_errors, axis=1)
        
        breakdown_df = pd.DataFrame(sq_errors, columns=self.feature_names_, index=index)
        series_res = pd.Series(weirdness_index, index=index, name="weirdness_index")
        
        return series_res, breakdown_df

    def fit_transform(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """Fit autoencoder on X and transform X, returning weirdness scores and feature error breakdown."""
        return self.fit(X).transform(X)

    def select_threshold(
        self,
        X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        method: str = "percentile",
        contamination: Optional[float] = None,
        percentile: Optional[float] = None,
    ) -> float:
        """Select and set anomaly threshold using statistical heuristics.
        
        Args:
            X: Optional dataset to compute error distribution from. If None, uses training errors.
            method: Thresholding heuristic:
                - 'percentile': based on percentile/contamination (e.g. (1 - contamination) * 100).
                - 'iqr' or 'tukey': Q3 + 1.5 * IQR.
                - 'gaussian' or 'std': mean + 3 * std.
            contamination: Fraction of expected outliers in data (used if method='percentile').
            percentile: Specific percentile in [0, 100] (overrides contamination if method='percentile').
            
        Returns:
            Computed threshold value (float).
        """
        if X is not None:
            scores, _ = self.transform(X)
            values = scores.to_numpy(dtype=float)
        elif self.train_weirdness_ is not None:
            values = np.asarray(self.train_weirdness_, dtype=float)
        else:
            raise ValueError("Detector must be fitted before selecting a threshold, or pass X explicitly.")
            
        method_norm = method.lower().strip()
        if method_norm in ("percentile", "quantile"):
            if percentile is not None:
                pct = percentile
            else:
                c = contamination if contamination is not None else self.contamination
                pct = (1.0 - c) * 100.0
            thresh = float(np.percentile(values, pct))
        elif method_norm in ("iqr", "tukey"):
            q25, q75 = np.percentile(values, [25, 75])
            iqr = q75 - q25
            thresh = float(q75 + 1.5 * iqr)
        elif method_norm in ("gaussian", "std", "zscore", "normal"):
            mean_val = float(np.mean(values))
            std_val = float(np.std(values))
            thresh = float(mean_val + 3.0 * std_val)
        else:
            raise ValueError(
                f"Unknown threshold selection method: '{method}'. Choose 'percentile', 'iqr', or 'gaussian'."
            )
            
        self.threshold_ = thresh
        return thresh

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.Series:
        """Predict whether each sample is an operational anomaly (weirdness >= threshold_).
        
        Returns:
            pd.Series of boolean flags (True for anomaly, False otherwise).
        """
        if self.model_ is None:
            raise ValueError("AutoencoderWeirdnessDetector is not fitted. Call fit() first.")
        if self.threshold_ is None:
            raise ValueError("Threshold is not set. Call fit() or select_threshold() first.")
            
        scores, _ = self.transform(X)
        is_anomaly = scores >= self.threshold_
        is_anomaly.name = "is_anomaly"
        return is_anomaly

    def decision_function(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.Series:
        """Compute anomaly score (weirdness index) for each sample."""
        scores, _ = self.transform(X)
        return scores

    def to_dict(self) -> Dict[str, Any]:
        """Serialize detector configuration, learned parameters, and PyTorch weights to a dict."""
        state_dict_serializable = None
        if self.model_ is not None and _TORCH_AVAILABLE:
            state_dict_serializable = {
                k: v.cpu().detach().numpy().tolist() for k, v in self.model_.state_dict().items()
            }
        
        return {
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
            "lr": self.lr,
            "epochs": self.epochs,
            "seed": self.seed,
            "contamination": self.contamination,
            "threshold": self.threshold,
            "threshold_": self.threshold_,
            "feature_names_": self.feature_names_,
            "mean_": self.mean_.tolist() if self.mean_ is not None else None,
            "std_": self.std_.tolist() if self.std_ is not None else None,
            "train_weirdness_": self.train_weirdness_.tolist() if self.train_weirdness_ is not None else None,
            "state_dict": state_dict_serializable,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AutoencoderWeirdnessDetector":
        """Reconstruct detector from dictionary."""
        detector = cls(
            latent_dim=data.get("latent_dim", 4),
            hidden_dim=data.get("hidden_dim", 12),
            lr=data.get("lr", 0.005),
            epochs=data.get("epochs", 150),
            seed=data.get("seed", 42),
            contamination=data.get("contamination", 0.05),
            threshold=data.get("threshold", None),
        )
        detector.threshold_ = data.get("threshold_")
        detector.feature_names_ = data.get("feature_names_", [])
        
        mean_list = data.get("mean_")
        if mean_list is not None:
            detector.mean_ = np.array(mean_list, dtype=np.float32)
            
        std_list = data.get("std_")
        if std_list is not None:
            detector.std_ = np.array(std_list, dtype=np.float32)
            
        tw_list = data.get("train_weirdness_")
        if tw_list is not None:
            detector.train_weirdness_ = np.array(tw_list, dtype=np.float32)
            
        state_dict_raw = data.get("state_dict")
        if state_dict_raw is not None:
            if not _TORCH_AVAILABLE:
                raise ImportError(
                    "AutoencoderWeirdnessDetector requires 'torch' to restore model weights. "
                    "Install with: pip install 'people-analytics-toolkit[deeplearning]'"
                )
            in_d = len(detector.feature_names_) if detector.feature_names_ else len(mean_list[0])
            detector.model_ = _AutoencoderModule(in_d, detector.hidden_dim, detector.latent_dim)
            loaded_state = {
                k: torch.tensor(np.array(v), dtype=torch.float32)
                for k, v in state_dict_raw.items()
            }
            detector.model_.load_state_dict(loaded_state)
            detector.model_.eval()
            
        return detector

    def to_json(self, filepath: Optional[Union[str, Path]] = None, indent: int = 2) -> str:
        """Export detector configuration and weights as JSON string or file."""
        data = self.to_dict()
        json_str = json.dumps(data, indent=indent)
        if filepath is not None:
            p = Path(filepath)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                f.write(json_str)
        return json_str

    @classmethod
    def from_json(cls, json_str_or_filepath: Union[str, Path]) -> "AutoencoderWeirdnessDetector":
        """Load detector from JSON string or file path."""
        p = Path(json_str_or_filepath)
        if p.exists() and p.is_file():
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = json.loads(str(json_str_or_filepath))
        return cls.from_dict(data)

    def save(self, filepath: Union[str, Path]) -> None:
        """Serialize detector to a file (JSON or binary pickle depending on extension)."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        if filepath.suffix.lower() == ".json":
            self.to_json(filepath)
        else:
            with open(filepath, "wb") as f:
                pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "AutoencoderWeirdnessDetector":
        """Deserialize detector from a file (JSON or binary pickle)."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        if filepath.suffix.lower() == ".json":
            return cls.from_json(filepath)
        
        # Try unpickling first, fallback to JSON if pickle fails
        try:
            with open(filepath, "rb") as f:
                obj = pickle.load(f)
            if isinstance(obj, cls):
                return obj
        except Exception:
            pass
        return cls.from_json(filepath)


class LocalOutlierFactorDetector:
    """Local Outlier Factor (LOF) estimator for peer density anomaly detection.

    Measures the local density deviation of a data point relative to its k-nearest neighbors.
    Points in sparse local neighborhoods receive high LOF anomaly scores.

    Parameters
    ----------
    n_neighbors : int, default=20
        Number of neighbors used in local density estimation.
    contamination : float, default=0.05
        Expected proportion of outliers in the data set.
    feature_cols : list of str, optional
        Subset of numerical features to fit on. If None, uses all numeric columns when given a DataFrame.
    """

    def __init__(
        self,
        n_neighbors: int = 20,
        contamination: float = 0.05,
        feature_cols: Optional[List[str]] = None,
    ):
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.feature_cols = feature_cols
        self.scaler_ = None
        self.model_ = None
        self.threshold_ = None
        self.decision_scores_ = None
        self.labels_ = None
        self._is_fitted = False
        self.is_fitted_ = False

    @property
    def is_fitted(self) -> bool:
        """Whether the estimator has been fitted."""
        return self._is_fitted

    def _extract_array(self, X: Union[pd.DataFrame, np.ndarray]) -> Tuple[np.ndarray, Optional[pd.Index]]:
        if isinstance(X, pd.DataFrame):
            cols = self.feature_cols if self.feature_cols is not None else X.select_dtypes(include=[np.number]).columns.tolist()
            missing = [c for c in cols if c not in X.columns]
            if missing:
                raise KeyError(f"Feature columns not found in DataFrame: {missing}")
            return X[cols].to_numpy(dtype=float), X.index
        elif isinstance(X, pd.Series):
            return X.to_numpy(dtype=float).reshape(-1, 1), X.index
        else:
            arr = np.asarray(X, dtype=float)
            return arr.reshape(-1, 1) if arr.ndim == 1 else arr, None

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Any = None):
        """Fit the LOF detector on training data X."""
        try:
            from pyod.models.lof import LOF
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            raise ImportError(
                "LocalOutlierFactorDetector requires 'pyod'. "
                "Install with: pip install 'people-analytics-toolkit[anomalies]'"
            )

        X_mat, idx = self._extract_array(X)
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_mat)

        self.model_ = LOF(n_neighbors=self.n_neighbors, contamination=self.contamination)
        self.model_.fit(X_scaled)

        self.threshold_ = float(self.model_.threshold_)
        self.decision_scores_ = pd.Series(self.model_.decision_scores_, index=idx, name="lof_anomaly_score")
        self.labels_ = pd.Series(self.model_.labels_, index=idx, name="is_lof_outlier")
        self._is_fitted = True
        self.is_fitted_ = True
        return self

    def decision_function(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.Series:
        """Compute continuous anomaly scores for observations in X (higher = more anomalous)."""
        if not self._is_fitted or self.model_ is None or self.scaler_ is None:
            raise ValueError("LocalOutlierFactorDetector is not fitted. Call fit() first.")
        X_mat, idx = self._extract_array(X)
        X_scaled = self.scaler_.transform(X_mat)
        scores = self.model_.decision_function(X_scaled)
        return pd.Series(scores, index=idx, name="lof_anomaly_score")

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.Series:
        """Predict whether observations in X are outliers (1=outlier, 0=inlier)."""
        if not self._is_fitted or self.model_ is None or self.scaler_ is None:
            raise ValueError("LocalOutlierFactorDetector is not fitted. Call fit() first.")
        X_mat, idx = self._extract_array(X)
        X_scaled = self.scaler_.transform(X_mat)
        preds = self.model_.predict(X_scaled)
        return pd.Series(preds, index=idx, name="is_lof_outlier")

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """Transform X into DataFrame containing anomaly score and outlier indicator."""
        scores = self.decision_function(X)
        preds = self.predict(X)
        return pd.DataFrame({"lof_anomaly_score": scores, "is_lof_outlier": preds}, index=scores.index)

    def fit_predict(self, X: Union[pd.DataFrame, np.ndarray], y: Any = None) -> Tuple[pd.Series, pd.Series, dict]:
        """Fit detector and return in-sample (scores, is_outlier, metadata)."""
        self.fit(X)
        meta = {
            "threshold": self.threshold_,
            "n_neighbors": self.n_neighbors,
            "contamination": self.contamination,
            "model": self.model_,
            "detector": self,
        }
        return self.decision_scores_, self.labels_, meta

    def fit_transform(self, X: Union[pd.DataFrame, np.ndarray], y: Any = None) -> Tuple[pd.Series, pd.Series, dict]:
        """Alias for fit_predict adhering to transformer convention."""
        return self.fit_predict(X, y=y)


PyODLocalOutlierFactor = LocalOutlierFactorDetector


def fit_pyod_local_outlier_factor(
    df: pd.DataFrame,
    feature_cols: List[str],
    n_neighbors: int = 20,
    contamination: float = 0.05,
) -> Tuple[pd.Series, pd.Series, dict]:
    """Fit Local Outlier Factor (LOF) using PyOD for density-based peer anomaly detection.

    Measures the local density deviation of a data point relative to its k-nearest neighbors.
    Points in sparse local neighborhoods receive high LOF anomaly scores.

    Returns:
        (lof_scores_series, is_outlier_series, model_metadata)
    """
    detector = LocalOutlierFactorDetector(
        n_neighbors=n_neighbors,
        contamination=contamination,
        feature_cols=feature_cols,
    )
    return detector.fit_predict(df)


def compute_lcss_distance(
    series1: Union[np.ndarray, pd.Series, List[float]],
    series2: Union[np.ndarray, pd.Series, List[float]],
    epsilon: float = 2.5,
    delta: Optional[int] = 4,
) -> Tuple[int, float]:
    """Compute Longest Common Subsequence (LCSS) on Real Sequences with tolerance epsilon and window delta.
    
    Tolerates missing gaps (such as vacation leave or leave of absence) without severe DTW distortion.
    Accelerated with Sakoe-Chiba time matching window constraints and JIT compilation.
    
    Parameters
    ----------
    series1 : array-like
        First sequence.
    series2 : array-like
        Second sequence.
    epsilon : float, default=2.5
        Spatial value matching tolerance threshold.
    delta : int, optional, default=4
        Temporal matching window constraint |i - j| <= delta. If None or 0, unconstrained.

    Returns
    -------
    lcss_length : int
        Number of matched elements.
    normalized_distance : float
        Normalized distance in [0, 1] defined as 1.0 - (lcss_length / min(len1, len2)).
    """
    s1 = np.asarray(series1, dtype=float)
    s2 = np.asarray(series2, dtype=float)
    n, m = len(s1), len(s2)
    if n == 0 or m == 0:
        return 0, 1.0
        
    d = int(delta) if delta is not None and delta > 0 else 0
    lcss_len = int(_lcss_dp(s1, s2, float(epsilon), d))
    norm_dist = 1.0 - (lcss_len / max(1, min(n, m)))
    return lcss_len, float(norm_dist)


def compute_edr_distance(
    series1: Union[np.ndarray, pd.Series, List[float]],
    series2: Union[np.ndarray, pd.Series, List[float]],
    epsilon: float = 2.5,
    window: Optional[int] = None,
) -> float:
    """Compute Edit Distance on Real Sequences (EDR) with spatial matching threshold epsilon.
    
    Cost function: subcost is 0 if |s1[i] - s2[j]| <= epsilon, else 1. Insertion/deletion cost is 1.
    Normalized by max(len1, len2). Supports Sakoe-Chiba window constraints and JIT acceleration.

    Parameters
    ----------
    series1 : array-like
        First sequence.
    series2 : array-like
        Second sequence.
    epsilon : float, default=2.5
        Spatial value matching threshold.
    window : int, optional
        Temporal Sakoe-Chiba band width |i - j| <= window. If None, unconstrained.

    Returns
    -------
    float
        Normalized edit distance in [0, 1].
    """
    s1 = np.asarray(series1, dtype=float)
    s2 = np.asarray(series2, dtype=float)
    n, m = len(s1), len(s2)
    if n == 0 or m == 0:
        return 1.0 if max(n, m) > 0 else 0.0
        
    w = int(window) if window is not None and window > 0 else 0
    cost = float(_edr_dp(s1, s2, float(epsilon), w))
    return float(cost / max(n, m))


def calculate_mahalanobis_distance(
    data: Union[pd.DataFrame, pd.Series, np.ndarray, Sequence],
    v: Optional[Union[np.ndarray, Sequence[float], pd.Series]] = None,
    feature_cols: Optional[List[str]] = None,
    reference_data: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    mean: Optional[Union[np.ndarray, Sequence[float], pd.Series]] = None,
    cov: Optional[Union[np.ndarray, Sequence[Sequence[float]], pd.DataFrame]] = None,
    inv_cov: Optional[Union[np.ndarray, Sequence[Sequence[float]], pd.DataFrame]] = None,
    robust: bool = False,
    regularization: float = 1e-6,
    significance_level: float = 0.05,
    group_col: Optional[Union[str, pd.Series, Sequence]] = None,
) -> Tuple[pd.Series, pd.Series, Dict[str, Any]]:
    r"""Compute Mahalanobis distance, chi-squared anomaly p-values, and outlier flags.

    Measures multivariate statistical distance between data points and a reference
    centroid, explicitly accounting for the covariance structure across features.
    Under the assumption of multivariate normality, squared Mahalanobis distance
    follows a Chi-Squared distribution with degrees of freedom equal to the dimensionality
    :math:`p`:

    .. math::
        D_M^2(x, \mu, \Sigma) = (x - \mu)^T \Sigma^{-1} (x - \mu) \sim \chi^2_p

    An observation is classified as a multivariate anomaly when:

    .. math::
        D_M^2(x) > \chi^2_{p, 1 - \alpha} \iff p\text{-value} = 1 - F_{\chi^2_p}(D_M^2) < \alpha

    Parameters
    ----------
    data : pd.DataFrame, pd.Series, np.ndarray, or sequence
        Input data points to evaluate. Can be a DataFrame of observations, a 2D array,
        or a single 1D vector.
    v : np.ndarray, sequence, or pd.Series, optional
        Secondary reference vector when evaluating pairwise distance between two points :math:`(u, v)`.
        If provided, `data` is treated as vector :math:`u` and `v` is treated as the centroid :math:`\mu`.
    feature_cols : list of str, optional
        Subset of numerical column names to evaluate when `data` is a DataFrame.
        Defaults to all numeric columns.
    reference_data : pd.DataFrame or np.ndarray, optional
        Baseline or uncorrupted reference dataset used to estimate the centroid :math:`\mu`
        and covariance matrix :math:`\Sigma`. If None, parameters are estimated from `data`.
    mean : np.ndarray, sequence, or pd.Series, optional
        Precomputed centroid vector :math:`\mu`.
    cov : np.ndarray, sequence of sequences, or pd.DataFrame, optional
        Precomputed covariance matrix :math:`\Sigma`.
    inv_cov : np.ndarray, sequence of sequences, or pd.DataFrame, optional
        Precomputed precision / inverse covariance matrix :math:`\Sigma^{-1}`.
    robust : bool, default=False
        If True, estimates location and dispersion using Minimum Covariance Determinant (MCD)
        to prevent masking effects from extreme multivariate outliers.
    regularization : float, default=1e-6
        Ridge shrinkage parameter added to covariance diagonal (:math:`\Sigma + \lambda I`)
        to ensure well-conditioned invertibility when variables are collinear or :math:`N < p`.
    significance_level : float, default=0.05
        Statistical threshold :math:`\alpha` for the :math:`\chi^2_p` hypothesis test.
    group_col : str, pd.Series, or sequence, optional
        Grouping identifier (e.g. 'store_id' or 'department') to evaluate peer-relative
        multivariate distance within each cohort independently.

    Returns
    -------
    Tuple[pd.Series, pd.Series, Dict[str, Any]]
        - distances : pd.Series of Mahalanobis distances :math:`D_M`
        - is_outlier : pd.Series of boolean outlier flags (:math:`D_M^2 > \chi^2_{p, 1 - \alpha}`)
        - metadata : dictionary containing:
            - 'd_squared': squared distances :math:`D_M^2`
            - 'p_values': chi-squared p-values
            - 'degrees_of_freedom': dimensionality :math:`p`
            - 'critical_value': :math:`\chi^2` threshold at :math:`(1 - \alpha)`
            - 'threshold_distance': :math:`\sqrt{\chi^2_{p, 1 - \alpha}}`
            - 'significance_level': :math:`\alpha`
            - 'mean': centroid :math:`\mu`
            - 'covariance': regularized covariance :math:`\Sigma`
            - 'inv_covariance': precision matrix :math:`\Sigma^{-1}`
            - 'n_outliers': total flagged outliers
            - 'outlier_fraction': proportion of outliers
            - 'method': estimation method ('robust_mcd' or 'empirical')
            - (optional) 'group_metrics': dict of per-group metadata if `group_col` specified.
    """
    if not (0.0 < significance_level < 1.0):
        raise ValueError(f"significance_level must be in (0, 1), got {significance_level}")
    if regularization < 0.0:
        raise ValueError(f"regularization must be non-negative, got {regularization}")

    # 1. Grouped Evaluation Path
    if group_col is not None:
        if isinstance(data, pd.DataFrame):
            if isinstance(group_col, str):
                if group_col not in data.columns:
                    raise KeyError(f"group_col '{group_col}' not found in DataFrame")
                g_series = data[group_col]
            else:
                g_series = pd.Series(group_col, index=data.index)
            data_idx = data.index
        else:
            g_series = pd.Series(group_col)
            data_idx = pd.RangeIndex(len(data))

        dist_res = pd.Series(index=data_idx, dtype=float, name="mahalanobis_distance")
        outlier_res = pd.Series(index=data_idx, dtype=bool, name="is_mahalanobis_outlier")
        d_sq_res = pd.Series(index=data_idx, dtype=float, name="mahalanobis_d_squared")
        p_val_res = pd.Series(index=data_idx, dtype=float, name="mahalanobis_p_value")
        group_meta = {}

        for g_val, idxs in g_series.groupby(g_series).groups.items():
            sub_data = data.loc[idxs] if isinstance(data, pd.DataFrame) else (
                data.iloc[idxs] if isinstance(data, pd.Series) else np.asarray(data)[idxs]
            )
            sub_d, sub_out, sub_m = calculate_mahalanobis_distance(
                data=sub_data,
                feature_cols=feature_cols,
                robust=robust,
                regularization=regularization,
                significance_level=significance_level,
                group_col=None,
            )
            dist_res.loc[idxs] = sub_d
            outlier_res.loc[idxs] = sub_out
            d_sq_res.loc[idxs] = sub_m["d_squared"]
            p_val_res.loc[idxs] = sub_m["p_values"]
            group_meta[g_val] = sub_m

        first_meta = next(iter(group_meta.values())) if group_meta else {}
        p_dim = first_meta.get("degrees_of_freedom", 0)
        crit_val = float(chi2.ppf(1.0 - significance_level, df=p_dim)) if p_dim > 0 else np.nan

        meta = {
            "d_squared": d_sq_res,
            "p_values": p_val_res,
            "degrees_of_freedom": p_dim,
            "critical_value": crit_val,
            "threshold_distance": float(np.sqrt(crit_val)) if not np.isnan(crit_val) else np.nan,
            "significance_level": significance_level,
            "n_outliers": int(outlier_res.sum()),
            "outlier_fraction": float(outlier_res.mean()) if len(outlier_res) > 0 else 0.0,
            "method": f"grouped_{'robust_mcd' if robust else 'empirical'}",
            "group_metrics": group_meta,
        }
        return dist_res, outlier_res, meta

    # 2. Pairwise Vector Distance Path (u, v)
    if v is not None:
        u_arr = np.asarray(data, dtype=float).ravel()
        v_arr = np.asarray(v, dtype=float).ravel()
        if len(u_arr) != len(v_arr):
            raise ValueError(f"Vector dimensions do not match: {len(u_arr)} vs {len(v_arr)}")
        p = len(u_arr)

        if inv_cov is not None:
            inv_sigma = np.asarray(inv_cov, dtype=float)
            sigma = cov if cov is not None else np.nan
        elif cov is not None:
            sigma = np.asarray(cov, dtype=float)
            reg_sigma = sigma + np.eye(p) * regularization
            try:
                inv_sigma = np.linalg.inv(reg_sigma)
            except np.linalg.LinAlgError:
                inv_sigma = np.linalg.pinv(reg_sigma)
        elif reference_data is not None:
            ref_arr = np.asarray(reference_data, dtype=float)
            sigma = np.cov(ref_arr, rowvar=False)
            reg_sigma = sigma + np.eye(p) * regularization
            try:
                inv_sigma = np.linalg.inv(reg_sigma)
            except np.linalg.LinAlgError:
                inv_sigma = np.linalg.pinv(reg_sigma)
        else:
            raise ValueError(
                "When computing pairwise Mahalanobis distance between two vectors, "
                "'cov', 'inv_cov', or 'reference_data' must be provided."
            )

        diff = u_arr - v_arr
        d_sq = float((diff @ inv_sigma) @ diff)
        d_sq = max(0.0, d_sq)
        d_mahal = float(np.sqrt(d_sq))
        crit_val = float(chi2.ppf(1.0 - significance_level, df=p))
        p_val = float(1.0 - chi2.cdf(d_sq, df=p))
        is_out = bool(d_sq > crit_val)

        dist_series = pd.Series([d_mahal], name="mahalanobis_distance")
        outlier_series = pd.Series([is_out], name="is_mahalanobis_outlier")
        meta = {
            "d_squared": pd.Series([d_sq], name="mahalanobis_d_squared"),
            "p_values": pd.Series([p_val], name="mahalanobis_p_value"),
            "degrees_of_freedom": p,
            "critical_value": crit_val,
            "threshold_distance": float(np.sqrt(crit_val)),
            "significance_level": significance_level,
            "mean": v_arr,
            "covariance": sigma,
            "inv_covariance": inv_sigma,
            "n_outliers": int(is_out),
            "outlier_fraction": float(is_out),
            "method": "pairwise",
        }
        return dist_series, outlier_series, meta

    # 3. Tabular Dataset Evaluation Path
    if isinstance(data, (list, tuple)) and len(data) == 0:
        raise ValueError("data cannot be empty")
    if hasattr(data, "__len__") and len(data) == 0:
        raise ValueError("data cannot be empty")

    if isinstance(data, pd.DataFrame):
        data_index = data.index
        if feature_cols is not None:
            missing = [c for c in feature_cols if c not in data.columns]
            if missing:
                raise KeyError(f"Feature columns not found in DataFrame: {missing}")
            cols = feature_cols
        else:
            cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if not cols:
                raise ValueError("No numerical columns found in DataFrame")
        nan_cols = [c for c in cols if data[c].isna().any()]
        if nan_cols:
            raise ValueError(f"Input DataFrame contains NaNs in columns: {nan_cols}")
        X = data[cols].to_numpy(dtype=float)
        col_names = cols
    elif isinstance(data, pd.Series):
        data_index = data.index
        if data.isna().any():
            raise ValueError("Input Series contains NaNs")
        X = data.to_numpy(dtype=float).reshape(-1, 1)
        col_names = [data.name or "feature_0"]
    else:
        arr = np.asarray(data, dtype=float)
        if np.isnan(arr).any():
            raise ValueError("Input array contains NaNs")
        if arr.ndim == 1:
            if mean is not None and len(mean) == len(arr) and len(arr) > 1:
                X = arr.reshape(1, -1)
            else:
                X = arr.reshape(-1, 1)
        else:
            X = arr
        data_index = pd.RangeIndex(len(X))
        col_names = [f"feature_{i}" for i in range(X.shape[1])]

    N, p = X.shape

    # Reference data extraction
    if reference_data is not None:
        if isinstance(reference_data, pd.DataFrame):
            if feature_cols is not None:
                X_ref = reference_data[feature_cols].to_numpy(dtype=float)
            else:
                ref_cols = [c for c in col_names if c in reference_data.columns]
                if len(ref_cols) != len(col_names):
                    ref_cols = reference_data.select_dtypes(include=[np.number]).columns.tolist()
                X_ref = reference_data[ref_cols].to_numpy(dtype=float)
        else:
            ref_arr = np.asarray(reference_data, dtype=float)
            X_ref = ref_arr.reshape(-1, 1) if ref_arr.ndim == 1 else ref_arr
    else:
        X_ref = X

    # Centroid (mean) & Covariance estimation
    est_method = "empirical"
    if inv_cov is not None:
        inv_sigma = np.asarray(inv_cov, dtype=float)
        sigma = np.asarray(cov, dtype=float) if cov is not None else np.nan
        mu = np.asarray(mean, dtype=float) if mean is not None else np.mean(X_ref, axis=0)
    elif cov is not None:
        sigma = np.asarray(cov, dtype=float)
        mu = np.asarray(mean, dtype=float) if mean is not None else np.mean(X_ref, axis=0)
        reg_sigma = sigma + np.eye(p) * regularization
        try:
            inv_sigma = np.linalg.inv(reg_sigma)
        except np.linalg.LinAlgError:
            inv_sigma = np.linalg.pinv(reg_sigma)
    else:
        if robust and len(X_ref) > p + 1:
            try:
                from sklearn.covariance import MinCovDet
                mcd = MinCovDet(random_state=42)
                mcd.fit(X_ref)
                mu = mcd.location_ if mean is None else np.asarray(mean, dtype=float)
                sigma = mcd.covariance_
                est_method = "robust_mcd"
            except Exception:
                mu = np.median(X_ref, axis=0) if mean is None else np.asarray(mean, dtype=float)
                sigma = np.cov(X_ref, rowvar=False)
                est_method = "empirical"
        else:
            mu = np.mean(X_ref, axis=0) if mean is None else np.asarray(mean, dtype=float)
            sigma = np.cov(X_ref, rowvar=False)
            est_method = "empirical"

        if sigma.ndim == 0:
            sigma = np.array([[float(sigma)]])
        elif sigma.ndim == 1 and p == 1:
            sigma = sigma.reshape(1, 1)

        reg_sigma = sigma + np.eye(p) * regularization
        try:
            inv_sigma = np.linalg.inv(reg_sigma)
        except np.linalg.LinAlgError:
            inv_sigma = np.linalg.pinv(reg_sigma)

    diff = X - mu
    d_sq = np.sum((diff @ inv_sigma) * diff, axis=1)
    d_sq = np.maximum(0.0, d_sq)
    d_mahal = np.sqrt(d_sq)

    crit_val = float(chi2.ppf(1.0 - significance_level, df=p))
    p_vals = 1.0 - chi2.cdf(d_sq, df=p)
    is_outlier = d_sq > crit_val

    dist_series = pd.Series(d_mahal, index=data_index, name="mahalanobis_distance")
    outlier_series = pd.Series(is_outlier, index=data_index, name="is_mahalanobis_outlier")
    d_sq_series = pd.Series(d_sq, index=data_index, name="mahalanobis_d_squared")
    p_val_series = pd.Series(p_vals, index=data_index, name="mahalanobis_p_value")

    meta = {
        "d_squared": d_sq_series,
        "p_values": p_val_series,
        "degrees_of_freedom": p,
        "critical_value": crit_val,
        "threshold_distance": float(np.sqrt(crit_val)),
        "significance_level": significance_level,
        "mean": pd.Series(mu, index=col_names, name="centroid") if col_names else mu,
        "covariance": pd.DataFrame(sigma, index=col_names, columns=col_names) if (col_names and hasattr(sigma, "shape") and sigma.ndim == 2) else sigma,
        "inv_covariance": pd.DataFrame(inv_sigma, index=col_names, columns=col_names) if (col_names and hasattr(inv_sigma, "shape") and inv_sigma.ndim == 2) else inv_sigma,
        "n_outliers": int(np.sum(is_outlier)),
        "outlier_fraction": float(np.mean(is_outlier)),
        "method": est_method,
    }

    return dist_series, outlier_series, meta


# API Consistency Aliases: compute_* <=> calculate_*
compute_kl_divergence = calculate_kl_divergence
compute_jensen_shannon_divergence = calculate_jensen_shannon_divergence
compute_mahalanobis_distance = calculate_mahalanobis_distance
calculate_dtw_distance = deprecated_alias(
    compute_dtw_distance,
    "calculate_dtw_distance",
    "compute_dtw_distance",
)
calculate_lcss_distance = deprecated_alias(
    compute_lcss_distance,
    "calculate_lcss_distance",
    "compute_lcss_distance",
)
calculate_edr_distance = deprecated_alias(
    compute_edr_distance,
    "calculate_edr_distance",
    "compute_edr_distance",
)

__all__ = [
    "calculate_kl_divergence",
    "compute_kl_divergence",
    "calculate_jensen_shannon_divergence",
    "compute_jensen_shannon_divergence",
    "compute_dtw_distance",
    "calculate_dtw_distance",
    "compute_lcss_distance",
    "calculate_lcss_distance",
    "compute_edr_distance",
    "calculate_edr_distance",
    "AutoencoderWeirdnessDetector",
    "LocalOutlierFactorDetector",
    "PyODLocalOutlierFactor",
    "fit_pyod_local_outlier_factor",
    "calculate_mahalanobis_distance",
    "compute_mahalanobis_distance",
]
