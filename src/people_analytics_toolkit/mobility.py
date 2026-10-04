"""Career Mobility and Movement Graph Embeddings.

Contains:
1. CareerMovementEmbeddings: Projects job roles and associate career trajectories
   into a shared low-dimensional embedding space using directional transition graphs,
   multi-step path diffusion, and spectral matrix decomposition.
2. Succession Planning & Talent Discovery: Uncovers non-obvious candidate talent pools
   for leadership and specialized roles via path reachability and cosine similarity.
"""

import json
import pickle
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds

from people_analytics_toolkit._deprecation import deprecated_alias

__all__ = [
    "CareerMovementEmbeddings",
    "compute_role_similarity_matrix",
    "get_role_similarity_matrix",
    "calculate_role_similarity_matrix",
    "calculate_time_in_position",
    "compute_time_in_position",
    "calculate_stagnation_index",
    "compute_stagnation_index",
]


class CareerMovementEmbeddings:
    """Generates low-dimensional dense embeddings for job roles and associate trajectories.
    
    Transforms discrete career sequences into continuous representations via
    transition graph random walk diffusion and spectral decomposition.
    """
    
    def __init__(
        self,
        embedding_dim: int = 8,
        max_steps: int = 3,
        decay_factor: float = 0.6,
        decay: Optional[float] = None,
        seed: int = 42,
    ):
        self.embedding_dim = embedding_dim
        self.max_steps = max_steps
        self.decay_factor = decay if decay is not None else decay_factor
        self.seed = seed
        self.role_to_idx_: Dict[str, int] = {}
        self.idx_to_role_: Dict[int, str] = {}
        self.role_embeddings_: Optional[np.ndarray] = None
        self.transition_matrix_: Optional[np.ndarray] = None
        self.diffusion_matrix_: Optional[np.ndarray] = None
        
    def fit(self, career_sequences: List[List[str]]):
        """Fit role embeddings from a collection of associate career sequences.
        
        Args:
            career_sequences: List of career histories, e.g.
                              [['Cashier', 'Stocker', 'Dept_Lead'], ['Stocker', 'Specialist', ...]]
        """
        # Build vocabulary of unique roles
        unique_roles = sorted(list(set(role for seq in career_sequences for role in seq)))
        self.role_to_idx_ = {role: idx for idx, role in enumerate(unique_roles)}
        self.idx_to_role_ = {idx: role for idx, role in enumerate(unique_roles)}
        V = len(unique_roles)
        
        if V < 3:
            raise ValueError("Need at least 3 distinct roles to build embedding space.")
            
        # 1. Directional transition counts: role_i -> role_j
        trans_counts = np.zeros((V, V), dtype=float)
        for seq in career_sequences:
            for i in range(len(seq) - 1):
                u = self.role_to_idx_[seq[i]]
                v = self.role_to_idx_[seq[i + 1]]
                trans_counts[u, v] += 1.0
                
        # Row-normalize to transition probability matrix P
        row_sums = trans_counts.sum(axis=1, keepdims=True)
        # For absorbing / terminal roles, keep self-loop
        row_sums[row_sums == 0] = 1.0
        P = trans_counts / row_sums
        self.transition_matrix_ = P.copy()
        
        # 2. Multi-step transition diffusion: M = sum_{k=1}^K gamma^(k-1) * P^k
        M = np.zeros((V, V), dtype=float)
        current_P = P.copy()
        weight = 1.0
        
        for k in range(1, self.max_steps + 1):
            M += weight * current_P
            weight *= self.decay_factor
            current_P = current_P @ P
            
        self.diffusion_matrix_ = M.copy()
        
        # 3. Spectral decomposition on affinity kernel S = (M + M^T)/2 + I*0.1
        affinity = 0.5 * (M + M.T) + np.eye(V) * 0.1
        actual_dim = min(self.embedding_dim, V - 2)
        actual_dim = max(2, actual_dim)
        
        try:
            U, S, _ = svds(affinity, k=actual_dim)
            raw_embeddings = U * np.sqrt(np.maximum(1e-6, S))
        except Exception:
            U, S, _ = np.linalg.svd(affinity)
            raw_embeddings = U[:, :actual_dim] * np.sqrt(np.maximum(1e-6, S[:actual_dim]))
            
        # L2 normalize
        norms = np.linalg.norm(raw_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.role_embeddings_ = raw_embeddings / norms
        self.embedding_dim = actual_dim
        
        return self
        
    def get_fallback_embedding(self, method: str = "mean") -> np.ndarray:
        """Compute a fallback embedding vector for unseen roles."""
        if self.role_embeddings_ is None or len(self.role_embeddings_) == 0:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        if method == "zero":
            return np.zeros(self.embedding_dim, dtype=float)
        elif method == "mean":
            bar_e = np.mean(self.role_embeddings_, axis=0)
            norm = float(np.linalg.norm(bar_e))
            return bar_e / norm if norm > 0 else bar_e
        else:
            raise ValueError(f"Unknown fallback method '{method}'. Choose 'mean' or 'zero'.")

    def get_role_embedding(self, role: str, fallback: Optional[str] = "mean") -> np.ndarray:
        """Retrieve unit-norm embedding vector for a specific job title.
        
        Parameters:
            role: Job role title.
            fallback: Strategy when role is unseen ('mean', 'zero', or None to raise KeyError).
        """
        if self.role_embeddings_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        if role in self.role_to_idx_:
            return self.role_embeddings_[self.role_to_idx_[role]]
        if fallback is None:
            raise KeyError(f"Role '{role}' not found in vocabulary.")
        return self.get_fallback_embedding(method=fallback)
        
    def get_associate_embedding(
        self,
        career_history: List[str],
        decay: float = 0.7,
        fallback: Optional[str] = "mean",
    ) -> np.ndarray:
        """Compute an associate's vector embedding from their sequence of past roles with recency decay."""
        if self.role_embeddings_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        vectors = []
        weights = []
        T = len(career_history)
        for t, role in enumerate(career_history):
            if role in self.role_to_idx_:
                vectors.append(self.get_role_embedding(role))
                weights.append(decay ** (T - 1 - t))
            elif fallback is not None:
                vectors.append(self.get_role_embedding(role, fallback=fallback))
                weights.append(decay ** (T - 1 - t))
                
        if not vectors:
            if fallback is not None:
                return self.get_fallback_embedding(method=fallback)
            return np.zeros(self.embedding_dim, dtype=float)
            
        weights = np.array(weights)[:, None]
        combined = np.sum(np.array(vectors) * weights, axis=0)
        norm = float(np.linalg.norm(combined))
        return combined / norm if norm > 0 else combined

    def transform_new_roles(
        self,
        new_roles_or_sequences: Union[List[str], List[List[str]], Dict[str, List[str]]],
        method: str = "context",
        fallback: str = "mean",
        update_vocab: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Project new unseen roles into the embedding space using local context or fallback.
        
        Parameters
        ----------
        new_roles_or_sequences : Union[List[str], List[List[str]], Dict[str, List[str]]]
            Either:
            - A list of new role names (e.g. ['NewRoleA', 'NewRoleB'])
            - A list of career sequences containing new roles (e.g. [['Cashier', 'NewRoleA', 'Dept_Supervisor']])
            - A dictionary mapping new roles to their transition neighbor roles (e.g. {'NewRoleA': ['Cashier', 'Dept_Supervisor']})
        method : str, default="context"
            Strategy for embedding new roles:
            - 'context': Averages embeddings of adjacent known neighbor roles in transitions.
            - 'fallback': Uses global fallback embedding (centroid or zero).
        fallback : str, default="mean"
            Fallback strategy if a new role has no known transition neighbors ('mean' or 'zero').
        update_vocab : bool, default=False
            If True, adds the newly projected roles into the model's vocabulary and embedding matrix.
            
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary mapping each new role to its projected unit-norm embedding vector.
        """
        if self.role_embeddings_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
            
        neighbors: Dict[str, List[str]] = {}
        if isinstance(new_roles_or_sequences, dict):
            for r, nbs in new_roles_or_sequences.items():
                if r not in self.role_to_idx_:
                    neighbors[r] = list(nbs)
        elif (
            isinstance(new_roles_or_sequences, list)
            and len(new_roles_or_sequences) > 0
            and isinstance(new_roles_or_sequences[0], list)
        ):
            for seq in new_roles_or_sequences:
                for i, r in enumerate(seq):
                    if r not in self.role_to_idx_:
                        if r not in neighbors:
                            neighbors[r] = []
                        if i > 0 and seq[i - 1] in self.role_to_idx_:
                            neighbors[r].append(seq[i - 1])
                        if i < len(seq) - 1 and seq[i + 1] in self.role_to_idx_:
                            neighbors[r].append(seq[i + 1])
        elif isinstance(new_roles_or_sequences, list):
            for r in new_roles_or_sequences:
                if isinstance(r, str) and r not in self.role_to_idx_:
                    neighbors[r] = []
        else:
            raise TypeError("new_roles_or_sequences must be a list of roles, list of sequences, or dictionary.")
            
        new_embeddings: Dict[str, np.ndarray] = {}
        for r, nbs in neighbors.items():
            valid_nbs = [self.get_role_embedding(nb, fallback=None) for nb in nbs if nb in self.role_to_idx_]
            if valid_nbs and method == "context":
                avg = np.mean(valid_nbs, axis=0)
                norm = float(np.linalg.norm(avg))
                emb = avg / norm if norm > 0 else self.get_fallback_embedding(method=fallback)
            else:
                emb = self.get_fallback_embedding(method=fallback)
                
            new_embeddings[r] = emb
            
            if update_vocab:
                new_idx = len(self.role_to_idx_)
                self.role_to_idx_[r] = new_idx
                self.idx_to_role_[new_idx] = r
                self.role_embeddings_ = np.vstack([self.role_embeddings_, emb])
                
        return new_embeddings

    def to_dict(self) -> dict:
        """Serialize model configuration and learned embeddings to a dictionary."""
        return {
            "params": {
                "embedding_dim": int(self.embedding_dim),
                "max_steps": int(self.max_steps),
                "decay_factor": float(self.decay_factor),
                "seed": int(self.seed),
            },
            "fitted": {
                "is_fitted": bool(self.role_embeddings_ is not None),
                "role_to_idx": self.role_to_idx_,
                "idx_to_role": {int(k): v for k, v in self.idx_to_role_.items()},
                "role_embeddings": self.role_embeddings_.tolist() if self.role_embeddings_ is not None else None,
                "transition_matrix": self.transition_matrix_.tolist() if self.transition_matrix_ is not None else None,
                "diffusion_matrix": self.diffusion_matrix_.tolist() if self.diffusion_matrix_ is not None else None,
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CareerMovementEmbeddings":
        """Reconstruct a CareerMovementEmbeddings instance from a dictionary."""
        params = data.get("params", {})
        instance = cls(**params)
        fitted = data.get("fitted", {})
        instance.role_to_idx_ = fitted.get("role_to_idx", {})
        instance.idx_to_role_ = {int(k): v for k, v in fitted.get("idx_to_role", {}).items()}
        re = fitted.get("role_embeddings")
        instance.role_embeddings_ = np.array(re, dtype=float) if re is not None else None
        tm = fitted.get("transition_matrix")
        instance.transition_matrix_ = np.array(tm, dtype=float) if tm is not None else None
        dm = fitted.get("diffusion_matrix")
        instance.diffusion_matrix_ = np.array(dm, dtype=float) if dm is not None else None
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
    def from_json(cls, json_str_or_filepath: Union[str, Any]) -> "CareerMovementEmbeddings":
        """Restore CareerMovementEmbeddings from a JSON string or file path."""
        if isinstance(json_str_or_filepath, str):
            if json_str_or_filepath.strip().startswith("{"):
                data = json.loads(json_str_or_filepath)
            else:
                with open(json_str_or_filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
        else:
            data = json.load(json_str_or_filepath)
        return cls.from_dict(data)

    def save(self, filepath: str):
        """Save model to disk as JSON or Pickle."""
        if filepath.endswith(".json"):
            self.to_json(filepath)
        else:
            with open(filepath, "wb") as f:
                pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str) -> "CareerMovementEmbeddings":
        """Load model from disk (JSON or Pickle)."""
        if filepath.endswith(".json"):
            return cls.from_json(filepath)
        try:
            with open(filepath, "rb") as f:
                return pickle.load(f)
        except Exception:
            return cls.from_json(filepath)
        
    def compute_role_similarity_matrix(
        self,
        roles: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """Compute complete or subset pairwise cosine similarity matrix between organizational roles.

        Parameters
        ----------
        roles : sequence of str, optional
            Optional subset of roles to include in the similarity matrix.
            If None, computes similarities across all vocabulary roles.

        Returns
        -------
        pd.DataFrame
            Square symmetric similarity matrix with cosine similarity scores in [-1.0, 1.0].

        Raises
        ------
        ValueError
            If the model has not been fitted yet.
        KeyError
            If any role in `roles` is not found in the fitted vocabulary.
        """
        if self.role_embeddings_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        if roles is None:
            sims = self.role_embeddings_ @ self.role_embeddings_.T
            role_names = [self.idx_to_role_[i] for i in range(len(self.idx_to_role_))]
            return pd.DataFrame(sims, index=role_names, columns=role_names)

        missing = [r for r in roles if r not in self.role_to_idx_]
        if missing:
            raise KeyError(f"Roles not found in vocabulary: {missing}")
        indices = [self.role_to_idx_[r] for r in roles]
        sub_embeds = self.role_embeddings_[indices]
        sims = sub_embeds @ sub_embeds.T
        return pd.DataFrame(sims, index=list(roles), columns=list(roles))

    # Deprecated method aliases emitting FutureWarnings
    get_role_similarity_matrix = deprecated_alias(
        compute_role_similarity_matrix,
        "get_role_similarity_matrix",
        "compute_role_similarity_matrix",
    )
    calculate_role_similarity_matrix = deprecated_alias(
        compute_role_similarity_matrix,
        "calculate_role_similarity_matrix",
        "compute_role_similarity_matrix",
    )
        
    def recommend_succession_candidates(
        self,
        target_role: str,
        associates_df: pd.DataFrame,
        history_col: str = "career_history",
        current_role_col: str = "current_role",
        top_k: int = 5,
        cosine_weight: float = 0.5,
        reachability_weight: float = 0.5,
    ) -> pd.DataFrame:
        """Rank candidate associates for succession into a target role.
        
        Combines trajectory cosine similarity in embedding space with directional transition reachability:
            Succession Score = cosine_weight * Cosine_Similarity + reachability_weight * (Reachability / Max_Reachability)
            
        By default (cosine_weight=0.5, reachability_weight=0.5):
            Succession Score = 0.5 * Cosine_Similarity + 0.5 * (Reachability / Max_Reachability)

        Parameters
        ----------
        target_role : str
            Target role for which succession candidates are being sought.
        associates_df : pd.DataFrame
            Employee DataFrame containing associate identifiers and career history.
        history_col : str, default="career_history"
            Column name containing past career sequence (list of role strings or delimiter-separated string).
        current_role_col : str, default="current_role"
            Column name containing the candidate's current organizational role.
        top_k : int, default=5
            Maximum number of top-ranked candidates to return.
        cosine_weight : float, default=0.5
            Relative weight assigned to trajectory embedding cosine similarity.
        reachability_weight : float, default=0.5
            Relative weight assigned to diffusion transition graph reachability.

        Returns
        -------
        pd.DataFrame
            Ranked candidate DataFrame with scores and diagnostic columns.

        Raises
        ------
        ValueError
            If model is not fitted, or if cosine_weight or reachability_weight is negative.
        KeyError
            If target_role is not found in the model vocabulary.
        """
        if self.diffusion_matrix_ is None or not self.role_to_idx_:
            raise ValueError("CareerMovementEmbeddings model has not been fitted. Call fit() first.")
            
        if target_role not in self.role_to_idx_:
            raise KeyError(f"Target role '{target_role}' not in model vocabulary.")
            
        if cosine_weight < 0 or reachability_weight < 0:
            raise ValueError(
                f"Weights must be non-negative, got cosine_weight={cosine_weight}, "
                f"reachability_weight={reachability_weight}"
            )
            
        output_cols = [
            "employee_id",
            "current_role",
            "career_history",
            "embedding_cosine_similarity",
            "transition_reachability",
            "succession_similarity",
        ]
        if len(associates_df) == 0:
            return pd.DataFrame(columns=output_cols)

        target_idx = self.role_to_idx_[target_role]
        target_vec = self.get_role_embedding(target_role)

        # Max reachability into target role across all non-target roles
        all_reach = self.diffusion_matrix_[:, target_idx].copy()
        all_reach[target_idx] = 0.0
        max_reach = float(np.max(all_reach)) if np.max(all_reach) > 0 else 1.0
        reach_lookup = {
            role: float(self.diffusion_matrix_[idx, target_idx] / max_reach)
            for role, idx in self.role_to_idx_.items()
        }

        # Fast vector extraction without per-row iterrows overhead
        if current_role_col in associates_df.columns:
            curr_roles = associates_df[current_role_col].tolist()
        else:
            curr_roles = [""] * len(associates_df)

        valid_mask = [r != target_role for r in curr_roles]
        cand_indices = [i for i, is_valid in enumerate(valid_mask) if is_valid]
        if not cand_indices:
            return pd.DataFrame(columns=output_cols)

        if "employee_id" in associates_df.columns:
            emp_ids = associates_df["employee_id"].tolist()
        else:
            emp_ids = [f"EMP_{i}" for i in range(len(associates_df))]

        if history_col in associates_df.columns:
            histories = associates_df[history_col].tolist()
        else:
            histories = [[r] for r in curr_roles]

        cand_emp_ids = [emp_ids[i] for i in cand_indices]
        cand_curr_roles = [curr_roles[i] for i in cand_indices]
        cand_histories = [histories[i] for i in cand_indices]

        # Vectorized reachability scoring
        reach_scores = np.array([reach_lookup.get(r, 0.0) for r in cand_curr_roles], dtype=np.float64)

        # Career history trajectory embedding calculation with dict memoization
        history_cache: Dict[Tuple[str, ...], float] = {}
        cos_sims = np.zeros(len(cand_indices), dtype=np.float64)
        norm_histories: List[List[str]] = []

        for i, h in enumerate(cand_histories):
            if isinstance(h, list):
                h_tuple = tuple(h)
                norm_h = h
            elif isinstance(h, tuple):
                h_tuple = h
                norm_h = list(h)
            elif isinstance(h, np.ndarray):
                h_tuple = tuple(h.tolist())
                norm_h = h.tolist()
            elif pd.isna(h):
                curr_r = cand_curr_roles[i]
                h_tuple = (curr_r,)
                norm_h = [curr_r]
            else:
                h_str = str(h)
                h_tuple = (h_str,)
                norm_h = [h_str]

            norm_histories.append(norm_h)

            if h_tuple not in history_cache:
                assoc_vec = self.get_associate_embedding(norm_h)
                history_cache[h_tuple] = float(np.dot(assoc_vec, target_vec))
            cos_sims[i] = history_cache[h_tuple]

        # Combined succession potential
        combined_scores = cosine_weight * cos_sims + reachability_weight * reach_scores
        round_combined = np.round(combined_scores, 4)

        k = min(top_k, len(round_combined))
        top_order = np.argsort(-round_combined, kind="mergesort")[:k]

        records = []
        for idx in top_order:
            norm_h = norm_histories[idx]
            hist_str = " -> ".join(norm_h) if isinstance(norm_h, list) else str(norm_h)
            records.append({
                "employee_id": cand_emp_ids[idx],
                "current_role": cand_curr_roles[idx],
                "career_history": hist_str,
                "embedding_cosine_similarity": round(float(cos_sims[idx]), 4),
                "transition_reachability": round(float(reach_scores[idx]), 4),
                "succession_similarity": round(float(combined_scores[idx]), 4),
            })

        return pd.DataFrame(records)


def calculate_time_in_position(
    df: pd.DataFrame,
    last_role_change_col: str = "last_role_change_date",
    as_of_date: Optional[Union[str, pd.Timestamp]] = None,
    output_unit: str = "months",
) -> pd.Series:
    """Calculate Time-in-Position as running elapsed duration since the last job code change.

    Computes elapsed time in months or days between an associate's last promotion or lateral transfer
    and the designated evaluation date.

    Formulas:
        Time_in_Position_days = As_Of_Date - Last_Role_Change_Date
        Time_in_Position_months = Time_in_Position_days / 30.4375

    Parameters
    ----------
    df : pd.DataFrame
        Employee dataset containing date column for last job code change.
    last_role_change_col : str, default='last_role_change_date'
        Column name with datetime strings or timestamps of the last role transition.
    as_of_date : str or pd.Timestamp, optional
        Reference evaluation date. Defaults to current date or maximum date in column.
    output_unit : str, default='months'
        Unit of duration ('months' or 'days').

    Returns
    -------
    pd.Series
        Time-in-position in specified units.

    Raises
    ------
    ValueError
        If `last_role_change_col` is not found in `df`, or if `output_unit` is not 'months' or 'days'.
    """
    if last_role_change_col not in df.columns:
        raise ValueError(f"Column '{last_role_change_col}' not found in DataFrame.")

    role_dates = pd.to_datetime(df[last_role_change_col])
    if as_of_date is None:
        eval_dt = role_dates.max()
    else:
        eval_dt = pd.to_datetime(as_of_date)

    delta_days = (eval_dt - role_dates).dt.total_seconds() / (24 * 3600)
    delta_days = np.maximum(0.0, delta_days)

    if output_unit == "days":
        return pd.Series(np.round(delta_days, 1), index=df.index, name="time_in_position_days")
    elif output_unit == "months":
        # Average days in a month: 365.25 / 12 = 30.4375
        months = delta_days / 30.4375
        return pd.Series(np.round(months, 1), index=df.index, name="time_in_position_months")
    else:
        raise ValueError(f"Unsupported output_unit '{output_unit}'. Use 'months' or 'days'.")


def calculate_stagnation_index(
    df: pd.DataFrame,
    time_in_pos_col: str = "time_in_position_months",
    cohort_col: str = "department",
    perf_col: str = "performance_rating",
    high_perf_threshold: float = 4.0,
    stagnation_quantile: float = 0.90,
    emerging_quantile: float = 0.75,
    base_flight_risk_col: Optional[str] = None,
    stagnation_sensitivity: float = 0.25,
) -> pd.DataFrame:
    """Calculate the Stagnation Index and flag stagnant top performers for dynamic career pathing.

    Evaluates associates against localized department cohort tenure distributions,
    identifying high performers whose Time-in-Position exceeds the 90th percentile.
    Such associates face acute promotional bottlenecks and high voluntary flight risk,
    serving as dynamic career-pathing intervention triggers.

    Formulas:
        Cohort_Percentile_i = F_C(Time_in_Position_i)
        Is_Cohort_Stagnant_i = I(Time_in_Position_i >= Q_90(Time_in_Position | C))
        Is_Stagnant_Top_Performer_i = Is_Cohort_Stagnant_i AND (Perf_Rating_i >= high_perf_threshold)
        Delta_Flight_Risk_stagnation = stagnation_sensitivity * Perf_Factor * (1 - exp(-Excess / 6.0))

    Parameters
    ----------
    df : pd.DataFrame
        Employee dataset.
    time_in_pos_col : str, default='time_in_position_months'
        Column name with months in current position.
    cohort_col : str, default='department'
        Column name defining the peer comparison cohort (e.g. department or job family).
    perf_col : str, default='performance_rating'
        Column name with employee performance rating.
    high_perf_threshold : float, default=4.0
        Threshold for high performer status.
    stagnation_quantile : float, default=0.90
        Cohort quantile benchmark defining acute stagnation.
    emerging_quantile : float, default=0.75
        Cohort quantile benchmark defining emerging stagnation warning.
    base_flight_risk_col : str, optional
        Column name with baseline flight risk to calculate post-stagnation flight risk.
    stagnation_sensitivity : float, default=0.25
        Scaling factor for flight risk surge.

    Returns
    -------
    pd.DataFrame
        DataFrame with cohort benchmarks, stagnation metrics, and career-pathing intervention tags.

    Raises
    ------
    ValueError
        If `time_in_pos_col`, `cohort_col`, or `perf_col` is missing from `df`,
        or if stagnation quantile parameters are outside the interval [0.0, 1.0].
    """
    res = df.copy()
    if time_in_pos_col not in res.columns:
        raise ValueError(f"Column '{time_in_pos_col}' not found in DataFrame.")
    if cohort_col not in res.columns:
        raise ValueError(f"Column '{cohort_col}' not found in DataFrame.")
    if perf_col not in res.columns:
        raise ValueError(f"Column '{perf_col}' not found in DataFrame.")

    # Calculate cohort percentiles and benchmarks
    res["cohort_median_months"] = res.groupby(cohort_col)[time_in_pos_col].transform("median")
    res["cohort_p90_months"] = res.groupby(cohort_col)[time_in_pos_col].transform(lambda s: s.quantile(stagnation_quantile))
    res["cohort_p75_months"] = res.groupby(cohort_col)[time_in_pos_col].transform(lambda s: s.quantile(emerging_quantile))

    # Empirical percentile within cohort (0.0 to 1.0)
    res["cohort_time_in_pos_percentile"] = res.groupby(cohort_col)[time_in_pos_col].transform(lambda s: s.rank(pct=True))
    res["cohort_time_in_pos_percentile"] = np.round(res["cohort_time_in_pos_percentile"], 4)

    # Boolean flags
    res["is_cohort_stagnant"] = res[time_in_pos_col] >= res["cohort_p90_months"]
    res["is_high_performer"] = res[perf_col] >= high_perf_threshold
    res["is_stagnant_top_performer"] = res["is_cohort_stagnant"] & res["is_high_performer"]

    # Flight risk surge spikes sharply for stagnant top performers
    # Tenured low/average performers who are stagnant do not face this sharp voluntary attrition surge
    excess_months = np.maximum(0.0, res[time_in_pos_col] - res["cohort_p90_months"])
    risk_surge = np.where(
        res["is_cohort_stagnant"] & res["is_high_performer"],
        (0.08 + stagnation_sensitivity * (1.0 - np.exp(-excess_months / 6.0))) * (res[perf_col] / 4.0),
        0.0,
    )
    res["stagnation_flight_risk_delta"] = np.round(np.clip(risk_surge, 0.0, 0.45), 4)

    if base_flight_risk_col and base_flight_risk_col in res.columns:
        post_risk = np.clip(res[base_flight_risk_col] + res["stagnation_flight_risk_delta"], 0.0, 1.0)
        res["post_stagnation_flight_risk"] = np.round(post_risk, 4)

    # Vectorized actionable career pathing intervention tag via np.select
    cond_urgent = res["is_high_performer"] & res["is_cohort_stagnant"]
    cond_emerging = res["is_high_performer"] & (res[time_in_pos_col] >= res["cohort_p75_months"])
    cond_tenured = (~res["is_high_performer"]) & res["is_cohort_stagnant"]

    res["career_pathing_intervention"] = np.select(
        [cond_urgent, cond_emerging, cond_tenured],
        [
            "Urgent Career-Pathing Trigger",
            "Emerging Stagnation Warning",
            "Tenured Core Contributor",
        ],
        default="Normal Progression",
    )
    return res


# API Consistency Aliases: compute_* <=> calculate_*
compute_time_in_position = calculate_time_in_position
compute_stagnation_index = calculate_stagnation_index


def compute_role_similarity_matrix(
    model_or_embeddings: Union[CareerMovementEmbeddings, pd.DataFrame, np.ndarray],
    roles: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Compute complete or subset pairwise cosine similarity matrix between organizational roles.

    Parameters
    ----------
    model_or_embeddings : CareerMovementEmbeddings, pd.DataFrame, or np.ndarray
        Fitted CareerMovementEmbeddings instance, or 2D array/DataFrame of role embeddings.
    roles : sequence of str, optional
        Role names corresponding to rows of embeddings (required if `model_or_embeddings` is np.ndarray).

    Returns
    -------
    pd.DataFrame
        Pairwise cosine similarity matrix.

    Raises
    ------
    TypeError
        If `model_or_embeddings` is not a CareerMovementEmbeddings instance, DataFrame, or ndarray.
    KeyError
        If any role in `roles` is not found in the embedding index.
    """
    if isinstance(model_or_embeddings, CareerMovementEmbeddings):
        return model_or_embeddings.compute_role_similarity_matrix(roles=roles)
    elif isinstance(model_or_embeddings, pd.DataFrame):
        role_names = list(roles) if roles is not None else model_or_embeddings.index.tolist()
        embeds = model_or_embeddings.to_numpy(dtype=float)
        norms = np.linalg.norm(embeds, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        normed = embeds / norms
        sims = normed @ normed.T
        return pd.DataFrame(sims, index=role_names, columns=role_names)
    elif isinstance(model_or_embeddings, np.ndarray):
        if roles is None:
            roles = [f"role_{i}" for i in range(len(model_or_embeddings))]
        embeds = np.asarray(model_or_embeddings, dtype=float)
        norms = np.linalg.norm(embeds, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        normed = embeds / norms
        sims = normed @ normed.T
        return pd.DataFrame(sims, index=list(roles), columns=list(roles))
    else:
        raise TypeError(
            f"model_or_embeddings must be CareerMovementEmbeddings, pd.DataFrame, or np.ndarray, got {type(model_or_embeddings).__name__}"
        )


get_role_similarity_matrix = deprecated_alias(
    compute_role_similarity_matrix,
    "get_role_similarity_matrix",
    "compute_role_similarity_matrix",
)
calculate_role_similarity_matrix = deprecated_alias(
    compute_role_similarity_matrix,
    "calculate_role_similarity_matrix",
    "compute_role_similarity_matrix",
)


