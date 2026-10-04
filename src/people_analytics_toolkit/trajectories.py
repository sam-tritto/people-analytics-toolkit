"""Continuous-Time Trajectories and Heterogeneous Graph Representation Learning.

This module implements continuous-time stochastic processes and heterogeneous representation learning
for People Analytics and workforce intelligence:
1. Continuous-Time Markov Chains (CTMC): Generator matrix Q estimation, continuous matrix exponential
   transition probability P(t) = exp(Qt), and expected sojourn time bottleneck identification.
2. Meta-Path Biased Random Walks: Schema-governed stochastic traversals on heterogeneous HCM multigraphs.
3. Metapath2Vec: Dense heterogeneous graph representation learning via shifted Pointwise Mutual
   Information (SPPMI) low-rank factorization for skill adjacency and team complementarity.
4. Dynamic Entity & Trajectory Self-Attention: Multi-head scaled dot-product attention dynamically
   weighting heterogeneous entity tokens based on operational context.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import json
import pickle
import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.spatial.distance import cdist
from sklearn.decomposition import TruncatedSVD
import networkx as nx
 
__all__ = [
    "ContinuousTimeMarkovChain",
    "generate_metapath_walks",
    "Metapath2Vec",
    "DynamicEntitySelfAttention",
]

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
    class _DynamicAttentionModule(nn.Module):
        """PyTorch Module for multi-head dynamic entity attention optimization."""
        def __init__(self, d_model: int, n_heads: int):
            super().__init__()
            self.d_model = d_model
            self.n_heads = n_heads
            self.head_dim = d_model // n_heads
            self.W_q = nn.Linear(d_model, d_model, bias=False)
            self.W_k = nn.Linear(d_model, d_model, bias=False)
            self.W_v = nn.Linear(d_model, d_model, bias=False)
            self.W_o = nn.Linear(d_model, d_model, bias=False)
            self.pred_head = nn.Linear(d_model, 1)

        def forward(self, tokens: torch.Tensor, query: Optional[torch.Tensor] = None):
            n_tokens = tokens.shape[0]
            if query is None:
                q = tokens.mean(dim=0, keepdim=True)
            else:
                q = query.view(1, -1)

            Q = self.W_q(q).view(1, self.n_heads, self.head_dim).transpose(0, 1)
            K = self.W_k(tokens).view(n_tokens, self.n_heads, self.head_dim).transpose(0, 1)
            V = self.W_v(tokens).view(n_tokens, self.n_heads, self.head_dim).transpose(0, 1)

            scale = np.sqrt(self.head_dim)
            scores = torch.bmm(Q, K.transpose(1, 2)) / scale
            attn = torch.softmax(scores, dim=-1)
            head_out = torch.bmm(attn, V)
            concat = head_out.transpose(0, 1).contiguous().view(1, self.d_model)
            ctx = self.W_o(concat).squeeze(0)
            return ctx, attn.squeeze(1)
else:
    _DynamicAttentionModule = None


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


class ContinuousTimeMarkovChain:
    r"""Continuous-Time Markov Chain (CTMC) for career trajectories and mobility dynamics.

    A CTMC models continuous-time state transitions governed by an infinitesimal generator
    rate matrix :math:`Q \in \mathbb{R}^{M \times M}`.
    - Off-diagonal elements :math:`q_{ij} \ge 0` (:math:`i \neq j`) represent instantaneous
      transition rates from state :math:`i` to state :math:`j`.
    - Diagonal elements :math:`q_{ii} = -\sum_{j \neq i} q_{ij} \le 0` ensure that each row sums to zero.

    The continuous transition probability matrix over any arbitrary continuous time horizon :math:`t \ge 0`
    is calculated via the matrix exponential:
    .. math::
        P(t) = \exp(Q t) = \sum_{k=0}^{\infty} \frac{(Q t)^k}{k!}

    The expected sojourn time in transient state :math:`i` before transitioning is:
    .. math::
        \mathbb{E}[T_i] = \frac{1}{-q_{ii}} = \frac{1}{\sum_{j \neq i} q_{ij}}
    """

    def __init__(self, state_labels: Optional[List[str]] = None):
        self.state_labels: List[str] = state_labels or []
        self.state_to_idx: Dict[str, int] = {s: i for i, s in enumerate(self.state_labels)}
        self.generator_matrix_: Optional[np.ndarray] = None
        self.transition_counts_: Optional[pd.DataFrame] = None
        self.total_sojourn_times_: Optional[pd.Series] = None

    def fit_from_event_log(
        self,
        df: pd.DataFrame,
        employee_id_col: str = "employee_id",
        state_col: str = "role_level",
        start_time_col: str = "start_year",
        end_time_col: str = "end_year",
    ) -> "ContinuousTimeMarkovChain":
        """Fit generator matrix Q from an irregular employee career event log.

        Parameters
        ----------
        df : pd.DataFrame
            Event log containing historical transitions.
        employee_id_col : str, default='employee_id'
            Column identifying individual employees.
        state_col : str, default='role_level'
            Column identifying job level, title, or department state.
        start_time_col : str, default='start_year'
            Continuous timestamp when employee entered the state.
        end_time_col : str, default='end_year'
            Continuous timestamp when employee exited the state.
        """
        sorted_df = df.sort_values(by=[employee_id_col, start_time_col]).copy()
        states = sorted(sorted_df[state_col].unique())
        self.state_labels = list(states)
        self.state_to_idx = {s: i for i, s in enumerate(self.state_labels)}
        m = len(self.state_labels)

        transition_counts = np.zeros((m, m), dtype=np.float64)
        total_time_in_state = np.zeros(m, dtype=np.float64)

        for _, group in sorted_df.groupby(employee_id_col):
            rows = group.to_dict("records")
            for idx, r in enumerate(rows):
                curr_s = r[state_col]
                s_idx = self.state_to_idx[curr_s]
                duration = max(0.01, float(r[end_time_col]) - float(r[start_time_col]))
                total_time_in_state[s_idx] += duration

                if idx + 1 < len(rows):
                    next_s = rows[idx + 1][state_col]
                    next_idx = self.state_to_idx[next_s]
                    if s_idx != next_idx:
                        transition_counts[s_idx, next_idx] += 1.0

        Q = np.zeros((m, m), dtype=np.float64)
        for i in range(m):
            t_i = total_time_in_state[i]
            if t_i > 0:
                for j in range(m):
                    if i != j:
                        Q[i, j] = transition_counts[i, j] / t_i
                Q[i, i] = -np.sum(Q[i, :])
            else:
                Q[i, i] = 0.0

        self.generator_matrix_ = Q
        self.transition_counts_ = pd.DataFrame(transition_counts, index=self.state_labels, columns=self.state_labels)
        self.total_sojourn_times_ = pd.Series(total_time_in_state, index=self.state_labels)
        return self

    def fit_from_matrix(
        self,
        generator_matrix: np.ndarray,
        state_labels: List[str],
    ) -> "ContinuousTimeMarkovChain":
        """Initialize CTMC with a known generator rate matrix Q."""
        Q = np.asarray(generator_matrix, dtype=np.float64)
        m = len(state_labels)
        if Q.shape != (m, m):
            raise ValueError(f"generator_matrix shape {Q.shape} must match ({m}, {m}).")
        # Ensure row sums are zero
        for i in range(m):
            Q[i, i] = 0.0
            Q[i, i] = -np.sum(Q[i, :])

        self.state_labels = list(state_labels)
        self.state_to_idx = {s: i for i, s in enumerate(self.state_labels)}
        self.generator_matrix_ = Q
        return self

    def transition_probability_matrix(self, t: float) -> pd.DataFrame:
        """Compute transition probability matrix P(t) = exp(Q * t) for horizon t."""
        if self.generator_matrix_ is None:
            raise ValueError("CTMC model must be fitted before computing transition probabilities.")
        if t < 0:
            raise ValueError("Time horizon t must be non-negative.")

        if t == 0:
            p_mat = np.eye(len(self.state_labels))
        else:
            p_mat = expm(self.generator_matrix_ * t)
            # Guard against minor numerical floating-point negative values
            p_mat = np.maximum(p_mat, 0.0)
            row_sums = np.sum(p_mat, axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            p_mat = p_mat / row_sums

        return pd.DataFrame(p_mat, index=self.state_labels, columns=self.state_labels)

    def expected_sojourn_times(self) -> pd.Series:
        """Calculate the expected continuous duration E[T_i] spent in each state before transitioning."""
        if self.generator_matrix_ is None:
            raise ValueError("CTMC model must be fitted first.")

        diag = np.diag(self.generator_matrix_)
        sojourn = np.zeros(len(diag))
        for i, q_ii in enumerate(diag):
            if abs(q_ii) > 1e-9:
                sojourn[i] = 1.0 / (-q_ii)
            else:
                sojourn[i] = np.inf  # Absorbing state

        return pd.Series(sojourn, index=self.state_labels, name="expected_sojourn_years")

    def predict_trajectory_distribution(
        self,
        initial_state: str,
        t: float,
    ) -> pd.Series:
        """Predict the state probability distribution at continuous time t given initial state."""
        if initial_state not in self.state_to_idx:
            raise ValueError(f"Unknown initial state '{initial_state}'. Valid states: {self.state_labels}")

        p_df = self.transition_probability_matrix(t)
        return p_df.loc[initial_state]

    def identify_pipeline_bottlenecks(
        self,
        sojourn_threshold_multiplier: float = 1.4,
    ) -> pd.DataFrame:
        """Identify internal mobility bottlenecks where expected sojourn time exceeds normal levels."""
        sojourn = self.expected_sojourn_times()
        transient_sojourn = sojourn[np.isfinite(sojourn)]
        median_sojourn = float(transient_sojourn.median())

        is_bottleneck = (sojourn > median_sojourn * sojourn_threshold_multiplier) & np.isfinite(sojourn)
        relative_drag = sojourn / (median_sojourn + 1e-6)

        return pd.DataFrame({
            "expected_sojourn_years": sojourn,
            "relative_drag_ratio": relative_drag,
            "is_mobility_bottleneck": is_bottleneck,
        })

    def stationary_distribution(self) -> pd.Series:
        r"""Compute the long-run equilibrium stationary distribution \pi.

        For a continuous-time Markov chain, the stationary distribution satisfies:
        .. math::
            \pi Q = 0 \quad \text{and} \quad \sum_{i=1}^M \pi_i = 1

        Returns
        -------
        pd.Series
            Stationary probability for each state in the Markov chain.
        """
        if self.generator_matrix_ is None:
            raise ValueError("CTMC model must be fitted before computing stationary distribution.")

        m = len(self.state_labels)
        if m == 0:
            return pd.Series(dtype=np.float64)

        # Form system: Q^T \pi^T = 0 subject to \sum \pi_i = 1
        A = np.vstack([self.generator_matrix_.T, np.ones((1, m))])
        b = np.zeros(m + 1, dtype=np.float64)
        b[-1] = 1.0

        pi, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        # Numerical guard: probabilities must be non-negative and sum to 1
        pi = np.maximum(pi, 0.0)
        pi[pi < 1e-12] = 0.0
        total = np.sum(pi)
        if total > 0:
            pi = pi / total
        else:
            pi = np.ones(m) / m

        return pd.Series(pi, index=self.state_labels, name="stationary_probability")

    def simulate_trajectory(
        self,
        initial_state: str,
        max_time: float = 10.0,
        max_steps: int = 100,
        seed: Optional[int] = None,
        random_state: Optional[Union[int, np.random.Generator, np.random.RandomState]] = None,
    ) -> pd.DataFrame:
        """Simulate a single continuous-time career trajectory using the Gillespie / jump process algorithm.

        Parameters
        ----------
        initial_state : str
            Starting state label for the trajectory.
        max_time : float, default=10.0
            Maximum continuous time horizon for the simulation (e.g. years).
        max_steps : int, default=100
            Maximum number of transition steps to simulate.
        seed : Optional[int], default=None
            Random seed for reproducibility.
        random_state : Optional[int, Generator, RandomState], default=None
            Random state object or integer seed.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
            - 'step': Transition step index (0, 1, 2, ...).
            - 'state': State label during the interval.
            - 'start_time': Continuous timestamp of state entry.
            - 'end_time': Continuous timestamp of state exit (or max_time).
            - 'duration': Continuous time spent in the state.
            - 'is_absorbing': Whether the state is an absorbing state.
        """
        if self.generator_matrix_ is None:
            raise ValueError("CTMC model must be fitted before simulating trajectories.")
        if initial_state not in self.state_to_idx:
            raise ValueError(f"Unknown initial state '{initial_state}'. Valid states: {self.state_labels}")
        if max_time <= 0:
            raise ValueError(f"max_time must be positive, got {max_time}.")
        if max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {max_steps}.")

        if seed is not None:
            rng = np.random.default_rng(seed)
        elif isinstance(random_state, (np.random.Generator, np.random.RandomState)):
            rng = random_state
        elif random_state is not None:
            rng = np.random.default_rng(random_state)
        else:
            rng = np.random.default_rng()

        curr_state = initial_state
        curr_time = 0.0
        step = 0
        records: List[Dict[str, Any]] = []

        while curr_time < max_time and step < max_steps:
            curr_idx = self.state_to_idx[curr_state]
            q_ii = self.generator_matrix_[curr_idx, curr_idx]
            exit_rate = -q_ii

            # Check for absorbing state (exit_rate == 0)
            if exit_rate <= 1e-9:
                end_time = max_time
                duration = end_time - curr_time
                records.append({
                    "step": step,
                    "state": curr_state,
                    "start_time": round(float(curr_time), 4),
                    "end_time": round(float(end_time), 4),
                    "duration": round(float(duration), 4),
                    "is_absorbing": True,
                })
                break

            # Sample continuous holding time from Exponential(exit_rate)
            holding_time = float(rng.exponential(scale=1.0 / exit_rate))
            end_time = min(curr_time + holding_time, max_time)
            duration = end_time - curr_time
            reached_horizon = (curr_time + holding_time >= max_time)

            records.append({
                "step": step,
                "state": curr_state,
                "start_time": round(float(curr_time), 4),
                "end_time": round(float(end_time), 4),
                "duration": round(float(duration), 4),
                "is_absorbing": False,
            })

            if reached_horizon:
                break

            # Sample next state from embedded jump probabilities: P(i -> j) = q_ij / sum_{k != i} q_ik
            off_diag_rates = np.copy(self.generator_matrix_[curr_idx, :])
            off_diag_rates[curr_idx] = 0.0
            off_diag_rates = np.maximum(off_diag_rates, 0.0)
            total_off_diag = np.sum(off_diag_rates)

            if total_off_diag <= 1e-9:
                break

            jump_probs = off_diag_rates / total_off_diag
            next_idx = int(rng.choice(len(self.state_labels), p=jump_probs))
            curr_state = self.state_labels[next_idx]
            curr_time = end_time
            step += 1

        return pd.DataFrame(records)

    def simulate_trajectories(
        self,
        initial_state: str,
        n_trajectories: int = 50,
        max_time: float = 10.0,
        max_steps: int = 100,
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """Simulate multiple continuous-time career trajectories.

        Parameters
        ----------
        initial_state : str
            Starting state label for each trajectory.
        n_trajectories : int, default=50
            Number of individual trajectories to simulate.
        max_time : float, default=10.0
            Maximum continuous time horizon for each trajectory.
        max_steps : int, default=100
            Maximum number of transitions per trajectory.
        seed : Optional[int], default=None
            Random seed for reproducibility.

        Returns
        -------
        pd.DataFrame
            Concatenated DataFrame of all trajectories with a 'trajectory_id' column.
        """
        rng = np.random.default_rng(seed)
        all_dfs = []
        for i in range(n_trajectories):
            df_i = self.simulate_trajectory(
                initial_state=initial_state,
                max_time=max_time,
                max_steps=max_steps,
                random_state=rng,
            )
            df_i.insert(0, "trajectory_id", f"traj_{i:04d}")
            all_dfs.append(df_i)

        if all_dfs:
            return pd.concat(all_dfs, ignore_index=True)
        return pd.DataFrame(columns=["trajectory_id", "step", "state", "start_time", "end_time", "duration", "is_absorbing"])


def generate_metapath_walks(
    G: nx.Graph,
    meta_paths: List[List[str]],
    walk_length: int = 20,
    num_walks: int = 10,
    random_state: int = 42,
) -> List[List[str]]:
    """Generate meta-path-biased random walks over a heterogeneous multigraph.

    Parameters
    ----------
    G : nx.Graph
        Heterogeneous graph where each node possesses a 'node_type' attribute.
    meta_paths : list of list of str
        List of schema sequences, e.g. [['Employee', 'Project', 'Skill', 'Employee']].
    walk_length : int, default=20
        Total length of each random walk sequence.
    num_walks : int, default=10
        Number of walks started from each valid source node.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    list of list of str
        Corpus of node sequences strictly adhering to the meta-path schemas.
    """
    rng = np.random.RandomState(random_state)
    node_types = nx.get_node_attributes(G, "node_type")
    if not node_types:
        raise ValueError("Graph nodes must possess a 'node_type' attribute.")

    walks: List[List[str]] = []

    for path_schema in meta_paths:
        if len(path_schema) < 2:
            raise ValueError(f"Each meta-path schema must contain at least 2 node types, got {path_schema}")

        start_type = path_schema[0]
        # Normalize schema into repeating cyclic pattern:
        # If the schema is closed (e.g. ['A', 'B', 'C', 'A']), the repeating unit is ['A', 'B', 'C'].
        # If the schema is open (e.g. ['A', 'B', 'C'] or ['A', 'B']), the full schema repeats ['A', 'B', 'C'].
        if path_schema[0] == path_schema[-1] and len(path_schema) > 2:
            cycle_pattern = list(path_schema[:-1])
        else:
            cycle_pattern = list(path_schema)

        pattern_len = len(cycle_pattern)
        candidate_starts = [n for n, t in node_types.items() if t == start_type]

        for start_node in candidate_starts:
            for _ in range(num_walks):
                walk = [str(start_node)]
                curr_node = start_node

                for step in range(1, walk_length):
                    # Schema target node type at this step
                    target_type = cycle_pattern[step % pattern_len]

                    # Find neighbors of target type
                    neighbors = list(G.neighbors(curr_node))
                    valid_neighbors = [nb for nb in neighbors if node_types.get(nb) == target_type]

                    if not valid_neighbors:
                        break

                    next_node = valid_neighbors[rng.randint(len(valid_neighbors))]
                    walk.append(str(next_node))
                    curr_node = next_node

                if len(walk) >= min(3, walk_length):
                    walks.append(walk)

    return walks


class Metapath2Vec:
    r"""Metapath2Vec representation learning for heterogeneous organizational networks.

    Embeds multi-entity organizational networks (Employees, Projects, Skills, Departments)
    into a low-dimensional metric space using meta-path-biased random walks and shifted Positive
    Pointwise Mutual Information (SPPMI) low-rank matrix decomposition:
    .. math::
        \text{SPPMI}(u, v) = \max\left(0, \log\left(\frac{\#(u, v) \cdot |\mathcal{D}|}{\#(u) \cdot \#(v) \cdot k}\right)\right)

    Parameters
    ----------
    embedding_dim : int, default=32
        Dimensionality of the learned dense entity vectors.
    walk_length : int, default=20
        Length of each meta-path random walk.
    num_walks : int, default=10
        Number of walks started per source node.
    window_size : int, default=5
        Context window size for co-occurrence counting.
    negative_rate : float, default=5.0
        Negative sampling shift parameter k.
    random_state : int, default=42
        Random seed for reproducibility.
    """

    def __init__(
        self,
        embedding_dim: int = 32,
        walk_length: int = 20,
        num_walks: int = 10,
        window_size: int = 5,
        negative_rate: float = 5.0,
        random_state: int = 42,
    ):
        self.embedding_dim = embedding_dim
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.window_size = window_size
        self.negative_rate = negative_rate
        self.random_state = random_state

        self.embeddings_: Optional[pd.DataFrame] = None
        self.node_types_: Dict[str, str] = {}
        self.node_list_: List[str] = []

    def fit(
        self,
        G: nx.Graph,
        meta_paths: Optional[List[List[str]]] = None,
    ) -> "Metapath2Vec":
        """Fit Metapath2Vec embeddings on the heterogeneous graph G."""
        node_types = nx.get_node_attributes(G, "node_type")
        self.node_types_ = {str(k): str(v) for k, v in node_types.items()}
        self.node_list_ = sorted(list(self.node_types_.keys()))
        n_nodes = len(self.node_list_)
        node_to_idx = {n: i for i, n in enumerate(self.node_list_)}

        if meta_paths is None:
            # Default schema: Employee -> Project -> Skill -> Employee
            meta_paths = [
                ["Employee", "Project", "Skill", "Employee"],
                ["Employee", "Department", "Employee"],
            ]

        # 1. Generate biased walks
        walks = generate_metapath_walks(
            G=G,
            meta_paths=meta_paths,
            walk_length=self.walk_length,
            num_walks=self.num_walks,
            random_state=self.random_state,
        )

        # 2. Build co-occurrence matrix within window_size
        co_counts = np.zeros((n_nodes, n_nodes), dtype=np.float64)
        node_freq = np.zeros(n_nodes, dtype=np.float64)

        for walk in walks:
            w_len = len(walk)
            for i, target in enumerate(walk):
                if target not in node_to_idx:
                    continue
                t_idx = node_to_idx[target]
                node_freq[t_idx] += 1.0

                start = max(0, i - self.window_size)
                end = min(w_len, i + self.window_size + 1)
                for j in range(start, end):
                    if i != j:
                        ctx = walk[j]
                        if ctx in node_to_idx:
                            c_idx = node_to_idx[ctx]
                            co_counts[t_idx, c_idx] += 1.0

        total_pairs = float(np.sum(co_counts))
        if total_pairs == 0:
            total_pairs = 1.0

        # 3. Compute Shifted Positive PMI (SPPMI)
        p_uv = co_counts / total_pairs
        p_u = np.sum(co_counts, axis=1, keepdims=True) / total_pairs
        p_v = np.sum(co_counts, axis=0, keepdims=True) / total_pairs

        expected = np.dot(p_u, p_v) * self.negative_rate
        ratio = np.divide(p_uv, expected, out=np.zeros_like(p_uv), where=expected > 0)
        sppmi = np.maximum(0.0, np.log(np.maximum(ratio, 1e-9)))

        # 4. Low-Rank Matrix Factorization via Truncated SVD (Levy & Goldberg, 2014)
        k_dim = min(self.embedding_dim, n_nodes - 1)
        svd = TruncatedSVD(n_components=k_dim, random_state=self.random_state)
        U_sigma = svd.fit_transform(sppmi)

        # Normalize embedding rows to unit vectors
        norms = np.linalg.norm(U_sigma, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normed_embeds = U_sigma / norms

        self.embeddings_ = pd.DataFrame(
            normed_embeds,
            index=self.node_list_,
            columns=[f"dim_{d}" for d in range(k_dim)],
        )
        return self

    def get_embedding(self, node_id: str) -> np.ndarray:
        """Retrieve the dense embedding vector for a node."""
        if self.embeddings_ is None:
            raise ValueError("Metapath2Vec model has not been fitted.")
        node_str = str(node_id)
        if node_str not in self.embeddings_.index:
            raise KeyError(f"Node '{node_str}' not found in embedding vocabulary.")
        return self.embeddings_.loc[node_str].values

    def find_similar_nodes(
        self,
        node_id: str,
        top_k: int = 5,
        target_node_type: Optional[str] = None,
    ) -> pd.DataFrame:
        """Find the top-k most structurally similar nodes via cosine similarity."""
        if self.embeddings_ is None:
            raise ValueError("Metapath2Vec model has not been fitted.")
        v = self.get_embedding(node_id).reshape(1, -1)

        candidate_df = self.embeddings_.copy()
        if target_node_type is not None:
            matching_nodes = [n for n in candidate_df.index if self.node_types_.get(n) == target_node_type]
            candidate_df = candidate_df.loc[matching_nodes]

        # Cosine similarity on unit vectors is dot product
        similarities = np.dot(candidate_df.values, v.T).ravel()
        candidate_df["cosine_similarity"] = np.round(similarities, 4)
        candidate_df["node_type"] = [self.node_types_.get(n, "Unknown") for n in candidate_df.index]

        # Exclude self
        res = candidate_df[candidate_df.index != str(node_id)].sort_values(by="cosine_similarity", ascending=False)
        return res[["node_type", "cosine_similarity"]].head(top_k)

    def predict_team_complementarity(self, employee_ids: List[str]) -> Dict[str, float]:
        """Evaluate structural complementarity and skill coverage for a proposed project team."""
        if self.embeddings_ is None:
            raise ValueError("Metapath2Vec model has not been fitted.")

        valid_ids = [str(e) for e in employee_ids if str(e) in self.embeddings_.index]
        if len(valid_ids) < 2:
            return {"pairwise_cohesion": 1.0, "diversity_spread": 0.0, "structural_coverage": 0.0}

        team_vecs = self.embeddings_.loc[valid_ids].values
        # Pairwise cosine distances
        dists = cdist(team_vecs, team_vecs, metric="cosine")
        upper_tri = dists[np.triu_indices(len(valid_ids), k=1)]

        mean_distance = float(np.mean(upper_tri))
        cohesion = float(np.clip((2.0 - mean_distance) / 2.0, 0.0, 1.0))
        diversity = float(np.clip(mean_distance / 2.0, 0.0, 1.0))
        coverage = float(np.linalg.norm(np.sum(team_vecs, axis=0))) / len(valid_ids)

        return {
            "pairwise_cohesion": float(np.round(cohesion, 4)),
            "diversity_spread": float(np.round(diversity, 4)),
            "structural_coverage": float(np.round(coverage, 4)),
        }



class DynamicEntitySelfAttention:
    r"""Dynamic Entity & Trajectory Self-Attention Aggregator with learnable multi-head projections.

    Multi-head scaled dot-product attention formulation:
    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O
    where:
    .. math::
        \text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V) = \text{softmax}\left(\frac{(Q W_i^Q) (K W_i^K)^T}{\sqrt{d_k}}\right) (V W_i^V)

    Dynamically weights an employee's career trajectory tokens, project exposure vectors,
    and skill tokens to construct context-adaptive workforce representations using
    trainable projection parameter matrices :math:`W^Q, W^K, W^V, W^O`.
    """

    def __init__(
        self,
        n_heads: int = 2,
        d_model: Optional[int] = None,
        random_state: int = 42,
    ):
        """Initialize DynamicEntitySelfAttention with multi-head projection parameters.

        Parameters
        ----------
        n_heads : int, default=2
            Number of parallel attention heads. Embedding dimension d_model must be divisible by n_heads.
        d_model : int, optional
            Embedding dimension. If None, inferred upon first call to fit() or aggregate().
        random_state : int, default=42
            Random seed for Glorot uniform projection matrix initialization.
        """
        if not isinstance(n_heads, (int, np.integer)) or n_heads < 1:
            raise ValueError(f"n_heads must be a positive integer >= 1, got {n_heads}")

        self.n_heads = int(n_heads)
        self.d_model = d_model
        self.random_state = random_state
        self.head_dim: Optional[int] = None
        self.W_q_: Optional[np.ndarray] = None
        self.W_k_: Optional[np.ndarray] = None
        self.W_v_: Optional[np.ndarray] = None
        self.W_o_: Optional[np.ndarray] = None
        self.is_fitted_: bool = False
        self.training_loss_history_: List[float] = []

        if self.d_model is not None:
            self._init_weights(self.d_model)

    def _init_weights(self, d_model: int) -> None:
        """Initialize projection matrices W_q, W_k, W_v, W_o using Glorot uniform initialization."""
        if d_model % self.n_heads != 0:
            raise ValueError(
                f"Embedding dimension d_model={d_model} must be divisible by n_heads={self.n_heads}."
            )
        self.d_model = d_model
        self.head_dim = d_model // self.n_heads

        rng = np.random.RandomState(self.random_state)
        bound = np.sqrt(6.0 / (2.0 * d_model))
        self.W_q_ = rng.uniform(-bound, bound, size=(d_model, d_model)).astype(np.float64)
        self.W_k_ = rng.uniform(-bound, bound, size=(d_model, d_model)).astype(np.float64)
        self.W_v_ = rng.uniform(-bound, bound, size=(d_model, d_model)).astype(np.float64)
        self.W_o_ = rng.uniform(-bound, bound, size=(d_model, d_model)).astype(np.float64)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this estimator."""
        return {
            "n_heads": self.n_heads,
            "d_model": self.d_model,
            "random_state": self.random_state,
        }

    def set_params(self, **params) -> "DynamicEntitySelfAttention":
        """Set parameters for this estimator."""
        for k, v in params.items():
            if hasattr(self, k):
                if k == "n_heads":
                    if not isinstance(v, (int, np.integer)) or v < 1:
                        raise ValueError(f"n_heads must be a positive integer >= 1, got {v}")
                    v = int(v)
                setattr(self, k, v)
            else:
                raise ValueError(f"Invalid parameter {k} for DynamicEntitySelfAttention")
        return self

    def fit(
        self,
        tokens_data: Union[np.ndarray, List[np.ndarray]],
        targets: Optional[Union[np.ndarray, pd.Series, List[float]]] = None,
        epochs: int = 30,
        lr: float = 0.01,
        weight_decay: float = 1e-4,
    ) -> "DynamicEntitySelfAttention":
        """Fit learnable attention projection parameters W_q, W_k, W_v, W_o.

        Parameters
        ----------
        tokens_data : np.ndarray or list of np.ndarray
            Single token matrix (n_tokens, d_model) or list of token matrices for multiple entities.
        targets : Optional array-like, default=None
            Supervision targets (e.g. team performance rating, flight risk, project outcome).
            If None, fits using self-supervised reconstruction of the token centroid.
        epochs : int, default=30
            Number of gradient descent optimization epochs.
        lr : float, default=0.01
            Learning rate.
        weight_decay : float, default=1e-4
            L2 regularization strength.

        Returns
        -------
        self : DynamicEntitySelfAttention
        """
        if isinstance(tokens_data, list):
            data_list = [np.asarray(t, dtype=np.float32) for t in tokens_data]
        elif isinstance(tokens_data, np.ndarray):
            if tokens_data.ndim == 2:
                data_list = [tokens_data.astype(np.float32)]
            elif tokens_data.ndim == 3:
                data_list = [tokens_data[i].astype(np.float32) for i in range(len(tokens_data))]
            else:
                raise ValueError(f"tokens_data array must have 2 or 3 dimensions, got {tokens_data.ndim}")
        else:
            raise TypeError("tokens_data must be a NumPy array or list of arrays.")

        d_model = data_list[0].shape[1]
        if self.W_q_ is None or self.d_model != d_model:
            self._init_weights(d_model)

        if not _TORCH_AVAILABLE:
            self.is_fitted_ = True
            self.training_loss_history_ = [0.0]
            return self

        _configure_torch_reproducibility(self.random_state)
        module = _DynamicAttentionModule(d_model, self.n_heads)

        # Initialize module with current weights (transposed for nn.Linear)
        with torch.no_grad():
            module.W_q.weight.copy_(torch.from_numpy(self.W_q_.T.astype(np.float32)))
            module.W_k.weight.copy_(torch.from_numpy(self.W_k_.T.astype(np.float32)))
            module.W_v.weight.copy_(torch.from_numpy(self.W_v_.T.astype(np.float32)))
            module.W_o.weight.copy_(torch.from_numpy(self.W_o_.T.astype(np.float32)))

        optimizer = optim.Adam(module.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()

        y_tensors = None
        if targets is not None:
            y_arr = np.asarray(targets, dtype=np.float32).ravel()
            y_tensors = [torch.tensor(y_arr[i], dtype=torch.float32) for i in range(len(data_list))]

        losses = []
        module.train()

        for epoch in range(epochs):
            epoch_loss = 0.0
            optimizer.zero_grad()
            for i, toks_np in enumerate(data_list):
                toks_torch = torch.from_numpy(toks_np)
                ctx, _ = module(toks_torch)

                if y_tensors is not None:
                    pred = module.pred_head(ctx).squeeze()
                    loss = criterion(pred, y_tensors[i])
                else:
                    # Self-supervised centroid reconstruction
                    target_centroid = toks_torch.mean(dim=0)
                    loss = criterion(ctx, target_centroid)

                loss.backward()
                epoch_loss += loss.item()

            optimizer.step()
            losses.append(epoch_loss / len(data_list))

        # Copy learned weights back to NumPy attributes
        with torch.no_grad():
            self.W_q_ = module.W_q.weight.detach().cpu().numpy().T.astype(np.float64)
            self.W_k_ = module.W_k.weight.detach().cpu().numpy().T.astype(np.float64)
            self.W_v_ = module.W_v.weight.detach().cpu().numpy().T.astype(np.float64)
            self.W_o_ = module.W_o.weight.detach().cpu().numpy().T.astype(np.float64)

        self.training_loss_history_ = losses
        self.is_fitted_ = True
        return self

    def aggregate(
        self,
        entity_tokens: np.ndarray,
        query: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Aggregate heterogeneous tokens into a context-adapted representation using multi-head attention.

        Parameters
        ----------
        entity_tokens : np.ndarray of shape (n_tokens, d_model)
            Array of entity vectors (e.g. employee token, project tokens, skill tokens).
        query : np.ndarray of shape (d_model,), optional
            Context query vector (e.g. staffing requirements or leadership criterion).
            If None, self-attention across tokens is performed.

        Returns
        -------
        dict
            Dictionary containing:
            - 'context_vector': Aggregated output vector (d_model,).
            - 'attention_weights': Normalized attention probability distribution over tokens.
            - 'head_attention_weights': Multi-head attention matrix of shape (n_heads, n_tokens).
        """
        tokens = np.asarray(entity_tokens, dtype=np.float64)
        if tokens.ndim != 2:
            raise ValueError(f"entity_tokens must be 2D array of shape (n_tokens, d_model), got shape {tokens.shape}")

        n_tokens, d_model = tokens.shape
        if self.W_q_ is None or self.d_model != d_model:
            self._init_weights(d_model)

        head_dim = self.head_dim
        scale = np.sqrt(head_dim)

        if query is None:
            q_vec = np.mean(tokens, axis=0, keepdims=True)
        else:
            q_vec = np.asarray(query, dtype=np.float64).reshape(1, -1)
            if q_vec.shape[1] != d_model:
                raise ValueError(f"query dimension {q_vec.shape[1]} must match token dimension {d_model}")

        # Linear projections via learnable matrices W_q, W_k, W_v
        Q_proj = np.dot(q_vec, self.W_q_)    # (1, d_model)
        K_proj = np.dot(tokens, self.W_k_)   # (n_tokens, d_model)
        V_proj = np.dot(tokens, self.W_v_)   # (n_tokens, d_model)

        head_contexts = []
        head_weights = []

        for h in range(self.n_heads):
            start_idx = h * head_dim
            end_idx = (h + 1) * head_dim

            Q_h = Q_proj[:, start_idx:end_idx]  # (1, head_dim)
            K_h = K_proj[:, start_idx:end_idx]  # (n_tokens, head_dim)
            V_h = V_proj[:, start_idx:end_idx]  # (n_tokens, head_dim)

            # Scaled dot-product: (1, head_dim) x (head_dim, n_tokens) -> (1, n_tokens)
            scores_h = np.dot(Q_h, K_h.T) / scale
            exp_scores = np.exp(scores_h - np.max(scores_h))
            attn_h = (exp_scores / np.sum(exp_scores)).ravel()  # (n_tokens,)

            ctx_h = np.dot(attn_h, V_h).reshape(1, -1)  # (1, head_dim)
            head_contexts.append(ctx_h)
            head_weights.append(attn_h)

        # Concatenate heads and project through learnable W_o
        concat_heads = np.hstack(head_contexts)  # (1, d_model)
        context_vec = np.dot(concat_heads, self.W_o_).ravel()  # (d_model,)

        head_weights_mat = np.array(head_weights)  # (n_heads, n_tokens)
        avg_attention = np.mean(head_weights_mat, axis=0)  # (n_tokens,)
        token_index = getattr(entity_tokens, "index", None)

        return {
            "context_vector": context_vec,
            "attention_weights": pd.Series(avg_attention, index=token_index, name="attention_weight"),
            "head_attention_weights": head_weights_mat,
        }

    def transform(
        self,
        entity_tokens: Union[np.ndarray, List[np.ndarray], pd.DataFrame],
        query: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Transform token sequence(s) into multi-head aggregated context vector(s).

        Parameters
        ----------
        entity_tokens : np.ndarray, list of np.ndarray, or pd.DataFrame
            Single 2D token array (n_tokens, d_model), DataFrame, or list/batch of 2D token arrays.
        query : np.ndarray, optional
            Query vector of shape (d_model,).

        Returns
        -------
        np.ndarray
            Aggregated context vector (d_model,) or matrix (n_entities, d_model).
        """
        if isinstance(entity_tokens, list):
            return np.vstack([self.aggregate(t, query=query)["context_vector"] for t in entity_tokens])
        elif isinstance(entity_tokens, np.ndarray) and entity_tokens.ndim == 3:
            return np.vstack([self.aggregate(entity_tokens[i], query=query)["context_vector"] for i in range(len(entity_tokens))])
        else:
            return self.aggregate(entity_tokens, query=query)["context_vector"]

    def fit_transform(
        self,
        tokens_data: Union[np.ndarray, List[np.ndarray]],
        targets: Optional[Union[np.ndarray, pd.Series, List[float]]] = None,
        query: Optional[np.ndarray] = None,
        epochs: int = 30,
        lr: float = 0.01,
        weight_decay: float = 1e-4,
    ) -> np.ndarray:
        """Fit learnable multi-head projections and transform tokens into context vectors."""
        return self.fit(tokens_data, targets=targets, epochs=epochs, lr=lr, weight_decay=weight_decay).transform(tokens_data, query=query)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize attention model parameters to dictionary."""
        return {
            "n_heads": self.n_heads,
            "d_model": self.d_model,
            "random_state": self.random_state,
            "is_fitted_": getattr(self, "is_fitted_", False),
            "training_loss_history_": getattr(self, "training_loss_history_", []),
            "W_q_": self.W_q_.tolist() if self.W_q_ is not None else None,
            "W_k_": self.W_k_.tolist() if self.W_k_ is not None else None,
            "W_v_": self.W_v_.tolist() if self.W_v_ is not None else None,
            "W_o_": self.W_o_.tolist() if self.W_o_ is not None else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DynamicEntitySelfAttention":
        """Reconstruct attention model from dictionary."""
        model = cls(
            n_heads=data["n_heads"],
            d_model=data.get("d_model"),
            random_state=data.get("random_state", 42),
        )
        model.is_fitted_ = data.get("is_fitted_", False)
        model.training_loss_history_ = data.get("training_loss_history_", [])
        for k in ["W_q_", "W_k_", "W_v_", "W_o_"]:
            v = data.get(k)
            if v is not None:
                setattr(model, k, np.array(v, dtype=np.float64))
        if model.d_model is not None:
            model.head_dim = model.d_model // model.n_heads
        return model

    def save(self, filepath: Union[str, Path]) -> None:
        """Save model parameters to JSON or pickle file."""
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.suffix.lower() == ".json":
            with open(p, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2)
        else:
            with open(p, "wb") as f:
                pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "DynamicEntitySelfAttention":
        """Load model from file."""
        p = Path(filepath)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")
        if p.suffix.lower() == ".json":
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls.from_dict(data)
        try:
            with open(p, "rb") as f:
                obj = pickle.load(f)
            if isinstance(obj, cls):
                return obj
        except Exception:
            pass
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
