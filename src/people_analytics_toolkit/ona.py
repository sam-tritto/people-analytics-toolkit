"""Organizational Network Analysis (ONA) & Structural Capital Features.

Implements network centrality, Burt's structural hole constraint index,
Leinster magnitude / Euler centrality, collaboration overload quantification,
and network contagion flight risk simulation.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import scipy.sparse as sp
import networkx as nx

from people_analytics_toolkit._deprecation import deprecated_alias

__all__ = [
    "calculate_degree_centrality",
    "compute_degree_centrality",
    "calculate_closeness_centrality",
    "compute_closeness_centrality",
    "calculate_betweenness_centrality",
    "compute_betweenness_centrality",
    "calculate_eigenvector_centrality",
    "compute_eigenvector_centrality",
    "calculate_burt_constraint",
    "compute_burt_constraint",
    "calculate_euler_centrality",
    "compute_euler_centrality",
    "calculate_collaboration_overload",
    "compute_collaboration_overload",
    "simulate_attrition_contagion",
    "compute_complete_ona_profile",
    "calculate_complete_ona_profile",
    "calculate_span_of_control",
    "compute_span_of_control",
    "evaluate_span_reorganization_shock",
]



def _ensure_graph(
    graph_or_edges: Union[nx.Graph, nx.DiGraph, pd.DataFrame, List[Tuple]],
    weight: Optional[str] = "weight",
    directed: bool = False,
) -> nx.Graph:
    """Convert flexible input structures into a standardized NetworkX graph."""
    if isinstance(graph_or_edges, (nx.Graph, nx.DiGraph)):
        return graph_or_edges
    
    G = nx.DiGraph() if directed else nx.Graph()
    if isinstance(graph_or_edges, pd.DataFrame):
        df = graph_or_edges
        src_col = "source" if "source" in df.columns else df.columns[0]
        tgt_col = "target" if "target" in df.columns else df.columns[1]
        
        has_weight = weight in df.columns if weight else False
        for _, row in df.iterrows():
            w = float(row[weight]) if has_weight else 1.0
            G.add_edge(row[src_col], row[tgt_col], weight=w)
    elif isinstance(graph_or_edges, list):
        for item in graph_or_edges:
            if len(item) == 2:
                G.add_edge(item[0], item[1], weight=1.0)
            elif len(item) >= 3:
                G.add_edge(item[0], item[1], weight=float(item[2]))
    else:
        raise TypeError(f"Unsupported graph input type: {type(graph_or_edges)}")
        
    return G


def calculate_degree_centrality(
    graph_or_edges: Union[nx.Graph, nx.DiGraph, pd.DataFrame, List[Tuple]],
    weight: Optional[str] = None,
    normalized: bool = True,
    kind: str = "total",
) -> pd.Series:
    r"""Calculate Degree Centrality (identifies organizational 'Connectors' and direct tie volume).

    Measures the volume of direct connections an employee maintains across the
    organization. In human capital management (HCM) and organizational network analysis,
    high-degree individuals serve as immediate communication hubs, team integrators,
    and direct operational touchpoints.

    Mathematical Formulations:
        - Unweighted Normalized:
          .. math::
              C_D(v) = \frac{\text{deg}(v)}{|V| - 1}
        - Unnormalized Raw Degree:
          .. math::
              C_{D,\text{raw}}(v) = \text{deg}(v)
        - Weighted Degree (Tie Strength / Volume):
          .. math::
              S(v) = \sum_{u \in N(v)} w_{vu}
          When `normalized=True`, scaled by :math:`(|V| - 1)`.

    For directed networks (nx.DiGraph):
        - `kind='total'` or `'all'`: Combined in- and out-degree.
        - `kind='in'`: In-degree centrality (incoming advice/information requests).
        - `kind='out'`: Out-degree centrality (outgoing messages/initiative pushes).

    Parameters
    ----------
    graph_or_edges : nx.Graph, nx.DiGraph, pd.DataFrame, or list of tuples
        Network structure as a NetworkX graph or edge list DataFrame with [source, target, (weight)].
    weight : str, optional
        Edge attribute storing collaboration intensity/tie weight.
        If None, computes standard unweighted degree centrality.
    normalized : bool, default=True
        Whether to normalize degree by (|V| - 1).
    kind : {'total', 'in', 'out', 'all'}, default='total'
        Directionality option for directed graphs (in-degree, out-degree, or total degree).
        Ignored for undirected graphs.

    Returns
    -------
    pd.Series
        Degree centrality scores indexed by employee ID, sorted by index.
    """
    G = _ensure_graph(graph_or_edges, weight=weight)
    n = len(G)
    if n <= 1:
        return pd.Series({node: 0.0 for node in G.nodes()}, name="degree_centrality", dtype=float)

    denom = float(n - 1) if normalized else 1.0

    if G.is_directed():
        k = str(kind).strip().lower()
        if k in ("in", "in_degree"):
            if weight is not None:
                deg_dict = dict(G.in_degree(weight=weight))
                res = {node: float(val) / denom for node, val in deg_dict.items()}
            else:
                if normalized:
                    res = nx.in_degree_centrality(G)
                else:
                    res = {node: float(val) for node, val in dict(G.in_degree()).items()}
            name = "in_degree_centrality"
        elif k in ("out", "out_degree"):
            if weight is not None:
                deg_dict = dict(G.out_degree(weight=weight))
                res = {node: float(val) / denom for node, val in deg_dict.items()}
            else:
                if normalized:
                    res = nx.out_degree_centrality(G)
                else:
                    res = {node: float(val) for node, val in dict(G.out_degree()).items()}
            name = "out_degree_centrality"
        else:
            if weight is not None:
                in_d = dict(G.in_degree(weight=weight))
                out_d = dict(G.out_degree(weight=weight))
                res = {node: float(in_d.get(node, 0.0) + out_d.get(node, 0.0)) / (2.0 * denom if normalized else 1.0) for node in G.nodes()}
            else:
                if normalized:
                    in_cent = nx.in_degree_centrality(G)
                    out_cent = nx.out_degree_centrality(G)
                    res = {node: (in_cent[node] + out_cent[node]) / 2.0 for node in G.nodes()}
                else:
                    res = {node: float(val) for node, val in dict(G.degree()).items()}
            name = "degree_centrality"
    else:
        if weight is not None:
            deg_dict = dict(G.degree(weight=weight))
            res = {node: float(val) / denom for node, val in deg_dict.items()}
        else:
            if normalized:
                res = nx.degree_centrality(G)
            else:
                res = {node: float(val) for node, val in dict(G.degree()).items()}
        name = "degree_centrality"

    return pd.Series(res, name=name, dtype=float).sort_index()


def calculate_closeness_centrality(
    graph_or_edges: Union[nx.Graph, nx.DiGraph, pd.DataFrame, List[Tuple]],
    weight: Optional[str] = "weight",
    invert_weight_for_distance: bool = True,
    normalized: bool = True,
    wf_improved: bool = True,
) -> pd.Series:
    r"""Calculate Closeness Centrality (identifies organizational 'Information Disseminators').

    Measures how quickly an associate can access or spread information across the
    entire enterprise based on the reciprocal of geodesic shortest-path distances
    to all other members. In workforce networks, high-closeness individuals operate
    with low communication friction, enabling rapid organizational pulse sensing
    and agile knowledge propagation.

    Mathematical Formulations:
        - For a fully connected graph:
          .. math::
              C_C(u) = \frac{|V| - 1}{\sum_{v \ne u} d(u, v)}
        - Wasserman & Faust (1994) formulation for disconnected components:
          .. math::
              C_C(u) = \frac{|R(u)| - 1}{|V| - 1} \cdot \frac{|R(u)| - 1}{\sum_{v \in R(u)} d(u, v)}
          where :math:`R(u)` is the set of nodes reachable from :math:`u`.

    Parameters
    ----------
    graph_or_edges : nx.Graph, nx.DiGraph, pd.DataFrame, or list of tuples
        Network structure as a NetworkX graph or edge list DataFrame with [source, target, (weight)].
    weight : str, optional, default='weight'
        Edge attribute storing collaboration intensity/tie volume. If None, computes unweighted hop distances.
    invert_weight_for_distance : bool, default=True
        In shortest-path routing, higher collaboration volume indicates
        closer communication distance. When True, inverts positive weights (:math:`d = 1 / w`).
    normalized : bool, default=True
        Whether to normalize closeness by network size (|V| - 1).
    wf_improved : bool, default=True
        Use Wasserman and Faust improved formula for graphs with multiple components.

    Returns
    -------
    pd.Series
        Closeness centrality scores indexed by employee ID, sorted by index.
    """
    G = _ensure_graph(graph_or_edges, weight=weight)
    n = len(G)
    if n <= 1:
        return pd.Series({node: 0.0 for node in G.nodes()}, name="closeness_centrality", dtype=float)

    if weight and invert_weight_for_distance:
        H = G.copy()
        weights = [float(data.get(weight, 1.0)) for _, _, data in H.edges(data=True)]
        max_w = max(weights) if weights else 1.0
        max_w = max_w if max_w > 0 else 1.0
        for u, v, data in H.edges(data=True):
            w = float(data.get(weight, 1.0)) if weight else 1.0
            data["_dist"] = max_w / max(w, 1e-6)
        cc = nx.closeness_centrality(H, distance="_dist", wf_improved=wf_improved)
    elif weight:
        cc = nx.closeness_centrality(G, distance=weight, wf_improved=wf_improved)
    else:
        cc = nx.closeness_centrality(G, distance=None, wf_improved=wf_improved)

    if not normalized:
        cc_unnorm = {}
        for node, c_val in cc.items():
            cc_unnorm[node] = c_val / (n - 1.0) if n > 1 else 0.0
        cc = cc_unnorm

    return pd.Series(cc, name="closeness_centrality", dtype=float).sort_index()


def calculate_betweenness_centrality(
    graph_or_edges: Union[nx.Graph, nx.DiGraph, pd.DataFrame, List[Tuple]],
    weight: Optional[str] = "weight",
    invert_weight_for_distance: bool = True,
    normalized: bool = True,
) -> pd.Series:
    """Calculate Betweenness Centrality (identifies organizational 'Brokers').
    
    Formula:
        C_B(v) = sum_{s != v != t} (sigma_{st}(v) / sigma_{st})
        
    Brokers act as critical bridges spanning shortest communication paths
    between otherwise decoupled business units, teams, or departments.
    
    Parameters
    ----------
    graph_or_edges : nx.Graph or DataFrame with [source, target, weight]
    weight : str, default='weight'
        Edge attribute name storing collaboration intensity.
    invert_weight_for_distance : bool, default=True
        In shortest-path routing, higher collaboration volume indicates
        closer communication distance. When True, inverts positive weights (d = 1 / w).
    normalized : bool, default=True
        Scales betweenness between 0 and 1.
        
    Returns
    -------
    pd.Series
        Betweenness centrality indexed by employee ID.
    """
    G = _ensure_graph(graph_or_edges, weight=weight)
    
    if weight and invert_weight_for_distance:
        H = G.copy()
        for u, v, data in H.edges(data=True):
            w = data.get(weight, 1.0)
            data["_dist"] = 1.0 / max(float(w), 1e-6)
        bc = nx.betweenness_centrality(H, weight="_dist", normalized=normalized)
    else:
        bc = nx.betweenness_centrality(G, weight=weight, normalized=normalized)
        
    return pd.Series(bc, name="betweenness_centrality").sort_index()


def calculate_eigenvector_centrality(
    graph_or_edges: Union[nx.Graph, nx.DiGraph, pd.DataFrame, List[Tuple]],
    weight: Optional[str] = "weight",
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> pd.Series:
    """Calculate Eigenvector Centrality (identifies organizational 'Hubs').
    
    Formula:
        lambda * x_v = sum_{u in N(v)} A_{vu} * x_u
        
    Hubs possess deep institutional capital and informal influence by virtue
    of being connected to other highly central associates.
    
    Parameters
    ----------
    graph_or_edges : nx.Graph or DataFrame with [source, target, weight]
    weight : str, default='weight'
    max_iter : int, default=1000
    tol : float, default=1e-6
    
    Returns
    -------
    pd.Series
        Eigenvector centrality scores indexed by employee ID.
    """
    G = _ensure_graph(graph_or_edges, weight=weight)
    try:
        ec = nx.eigenvector_centrality(G, weight=weight, max_iter=max_iter, tol=tol)
    except nx.PowerIterationFailedConvergence:
        # Fallback to numpy spectral decomposition
        ec = nx.eigenvector_centrality_numpy(G, weight=weight)
        
    return pd.Series(ec, name="eigenvector_centrality").sort_index()


def calculate_burt_constraint(
    graph_or_edges: Union[nx.Graph, nx.DiGraph, pd.DataFrame, List[Tuple]],
    weight: Optional[str] = "weight",
) -> pd.Series:
    """Calculate Ronald Burt's Network Constraint Index (Structural Holes).
    
    Formula:
        C_i = sum_{j in N_i} [ p_{ij} + sum_{q in N_i, q != j} (p_{iq} * p_{qj}) ]^2
        where p_{ij} = (z_{ij} + z_{ji}) / sum_k (z_{ik} + z_{ki})
        
    Constraint measures the redundancy of an associate's egocentric network:
    - Low Constraint (< 0.25): High access to structural holes, non-redundant contacts,
      superior innovation access, and elevated career mobility.
    - High Constraint (> 0.50): Insular, redundant network where contacts all communicate
      directly with one another (echo chamber with limited external visibility).
      
    Parameters
    ----------
    graph_or_edges : nx.Graph or DataFrame with [source, target, weight]
    weight : str, default='weight'
        Edge attribute storing interaction intensity.
        
    Returns
    -------
    pd.Series
        Burt's constraint index C_i indexed by employee ID.
    """
    G = _ensure_graph(graph_or_edges, weight=weight)
    nodes = list(G.nodes())
    n = len(nodes)
    if n == 0:
        return pd.Series({}, name="burt_constraint", dtype=float)
    if n == 1:
        return pd.Series({nodes[0]: np.nan}, name="burt_constraint", dtype=float)

    edge_weight = "weight" if isinstance(graph_or_edges, (pd.DataFrame, list)) and weight else weight

    # High-performance sparse matrix formulation of Burt's constraint (O(E) sparse graph ops)
    P = nx.to_scipy_sparse_array(G, weight=edge_weight, dtype=float, format="csr")
    mutual = (P + P.T) if G.is_directed() else P.copy()
    mutual.setdiag(0)
    mutual.eliminate_zeros()

    row_sums = np.array(mutual.sum(axis=1)).flatten()
    nonzero = row_sums > 0

    inv_sums = np.zeros_like(row_sums)
    inv_sums[nonzero] = 1.0 / row_sums[nonzero]

    D_inv = sp.diags(inv_sums)
    P_norm = D_inv @ mutual
    P2 = P_norm @ P_norm

    M = P_norm + P2

    # Restrict local constraints to immediate neighbors (mutual > 0)
    adj_bin = mutual.copy()
    adj_bin.data = np.ones_like(adj_bin.data)

    M_neighbors = M.multiply(adj_bin)
    M_sq = M_neighbors.multiply(M_neighbors)
    c_vals = np.array(M_sq.sum(axis=1)).flatten()
    c_vals[~nonzero] = np.nan

    return pd.Series(c_vals, index=nodes, name="burt_constraint", dtype=float).sort_index()


def calculate_euler_centrality(
    graph_or_edges: Union[nx.Graph, nx.DiGraph, pd.DataFrame, List[Tuple]],
    scale: Optional[float] = None,
    regularization: float = 1e-4,
) -> pd.Series:
    """Calculate Euler Centrality via Leinster Metric Space Magnitude.
    
    Formula:
        Z_{ij} = exp(-c * d(i, j))
        Z * w = 1  ==>  w = (Z + reg * I)^(-1) * 1
        
    In Leinster magnitude theory, the Euler characteristic of a graph or
    similarity space measures the effective number of independent vertices.
    The individual weight w_i (Euler Centrality) quantifies an associate's
    global contribution to structural distinctiveness across the entire enterprise.
    
    - High Euler Centrality: Associate occupies a structurally unique position in the
      global organizational topology.
    - Low Euler Centrality: Associate's structural position is redundant with peers.
    
    Parameters
    ----------
    graph_or_edges : nx.Graph or DataFrame
    scale : float, optional
        Decay constant c. If None, set to 1 / mean_geodesic_distance.
    regularization : float, default=1e-4
        Tikhonov ridge regularization parameter for numerical stability.
        
    Returns
    -------
    pd.Series
        Euler centrality weights indexed by employee ID.
    """
    G = _ensure_graph(graph_or_edges)
    nodes = list(G.nodes())
    n = len(nodes)
    
    if n == 0:
        return pd.Series(dtype=float, name="euler_centrality")
    if n == 1:
        return pd.Series({nodes[0]: 1.0}, name="euler_centrality")
        
    # All-pairs shortest path length
    path_lengths = dict(nx.all_pairs_shortest_path_length(G))
    D = np.full((n, n), np.nan, dtype=float)
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    for u, paths in path_lengths.items():
        i = node_to_idx[u]
        for v, dist in paths.items():
            j = node_to_idx[v]
            D[i, j] = dist
            
    # For disconnected components, set distance to 2 * max observed distance
    finite_mask = ~np.isnan(D)
    max_d = np.max(D[finite_mask]) if np.any(finite_mask) else 1.0
    D[np.isnan(D)] = max_d * 2.0
    
    if scale is None:
        mean_d = np.mean(D)
        scale = 1.0 / max(mean_d, 0.5)
        
    Z = np.exp(-scale * D)
    
    # Solve regularized system (Z + lambda * I) * w = 1
    I = np.eye(n)
    ones = np.ones(n)
    try:
        w = np.linalg.solve(Z + regularization * I, ones)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(Z + regularization * I, ones, rcond=None)[0]
        
    return pd.Series(w, index=nodes, name="euler_centrality").sort_index()


def calculate_collaboration_overload(
    graph_or_edges: Union[nx.Graph, nx.DiGraph, pd.DataFrame, List[Tuple]],
    weight: Optional[str] = "weight",
    threshold_z: float = 2.0,
) -> pd.DataFrame:
    """Calculate Collaboration Overload indices to flag communication burnout.
    
    Computes total interaction degree volume, in-degree demand, and standardizes
    into Z-scores to isolate organizational bottlenecks at risk of exhaustion.
    
    Parameters
    ----------
    graph_or_edges : nx.Graph or DataFrame
    weight : str, default='weight'
    threshold_z : float, default=2.0
        Standard deviations above mean to flag critical overload.
        
    Returns
    -------
    pd.DataFrame
        Columns: [total_interaction_volume, overload_z_score, is_overloaded]
    """
    G = _ensure_graph(graph_or_edges, weight=weight, directed=True)
    nodes = list(G.nodes())
    
    volumes = {}
    for node in nodes:
        # Sum of incident weights
        if G.is_directed():
            in_vol = sum(data.get(weight, 1.0) for _, _, data in G.in_edges(node, data=True))
            out_vol = sum(data.get(weight, 1.0) for _, _, data in G.out_edges(node, data=True))
            tot_vol = in_vol + out_vol
        else:
            tot_vol = sum(data.get(weight, 1.0) for _, _, data in G.edges(node, data=True))
            in_vol = tot_vol
        volumes[node] = tot_vol
        
    df = pd.DataFrame(index=nodes)
    df["interaction_volume"] = pd.Series(volumes)
    
    mean_v = df["interaction_volume"].mean()
    std_v = df["interaction_volume"].std(ddof=1) if len(df) > 1 else 1.0
    if std_v == 0 or np.isnan(std_v):
        std_v = 1.0
        
    df["overload_z_score"] = (df["interaction_volume"] - mean_v) / std_v
    df["is_overloaded"] = df["overload_z_score"] >= threshold_z
    return df.sort_index()


def simulate_attrition_contagion(
    graph_or_edges: Union[nx.Graph, nx.DiGraph, pd.DataFrame, List[Tuple]],
    departed_node: str,
    weight: Optional[str] = "weight",
    contagion_base_rate: float = 0.50,
    distance_decay: float = 0.60,
    max_hops: int = 2,
) -> pd.DataFrame:
    """Simulate flight risk contagion propagation following an employee departure.
    
    Models how the sudden departure of a key hub, broker, or team member spreads
    operational strain and voluntary turnover contagion through the network.
    
    Formula:
        Contagion_Risk_j = Base_Rate * Relative_Tie_Strength_{j, departed} * (Decay)^d
        
    Parameters
    ----------
    graph_or_edges : nx.Graph or DataFrame
    departed_node : str
        ID of employee who has resigned or is modeled as departing.
    weight : str, default='weight'
    contagion_base_rate : float, default=0.50
        Maximum additional flight risk delta on direct collaborators.
    distance_decay : float, default=0.60
        Attenuation multiplier per geodesic graph hop.
    max_hops : int, default=2
        Maximum propagation distance.
        
    Returns
    -------
    pd.DataFrame
        Columns: [distance_to_departed, tie_strength_to_departed, contagion_risk_delta]
    """
    G = _ensure_graph(graph_or_edges, weight=weight)
    if departed_node not in G:
        raise ValueError(f"Departed node '{departed_node}' not found in organizational graph.")
        
    shortest_paths = nx.single_source_shortest_path_length(G, departed_node, cutoff=max_hops)
    
    results = []
    for node, hop in shortest_paths.items():
        if node == departed_node:
            continue
            
        # Tie strength to departed
        tot_w = sum(d.get(weight, 1.0) for _, _, d in G.edges(node, data=True))
        if G.has_edge(node, departed_node):
            edge_data = G.get_edge_data(node, departed_node)
            w = float(edge_data.get(weight, 1.0)) if weight else 1.0
            rel_strength = w / max(tot_w, 1e-6)
        else:
            # Indirect tie strength mediated through mutual neighbors
            common = list(nx.common_neighbors(G, node, departed_node)) if not G.is_directed() else []
            if common:
                indirect_vals = []
                for m in common:
                    w_nm = float(G.get_edge_data(node, m).get(weight, 1.0)) if weight else 1.0
                    w_md = float(G.get_edge_data(m, departed_node).get(weight, 1.0)) if weight else 1.0
                    tot_m = sum(d.get(weight, 1.0) for _, _, d in G.edges(m, data=True))
                    indirect_vals.append((w_nm / max(tot_w, 1e-6)) * (w_md / max(tot_m, 1e-6)))
                rel_strength = max(indirect_vals) if indirect_vals else 0.02
            else:
                rel_strength = 0.02
            
        decay = distance_decay ** hop
        risk_delta = contagion_base_rate * rel_strength * decay
        
        results.append({
            "employee_id": node,
            "hop_distance": hop,
            "relative_tie_to_departed": round(rel_strength, 3),
            "contagion_risk_delta": round(risk_delta, 4),
        })
        
    res_df = pd.DataFrame(results)
    if not res_df.empty:
        res_df = res_df.sort_values(by="contagion_risk_delta", ascending=False).reset_index(drop=True)
    return res_df


def compute_complete_ona_profile(
    graph_or_edges: Union[nx.Graph, nx.DiGraph, pd.DataFrame, List[Tuple]],
    weight: Optional[str] = "weight",
) -> pd.DataFrame:
    """Compute an end-to-end ONA capital profile across all enterprise employees.
    
    Assembles betweenness centrality, eigenvector centrality, Burt's constraint,
    Euler centrality, collaboration overload, and classifies HCM network archetypes:
    - 'Broker / Bridge' (High betweenness, low constraint)
    - 'Informal Hub' (High eigenvector, central connector)
    - 'Cohesive Core' (High constraint, high degree)
    - 'Autonomous Specialist' (Low constraint, moderate degree)
    - 'Peripheral' (Low degree, elevated constraint)
    
    Returns
    -------
    pd.DataFrame
        Complete multidimensional ONA metrics with structural archetype labels.
    """
    G = _ensure_graph(graph_or_edges, weight=weight)
    
    deg = calculate_degree_centrality(G, weight=weight)
    cc = calculate_closeness_centrality(G, weight=weight)
    bc = calculate_betweenness_centrality(G, weight=weight)
    ec = calculate_eigenvector_centrality(G, weight=weight)
    constraint = calculate_burt_constraint(G, weight=weight)
    euler = calculate_euler_centrality(G)
    overload = calculate_collaboration_overload(G, weight=weight)
    
    df = pd.DataFrame({
        "degree_centrality": deg,
        "closeness_centrality": cc,
        "betweenness_centrality": bc,
        "eigenvector_centrality": ec,
        "burt_constraint": constraint,
        "euler_centrality": euler,
        "interaction_volume": overload["interaction_volume"],
        "overload_z_score": overload["overload_z_score"],
        "is_overloaded": overload["is_overloaded"],
    })
    
    # Archetype classification
    bc_p75 = df["betweenness_centrality"].quantile(0.75)
    ec_p75 = df["eigenvector_centrality"].quantile(0.75)
    c_p50 = df["burt_constraint"].median()
    
    archetypes = []
    for _, row in df.iterrows():
        b = row["betweenness_centrality"]
        e = row["eigenvector_centrality"]
        c = row["burt_constraint"]
        
        if b >= bc_p75 and (c < c_p50 or np.isnan(c)):
            archetypes.append("Broker / Bridge")
        elif e >= ec_p75:
            archetypes.append("Informal Hub")
        elif c >= c_p50 and row["overload_z_score"] >= 0:
            archetypes.append("Cohesive Core")
        elif c < c_p50:
            archetypes.append("Autonomous Specialist")
        else:
            archetypes.append("Peripheral")
            
    df["network_archetype"] = archetypes
    return df


def calculate_span_of_control(
    hierarchy_or_edges: Union[pd.DataFrame, nx.DiGraph, List[Tuple[str, str]]],
    employee_col: str = "employee_id",
    manager_col: str = "manager_id",
    weekly_manager_budget_hours: float = 10.0,
    optimal_range: Tuple[int, int] = (4, 10),
    critical_threshold: int = 15,
) -> pd.DataFrame:
    """Calculate Span of Control (Managerial Load) and Associate Attention Dilution.

    Quantifies the absolute number of direct reports assigned to each leader,
    evaluates organizational layers against optimal managerial load benchmarks,
    and estimates the resulting 1-on-1 attention dilution experienced by associates.

    Formulas:
        Span(m) = sum_{e in E} I(manager(e) == m)
        Assigned_Span(e) = Span(manager(e))
        Weekly_1on1_Minutes(e) = (weekly_budget_hours * 60) / max(Assigned_Span(e), 1)
        Attention_Dilution_Score(e) = 1 / (1 + exp(-(Assigned_Span(e) - optimal_max) / 2.5))

    Parameters
    ----------
    hierarchy_or_edges : pd.DataFrame, nx.DiGraph, or List of (manager, report) / (report, manager)
        Reporting hierarchy data. If DataFrame, requires `employee_col` and `manager_col`.
    employee_col : str, default='employee_id'
        Column name for employee ID.
    manager_col : str, default='manager_id'
        Column name for manager ID.
    weekly_manager_budget_hours : float, default=10.0
        Total weekly hours an average manager allocates to direct report 1-on-1s and coaching.
    optimal_range : tuple of (int, int), default=(4, 10)
        Healthy lower and upper bounds for knowledge-worker and operational manager spans.
    critical_threshold : int, default=15
        Span threshold above which managerial dilution causes severe operational burnout.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by employee_id with columns:
        - manager_id: Direct manager ID
        - is_manager: Boolean indicator whether employee manages direct reports
        - direct_reports_count: Number of direct reports reporting to this employee
        - manager_span: Direct reports count of this employee's direct manager
        - managerial_load_category: Classification of manager load ('Under-leveraged', 'Optimal', 'Stretched', 'Critical Overload')
        - est_weekly_1on1_minutes: Estimated dedicated 1-on-1 coaching minutes per associate
        - attention_dilution_score: Continuous risk index in [0, 1] of managerial dilution
    """
    if isinstance(hierarchy_or_edges, pd.DataFrame):
        df = hierarchy_or_edges.copy()
        if employee_col not in df.columns or manager_col not in df.columns:
            raise ValueError(f"DataFrame must contain '{employee_col}' and '{manager_col}' columns.")
        records = df[[employee_col, manager_col]].dropna(subset=[employee_col]).copy()
    elif isinstance(hierarchy_or_edges, nx.DiGraph):
        # Assume edges directed from manager -> report or report -> manager
        # If out-degree > in-degree on roots, it is manager -> report
        records_list = []
        for u, v in hierarchy_or_edges.edges():
            records_list.append({employee_col: v, manager_col: u})
        records = pd.DataFrame(records_list)
    elif isinstance(hierarchy_or_edges, list):
        records = pd.DataFrame(hierarchy_or_edges, columns=[manager_col, employee_col])
    else:
        raise TypeError(f"Unsupported hierarchy input type: {type(hierarchy_or_edges)}")

    def _clean_id(val: Any) -> Optional[str]:
        if val is None or pd.isna(val):
            return None
        s = str(val).strip()
        if s in ("None", "nan", "<NA>", ""):
            return None
        return s

    records[employee_col] = records[employee_col].apply(_clean_id)
    records = records[records[employee_col].notna()].copy()
    records[manager_col] = records[manager_col].apply(_clean_id)

    # All unique individuals appearing as employees or managers
    emp_ids = set(records[employee_col].dropna())
    mgr_ids = set(records[manager_col].dropna())
    all_employees = sorted(list(emp_ids | mgr_ids))

    # Calculate direct reports for every individual
    valid_mgr_records = records[records[manager_col].isin(all_employees)]
    span_counts = valid_mgr_records.groupby(manager_col)[employee_col].nunique().to_dict()

    # Build employee -> manager mapping
    emp_to_mgr = dict(zip(records[employee_col], records[manager_col]))

    res_rows = []
    budget_minutes = float(weekly_manager_budget_hours) * 60.0
    opt_min, opt_max = optimal_range

    for emp in all_employees:
        mgr = _clean_id(emp_to_mgr.get(emp, None))

        reports_cnt = int(span_counts.get(emp, 0))
        is_mgr = reports_cnt > 0

        # Classification for managers
        if is_mgr:
            if reports_cnt < opt_min:
                load_cat = "Under-leveraged"
            elif reports_cnt <= opt_max:
                load_cat = "Optimal"
            elif reports_cnt <= critical_threshold:
                load_cat = "Stretched"
            else:
                load_cat = "Critical Overload"
        else:
            load_cat = "Individual Contributor"

        # Attention metrics from direct manager
        if mgr is not None:
            m_span = int(span_counts.get(mgr, 1))
            weekly_1on1 = round(budget_minutes / max(m_span, 1), 1)
            # Continuous sigmoidal dilution score
            dilution = 1.0 / (1.0 + np.exp(-(m_span - opt_max) / 2.5))
            dilution = round(float(dilution), 4)
        else:
            m_span = np.nan
            weekly_1on1 = np.nan
            dilution = np.nan

        res_rows.append({
            "employee_id": emp,
            "manager_id": mgr,
            "is_manager": is_mgr,
            "direct_reports_count": reports_cnt,
            "manager_span": m_span,
            "managerial_load_category": load_cat,
            "est_weekly_1on1_minutes": weekly_1on1,
            "attention_dilution_score": dilution,
        })

    out_df = pd.DataFrame(res_rows).set_index("employee_id").sort_index()
    return out_df


def evaluate_span_reorganization_shock(
    df_pre: pd.DataFrame,
    df_post: pd.DataFrame,
    employee_col: str = "employee_id",
    manager_col: str = "manager_id",
    base_flight_risk_col: Optional[str] = None,
    shock_sensitivity: float = 0.15,
) -> pd.DataFrame:
    """Evaluate associate flight risk surge induced by managerial reorganization shock.

    When an organizational restructuring abruptly surges a manager's span of control
    (e.g., from 8 to 25 direct reports), direct reports experience acute managerial
    attention starvation, triggering flight risk increases independent of individual performance.

    Formulas:
        Span_Delta(e) = Post_Manager_Span(e) - Pre_Manager_Span(e)
        Expansion_Factor(e) = max(0, Span_Delta(e) / max(Pre_Manager_Span(e), 4))
        Overload_Excess(e) = max(0, Post_Manager_Span(e) - 10)
        Delta_Flight_Risk(e) = shock_sensitivity * (Overload_Excess / 15.0) * (1.0 + 0.5 * Expansion_Factor)
        Post_Reorg_Flight_Risk(e) = clip(Base_Flight_Risk(e) + Delta_Flight_Risk(e), 0.0, 1.0)

    Parameters
    ----------
    df_pre : pd.DataFrame
        Baseline reporting hierarchy before reorganization.
    df_post : pd.DataFrame
        Updated reporting hierarchy after reorganization.
    employee_col : str, default='employee_id'
        Column name for employee ID.
    manager_col : str, default='manager_id'
        Column name for manager ID.
    base_flight_risk_col : str, optional
        Column name in df_pre containing existing baseline flight risk probabilities.
    shock_sensitivity : float, default=0.15
        Scaling parameter controlling sensitivity of flight risk surge to span expansion.

    Returns
    -------
    pd.DataFrame
        Comparative reorganization impact evaluation containing:
        - prev_manager_id, new_manager_id, manager_changed
        - prev_manager_span, new_manager_span, span_delta
        - prev_weekly_1on1_minutes, new_weekly_1on1_minutes, minutes_lost
        - reorg_shock_flight_risk_delta
        - post_reorg_flight_risk (if base_flight_risk_col provided)
    """
    pre_metrics = calculate_span_of_control(df_pre, employee_col=employee_col, manager_col=manager_col)
    post_metrics = calculate_span_of_control(df_post, employee_col=employee_col, manager_col=manager_col)

    # Identify common employees
    common_emps = sorted(list(set(pre_metrics.index).intersection(set(post_metrics.index))))

    rows = []
    base_risk_map = {}
    if base_flight_risk_col and base_flight_risk_col in df_pre.columns:
        base_risk_map = dict(zip(df_pre[employee_col].astype(str), df_pre[base_flight_risk_col]))

    for emp in common_emps:
        r_pre = pre_metrics.loc[emp]
        r_post = post_metrics.loc[emp]

        p_mgr = r_pre["manager_id"]
        n_mgr = r_post["manager_id"]
        if pd.isna(p_mgr) and pd.isna(n_mgr):
            mgr_changed = False
        else:
            mgr_changed = (p_mgr != n_mgr)

        p_span = r_pre["manager_span"]
        n_span = r_post["manager_span"]

        if pd.isna(n_span) or pd.isna(p_span):
            span_delta = np.nan
            minutes_lost = np.nan
            risk_delta = 0.0
        else:
            span_delta = float(n_span - p_span)
            minutes_lost = float(r_pre["est_weekly_1on1_minutes"] - r_post["est_weekly_1on1_minutes"])

            # Shock formulation
            overload_excess = max(0.0, float(n_span) - 10.0)
            expansion_factor = max(0.0, span_delta / max(float(p_span), 4.0))

            if overload_excess > 0.0 and span_delta > 0:
                raw_shock = shock_sensitivity * (overload_excess / 15.0) * (1.0 + 0.5 * expansion_factor)
                risk_delta = float(np.clip(raw_shock, 0.0, 0.45))
            else:
                risk_delta = 0.0

        item = {
            "employee_id": emp,
            "prev_manager_id": p_mgr,
            "new_manager_id": n_mgr,
            "manager_changed": mgr_changed,
            "prev_manager_span": p_span,
            "new_manager_span": n_span,
            "span_delta": span_delta,
            "prev_weekly_1on1_minutes": r_pre["est_weekly_1on1_minutes"],
            "new_weekly_1on1_minutes": r_post["est_weekly_1on1_minutes"],
            "weekly_1on1_minutes_lost": minutes_lost,
            "reorg_shock_flight_risk_delta": round(risk_delta, 4),
        }

        if base_flight_risk_col:
            base_risk = float(base_risk_map.get(emp, 0.15))
            post_risk = float(np.clip(base_risk + risk_delta, 0.0, 1.0))
            item["base_flight_risk"] = round(base_risk, 4)
            item["post_reorg_flight_risk"] = round(post_risk, 4)

        rows.append(item)

    res_df = pd.DataFrame(rows).set_index("employee_id")
    return res_df.sort_values(by="reorg_shock_flight_risk_delta", ascending=False)


# API Consistency Aliases: compute_* <=> calculate_*
compute_degree_centrality = calculate_degree_centrality
compute_closeness_centrality = calculate_closeness_centrality
compute_betweenness_centrality = calculate_betweenness_centrality
compute_eigenvector_centrality = calculate_eigenvector_centrality
compute_burt_constraint = calculate_burt_constraint
compute_euler_centrality = calculate_euler_centrality
compute_collaboration_overload = calculate_collaboration_overload
compute_span_of_control = calculate_span_of_control
calculate_complete_ona_profile = deprecated_alias(
    compute_complete_ona_profile,
    "calculate_complete_ona_profile",
    "compute_complete_ona_profile",
)
