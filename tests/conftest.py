"""Shared pytest fixtures for the People Analytics Toolkit test suite.

Provides standardized, seeded synthetic datasets for feature engineering, causal inference,
network analysis, survival modeling, and fairness auditing tests.
"""

import pytest
import pandas as pd
import numpy as np

from data.synthetic_generators import (
    generate_store_roster_data,
    generate_cohort_timeseries_overtime,
    generate_compensation_equity_data,
    generate_algorithmic_equity_and_fairness_data,
    generate_causal_and_uplift_data,
    generate_panel_policy_data,
    generate_organizational_network_data,
    generate_career_trajectory_event_log,
    generate_heterogeneous_hcm_multigraph,
    generate_reporting_hierarchy_with_reorg,
    generate_time_in_position_workforce,
    generate_bradford_factor_workforce,
    generate_retail_scheduling_pilot_data,
    generate_overtime_turnover_causal_dag_data,
    generate_store_department_labor_panel,
    generate_daily_store_foot_traffic,
    generate_store_profiles_multivariate,
    generate_career_movement_dataset,
    generate_turnover_inflection_dataset,
)


@pytest.fixture(scope="session")
def store_roster_df() -> pd.DataFrame:
    """Synthetic workforce roster across stores and departments."""
    return generate_store_roster_data(n_employees=500, seed=42)


@pytest.fixture(scope="session")
def cohort_timeseries_df() -> pd.DataFrame:
    """Overtime and labor hour weekly timeseries for cohort memory modeling."""
    return generate_cohort_timeseries_overtime(n_weeks=52, seed=42)


@pytest.fixture(scope="session")
def compensation_equity_df() -> pd.DataFrame:
    """Compensation equity dataset with salary grades, compa-ratio, and range penetration."""
    return generate_compensation_equity_data(n_employees=400, seed=42)


@pytest.fixture(scope="session")
def algorithmic_fairness_df() -> pd.DataFrame:
    """Workforce dataset with sensitive demographic attributes for algorithmic fairness auditing."""
    return generate_algorithmic_equity_and_fairness_data(n_employees=600, seed=42)


@pytest.fixture(scope="session")
def causal_uplift_df() -> pd.DataFrame:
    """Observational cohort for AIPW, Causal Forest, and Uplift Modeling."""
    return generate_causal_and_uplift_data(n_employees=800, seed=42)


@pytest.fixture(scope="session")
def panel_policy_df() -> pd.DataFrame:
    """Location x quarter panel data for Synthetic Difference-in-Differences policy evaluation."""
    return generate_panel_policy_data(
        n_units=12,
        n_periods=10,
        treated_units=("Office_North",),
        post_period_start=7,
        seed=42,
    )


@pytest.fixture(scope="session")
def org_network_data():
    """Tuple of (nodes_df, edges_df, NetworkX graph) for ONA centrality and structural hole metrics."""
    return generate_organizational_network_data(n_employees=100, seed=42)


@pytest.fixture(scope="session")
def career_event_log() -> pd.DataFrame:
    """Career trajectory transition log for Continuous-Time Markov Chains (CTMC)."""
    return generate_career_trajectory_event_log(n_employees=400, max_years=6.0, seed=42)


@pytest.fixture(scope="session")
def heterogeneous_multigraph_data():
    """Heterogeneous multigraph, nodes_df, edges_df for Metapath2Vec representation learning."""
    return generate_heterogeneous_hcm_multigraph(n_employees=80, n_projects=15, n_skills=20, seed=42)


@pytest.fixture(scope="session")
def hierarchy_reorg_data():
    """Tuple of (df_pre, df_post) reporting hierarchy snapshots for span of control shocks."""
    return generate_reporting_hierarchy_with_reorg(seed=42)


@pytest.fixture(scope="session")
def time_in_pos_df() -> pd.DataFrame:
    """Workforce tenure and role tenure data for Stagnation Index calculation."""
    return generate_time_in_position_workforce(n_employees=300, seed=42)


@pytest.fixture(scope="session")
def bradford_workforce_data():
    """Tuple of (spell_df, daily_attendance_df) for Bradford Factor friction analysis."""
    return generate_bradford_factor_workforce(n_employees=50, window_weeks=52, seed=42)


@pytest.fixture(scope="session")
def retail_scheduling_pilot_df() -> pd.DataFrame:
    """Store-level weekly panel data for synthetic twin optimization."""
    return generate_retail_scheduling_pilot_data(n_pilot_stores=3, n_donor_stores=15, n_weeks=40, seed=42)


@pytest.fixture(scope="session")
def causal_dag_df() -> pd.DataFrame:
    """Overtime and turnover structural equation dataset for NOTEARS DAG causal discovery."""
    return generate_overtime_turnover_causal_dag_data(n_samples=600, seed=42)


@pytest.fixture(scope="session")
def labor_panel_df() -> pd.DataFrame:
    """Multi-horizon Store x Department labor panel for configuration-driven rolling metrics."""
    return generate_store_department_labor_panel(n_stores=3, n_depts=2, n_weeks=55, seed=42)


@pytest.fixture(scope="session")
def foot_traffic_df() -> pd.DataFrame:
    """Daily store foot traffic series for Singular Spectrum Analysis (SSA) smoothing."""
    return generate_daily_store_foot_traffic(n_days=90, n_stores=2, seed=42)


@pytest.fixture(scope="session")
def multivariate_store_profiles() -> pd.DataFrame:
    """Multivariate store KPI profile dataframe for Mahalanobis and LOF anomaly detection."""
    return generate_store_profiles_multivariate(n_stores=200, seed=42)


@pytest.fixture(scope="session")
def career_movement_data():
    """Tuple of (nodes_df, edges_df, G) for CareerMovementEmbeddings."""
    return generate_career_movement_dataset(n_roles=15, seed=42)


@pytest.fixture(scope="session")
def turnover_inflection_data():
    """Tuple of (df, X, y, model, shap_values) for SHAP tipping point threshold detection."""
    return generate_turnover_inflection_dataset(n_samples=400, seed=42)
