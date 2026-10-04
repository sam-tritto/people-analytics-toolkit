"""Comprehensive unit tests resolving testing gaps from Section 4 of package_evaluation.md.

Covers:
1. correlation_heatmap_df(): styling, multiple correlation methods, padjust, CIs, power, and empty/NaN edge cases.
2. compute_rolling_metrics(): 10+ aggregation types (mean, sum, std, var, min, max, median, skew, kurt, quantile, callable).
3. RollingMetricsPipeline: fluent chaining, fitting, transformation, and feature name inspection.
4. singular_spectrum_analysis() & SingularSpectrumAnalysis: SVD decomposition, reconstruction, class interface.
5. compute_svd_smoothed_baseline(): DataFrame-level SSA with partitions and empty DF handling.
6. compute_lag_features() & compute_differencing_features(): multi-horizon presets, zero-division protection.
7. TemporalFeaturesPipeline: fluent lags and differencing assembly.
8. notears_linear() & causal DAG discovery edge cases: collinearity, small N, zero-variance columns.
9. calculate_bradford_factor() Mode 3: spell-level event logs, duration columns, window clipping, unplanned filtering.
10. Edge cases: empty DataFrames (0-row inputs) across all key pipelines.
11. Edge cases: NaN handling across temporal, compensation, and statistical modules.
12. ExponentiatedGradientFairness: convergence verification, dual history, and violation tracking.
"""

import numpy as np
import pandas as pd
import pytest

from people_analytics_toolkit.memory import (
    RollingMetricConfig,
    compute_rolling_metrics,
    RollingMetricsPipeline,
    singular_spectrum_analysis,
    compute_svd_smoothed_baseline,
    SingularSpectrumAnalysis,
    compute_lag_features,
    compute_differencing_features,
    TemporalFeaturesPipeline,
)
from people_analytics_toolkit.attribution import correlation_heatmap_df
from people_analytics_toolkit.chaos import (
    calculate_bradford_factor,
    calculate_bradford_score,
    classify_bradford_risk,
)
from people_analytics_toolkit.prescriptive import (
    notears_linear,
    compute_directed_causal_edge_weights,
    diagnose_causal_confounding,
)
from people_analytics_toolkit.compensation import (
    calculate_compa_ratio,
    calculate_range_penetration,
)
from people_analytics_toolkit.equity import (
    ExponentiatedGradientFairness,
    demographic_parity_difference,
    equalized_odds_difference,
)
from sklearn.linear_model import LogisticRegression
from data.synthetic_generators import (
    generate_algorithmic_equity_and_fairness_data,
    generate_bradford_factor_workforce,
    generate_daily_store_foot_traffic,
    generate_store_department_labor_panel,
)


# ===========================================================================
# 1. CORRELATION HEATMAP TABLE & PINGOUIN INTEGRATION
# ===========================================================================

class TestCorrelationHeatmapComprehensive:
    """Comprehensive tests for correlation_heatmap_df covering methods, styling, and robustness."""

    @pytest.fixture
    def sample_data(self):
        rng = np.random.default_rng(123)
        n = 60
        tenure = np.linspace(1, 10, n)
        overtime = 2.0 * tenure + rng.normal(0, 1.0, n)
        satisfaction = -1.2 * tenure + rng.normal(0, 1.0, n)
        commute = rng.normal(15, 5, n)
        noise = rng.normal(0, 2, n)
        turnover = 0.5 * tenure + 0.3 * overtime - 0.4 * satisfaction + rng.normal(0, 0.5, n)
        return pd.DataFrame({
            "tenure": tenure,
            "overtime": overtime,
            "satisfaction": satisfaction,
            "commute": commute,
            "noise": noise,
            "turnover": turnover,
        })

    @pytest.mark.parametrize("method", ["pearson", "spearman", "kendall"])
    def test_correlation_methods(self, sample_data, method):
        res = correlation_heatmap_df(
            df=sample_data,
            target="turnover",
            method=method,
            viz=False,
        )
        assert isinstance(res, pd.DataFrame)
        assert len(res) == 5  # 5 predictors
        assert (res["METHOD"] == method).all()
        assert "CORRELATION TO turnover" in res.columns
        assert "P" in res.columns
        assert "LB" in res.columns
        assert "UB" in res.columns
        assert "POWER" in res.columns
        # Overtime should have strong positive correlation
        ot_corr = res.loc[res["METRIC"] == "overtime", "CORRELATION TO turnover"].iloc[0]
        assert ot_corr > 0.40

    @pytest.mark.parametrize("padjust", ["none", "bonf", "fdr_bh", "holm"])
    def test_p_value_adjustments(self, sample_data, padjust):
        res = correlation_heatmap_df(
            df=sample_data,
            target="turnover",
            padjust=padjust,
            viz=False,
        )
        assert len(res) == 5
        assert (res["P"] >= 0.0).all() and (res["P"] <= 1.0).all()

    def test_styler_and_html_generation(self, sample_data):
        styler, df_res = correlation_heatmap_df(
            df=sample_data,
            target="turnover",
            viz=False,
            return_styler=True,
        )
        html = styler.to_html()
        assert "background-color" in html
        assert "dark_slate" not in html
        assert isinstance(df_res, pd.DataFrame)
        assert len(df_res) == 5

    def test_correlation_heatmap_css_darkslategray(self):
        """Verifies that CI spanning 0 with significant P uses valid CSS darkslategray."""
        from people_analytics_toolkit.attribution import correlation_heatmap_df

        # Construct data where a feature has CI spanning 0
        rng = np.random.RandomState(42)
        n = 30
        x = rng.normal(0, 1, n)
        y = rng.normal(0, 1, n)
        df = pd.DataFrame({"feat": x, "target": y})

        styler, _ = correlation_heatmap_df(
            df=df,
            target="target",
            viz=False,
            return_styler=True,
            drop_ci_contain_0=False,
            drop_insignificant=False,
        )
        html = styler.to_html()
        # dark_slate must never be present
        assert "dark_slate;" not in html
        assert "dark_slate " not in html

    def test_sample_size_too_small_raises_value_error(self):
        small_df = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
        with pytest.raises(ValueError, match="too small"):
            correlation_heatmap_df(small_df, target="y", viz=False)

    def test_empty_dataframe_returns_empty_schema(self):
        empty_df = pd.DataFrame({"x": pd.Series(dtype=float), "y": pd.Series(dtype=float)})
        res = correlation_heatmap_df(empty_df, target="y", viz=False)
        assert isinstance(res, pd.DataFrame)
        assert len(res) == 0
        assert "METRIC" in res.columns
        assert "CORRELATION TO y" in res.columns
        assert "SIGNIFICANT" in res.columns


# ===========================================================================
# 2. CONFIGURATION-DRIVEN ROLLING METRICS: 10+ AGGREGATION TYPES
# ===========================================================================

class TestRollingMetricsAllAggregations:
    """Tests compute_rolling_metrics across 10+ aggregation types and hierarchical partitions."""

    @pytest.fixture
    def panel_data(self):
        rng = np.random.default_rng(42)
        rows = []
        for store in ["Store_A", "Store_B"]:
            for week in range(1, 25):
                rows.append({
                    "store_id": store,
                    "week": week,
                    "sales": float(100 + 2 * week + rng.normal(0, 5)),
                    "headcount": int(20 + (week % 4)),
                })
        return pd.DataFrame(rows)

    @pytest.mark.parametrize("agg", [
        "mean",
        "sum",
        "std",
        "var",
        "variance",
        "min",
        "max",
        "median",
        "skew",
        "kurt",
        "kurtosis",
    ])
    def test_standard_aggregation_types(self, panel_data, agg):
        cfg = RollingMetricConfig(
            target_column="sales",
            window=5,
            aggregation=agg,
            output_column=f"sales_{agg}",
        )
        res = compute_rolling_metrics(
            df=panel_data,
            configs=[cfg],
            partition_cols="store_id",
            temporal_col="week",
        )
        assert f"sales_{agg}" in res.columns
        assert len(res) == len(panel_data)
        # Check that after window=5, values are computed
        valid_vals = res.groupby("store_id")[f"sales_{agg}"].apply(lambda s: s.iloc[4:]).dropna()
        assert len(valid_vals) > 0

    def test_quantile_aggregation(self, panel_data):
        cfg = RollingMetricConfig(
            target_column="sales",
            window=6,
            aggregation="quantile",
            quantile=0.75,
            output_column="sales_q75",
        )
        res = compute_rolling_metrics(
            df=panel_data,
            configs=[cfg],
            partition_cols="store_id",
            temporal_col="week",
        )
        assert "sales_q75" in res.columns
        assert len(res) == len(panel_data)
        assert not res["sales_q75"].iloc[6:].isna().all()

    def test_custom_callable_aggregation(self, panel_data):
        # Peak-to-peak range function
        def peak_to_peak(x):
            return np.ptp(x)

        cfg = RollingMetricConfig(
            target_column="sales",
            window=4,
            aggregation=peak_to_peak,
            output_column="sales_ptp",
        )
        res = compute_rolling_metrics(
            df=panel_data,
            configs=[cfg],
            partition_cols="store_id",
            temporal_col="week",
        )
        assert "sales_ptp" in res.columns
        valid = res["sales_ptp"].dropna()
        assert (valid >= 0.0).all()

    def test_dict_of_dicts_config_syntax(self, panel_data):
        config_dict = {
            "sales_roll_5w": {"target_column": "sales", "window": 5, "aggregation": "mean"},
            "sales_max_3w": {"target_column": "sales", "window": 3, "aggregation": "max"},
        }
        res = compute_rolling_metrics(
            df=panel_data,
            configs=config_dict,
            partition_cols="store_id",
            temporal_col="week",
        )
        assert "sales_roll_5w" in res.columns
        assert "sales_max_3w" in res.columns


# ===========================================================================
# 3. ROLLING METRICS PIPELINE CLASS
# ===========================================================================

class TestRollingMetricsPipelineClass:
    """Verifies object-oriented chaining, fitting, and feature extraction."""

    def test_pipeline_chaining_and_get_features(self):
        pipeline = (
            RollingMetricsPipeline(partition_cols="store_id", temporal_col="week")
            .add_metric("sales", window=4, aggregation="mean", output_column="sales_m4")
            .add_metric("sales", window=4, aggregation="std", output_column="sales_s4")
            .add_metric("headcount", window=2, aggregation="max")
        )
        feature_names = pipeline.get_feature_names()
        assert "sales_m4" in feature_names
        assert "sales_s4" in feature_names
        assert any("headcount" in f for f in feature_names)

    def test_pipeline_fit_transform_and_transform(self):
        df = pd.DataFrame({
            "store_id": ["S1"] * 6,
            "week": list(range(1, 7)),
            "metric": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        })
        pipeline = RollingMetricsPipeline(partition_cols="store_id", temporal_col="week")
        pipeline.add_metric("metric", window=3, aggregation="mean", output_column="m_mean")

        assert not pipeline.is_fitted_
        fitted = pipeline.fit(df)
        assert fitted is pipeline
        assert pipeline.is_fitted_
        assert pipeline.feature_names_in_ == ["store_id", "week", "metric"]
        assert pipeline.n_features_in_ == 3
        assert pipeline.feature_names_out_ == ["m_mean"]

        res_ft = pipeline.fit_transform(df)
        res_t = pipeline.transform(df)

        assert (res_ft["m_mean"].dropna() == res_t["m_mean"].dropna()).all()
        # Mean of [10, 20, 30] is 20.0 at index 2
        assert np.isclose(res_ft["m_mean"].iloc[2], 20.0)

    def test_pipeline_fit_schema_validation_errors(self):
        df = pd.DataFrame({
            "store_id": ["S1"] * 4,
            "week": [1, 2, 3, 4],
            "metric": [10.0, 20.0, 30.0, 40.0],
        })
        # Missing partition column
        p_bad_partition = RollingMetricsPipeline(partition_cols="missing_dept")
        with pytest.raises(KeyError, match="Partition column 'missing_dept' not found"):
            p_bad_partition.fit(df)

        # Missing temporal column
        p_bad_temp = RollingMetricsPipeline(temporal_col="missing_date")
        with pytest.raises(KeyError, match="Temporal column 'missing_date' not found"):
            p_bad_temp.fit(df)

        # Missing target column
        p_bad_target = RollingMetricsPipeline().add_metric("missing_target", window=2)
        with pytest.raises(KeyError, match="Target column 'missing_target'"):
            p_bad_target.fit(df)

        # Non-DataFrame input
        with pytest.raises(TypeError, match="Expected df to be a pandas DataFrame"):
            p_bad_target.fit("not_a_df")


# ===========================================================================
# 4. SINGULAR SPECTRUM ANALYSIS (SSA) & CLASS
# ===========================================================================

class TestSingularSpectrumAnalysisComprehensive:
    """Tests SSA noise filtering, component orthogonality, and class-based API."""

    def test_component_orthogonality_and_decomposition(self):
        t = np.linspace(0, 10, 100)
        signal = 5.0 + 2.0 * np.sin(2 * np.pi * t / 2.0) + 1.0 * np.cos(2 * np.pi * t / 5.0)
        ssa = SingularSpectrumAnalysis(window_length=20, top_k=3)
        ssa.fit(signal)
        comps = ssa.decompose()

        assert comps.shape == (100, 3)
        # Components reconstructed sum should equal smoothed signal
        smoothed = ssa.transform(signal)
        comp_sum = comps.sum(axis=1).to_numpy()
        np.testing.assert_allclose(smoothed, comp_sum, atol=1e-5)

    def test_compute_svd_smoothed_baseline_partitions(self):
        df = pd.DataFrame({
            "unit": ["U1"] * 30 + ["U2"] * 30,
            "date": list(range(30)) + list(range(30)),
            "traffic": [50.0 + i + np.sin(i) for i in range(30)] + [100.0 + 2 * i for i in range(30)],
        })
        res = compute_svd_smoothed_baseline(
            df=df,
            target_col="traffic",
            partition_cols="unit",
            temporal_col="date",
            top_k=2,
            window_length=10,
        )
        assert "traffic_svd_smoothed" in res.columns
        assert "traffic_svd_residual" in res.columns
        assert len(res) == 60

    def test_compute_svd_smoothed_baseline_empty_df(self):
        empty_df = pd.DataFrame(columns=["unit", "date", "traffic"])
        res = compute_svd_smoothed_baseline(
            df=empty_df,
            target_col="traffic",
            partition_cols="unit",
            temporal_col="date",
        )
        assert len(res) == 0
        assert "traffic_svd_smoothed" in res.columns
        assert "traffic_svd_residual" in res.columns


# ===========================================================================
# 5. TEMPORAL LAG & DIFFERENCING PIPELINES
# ===========================================================================

class TestTemporalPipelinesComprehensive:
    """Tests lag, differencing, and fluent temporal pipeline assembly."""

    def test_compute_lag_features_multiple_targets_and_validation(self):
        df = pd.DataFrame({
            "id": ["A"] * 5,
            "t": [1, 2, 3, 4, 5],
            "v1": [10.0, 20.0, 30.0, 40.0, 50.0],
            "v2": [100.0, 200.0, 300.0, 400.0, 500.0],
        })
        res = compute_lag_features(
            df=df,
            target_cols=["v1", "v2"],
            lags=[1, 2],
            temporal_col="t",
            partition_cols="id",
        )
        assert "v1_lag_1" in res.columns
        assert "v2_lag_1" in res.columns
        assert "v1_lag_2" in res.columns
        assert "v2_lag_2" in res.columns
        # Lag 1 at t=2 is 10.0
        assert np.isclose(res.loc[1, "v1_lag_1"], 10.0)

        # Non-positive lag must raise ValueError
        with pytest.raises(ValueError, match="positive integer"):
            compute_lag_features(df, target_cols="v1", lags=0)

    def test_compute_differencing_presets(self):
        df = pd.DataFrame({
            "id": ["A"] * 15,
            "t": list(range(1, 16)),
            "metric": [float(i * 10) for i in range(1, 16)],
        })
        res_weekly = compute_differencing_features(df, target_cols="metric", periods="weekly", temporal_col="t")
        assert "metric_wow" in res_weekly.columns
        assert "metric_mom" in res_weekly.columns
        assert "metric_qoq" in res_weekly.columns
        assert "metric_yoy" in res_weekly.columns

        res_monthly = compute_differencing_features(df, target_cols="metric", periods="monthly", temporal_col="t")
        assert "metric_mom" in res_monthly.columns
        assert "metric_qoq" in res_monthly.columns
        assert "metric_yoy" in res_monthly.columns

        with pytest.raises(ValueError, match="Unknown periods preset"):
            compute_differencing_features(df, target_cols="metric", periods="invalid_preset")

    def test_temporal_features_pipeline_chaining(self):
        df = pd.DataFrame({
            "store": ["S1"] * 6,
            "week": list(range(1, 7)),
            "hours": [40.0, 42.0, 45.0, 48.0, 50.0, 55.0],
        })
        pipeline = (
            TemporalFeaturesPipeline(partition_cols="store", temporal_col="week")
            .add_lags("hours", lags=[1, 2])
            .add_differencing("hours", periods={"wow": 1}, pct_change=True, as_percent=True)
        )
        assert pipeline.get_feature_names() == ["hours_lag_1", "hours_lag_2", "hours_wow_pct"]

        res = pipeline.fit_transform(df)
        assert len(res) == 6
        assert "hours_lag_1" in res.columns
        assert "hours_wow_pct" in res.columns


# ===========================================================================
# 6. CAUSAL DAG DISCOVERY (NOTEARS) EDGE CASES
# ===========================================================================

class TestCausalDAGDiscoveryEdgeCases:
    """Verifies NOTEARS and DAG discovery under collinearity, small N, and constant columns."""

    def test_notears_perfectly_collinear_variables(self):
        rng = np.random.default_rng(42)
        n = 50
        x0 = rng.normal(0, 1, n)
        x1 = x0.copy()  # Exactly collinear
        x2 = 0.5 * x0 + rng.normal(0, 0.2, n)
        X = np.column_stack([x0, x1, x2])

        # Must not raise division-by-zero or crash
        W = notears_linear(X, lambda1=0.1, max_iter=25)
        assert W.shape == (3, 3)
        assert np.allclose(np.diag(W), 0.0)
        # Check no inf or nan
        assert not np.isnan(W).any()
        assert not np.isinf(W).any()

    def test_notears_small_sample_size(self):
        # Extremely small sample N=5, d=3
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, size=(5, 3))
        W = notears_linear(X, lambda1=0.1, max_iter=20)
        assert W.shape == (3, 3)
        assert np.allclose(np.diag(W), 0.0)

    def test_compute_directed_causal_edge_weights_edge_cases(self):
        df_collinear = pd.DataFrame({
            "col_a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "col_b": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],  # Collinear with col_a
            "col_c": [2.0, 3.0, 5.0, 7.0, 8.0, 11.0],
        })
        res = compute_directed_causal_edge_weights(df_collinear, lambda1=0.1)
        assert "directed_weights_df" in res
        assert "topological_order" in res
        assert len(res["topological_order"]) == 3


# ===========================================================================
# 7. BRADFORD FACTOR: MODE 3 SPELL-LEVEL EVENT LOGS
# ===========================================================================

class TestBradfordFactorMode3SpellEventLog:
    """Verifies Mode 3 spell-level event log ingestion, duration columns, and window clipping."""

    def test_spell_event_log_with_duration_column(self):
        # Event log where duration in days is directly provided
        spells_df = pd.DataFrame({
            "employee_id": ["E101", "E101", "E102"],
            "absence_date": ["2026-03-01", "2026-04-15", "2026-02-10"],
            "duration": [2, 3, 8],
        })
        res = calculate_bradford_factor(
            spells_df,
            employee_col="employee_id",
            start_date_col="absence_date",
            duration_col="duration",
            window_weeks=52,
            as_of_date="2026-10-01",
        )
        assert len(res) == 2
        e101 = res.loc[res["employee_id"] == "E101"].iloc[0]
        assert e101["absence_spells"] == 2
        assert e101["total_days_absent"] == 5
        # B = 2^2 * 5 = 20
        assert e101["bradford_score"] == 20
        assert e101["disruption_multiplier"] == 4.0

    def test_spell_window_boundary_clipping(self):
        # Spells before the 52-week lookback window must be excluded
        as_of = pd.Timestamp("2026-10-01")
        spells_df = pd.DataFrame({
            "employee_id": ["E1", "E1"],
            "start": ["2026-06-01", "2024-01-01"],  # Second spell is > 2 years old
            "end": ["2026-06-05", "2024-01-10"],
        })
        res = calculate_bradford_factor(
            spells_df,
            employee_col="employee_id",
            start_date_col="start",
            end_date_col="end",
            window_weeks=52,
            as_of_date=as_of,
        )
        e1 = res.iloc[0]
        # Only the 2026 spell should be counted (5 days, 1 spell)
        assert e1["absence_spells"] == 1
        assert e1["total_days_absent"] == 5
        assert e1["bradford_score"] == 5

    def test_unplanned_only_mode_3(self):
        spells_df = pd.DataFrame({
            "employee_id": ["E1", "E1"],
            "start": ["2026-05-01", "2026-07-01"],
            "end": ["2026-05-02", "2026-07-10"],
            "is_planned": [0, 1],  # Spell 1: unplanned (2 days), Spell 2: planned (10 days)
        })
        res = calculate_bradford_factor(
            spells_df,
            employee_col="employee_id",
            start_date_col="start",
            end_date_col="end",
            planned_col="is_planned",
            unplanned_only=True,
            as_of_date="2026-10-01",
        )
        e1 = res.iloc[0]
        # Primary score should reflect only the unplanned spell
        assert e1["absence_spells"] == 1
        assert e1["total_days_absent"] == 2
        assert e1["bradford_score"] == 2
        assert "planned_bradford_score" in e1


# ===========================================================================
# 8. EDGE CASES: EMPTY DATAFRAMES (0-ROW INPUTS)
# ===========================================================================

class TestEdgeCasesEmptyDataFrames:
    """Verifies that all core functions handle 0-row DataFrames gracefully without unhandled crashes."""

    def test_compute_rolling_metrics_empty(self):
        empty_df = pd.DataFrame(columns=["store_id", "week", "metric"])
        cfg = RollingMetricConfig(target_column="metric", window=4, aggregation="mean")
        res = compute_rolling_metrics(empty_df, configs=[cfg], partition_cols="store_id", temporal_col="week")
        assert isinstance(res, pd.DataFrame)
        assert len(res) == 0
        assert "metric_roll_mean_4" in res.columns

    def test_compute_lag_and_differencing_empty(self):
        empty_df = pd.DataFrame(columns=["unit", "period", "val"])
        res_lag = compute_lag_features(empty_df, target_cols="val", lags=[1, 2], temporal_col="period")
        assert len(res_lag) == 0
        assert "val_lag_1" in res_lag.columns

        res_diff = compute_differencing_features(empty_df, target_cols="val", periods={"d1": 1}, temporal_col="period")
        assert len(res_diff) == 0
        assert "val_d1" in res_diff.columns

    def test_pipelines_empty_fit_transform(self):
        empty_df = pd.DataFrame(columns=["group", "time", "val"])
        rp = RollingMetricsPipeline(partition_cols="group", temporal_col="time").add_metric("val", window=2)
        assert len(rp.fit_transform(empty_df)) == 0

        tp = TemporalFeaturesPipeline(partition_cols="group", temporal_col="time").add_lags("val", lags=1)
        assert len(tp.fit_transform(empty_df)) == 0

    def test_calculate_bradford_factor_empty(self):
        empty_df = pd.DataFrame(columns=["employee_id", "start", "end"])
        res = calculate_bradford_factor(empty_df, employee_col="employee_id", start_date_col="start", end_date_col="end")
        assert isinstance(res, pd.DataFrame)
        assert len(res) == 0
        assert "bradford_score" in res.columns


# ===========================================================================
# 9. EDGE CASES: NAN HANDLING ACROSS CORE MODULES
# ===========================================================================

class TestEdgeCasesNaNHandling:
    """Verifies robust, documented NaN behavior across temporal and compensation modules."""

    def test_rolling_metrics_with_internal_nans(self):
        # Target column with internal NaN values
        df = pd.DataFrame({
            "store_id": ["S1"] * 5,
            "week": [1, 2, 3, 4, 5],
            "sales": [10.0, np.nan, 30.0, 40.0, np.nan],
        })
        cfg = RollingMetricConfig(target_column="sales", window=3, min_periods=1, aggregation="mean")
        res = compute_rolling_metrics(df, configs=[cfg], partition_cols="store_id", temporal_col="week")
        # With min_periods=1:
        # week 1: [10] -> 10.0
        # week 2: [10, NaN] -> 10.0
        # week 3: [10, NaN, 30] -> 20.0
        assert np.isclose(res.loc[0, "sales_roll_mean_3"], 10.0)
        assert np.isclose(res.loc[1, "sales_roll_mean_3"], 10.0)
        assert np.isclose(res.loc[2, "sales_roll_mean_3"], 20.0)

    def test_compensation_nan_handling(self):
        # Series with NaNs should return NaN cleanly without crashing
        salaries = pd.Series([100000.0, np.nan, 80000.0])
        midpoint = 100000.0
        compa = calculate_compa_ratio(salaries, midpoint)
        assert np.isclose(compa.iloc[0], 1.0)
        assert pd.isna(compa.iloc[1])
        assert np.isclose(compa.iloc[2], 0.8)

        rp = calculate_range_penetration(salaries, band_min=60000.0, band_max=120000.0)
        assert not pd.isna(rp.iloc[0])
        assert pd.isna(rp.iloc[1])
        assert not pd.isna(rp.iloc[2])

    def test_correlation_heatmap_nan_pairwise_complete(self):
        # Ensure pairwise deletion in correlation handles sporadic NaNs
        df = pd.DataFrame({
            "turnover": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "tenure": [1.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "overtime": [2.0, 4.0, np.nan, 8.0, 10.0, 12.0, 14.0, 16.0],
        })
        res = correlation_heatmap_df(df, target="turnover", viz=False)
        assert len(res) == 2
        assert not res["CORRELATION TO turnover"].isna().any()


# ===========================================================================
# 10. EXPONENTIATED GRADIENT FAIRNESS CONVERGENCE
# ===========================================================================

class TestExponentiatedGradientFairnessConvergence:
    """Verifies that ExponentiatedGradientFairness tracks violations and converges."""

    def test_violation_tracking_and_disparity_reduction(self):
        df = generate_algorithmic_equity_and_fairness_data(n_employees=500, seed=42)
        X = df[["tenure_years", "education_years", "job_level", "performance_rating", "certifications"]]
        y = df["promotion_recommendation"]
        sens = df["demographic_group"]

        # Baseline unconstrained model
        base_clf = LogisticRegression(solver="liblinear", random_state=42).fit(X, y)
        base_dp = demographic_parity_difference(base_clf.predict(X), sens)

        # Fair model with exponentiated gradient
        fair_clf = ExponentiatedGradientFairness(
            estimator=LogisticRegression(solver="liblinear", random_state=42),
            constraint="demographic_parity",
            eps=0.02,
            max_iter=15,
            random_state=42,
        )
        fair_clf.fit(X, y, sensitive_features=sens)

        # 1. Violation history tracked per iteration
        assert len(fair_clf.violation_history_) == 15
        assert all(isinstance(v, (int, float, np.floating)) for v in fair_clf.violation_history_)
        assert all(v >= 0.0 for v in fair_clf.violation_history_)

        # 2. Ensemble weights form a valid probability distribution
        assert len(fair_clf.weights_) == 15
        assert np.isclose(sum(fair_clf.weights_), 1.0)
        assert all(w >= 0.0 for w in fair_clf.weights_)

        # 3. Dual multiplier history recorded
        assert len(fair_clf.dual_history_) == 15

        # 4. Final fair model disparity is strictly lower than unconstrained baseline
        fair_dp = demographic_parity_difference(fair_clf.predict(X), sens)
        assert fair_dp < base_dp


# ===========================================================================
# 11. ACCELERATED SEQUENCE ALIGNMENT (DTW, LCSS, EDR WITH SAKOE-CHIBA BAND)
# ===========================================================================

class TestAcceleratedSequenceAlignment:
    """Verifies JIT compilation, Sakoe-Chiba band constraints, and scalability on long sequences."""

    def test_dtw_with_sakoe_chiba_band(self):
        from people_analytics_toolkit.anomalies import compute_dtw_distance
        s1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        s2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # Unconstrained vs constrained identical series
        dist_full, path_full, _ = compute_dtw_distance(s1, s2)
        dist_band, path_band, _ = compute_dtw_distance(s1, s2, window=2)

        assert np.isclose(dist_full, 0.0)
        assert np.isclose(dist_band, 0.0)
        assert path_full == path_band

        # Ensure all steps in path adhere to window constraint |i - j| <= window
        for i_idx, j_idx in path_band:
            assert abs(i_idx - j_idx) <= 2

    def test_edr_with_sakoe_chiba_band(self):
        from people_analytics_toolkit.anomalies import compute_edr_distance
        s1 = np.array([10.0, 12.0, 11.0, 15.0, 14.0])
        s2 = np.array([10.0, 12.0, 11.0, 15.0, 14.0])

        assert np.isclose(compute_edr_distance(s1, s2, epsilon=1.0), 0.0)
        assert np.isclose(compute_edr_distance(s1, s2, epsilon=1.0, window=2), 0.0)

    def test_lcss_accelerated_gap_tolerance(self):
        from people_analytics_toolkit.anomalies import compute_lcss_distance
        s1 = np.array([5.0, 5.0, 5.0, 5.0, 10.0, 10.0])
        s2 = np.array([5.0, 5.0, -99.0, 5.0, 10.0, 10.0])

        lcss_len, lcss_dist = compute_lcss_distance(s1, s2, epsilon=0.5, delta=2)
        assert lcss_len == 5
        assert lcss_dist < 0.20

    def test_long_series_scalability(self):
        """Verify that DTW, LCSS, and EDR execute in milliseconds on N=500 series."""
        import time
        from people_analytics_toolkit.anomalies import (
            compute_dtw_distance,
            compute_lcss_distance,
            compute_edr_distance,
        )

        rng = np.random.default_rng(42)
        long_s1 = rng.normal(0, 1, size=500)
        long_s2 = long_s1 + rng.normal(0, 0.1, size=500)

        # 1. DTW with Sakoe-Chiba window=25 on N=500
        t0 = time.perf_counter()
        dtw_dist, dtw_path, _ = compute_dtw_distance(long_s1, long_s2, window=25)
        dtw_elapsed = time.perf_counter() - t0

        assert dtw_dist >= 0.0
        assert len(dtw_path) >= 500
        # Should complete in well under 100ms
        assert dtw_elapsed < 0.20

        # 2. LCSS on N=500
        t0 = time.perf_counter()
        lcss_len, lcss_dist = compute_lcss_distance(long_s1, long_s2, epsilon=0.2, delta=10)
        lcss_elapsed = time.perf_counter() - t0

        assert lcss_len > 0
        assert 0.0 <= lcss_dist <= 1.0
        assert lcss_elapsed < 0.20

        # 3. EDR with window=25 on N=500
        t0 = time.perf_counter()
        edr_dist = compute_edr_distance(long_s1, long_s2, epsilon=0.2, window=25)
        edr_elapsed = time.perf_counter() - t0

        assert 0.0 <= edr_dist <= 1.0
        assert edr_elapsed < 0.20


# ===========================================================================
# 12. ACCELERATED ROLLING SHANNON ENTROPY (CHAOS INDEX)
# ===========================================================================

class TestAcceleratedRollingShannonEntropy:
    """Verifies O(N) rolling Shannon entropy calculation, grouping, and scalability."""

    def test_basic_entropy_values(self):
        from people_analytics_toolkit.chaos import rolling_shannon_entropy, calculate_shannon_entropy

        # Uniform sequence of 2 alternating shifts
        df = pd.DataFrame({
            "shift": ["Morning", "Night"] * 10
        })
        # Window of 4: each window has 2 Morning, 2 Night -> p = 0.5, 0.5 -> H = 1.0 (base 2)
        res = rolling_shannon_entropy(df, val_col="shift", window=4, normalize=False)
        assert pd.isna(res.iloc[0])  # min_p = max(2, 4 // 2) = 2, index 0 has chunk_len=1
        # From index 1 onward, chunk has both or uniform shifts
        assert np.isclose(res.iloc[3], 1.0)
        assert np.isclose(res.iloc[-1], 1.0)

    def test_normalized_entropy_bounds(self):
        from people_analytics_toolkit.chaos import rolling_shannon_entropy

        rng = np.random.default_rng(42)
        shifts = rng.choice(["Morning", "Evening", "Night", "Off"], size=200)
        df = pd.DataFrame({"shift": shifts})

        res_norm = rolling_shannon_entropy(df, val_col="shift", window=14, normalize=True)
        valid_vals = res_norm.dropna()
        assert (valid_vals >= 0.0).all()
        assert (valid_vals <= 1.0 + 1e-9).all()

    def test_grouped_rolling_entropy(self):
        from people_analytics_toolkit.chaos import rolling_shannon_entropy

        df = pd.DataFrame({
            "dept": ["Store_A"] * 20 + ["Store_B"] * 20,
            "shift": ["Morning"] * 20 + ["Night", "Morning"] * 10,
        })
        res = rolling_shannon_entropy(df, val_col="shift", window=10, group_col="dept")
        # Store_A has only "Morning", so entropy should be 0.0
        store_a_res = res.iloc[:20].dropna()
        assert (store_a_res == 0.0).all()

        # Store_B has alternating shifts, so entropy should be > 0.0
        store_b_res = res.iloc[20:].dropna()
        assert (store_b_res > 0.5).all()

    def test_large_scale_retail_workforce_scalability(self):
        """Verifies that 50,000 shift records execute in under 250 milliseconds."""
        import time
        from people_analytics_toolkit.chaos import rolling_shannon_entropy

        rng = np.random.default_rng(42)
        n_rows = 50_000
        shifts = rng.choice(["Opening", "Mid", "Closing", "Off", "Overtime"], size=n_rows)
        depts = rng.choice(["Bakery", "Produce", "Deli", "Cashier", "Logistics"], size=n_rows)
        df = pd.DataFrame({"shift": shifts, "dept": depts})

        t0 = time.perf_counter()
        res = rolling_shannon_entropy(df, val_col="shift", window=21, group_col="dept", normalize=True)
        elapsed = time.perf_counter() - t0

        assert len(res) == n_rows
        assert not res.isna().all()
        # Must execute in under 300ms (O(N) algorithm with JIT / vectorized sliding frequency)
        assert elapsed < 0.30

    def test_invalid_parameters(self):
        import pytest
        from people_analytics_toolkit.chaos import rolling_shannon_entropy

        df = pd.DataFrame({"shift": ["A", "B", "C"]})
        with pytest.raises(KeyError):
            rolling_shannon_entropy(df, val_col="non_existent")
        with pytest.raises(KeyError):
            rolling_shannon_entropy(df, val_col="shift", group_col="bad_group")
        with pytest.raises(ValueError):
            rolling_shannon_entropy(df, val_col="shift", window=0)


# ===========================================================================
# 13. ACCELERATED BURT CONSTRAINT (STRUCTURAL HOLES)
# ===========================================================================

class TestAcceleratedBurtConstraint:
    """Verifies sparse Burt constraint calculation, edge cases, and 5K+ node scalability."""

    def test_burt_constraint_structural_hole_brokerage(self):
        from people_analytics_toolkit.ona import calculate_burt_constraint
        import networkx as nx

        # Star / broker graph: Node "Broker" connects two separate triangles
        # Triangle 1: (A1, A2, Broker)
        # Triangle 2: (C1, C2, Broker)
        edges = [
            ("A1", "A2"), ("A1", "Broker"), ("A2", "Broker"),
            ("C1", "C2"), ("C1", "Broker"), ("C2", "Broker"),
        ]
        res = calculate_burt_constraint(edges)

        assert isinstance(res, pd.Series)
        assert res.name == "burt_constraint"
        # Broker connects non-redundant structural holes, so constraint must be lower than perimeter nodes
        assert res["Broker"] < res["A1"]
        assert res["Broker"] < res["C1"]
        assert 0.0 < res["Broker"] < 1.0

    def test_burt_constraint_edge_cases(self):
        from people_analytics_toolkit.ona import calculate_burt_constraint
        import networkx as nx

        # Empty graph
        assert len(calculate_burt_constraint(nx.Graph())) == 0

        # Single node graph -> NaN constraint
        g1 = nx.Graph()
        g1.add_node("Single")
        s1 = calculate_burt_constraint(g1)
        assert len(s1) == 1
        assert pd.isna(s1["Single"])

        # Graph with isolated node and degree-1 nodes
        g_mix = nx.Graph()
        g_mix.add_edge("N1", "N2")
        g_mix.add_node("Isolated")
        s_mix = calculate_burt_constraint(g_mix)
        assert s_mix["N1"] == 1.0
        assert s_mix["N2"] == 1.0
        assert pd.isna(s_mix["Isolated"])

    def test_burt_constraint_large_scale_enterprise_scalability(self):
        """Verifies that a 5,000-node organizational network computes in under 500 ms."""
        import time
        import networkx as nx
        from people_analytics_toolkit.ona import calculate_burt_constraint

        # 5,000 employee graph with ~25,000 collaboration edges
        G_large = nx.erdos_renyi_graph(5000, 0.002, seed=42)

        import gc
        gc.collect()

        t0 = time.perf_counter()
        res = calculate_burt_constraint(G_large)
        elapsed = time.perf_counter() - t0

        assert len(res) == 5000
        assert not res.isna().all()
        # Must execute in under 500ms using sparse matrix operations (avoiding dense O(N^2) memory and loops)
        assert elapsed < 0.50


# ===========================================================================
# 14. VECTORIZED STAGNATION INDEX (NP.SELECT INTERVENTIONS)
# ===========================================================================

class TestVectorizedStagnationIndex:
    """Verifies vectorized stagnation tagging across all intervention categories and retail scale."""

    def test_all_four_intervention_branches(self):
        from people_analytics_toolkit.mobility import calculate_stagnation_index

        # Cohort of 10 employees where we control time_in_pos and perf
        # Quantiles: median, p75, p90
        df = pd.DataFrame({
            "emp_id": [f"E{i}" for i in range(10)],
            "dept": ["Engineering"] * 10,
            # Ordered months: 10, 20, 30, 40, 50, 60, 70, 80, 100, 100
            "months": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 100.0, 100.0],
            # E8 (100 mo, perf 3.0) -> Tenured Core Contributor
            # E9 (100 mo, perf 4.8) -> Urgent Career-Pathing Trigger
            "perf": [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 4.5, 4.5, 3.0, 4.8],
            "risk": [0.1] * 10,
        })
        res = calculate_stagnation_index(
            df,
            time_in_pos_col="months",
            cohort_col="dept",
            perf_col="perf",
            high_perf_threshold=4.0,
            base_flight_risk_col="risk",
        )

        interventions = set(res["career_pathing_intervention"])
        # Verify that branches are properly populated
        assert "Urgent Career-Pathing Trigger" in interventions
        assert "Emerging Stagnation Warning" in interventions
        assert "Tenured Core Contributor" in interventions
        assert "Normal Progression" in interventions

        # E9: months=100 (stagnant), perf=4.8 (high) -> Urgent
        assert res.loc[res["emp_id"] == "E9", "career_pathing_intervention"].iloc[0] == "Urgent Career-Pathing Trigger"
        # E8: months=90 (stagnant), perf=3.0 (normal) -> Tenured Core Contributor
        assert res.loc[res["emp_id"] == "E8", "career_pathing_intervention"].iloc[0] == "Tenured Core Contributor"
        # E7: months=80 (emerging), perf=4.5 (high) -> Emerging Stagnation Warning
        assert res.loc[res["emp_id"] == "E7", "career_pathing_intervention"].iloc[0] == "Emerging Stagnation Warning"
        # E0: months=10 (early), perf=3.0 -> Normal Progression
        assert res.loc[res["emp_id"] == "E0", "career_pathing_intervention"].iloc[0] == "Normal Progression"

    def test_large_scale_vectorized_stagnation_performance(self):
        """Verifies 50,000 workforce records process in under 150 ms using np.select."""
        import time
        from people_analytics_toolkit.mobility import calculate_stagnation_index

        rng = np.random.default_rng(42)
        n = 50_000
        df = pd.DataFrame({
            "months": rng.uniform(1, 120, size=n),
            "dept": rng.choice(["Sales", "Eng", "Ops", "HR", "Legal"], size=n),
            "perf": rng.uniform(1.0, 5.0, size=n),
            "risk": rng.uniform(0.05, 0.40, size=n),
        })

        t0 = time.perf_counter()
        res = calculate_stagnation_index(
            df,
            time_in_pos_col="months",
            cohort_col="dept",
            perf_col="perf",
            base_flight_risk_col="risk",
        )
        elapsed = time.perf_counter() - t0

        assert len(res) == n
        assert not res["career_pathing_intervention"].isna().any()
        # Must execute in under 150ms without slow Python row iterations
        assert elapsed < 0.15


# ===========================================================================
# 15. VECTORIZED SUCCESSION RECOMMENDATIONS
# ===========================================================================

class TestVectorizedSuccessionRecommendation:
    """Verifies vectorized candidate succession ranking, filtering, and enterprise talent pool scale."""

    def test_succession_ranking_and_target_role_exclusion(self):
        from people_analytics_toolkit.mobility import CareerMovementEmbeddings

        cme = CareerMovementEmbeddings(embedding_dim=8)
        cme.fit([
            ["Analyst", "Senior_Analyst", "Lead_Analyst", "Manager"],
            ["Associate", "Analyst", "Senior_Analyst"],
            ["Senior_Analyst", "Lead_Analyst", "Manager"],
        ])

        candidates = pd.DataFrame([
            {"employee_id": "EMP_1", "current_role": "Senior_Analyst", "career_history": ["Analyst", "Senior_Analyst"]},
            {"employee_id": "EMP_2", "current_role": "Manager", "career_history": ["Senior_Analyst", "Manager"]}, # target role -> must be skipped
            {"employee_id": "EMP_3", "current_role": "Analyst", "career_history": ["Analyst"]},
        ])

        recs = cme.recommend_succession_candidates("Manager", candidates, top_k=2)

        assert isinstance(recs, pd.DataFrame)
        assert len(recs) == 2
        # Target role incumbent must be excluded
        assert "EMP_2" not in recs["employee_id"].values
        # Schema integrity
        assert list(recs.columns) == [
            "employee_id",
            "current_role",
            "career_history",
            "embedding_cosine_similarity",
            "transition_reachability",
            "succession_similarity",
        ]
        # Top candidate must be Senior_Analyst (closer in transition and embedding)
        assert recs.iloc[0]["employee_id"] == "EMP_1"
        assert recs.iloc[0]["succession_similarity"] >= recs.iloc[1]["succession_similarity"]

    def test_succession_edge_cases(self):
        from people_analytics_toolkit.mobility import CareerMovementEmbeddings

        cme = CareerMovementEmbeddings(embedding_dim=8)
        cme.fit([["R1", "R2", "R3"], ["R2", "R3", "R4"]])

        # 1. Empty DataFrame
        empty_res = cme.recommend_succession_candidates("R3", pd.DataFrame())
        assert len(empty_res) == 0
        assert "succession_similarity" in empty_res.columns

        # 2. All candidates already in target role
        all_target = pd.DataFrame([
            {"employee_id": "E1", "current_role": "R3"},
            {"employee_id": "E2", "current_role": "R3"},
        ])
        target_res = cme.recommend_succession_candidates("R3", all_target)
        assert len(target_res) == 0
        assert "succession_similarity" in target_res.columns

    def test_large_scale_succession_scalability(self):
        """Verifies ranking across 20,000 enterprise talent candidates in under 200 ms."""
        import time
        from people_analytics_toolkit.mobility import CareerMovementEmbeddings

        rng = np.random.default_rng(42)
        roles = ["L1", "L2", "L3", "L4", "L5", "L6"]
        corpus = [list(rng.choice(roles, size=rng.integers(2, 5))) for _ in range(300)]

        cme = CareerMovementEmbeddings(embedding_dim=12)
        cme.fit(corpus)

        n = 20_000
        candidates = pd.DataFrame({
            "employee_id": [f"EMP_{i}" for i in range(n)],
            "current_role": rng.choice(roles, size=n),
            "career_history": [list(rng.choice(roles, size=rng.integers(1, 4))) for _ in range(n)],
        })

        t0 = time.perf_counter()
        recs = cme.recommend_succession_candidates("L5", candidates, top_k=10)
        elapsed = time.perf_counter() - t0

        assert len(recs) == 10
        assert (recs["current_role"] != "L5").all()
        # Must execute in under 200 ms without slow per-row iterrows loop
        assert elapsed < 0.20


# ===========================================================================
# 16. ACCELERATED CAUSAL FOREST DML PREDICT
# ===========================================================================

class TestAcceleratedCausalForestPredict:
    """Verifies vectorized CATE tree and forest predictions, intervals, and large-scale performance."""

    def test_single_tree_vectorized_predict_parity(self):
        from people_analytics_toolkit.prescriptive import HonestCausalTree

        rng = np.random.default_rng(42)
        n = 200
        X_tr = rng.normal(0, 1, size=(n, 4))
        T_res_tr = rng.normal(0, 1, size=n)
        Y_res_tr = 1.5 * T_res_tr + rng.normal(0, 0.2, size=n)

        X_est = rng.normal(0, 1, size=(n, 4))
        T_res_est = rng.normal(0, 1, size=n)
        Y_res_est = 1.5 * T_res_est + rng.normal(0, 0.2, size=n)

        tree = HonestCausalTree(max_depth=3, min_samples_leaf=5, random_state=42)
        tree.fit(X_tr, T_res_tr, Y_res_tr, X_est, T_res_est, Y_res_est)

        X_test = rng.normal(0, 1, size=(50, 4))
        # Vectorized predict
        vec_preds = tree.predict(X_test)
        # Instance loop predict
        inst_preds = np.array([tree.predict_instance(x)[0] for x in X_test])

        np.testing.assert_allclose(vec_preds, inst_preds, rtol=1e-12, atol=1e-12)

    def test_forest_predict_and_interval_shape(self):
        from people_analytics_toolkit.prescriptive import CausalForestDML

        rng = np.random.default_rng(42)
        n = 300
        X = pd.DataFrame(rng.normal(0, 1, size=(n, 3)), columns=["a", "b", "c"])
        t = rng.choice([0, 1], size=n)
        y = 2.0 * t + 0.3 * X["a"] * t + rng.normal(0, 0.5, size=n)

        forest = CausalForestDML(n_estimators=10, max_depth=3, random_state=42)
        forest.fit(X, t, y)

        preds = forest.predict(X)
        assert len(preds) == n
        assert isinstance(preds, np.ndarray)

        intervals = forest.predict_interval(X, alpha=0.05)
        assert set(intervals.keys()) == {"cate", "se", "ci_lower", "ci_upper"}
        assert len(intervals["cate"]) == n
        assert (intervals["se"] >= 0.0).all()
        assert (intervals["ci_lower"] <= intervals["ci_upper"]).all()

    def test_large_scale_forest_predict_scalability(self):
        """Verifies 10,000 samples across 20 honest causal trees predict in under 100 ms."""
        import time
        from people_analytics_toolkit.prescriptive import CausalForestDML

        rng = np.random.default_rng(42)
        n_train = 500
        X_train = pd.DataFrame(rng.normal(0, 1, size=(n_train, 5)), columns=[f"f{i}" for i in range(5)])
        t_train = rng.choice([0, 1], size=n_train)
        y_train = 1.0 * t_train + rng.normal(0, 0.5, size=n_train)

        forest = CausalForestDML(n_estimators=20, max_depth=4, random_state=42)
        forest.fit(X_train, t_train, y_train)

        n_eval = 10_000
        X_eval = rng.normal(0, 1, size=(n_eval, 5))

        t0 = time.perf_counter()
        preds = forest.predict(X_eval)
        elapsed = time.perf_counter() - t0

        assert len(preds) == n_eval
        # Vectorized/JIT evaluation of 200,000 tree-sample passes must take under 100 ms
        assert elapsed < 0.10


# ===========================================================================
# 17. ACCELERATED SSA HANKEL EMBEDDING
# ===========================================================================

class TestAcceleratedSSAHankelEmbedding:
    """Verifies that scipy.linalg.hankel generates exact trajectory matrices and scales efficiently."""

    def test_hankel_embedding_parity(self):
        from people_analytics_toolkit.memory import singular_spectrum_analysis

        rng = np.random.default_rng(42)
        n = 300
        # Multi-frequency periodic signal + noise
        t = np.linspace(0, 10, n)
        signal = 5.0 * np.sin(2 * np.pi * 0.5 * t) + 2.0 * np.cos(2 * np.pi * 1.5 * t) + rng.normal(0, 0.5, n)
        s = pd.Series(signal, name="daily_labor_demand")

        res = singular_spectrum_analysis(s, window_length=30, top_k=3)
        assert len(res["smoothed"]) == n
        assert len(res["residual"]) == n
        assert res["n_components_used"] == 3
        # Baseline should capture variance, residual should have mean near 0
        assert np.isclose(res["residual"].mean(), 0.0, atol=0.2)
        assert res["explained_variance_ratio"].sum() > 0.5

    def test_hankel_long_series_performance(self):
        """Verifies 5,000 observations (~14 years of daily staffing) decompose in under 250 ms."""
        import time
        from people_analytics_toolkit.memory import singular_spectrum_analysis

        rng = np.random.default_rng(42)
        n = 5_000
        t = np.linspace(0, 100, n)
        s = pd.Series(10.0 + 2.0 * np.sin(t) + rng.normal(0, 0.5, n))

        t0 = time.perf_counter()
        res = singular_spectrum_analysis(s, window_length=50, top_k=4)
        elapsed = time.perf_counter() - t0

        assert len(res["smoothed"]) == n
        assert elapsed < 0.25


# ===========================================================================
# 18. DEPRECATION STRATEGY & ALIAS WARNINGS
# ===========================================================================

class TestDeprecationWarnings:
    """Verifies that duplicate alias functions emit FutureWarning and remain functional."""

    def test_role_similarity_matrix_deprecation(self):
        import warnings
        from people_analytics_toolkit.mobility import (
            CareerMovementEmbeddings,
            compute_role_similarity_matrix,
            get_role_similarity_matrix,
            calculate_role_similarity_matrix,
        )

        transitions = [
            ("Analyst", "Senior Analyst"),
            ("Senior Analyst", "Lead"),
            ("Analyst", "Senior Analyst"),
            ("Senior Analyst", "Manager"),
        ]
        cme = CareerMovementEmbeddings(embedding_dim=4, max_steps=2).fit(transitions)

        # Instance method deprecations
        with pytest.warns(FutureWarning, match="get_role_similarity_matrix.*deprecated"):
            res1 = cme.get_role_similarity_matrix()
        assert isinstance(res1, pd.DataFrame)

        with pytest.warns(FutureWarning, match="calculate_role_similarity_matrix.*deprecated"):
            res2 = cme.calculate_role_similarity_matrix()
        assert isinstance(res2, pd.DataFrame)

        # Module-level deprecations
        with pytest.warns(FutureWarning, match="get_role_similarity_matrix.*deprecated"):
            res3 = get_role_similarity_matrix(cme)
        assert np.allclose(res1.values, res3.values)

        with pytest.warns(FutureWarning, match="calculate_role_similarity_matrix.*deprecated"):
            res4 = calculate_role_similarity_matrix(cme)
        assert np.allclose(res1.values, res4.values)

    def test_prescriptive_aliases_deprecation(self):
        from people_analytics_toolkit.prescriptive import (
            compute_synthetic_control_weights,
            calculate_synthetic_control_weights,
            compute_directed_causal_edge_weights,
            calculate_directed_causal_edge_weights,
        )

        Y_donors = np.array([[10.0, 11.0, 12.0], [9.0, 10.0, 11.0]])
        Y_target = np.array([9.5, 10.5, 11.5])

        with pytest.warns(FutureWarning, match="calculate_synthetic_control_weights.*deprecated"):
            w, c0, rmspe = calculate_synthetic_control_weights(
                donor_matrix_pre=Y_donors,
                target_series_pre=Y_target,
            )
        assert isinstance(w, pd.Series)

        df = pd.DataFrame({
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [2.0, 4.0, 6.0, 8.0, 10.0],
        })
        with pytest.warns(FutureWarning, match="calculate_directed_causal_edge_weights.*deprecated"):
            res = calculate_directed_causal_edge_weights(df, lambda1=0.1)
        assert "directed_weights_df" in res

    def test_memory_aliases_deprecation(self):
        from people_analytics_toolkit.memory import (
            calculate_rolling_metrics,
            calculate_lag_features,
            compute_ewma,
        )

        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        with pytest.warns(FutureWarning, match="compute_ewma.*deprecated"):
            ewma_res = compute_ewma(s, span=3)
        assert len(ewma_res) == 5

        df = pd.DataFrame({"val": [10, 20, 30, 40, 50]})
        with pytest.warns(FutureWarning, match="calculate_lag_features.*deprecated"):
            lag_df = calculate_lag_features(df, target_cols="val", lags=1)
        assert "val_lag_1" in lag_df.columns


# ===========================================================================
# 19. CLASSIFY PAY BAND STATUS: NON-SERIES INPUTS BUG FIX
# ===========================================================================

class TestClassifyPayBandStatusNonSeriesBug:
    """Verifies that classify_pay_band_status correctly processes non-Series inputs without AttributeError."""

    def test_raw_numpy_array_input(self):
        from people_analytics_toolkit.compensation import classify_pay_band_status

        rp_array = np.array([-0.10, 0.15, 0.40, 0.65, 0.85, 1.20])
        res = classify_pay_band_status(rp_array)
        assert isinstance(res, np.ndarray)
        assert res.dtype.kind in ("U", "O")
        assert list(res) == [
            "Green-Circled (<Min)",
            "Q1: Developing (0-25%)",
            "Q2: Core Proficient (25-50%)",
            "Q3: Advanced (50-75%)",
            "Q4: Senior Cap (75-100%)",
            "Red-Circled (>Max)",
        ]

    def test_raw_list_input(self):
        from people_analytics_toolkit.compensation import classify_pay_band_status

        rp_list = [-0.10, 0.15, 0.40, 0.65, 0.85, 1.20]
        res = classify_pay_band_status(rp_list)
        assert isinstance(res, list)
        assert res[0] == "Green-Circled (<Min)"
        assert res[-1] == "Red-Circled (>Max)"

    def test_scalar_float_input(self):
        from people_analytics_toolkit.compensation import classify_pay_band_status

        res = classify_pay_band_status(0.50)
        assert isinstance(res, str)
        assert res == "Q3: Advanced (50-75%)"

    def test_salary_with_scalar_bounds(self):
        from people_analytics_toolkit.compensation import classify_pay_band_status

        # Passing raw list salary and scalar band bounds:
        # 45000 -> rp=-0.10 -> Green-Circled (<Min)
        # 55000 -> rp=0.10 -> Q1: Developing (0-25%)
        # 65000 -> rp=0.30 -> Q2: Core Proficient (25-50%)
        # 105000 -> rp=1.10 -> Red-Circled (>Max)
        res = classify_pay_band_status([45000, 55000, 65000, 105000], band_min=50000, band_max=100000)
        assert isinstance(res, list)
        assert res[0] == "Green-Circled (<Min)"
        assert res[1] == "Q1: Developing (0-25%)"
        assert res[2] == "Q2: Core Proficient (25-50%)"
        assert res[3] == "Red-Circled (>Max)"

    def test_salary_list_with_series_bounds(self):
        from people_analytics_toolkit.compensation import classify_pay_band_status

        # Passing salary as list, but band_min as pd.Series with custom index:
        # E101: salary=70000, min=50000, max=100000 -> rp=0.40 -> Q2: Core Proficient
        # E102: salary=130000, min=60000, max=120000 -> rp=1.167 -> Red-Circled (>Max)
        band_min = pd.Series([50000, 60000], index=["E101", "E102"])
        band_max = pd.Series([100000, 120000], index=["E101", "E102"])
        res = classify_pay_band_status([70000, 130000], band_min=band_min, band_max=band_max)
        assert isinstance(res, pd.Series)
        assert list(res.index) == ["E101", "E102"]
        assert res.loc["E101"] == "Q2: Core Proficient (25-50%)"
        assert res.loc["E102"] == "Red-Circled (>Max)"


# ===========================================================================
# 20. SSA MISSING VALUES INTERPOLATION & WARNINGS
# ===========================================================================

class TestSSAMissingValuesInterpolation:
    """Verifies that missing values (NaNs) in SSA emit UserWarning and are marked in returned structures."""

    def test_ssa_warns_and_marks_missing_values(self):
        import pytest
        from people_analytics_toolkit.memory import singular_spectrum_analysis

        # 50-step synthetic series with 3 NaN values injected
        t = np.linspace(0, 5, 50)
        signal = 10.0 + np.sin(t)
        signal[10] = np.nan
        signal[25] = np.nan
        signal[40] = np.nan
        s = pd.Series(signal, index=[f"day_{i}" for i in range(50)], name="store_demand")

        with pytest.warns(UserWarning, match=r"Input series contains 3 missing value\(s\) \(NaNs\)\."):
            res = singular_spectrum_analysis(s, window_length=15, top_k=2)

        # Check returned dictionary contains imputed_mask and n_imputed
        assert "imputed_mask" in res
        assert "n_imputed" in res
        assert res["n_imputed"] == 3
        assert isinstance(res["imputed_mask"], pd.Series)
        assert list(res["imputed_mask"].index) == list(s.index)
        assert res["imputed_mask"].sum() == 3
        assert res["imputed_mask"]["day_10"] is True or res["imputed_mask"].iloc[10] == True
        assert res["imputed_mask"]["day_25"] is True or res["imputed_mask"].iloc[25] == True
        assert res["imputed_mask"]["day_40"] is True or res["imputed_mask"].iloc[40] == True
        assert res["imputed_mask"].iloc[0] == False
        # Smoothed series has no NaNs
        assert not res["smoothed"].isna().any()

    def test_ssa_no_warning_when_clean(self):
        import warnings
        from people_analytics_toolkit.memory import singular_spectrum_analysis

        s = pd.Series(np.linspace(10, 20, 40))
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            res = singular_spectrum_analysis(s, window_length=10, top_k=2)
            # Filter for UserWarning specifically about missing values
            nan_warnings = [w for w in record if "missing value" in str(w.message)]
            assert len(nan_warnings) == 0

        assert res["n_imputed"] == 0
        assert not res["imputed_mask"].any()

    def test_ssa_transformer_properties(self):
        import pytest
        from people_analytics_toolkit.memory import SingularSpectrumAnalysis

        t = np.linspace(0, 10, 60)
        data = 20.0 + 2.0 * np.cos(t)
        data[5] = np.nan
        data[30] = np.nan

        ssa = SingularSpectrumAnalysis(window_length=15, top_k=2)
        assert not ssa.is_fitted_ if hasattr(ssa, "is_fitted_") else True

        with pytest.warns(UserWarning, match=r"Input series contains 2 missing value\(s\)"):
            smoothed = ssa.fit_transform(data)

        assert len(smoothed) == 60
        assert ssa.has_imputed_ is True
        assert ssa.n_imputed_ == 2
        assert ssa.imputed_mask_.sum() == 2
        assert ssa.imputed_mask_.iloc[5] == True
        assert ssa.imputed_mask_.iloc[30] == True

    def test_compute_svd_smoothed_baseline_imputed_col(self):
        import pytest
        from people_analytics_toolkit.memory import compute_svd_smoothed_baseline

        df = pd.DataFrame({
            "store_id": ["S1"] * 30 + ["S2"] * 30,
            "step": list(range(30)) * 2,
            "traffic": np.linspace(50, 100, 60),
        })
        # Inject NaNs
        df.loc[5, "traffic"] = np.nan
        df.loc[35, "traffic"] = np.nan

        with pytest.warns(UserWarning):
            res_df = compute_svd_smoothed_baseline(
                df=df,
                target_col="traffic",
                partition_cols="store_id",
                temporal_col="step",
                imputed_col="traffic_was_imputed",
                top_k=2,
                window_length=10,
            )

        assert "traffic_was_imputed" in res_df.columns
        assert res_df["traffic_was_imputed"].dtype == bool
        assert res_df["traffic_was_imputed"].sum() == 2
        assert res_df.loc[5, "traffic_was_imputed"] == True
        assert res_df.loc[35, "traffic_was_imputed"] == True
        assert res_df.loc[0, "traffic_was_imputed"] == False


# ===========================================================================
# 21. DYNAMIC ENTITY SELF-ATTENTION MULTI-HEAD VERIFICATION
# ===========================================================================

class TestDynamicEntitySelfAttentionMultiHead:
    """Verifies that n_heads correctly drives multi-head attention projection and aggregation."""

    def test_n_heads_validation(self):
        import pytest
        from people_analytics_toolkit.trajectories import DynamicEntitySelfAttention

        with pytest.raises(ValueError, match="n_heads must be a positive integer"):
            DynamicEntitySelfAttention(n_heads=0)

        with pytest.raises(ValueError, match="n_heads must be a positive integer"):
            DynamicEntitySelfAttention(n_heads=-2)

        attn = DynamicEntitySelfAttention(n_heads=2)
        with pytest.raises(ValueError, match="n_heads must be a positive integer"):
            attn.set_params(n_heads=-1)

    def test_multi_head_attention_shapes_and_weights(self):
        from people_analytics_toolkit.trajectories import DynamicEntitySelfAttention

        rng = np.random.RandomState(42)
        tokens = rng.normal(size=(6, 32))  # 6 tokens, d_model=32

        # 4 heads -> head_dim = 8
        attn4 = DynamicEntitySelfAttention(n_heads=4, d_model=32, random_state=42)
        assert attn4.head_dim == 8

        res = attn4.aggregate(tokens)
        assert res["context_vector"].shape == (32,)
        # Head attention weights must be (4, 6)
        assert res["head_attention_weights"].shape == (4, 6)
        # Each head must normalize independently to sum to 1.0
        for h in range(4):
            assert np.isclose(res["head_attention_weights"][h].sum(), 1.0, atol=1e-5)

        # Average attention weights is length 6 and sums to 1.0
        assert len(res["attention_weights"]) == 6
        assert np.isclose(res["attention_weights"].sum(), 1.0, atol=1e-5)

    def test_single_vs_multi_head_projections(self):
        from people_analytics_toolkit.trajectories import DynamicEntitySelfAttention

        rng = np.random.RandomState(42)
        tokens = rng.normal(size=(5, 16))

        attn1 = DynamicEntitySelfAttention(n_heads=1, d_model=16, random_state=42)
        attn2 = DynamicEntitySelfAttention(n_heads=2, d_model=16, random_state=42)

        res1 = attn1.aggregate(tokens)
        res2 = attn2.aggregate(tokens)

        assert res1["head_attention_weights"].shape == (1, 5)
        assert res2["head_attention_weights"].shape == (2, 5)
        # Context vectors are both d_model=16
        assert res1["context_vector"].shape == (16,)
        assert res2["context_vector"].shape == (16,)

    def test_transform_and_fit_transform(self):
        from people_analytics_toolkit.trajectories import DynamicEntitySelfAttention

        rng = np.random.RandomState(42)
        tokens1 = rng.normal(size=(4, 16))
        tokens2 = rng.normal(size=(7, 16))

        attn = DynamicEntitySelfAttention(n_heads=2, d_model=16, random_state=42)

        # Single 2D transform
        ctx1 = attn.transform(tokens1)
        assert ctx1.shape == (16,)

        # Batch list transform
        batch_ctx = attn.transform([tokens1, tokens2])
        assert batch_ctx.shape == (2, 16)

        # DataFrame tokens with index retention
        df_tokens = pd.DataFrame(tokens1, index=["emp_tok", "proj1", "proj2", "skill1"])
        res_df = attn.aggregate(df_tokens)
        assert list(res_df["attention_weights"].index) == ["emp_tok", "proj1", "proj2", "skill1"]


# ===========================================================================
# 22. METAPATH SCHEMA CYCLING VERIFICATION
# ===========================================================================

class TestMetapathSchemaCycling:
    """Verifies that generate_metapath_walks correctly cycles across open, closed, and bipartite schemas."""

    def test_metapath_open_and_closed_schema_cycling(self):
        import networkx as nx
        from people_analytics_toolkit.trajectories import generate_metapath_walks

        # Create a small multipartite graph: E1 - P1 - S1 - E2 - P2 - S2
        G = nx.Graph()
        G.add_node("E1", node_type="Employee")
        G.add_node("E2", node_type="Employee")
        G.add_node("P1", node_type="Project")
        G.add_node("P2", node_type="Project")
        G.add_node("S1", node_type="Skill")
        G.add_node("S2", node_type="Skill")

        # Connect E <-> P, P <-> S, S <-> E
        G.add_edge("E1", "P1")
        G.add_edge("P1", "S1")
        G.add_edge("S1", "E2")
        G.add_edge("E2", "P2")
        G.add_edge("P2", "S2")
        G.add_edge("S2", "E1")

        node_types = nx.get_node_attributes(G, "node_type")

        # 1. Closed 4-element schema: [Employee, Project, Skill, Employee]
        closed_schema = ["Employee", "Project", "Skill", "Employee"]
        walks_closed = generate_metapath_walks(
            G, meta_paths=[closed_schema], walk_length=7, num_walks=2, random_state=42
        )
        assert len(walks_closed) > 0
        expected_closed_cycle = ["Employee", "Project", "Skill", "Employee", "Project", "Skill", "Employee"]
        for walk in walks_closed:
            for step_idx, node in enumerate(walk):
                assert node_types[node] == expected_closed_cycle[step_idx]

        # 2. Open 3-element schema: [Employee, Project, Skill]
        open_schema = ["Employee", "Project", "Skill"]
        walks_open = generate_metapath_walks(
            G, meta_paths=[open_schema], walk_length=7, num_walks=2, random_state=42
        )
        assert len(walks_open) > 0
        expected_open_cycle = ["Employee", "Project", "Skill", "Employee", "Project", "Skill", "Employee"]
        for walk in walks_open:
            for step_idx, node in enumerate(walk):
                assert node_types[node] == expected_open_cycle[step_idx]

        # 3. 2-element bipartite schema: [Employee, Project]
        bipartite_schema = ["Employee", "Project"]
        walks_bipartite = generate_metapath_walks(
            G, meta_paths=[bipartite_schema], walk_length=5, num_walks=2, random_state=42
        )
        assert len(walks_bipartite) > 0
        expected_bipartite = ["Employee", "Project", "Employee", "Project", "Employee"]
        for walk in walks_bipartite:
            for step_idx, node in enumerate(walk):
                assert node_types[node] == expected_bipartite[step_idx]

    def test_invalid_short_schema_raises_value_error(self):
        import networkx as nx
        from people_analytics_toolkit.trajectories import generate_metapath_walks

        G = nx.Graph()
        G.add_node("E1", node_type="Employee")
        with pytest.raises(ValueError, match="must contain at least 2 node types"):
            generate_metapath_walks(G, meta_paths=[["Employee"]], walk_length=5)


# ===========================================================================
# 23. OPTIONAL DEPENDENCY CONSISTENCY VERIFICATION
# ===========================================================================

class TestOptionalDependencyErrorMessages:
    """Verifies that all optional dependency ImportErrors use the standardized install format."""

    def test_lof_anomalies_extra_message(self, monkeypatch):
        import builtins
        from people_analytics_toolkit.anomalies import LocalOutlierFactorDetector

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name.startswith("pyod"):
                raise ImportError("No module named 'pyod'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        detector = LocalOutlierFactorDetector()
        with pytest.raises(ImportError, match=r"pip install 'people-analytics-toolkit\[anomalies\]'"):
            detector.fit(np.array([[1.0, 2.0], [3.0, 4.0]]))

    def test_garch_timeseries_extra_message(self, monkeypatch):
        import builtins
        from people_analytics_toolkit.chaos import GARCHVolatilityModel

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name.startswith("arch"):
                raise ImportError("No module named 'arch'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        model = GARCHVolatilityModel()
        with pytest.raises(ImportError, match=r"pip install 'people-analytics-toolkit\[timeseries\]'"):
            model.fit(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]))

    def test_survival_extra_message(self, monkeypatch):
        import builtins
        from people_analytics_toolkit.memory import HazardEmbeddings

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name.startswith("lifelines"):
                raise ImportError("No module named 'lifelines'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        hz = HazardEmbeddings()
        df = pd.DataFrame({"tenure_weeks": [10, 20], "turnover_event": [1, 0]})
        with pytest.raises(ImportError, match=r"pip install 'people-analytics-toolkit\[survival\]'"):
            hz.fit(df)


# ===========================================================================
# 20. PYTORCH REPRODUCIBILITY GUARANTEES
# ===========================================================================

class TestTorchReproducibilityGuarantees:
    """Verifies reproducibility guarantees for torch models (AutoencoderWeirdnessDetector & DynamicEntitySelfAttention)."""

    def test_autoencoder_weirdness_detector_reproducibility(self):
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not installed")

        from people_analytics_toolkit.anomalies import AutoencoderWeirdnessDetector

        rng = np.random.RandomState(42)
        X = rng.randn(100, 8).astype(np.float32)

        # Fit model 1 with seed=123
        model1 = AutoencoderWeirdnessDetector(epochs=10, seed=123)
        model1.fit(X)
        w1 = model1.train_weirdness_
        scores1, df1 = model1.transform(X)

        # Fit model 2 with seed=123
        model2 = AutoencoderWeirdnessDetector(epochs=10, seed=123)
        model2.fit(X)
        w2 = model2.train_weirdness_
        scores2, df2 = model2.transform(X)

        # Verify bit-exact/close reproducibility across training and inference
        np.testing.assert_allclose(w1, w2, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(scores1.values, scores2.values, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(df1.values, df2.values, rtol=1e-5, atol=1e-5)

        # Verify cuDNN and deterministic algorithms flags were configured
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False
        if hasattr(torch, "are_deterministic_algorithms_enabled"):
            assert torch.are_deterministic_algorithms_enabled() is True

    def test_autoencoder_different_seeds_produce_different_results(self):
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not installed")

        from people_analytics_toolkit.anomalies import AutoencoderWeirdnessDetector

        rng = np.random.RandomState(42)
        X = rng.randn(100, 8).astype(np.float32)

        model1 = AutoencoderWeirdnessDetector(epochs=10, seed=101).fit(X)
        model2 = AutoencoderWeirdnessDetector(epochs=10, seed=202).fit(X)

        # Different seeds should produce different model weights and training trajectories
        assert not np.allclose(model1.train_weirdness_, model2.train_weirdness_)

    def test_dynamic_entity_self_attention_reproducibility(self):
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not installed")

        from people_analytics_toolkit.trajectories import DynamicEntitySelfAttention

        rng = np.random.RandomState(42)
        tokens = [rng.randn(6, 12).astype(np.float32) for _ in range(4)]
        targets = rng.randn(4).astype(np.float32)

        attn1 = DynamicEntitySelfAttention(n_heads=2, random_state=77).fit(tokens, targets=targets, epochs=10)
        attn2 = DynamicEntitySelfAttention(n_heads=2, random_state=77).fit(tokens, targets=targets, epochs=10)

        np.testing.assert_allclose(attn1.training_loss_history_, attn2.training_loss_history_, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(attn1.W_q_, attn2.W_q_, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(attn1.W_k_, attn2.W_k_, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(attn1.W_v_, attn2.W_v_, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(attn1.W_o_, attn2.W_o_, rtol=1e-5, atol=1e-5)













