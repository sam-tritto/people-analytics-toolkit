"""Unit tests for Advanced Feature Engineering algorithms in People Analytics."""

import numpy as np
import pandas as pd
import pytest
import networkx as nx
from sklearn.linear_model import LogisticRegression

from people_analytics_toolkit.cardinality import BayesianTargetEncoder, GroupedLOOZScore
from people_analytics_toolkit.memory import (
    fractional_difference,
    get_fractional_weights,
    weighted_weeks_since,
    calculate_ewma,
    scan_ewma_spans,
    calculate_ewms,
    calculate_poisson_ewma,
    double_exponential_smoothing,
    triple_exponential_smoothing,
)
from people_analytics_toolkit.compensation import (
    calculate_salary_band_midpoint,
    calculate_compa_ratio,
    calculate_range_penetration,
    classify_pay_band_status,
    is_red_circled,
    is_green_circled,
    calculate_pay_equity_gap,
    median_gap,
    calculate_median_pay_gap,
)
from people_analytics_toolkit.chaos import (
    calculate_shannon_entropy,
    rolling_shannon_entropy,
    fit_garch_volatility,
    rolling_garch_volatility,
    HawkesProcessContagion,
)
from people_analytics_toolkit.anomalies import (
    calculate_kl_divergence,
    calculate_jensen_shannon_divergence,
    compute_dtw_distance,
)
from people_analytics_toolkit.attribution import lmdi_rate_mix_decomposition, logarithmic_mean
from people_analytics_toolkit.ona import (
    calculate_degree_centrality,
    calculate_closeness_centrality,
    calculate_betweenness_centrality,
    calculate_eigenvector_centrality,
    calculate_burt_constraint,
    calculate_euler_centrality,
    calculate_collaboration_overload,
    simulate_attrition_contagion,
    compute_complete_ona_profile,
    calculate_span_of_control,
    evaluate_span_reorganization_shock,
)
from people_analytics_toolkit.equity import (
    oaxaca_blinder_decomposition,
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    ExponentiatedGradientFairness,
    calculate_individual_fairness_consistency,
)
from people_analytics_toolkit.prescriptive import (
    aipw_estimator,
    synthetic_difference_in_differences,
    CausalForestDML,
    qini_curve,
    area_under_uplift_curve,
    doubly_robust_uplift_eval,
)
from people_analytics_toolkit.trajectories import (
    ContinuousTimeMarkovChain,
    generate_metapath_walks,
    Metapath2Vec,
    DynamicEntitySelfAttention,
)
from people_analytics_toolkit.mobility import (
    calculate_time_in_position,
    calculate_stagnation_index,
)
from data.synthetic_generators import (
    generate_organizational_network_data,
    generate_algorithmic_equity_and_fairness_data,
    generate_causal_and_uplift_data,
    generate_panel_policy_data,
    generate_career_trajectory_event_log,
    generate_heterogeneous_hcm_multigraph,
    generate_reporting_hierarchy_with_reorg,
    generate_time_in_position_workforce,
)




class TestLMDI:
    def test_logarithmic_mean_boundary(self):
        assert logarithmic_mean(5.0, 5.0) == 5.0
        assert logarithmic_mean(0.0, 5.0) == 0.0
        val = logarithmic_mean(10.0, 2.0)
        assert np.isclose(val, (10 - 2) / (np.log(10) - np.log(2)))

    def test_lmdi_zero_residual(self):
        df = pd.DataFrame({
            "department": ["A", "B", "C"],
            "headcount_y0": [1000, 500, 200],
            "turnover_rate_y0": [0.20, 0.40, 0.10],
            "headcount_y1": [900, 300, 500],
            "turnover_rate_y1": [0.18, 0.35, 0.12],
        })
        decomp_df, summary = lmdi_rate_mix_decomposition(df)
        
        # Mathematical identity: Delta R == Rate Effect + Mix Effect
        assert abs(summary["residual"]) < 1e-9
        assert np.isclose(
            summary["delta_turnover_rate_actual"],
            summary["rate_effect"] + summary["mix_effect"],
            atol=1e-9
        )


class TestCardinality:
    def test_bayesian_target_encoder_shrinkage(self):
        # A tiny group of 2 with mean 1.0, and a large group of 100 with mean 0.0
        # Global mean is ~ 0.02. With m=10, the tiny group should shrink dramatically towards 0.02.
        df = pd.DataFrame({
            "group": ["TINY", "TINY"] + ["LARGE"] * 98,
            "target": [1, 1] + [0] * 98,
        })
        encoder = BayesianTargetEncoder(m=10.0)
        encoder.fit(df, "group", "target")
        
        # Tiny group mean is 1.0, but shrunk value: (2 * 1.0 + 10 * 0.02) / (2 + 10) = 2.2 / 12 ~ 0.183
        tiny_val = encoder.encoding_map_["TINY"]
        assert tiny_val < 0.25
        assert tiny_val > 0.10

    def test_bayesian_target_encoder_invalid_m(self):
        # m <= 0 in constructor should raise ValueError
        with pytest.raises(ValueError, match="m must be positive"):
            BayesianTargetEncoder(m=0.0)
        with pytest.raises(ValueError, match="m must be positive"):
            BayesianTargetEncoder(m=-5.0)

        # m mutated to <= 0 before fit should raise ValueError
        encoder = BayesianTargetEncoder(m=10.0)
        encoder.m = -1.0
        df = pd.DataFrame({"group": ["A", "B"], "target": [1, 0]})
        with pytest.raises(ValueError, match="m must be positive"):
            encoder.fit(df, "group", "target")

    def test_bayesian_target_encoder_missing_target_col(self):
        df = pd.DataFrame({"group": ["A", "B"], "target": [1, 0]})
        encoder = BayesianTargetEncoder(m=10.0)
        with pytest.raises(ValueError, match="target_col 'nonexistent' not found"):
            encoder.fit(df, "group", "nonexistent")

        with pytest.raises(ValueError, match="target_col 'nonexistent' not found"):
            encoder.fit_transform_oof(df, "group", "nonexistent")

    def test_bayesian_target_encoder_missing_group_col(self):
        df = pd.DataFrame({"group": ["A", "B"], "target": [1, 0]})
        encoder = BayesianTargetEncoder(m=10.0)
        with pytest.raises(ValueError, match="group_col 'wrong_group' not found"):
            encoder.fit(df, "wrong_group", "target")
        with pytest.raises(ValueError, match="group_col columns"):
            encoder.fit(df, ["group", "missing_col"], "target")

    def test_bayesian_target_encoder_transform_before_fit(self):
        from sklearn.exceptions import NotFittedError

        encoder = BayesianTargetEncoder(m=10.0)
        df = pd.DataFrame({"group": ["A", "B"]})
        with pytest.raises(NotFittedError, match="is not fitted"):
            encoder.transform(df, "group")

    def test_bayesian_target_encoder_oof_and_transform(self):
        df = pd.DataFrame({
            "dept": ["HR", "ENG", "HR", "ENG", "SALES"] * 10,
            "attrition": [0, 1, 0, 0, 1] * 10,
        })
        encoder = BayesianTargetEncoder(m=5.0, cv_folds=3, seed=42)
        oof = encoder.fit_transform_oof(df, "dept", "attrition")
        assert len(oof) == len(df)
        assert not oof.isna().any()

        # After fit_transform_oof, the model should be fitted and transformable
        test_df = pd.DataFrame({"dept": ["HR", "UNKNOWN"]})
        transformed = encoder.transform(test_df, "dept")
        assert len(transformed) == 2
        assert not transformed.isna().any()

    def test_bayesian_target_encoder_get_set_params_and_clone(self):
        from sklearn.base import clone

        encoder = BayesianTargetEncoder(
            m=7.5, cv_folds=4, seed=123, group_col="dept", target_col="salary", return_2d=True
        )
        params = encoder.get_params()
        assert params["m"] == 7.5
        assert params["cv_folds"] == 4
        assert params["seed"] == 123
        assert params["group_col"] == "dept"
        assert params["target_col"] == "salary"
        assert params["return_2d"] is True

        # Test set_params
        encoder.set_params(m=15.0, return_2d=False)
        assert encoder.m == 15.0
        assert encoder.return_2d is False

        # Test sklearn clone
        cloned = clone(encoder)
        assert cloned.m == 15.0
        assert cloned.group_col == "dept"
        assert cloned.is_fitted_ is False

    def test_bayesian_target_encoder_pipeline_and_gridsearch(self):
        from sklearn.pipeline import Pipeline
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import GridSearchCV

        df = pd.DataFrame({
            "dept": ["HR", "ENG", "SALES", "HR", "ENG", "SALES"] * 10,
            "salary": [50.0, 95.0, 70.0, 52.0, 98.0, 68.0] * 10,
        })
        X = df[["dept"]]
        y = df["salary"]

        pipe = Pipeline([
            ("encoder", BayesianTargetEncoder(group_col="dept", return_2d=True)),
            ("regressor", Ridge()),
        ])

        param_grid = {"encoder__m": [2.0, 10.0, 50.0]}
        grid = GridSearchCV(pipe, param_grid, cv=3)
        grid.fit(X, y)

        assert grid.best_params_["encoder__m"] in [2.0, 10.0, 50.0]
        preds = grid.predict(X)
        assert len(preds) == len(X)
        assert not np.isnan(preds).any()

    def test_bayesian_target_encoder_serialization(self, tmp_path):
        import pickle

        df = pd.DataFrame({
            "dept": ["HR", "ENG", "SALES"] * 20,
            "bonus": [5.0, 25.0, 15.0] * 20,
        })
        test_df = pd.DataFrame({"dept": ["HR", "ENG", "MARKETING"]})

        encoder = BayesianTargetEncoder(m=8.0, group_col="dept")
        encoder.fit(df, "dept", "bonus")
        expected_output = encoder.transform(test_df)

        # 1. to_dict / from_dict
        d = encoder.to_dict()
        assert d["params"]["m"] == 8.0
        assert d["fitted"]["is_fitted_"] is True
        from_dict_enc = BayesianTargetEncoder.from_dict(d)
        pd.testing.assert_series_equal(expected_output, from_dict_enc.transform(test_df))

        # 2. to_json string / from_json string
        json_str = encoder.to_json()
        assert isinstance(json_str, str)
        from_json_enc = BayesianTargetEncoder.from_json(json_str)
        pd.testing.assert_series_equal(expected_output, from_json_enc.transform(test_df))

        # 3. to_json file / from_json file
        json_path = tmp_path / "encoder.json"
        encoder.to_json(str(json_path))
        from_file_enc = BayesianTargetEncoder.from_json(str(json_path))
        pd.testing.assert_series_equal(expected_output, from_file_enc.transform(test_df))

        # 4. pickle round-trip
        pickled = pickle.dumps(encoder)
        unpickled_enc = pickle.loads(pickled)
        pd.testing.assert_series_equal(expected_output, unpickled_enc.transform(test_df))

    def test_bayesian_target_encoder_inverse_transform(self):
        from sklearn.exceptions import NotFittedError

        # 1. Unfitted error check
        unfitted = BayesianTargetEncoder(m=5.0)
        with pytest.raises(NotFittedError, match="is not fitted yet"):
            unfitted.inverse_transform(pd.Series([0.5, 0.2]))

        # 2. Exact round-trip recovery
        df = pd.DataFrame({
            "dept": ["HR", "ENG", "SALES", "HR", "ENG", "SALES"] * 10,
            "turnover": [0, 1, 0, 0, 1, 1] * 10,
        })
        encoder = BayesianTargetEncoder(m=5.0, group_col="dept")
        encoder.fit(df, "dept", "turnover")

        test_df = pd.DataFrame({"dept": ["HR", "ENG", "SALES", "ENG", "HR"]})
        encoded = encoder.transform(test_df, "dept")
        decoded = encoder.inverse_transform(encoded)

        assert isinstance(decoded, pd.Series)
        pd.testing.assert_series_equal(decoded, test_df["dept"], check_names=False)

        # 3. 2D array and return_2d=True
        encoder_2d = BayesianTargetEncoder(m=5.0, group_col="dept", return_2d=True)
        encoder_2d.fit(df, "dept", "turnover")
        encoded_2d = encoder_2d.transform(test_df)
        assert encoded_2d.ndim == 2
        decoded_2d = encoder_2d.inverse_transform(encoded_2d)
        assert decoded_2d.ndim == 2
        assert (decoded_2d.ravel() == test_df["dept"].to_numpy()).all()

        # 4. Nearest neighbor decoding with perturbed floats
        # Add slight floating-point noise
        noisy_vals = encoded.to_numpy() + np.array([0.001, -0.002, 0.0015, 0.0, -0.0005])
        decoded_noisy = encoder.inverse_transform(noisy_vals, strategy="nearest")
        assert (decoded_noisy == test_df["dept"].to_numpy()).all()

        # 5. Exact strategy with tolerance
        decoded_exact = encoder.inverse_transform(encoded.to_numpy(), strategy="exact", tolerance=1e-5)
        assert (decoded_exact == test_df["dept"].to_numpy()).all()

        # Values far from any category should yield None when strategy='exact'
        out_of_range = np.array([999.0, -50.0])
        decoded_out = encoder.inverse_transform(out_of_range, strategy="exact", tolerance=1e-3)
        assert decoded_out[0] is None
        assert decoded_out[1] is None

        # 6. NaN handling
        with_nan = pd.Series([encoded.iloc[0], np.nan, encoded.iloc[1]])
        decoded_nan = encoder.inverse_transform(with_nan)
        assert decoded_nan.iloc[0] == "HR"
        assert pd.isna(decoded_nan.iloc[1])
        assert decoded_nan.iloc[2] == "ENG"

        # 7. Tie-breaking with duplicate encoded values
        # Two categories with same smoothed value: "TIED_BIG" (count=100) vs "TIED_SMALL" (count=5)
        encoder.encoding_map_["TIED_BIG"] = 0.50
        encoder.counts_map_["TIED_BIG"] = 100
        encoder.encoding_map_["TIED_SMALL"] = 0.50
        encoder.counts_map_["TIED_SMALL"] = 5
        decoded_tie = encoder.inverse_transform([0.50])
        assert decoded_tie[0] == "TIED_BIG"

        # 8. Validation errors
        with pytest.raises(ValueError, match="strategy must be 'nearest' or 'exact'"):
            encoder.inverse_transform([0.5], strategy="invalid")
        with pytest.raises(ValueError, match="tolerance must be non-negative"):
            encoder.inverse_transform([0.5], tolerance=-0.1)
        with pytest.raises(ValueError, match="single-column"):
            encoder.inverse_transform(pd.DataFrame({"a": [1.0], "b": [2.0]}))

    def test_loo_zscore_unmasks_outlier(self):
        # Group of 5 where 4 people have modest OT [4.0, 5.0, 6.0, 5.0], and 1 person has OT=35.0
        df = pd.DataFrame({
            "cohort": ["TEAM_A"] * 5,
            "overtime": [4.0, 5.0, 6.0, 5.0, 35.0],
        })
        res = GroupedLOOZScore.compute(df, "cohort", "overtime")
        
        # In standard Z-score, the outlier inflates group mean and std, masking itself (Z ~ 1.78)
        # In LOO Z-score, the outlier is compared to the 4 peers (mean=5.0, std=0.816) -> Z ~ 36.7
        outlier_std_z = res.loc[4, "standard_z_score"]
        outlier_loo_z = res.loc[4, "loo_z_score"]
        assert outlier_loo_z > outlier_std_z * 5
        assert outlier_loo_z > 30.0

    def test_loo_zscore_validation(self):
        df = pd.DataFrame({"cohort": ["A", "B"], "metric": [10.0, 20.0]})
        with pytest.raises(ValueError, match="val_col 'nonexistent' not found"):
            GroupedLOOZScore.compute(df, "cohort", "nonexistent")
        with pytest.raises(ValueError, match="group_col 'wrong' not found"):
            GroupedLOOZScore.compute(df, "wrong", "metric")

    def test_grouped_loo_zscore_fit_transform_and_inference(self, tmp_path):
        from sklearn.base import clone
        from sklearn.exceptions import NotFittedError
        from sklearn.pipeline import Pipeline
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import GridSearchCV
        import pickle

        # 1. NotFittedError before fit()
        scorer = GroupedLOOZScore(group_col="cohort", val_col="overtime")
        test_df = pd.DataFrame({"cohort": ["TEAM_A"], "overtime": [10.0]})
        with pytest.raises(NotFittedError, match="is not fitted yet"):
            scorer.transform(test_df)

        # 2. get_params / set_params / clone
        params = scorer.get_params()
        assert params["group_col"] == "cohort"
        assert params["val_col"] == "overtime"
        assert params["min_cohort_size"] == 3
        scorer.set_params(min_cohort_size=4)
        assert scorer.min_cohort_size == 4
        cloned = clone(scorer)
        assert cloned.min_cohort_size == 4
        assert cloned.is_fitted_ is False

        # 3. Fit on training cohorts
        train_df = pd.DataFrame({
            "cohort": ["TEAM_A"] * 5 + ["TEAM_B"] * 5,
            "overtime": [4.0, 5.0, 6.0, 5.0, 35.0] + [10.0, 12.0, 11.0, 10.5, 11.5],
        })
        scorer = GroupedLOOZScore(group_col="cohort", val_col="overtime", min_cohort_size=3)
        res_train = scorer.fit_transform(train_df)
        assert scorer.is_fitted_ is True
        assert "TEAM_A" in scorer.cohort_stats_
        assert "TEAM_B" in scorer.cohort_stats_
        # In-sample LOO on outlier
        assert res_train.loc[4, "loo_z_score"] > 30.0

        # 4. Inference on new out-of-sample data
        # Even a single observation in test_df uses the fitted cohort statistics
        inference_df = pd.DataFrame({
            "cohort": ["TEAM_A", "TEAM_B", "NEW_UNKNOWN_COHORT"],
            "overtime": [5.0, 11.0, 100.0],
        })
        res_infer = scorer.transform(inference_df)
        assert len(res_infer) == 3
        # TEAM_A overtime=5.0 is exactly the peer mean, so z ~ -0.45 or close to 0 vs full mu=11.0
        assert res_infer.loc[0, "standard_z_score"] < 0.0
        # Unknown cohort falls back to global stats without error
        assert not np.isnan(res_infer.loc[2, "standard_z_score"])
        assert res_infer.loc[2, "cohort_size"] == 0

        # 5. Pipeline & GridSearchCV
        pipe = Pipeline([
            ("loo", GroupedLOOZScore(group_col="cohort", val_col="overtime", return_scores_only=True, return_2d=True)),
            ("reg", Ridge()),
        ])
        grid = GridSearchCV(pipe, {"loo__min_cohort_size": [2, 3]}, cv=2)
        y = np.array([1, 2, 3, 4, 5, 2, 3, 2, 3, 2])
        grid.fit(train_df, y)
        assert grid.best_params_["loo__min_cohort_size"] in [2, 3]
        preds = grid.predict(inference_df)
        assert len(preds) == len(inference_df)

        # 6. Serialization (to_dict/from_dict, to_json/from_json, pickle)
        d = scorer.to_dict()
        assert d["fitted"]["is_fitted_"] is True
        from_dict_scorer = GroupedLOOZScore.from_dict(d)
        pd.testing.assert_frame_equal(res_infer, from_dict_scorer.transform(inference_df))

        json_str = scorer.to_json()
        from_json_scorer = GroupedLOOZScore.from_json(json_str)
        pd.testing.assert_frame_equal(res_infer, from_json_scorer.transform(inference_df))

        json_file = tmp_path / "loo_scorer.json"
        scorer.to_json(str(json_file))
        from_file_scorer = GroupedLOOZScore.from_json(str(json_file))
        pd.testing.assert_frame_equal(res_infer, from_file_scorer.transform(inference_df))

        pickled = pickle.dumps(scorer)
        unpickled_scorer = pickle.loads(pickled)
        pd.testing.assert_frame_equal(res_infer, unpickled_scorer.transform(inference_df))


class TestMemory:
    def test_fractional_weights(self):
        weights = get_fractional_weights(d=0.5, length=10)
        assert weights[0] == 1.0
        assert weights[1] == -0.5
        assert len(weights) <= 10

    def test_fractional_weights_invalid_d(self):
        with pytest.raises(ValueError, match=r"Differencing order d must be in \[0, 1\]"):
            get_fractional_weights(d=-0.1, length=10)
        with pytest.raises(ValueError, match=r"Differencing order d must be in \[0, 1\]"):
            get_fractional_weights(d=1.5, length=10)
        with pytest.raises(ValueError, match=r"Differencing order d must be in \[0, 1\]"):
            get_fractional_weights(d=2.5, length=10)

    def test_fractional_difference_shape(self):
        s = pd.Series(np.arange(100, dtype=float))
        diff = fractional_difference(s, d=0.4)
        assert len(diff) == 100
        assert not np.all(np.isnan(diff))

    def test_fractional_difference_invalid_d(self):
        s = pd.Series(np.arange(20, dtype=float))
        with pytest.raises(ValueError, match=r"Differencing order d must be in \[0, 1\]"):
            fractional_difference(s, d=-0.5)
        with pytest.raises(ValueError, match=r"Differencing order d must be in \[0, 1\]"):
            fractional_difference(s, d=1.2)
        with pytest.raises(ValueError, match=r"Differencing order d must be in \[0, 1\]"):
            fractional_difference(s, d=3.0)

    def test_fractional_difference_boundary_d(self):
        s = pd.Series(np.arange(20, dtype=float))
        # d = 0.0 should preserve values (no differencing)
        diff_0 = fractional_difference(s, d=0.0)
        assert len(diff_0) == 20
        np.testing.assert_allclose(diff_0.to_numpy(), s.to_numpy())

        # d = 1.0 is full first differencing
        diff_1 = fractional_difference(s, d=1.0)
        # first element is NaN because window requires 2 values
        assert np.isnan(diff_1.iloc[0])
        np.testing.assert_allclose(diff_1.iloc[1:].to_numpy(), 1.0)

    def test_weighted_weeks_since_decay(self):
        events = pd.DataFrame({
            "employee_id": ["E1", "E1"],
            "award_week": [10, 20],
            "award_tier": ["Gold", "Silver"],
        })
        res_w20 = weighted_weeks_since(events, current_week=20, half_life_weeks=8.0)
        # At week 20, the week 20 Silver award has elapsed 0 weeks (pulse = 0.5)
        # and the week 10 Gold award has elapsed 10 weeks
        pulse = res_w20.loc[0, "total_decayed_recognition_pulse"]
        assert pulse > 0.5  # Silver + decayed Gold

    def test_scan_ewma_spans_unsupervised(self):
        rng = np.random.default_rng(42)
        series = pd.Series(rng.normal(50, 5, size=60))
        scan_df = scan_ewma_spans(series, spans=[3, 7, 14, 28])

        assert len(scan_df) == 4
        assert list(scan_df["span"]) == [3, 7, 14, 28]
        expected_cols = [
            "span", "alpha", "half_life", "rmse_one_step", "mae_one_step",
            "rmse_in_sample", "correlation_with_raw", "noise_reduction_ratio",
            "aic", "bic", "is_optimal"
        ]
        for col in expected_cols:
            assert col in scan_df.columns

        # Verify exactly one optimal candidate
        assert scan_df["is_optimal"].sum() == 1
        best_span = scan_df.attrs["best_span"]
        assert best_span in [3, 7, 14, 28]
        assert scan_df.loc[scan_df["is_optimal"], "span"].iloc[0] == best_span

        # Shorter span has higher noise reduction ratio than longer span (less smoothing)
        assert scan_df.loc[scan_df["span"] == 3, "noise_reduction_ratio"].iloc[0] > \
               scan_df.loc[scan_df["span"] == 28, "noise_reduction_ratio"].iloc[0]

    def test_scan_ewma_spans_with_half_lives_and_supervised(self):
        rng = np.random.default_rng(42)
        x = pd.Series(rng.normal(100, 10, size=50))
        # Supervised target generated with an 8-period half-life decay
        target = calculate_ewma(x, half_life=8.0) + rng.normal(0, 1, size=50)

        # Candidate scan using half_lives
        scan_hl = scan_ewma_spans(
            x,
            half_lives=[2.0, 4.0, 8.0, 16.0],
            target_series=target,
            criterion="target_correlation",
        )

        assert len(scan_hl) == 4
        assert "target_correlation" in scan_hl.columns
        assert "target_r2" in scan_hl.columns
        assert "target_rmse" in scan_hl.columns
        # Candidate closest to true generating half-life (8.0) should have high correlation and be selected
        assert (scan_hl["target_correlation"] > 0.7).any()
        assert scan_hl["is_optimal"].sum() == 1
        assert scan_hl.loc[scan_hl["is_optimal"], "half_life"].iloc[0] == 8.0

    def test_scan_ewma_spans_validation(self):
        # Short series
        with pytest.raises(ValueError, match="at least 4 observations"):
            scan_ewma_spans(pd.Series([1.0, 2.0, 3.0]))

        # Invalid span
        with pytest.raises(ValueError, match="Span must be > 1"):
            scan_ewma_spans(pd.Series(range(10)), spans=[1, 5])

        # Invalid half_life
        with pytest.raises(ValueError, match="half_life must be positive"):
            scan_ewma_spans(pd.Series(range(10)), half_lives=[-2.0, 4.0])

    def test_ewms_count_decay(self):
        # 1 incident at t=0, none until t=4 where another occurs
        counts = pd.Series([1, 0, 0, 0, 1])
        ewms = calculate_ewms(counts, half_life=4.0)
        # At t=0: exactly 1.0
        assert np.isclose(ewms.iloc[0], 1.0)
        # At t=2 (half a half-life): 1 / sqrt(2) ~ 0.7071
        assert np.isclose(ewms.iloc[2], 1.0 / np.sqrt(2.0), atol=1e-4)
        # At t=4: exactly 1 half-life after first incident (decayed to 0.5) + new incident (1.0) = 1.5
        assert np.isclose(ewms.iloc[4], 1.5, atol=1e-4)

    def test_ewms_grouped(self):
        df = pd.DataFrame({
            "employee_id": ["E1", "E1", "E2", "E2"],
            "incidents": [2, 0, 0, 1],
        })
        ewms = calculate_ewms(df["incidents"], half_life=2.0, group_col=df["employee_id"])
        # E1 at t=0 has 2 incidents
        assert np.isclose(ewms.iloc[0], 2.0)
        # E1 at t=1 has decayed from 2 to 2 * (1 / sqrt(2)) ~ 1.4142
        assert np.isclose(ewms.iloc[1], 2.0 / np.sqrt(2.0), atol=1e-4)
        # E2 at t=0 has 0
        assert np.isclose(ewms.iloc[2], 0.0)
        # E2 at t=1 has 1
        assert np.isclose(ewms.iloc[3], 1.0)

    def test_poisson_ewma(self):
        counts = pd.Series([1, 2, 1, 3, 2, 8])
        intensity, z_scores = calculate_poisson_ewma(counts, alpha=0.3, baseline_mean=2.0)
        assert len(intensity) == 6
        assert len(z_scores) == 6
        # At the spike of 8, z-score should rise significantly
        assert z_scores.iloc[-1] > z_scores.iloc[0]

    def test_double_exponential_smoothing(self):
        # Linear growth sequence: 10, 20, 30, 40, 50, 60...
        t = np.arange(40, dtype=float)
        trend_series = pd.Series(50.0 + 2.5 * t)
        fitted, decomp, meta = double_exponential_smoothing(trend_series, forecast_periods=5)
        
        assert len(fitted) == 40
        assert "level" in decomp.columns
        assert "trend" in decomp.columns
        # Trend component should be close to 2.5
        assert np.isclose(decomp["trend"].iloc[-1], 2.5, atol=0.5)
        assert len(meta["forecast"]) == 5

    def test_triple_exponential_smoothing(self):
        # 3 cycles of length 12 with linear trend and seasonal variation
        t = np.arange(48, dtype=float)
        seasonal_series = pd.Series(100.0 + 1.5 * t + 10.0 * np.sin(2 * np.pi * t / 12.0))
        fitted, decomp, meta = triple_exponential_smoothing(
            seasonal_series, seasonal_periods=12, forecast_periods=6
        )
        
        assert len(fitted) == 48
        assert "level" in decomp.columns
        assert "trend" in decomp.columns
        assert "seasonal" in decomp.columns
        assert len(meta["forecast"]) == 6
        assert meta["seasonal_periods"] == 12




class TestChaos:
    def test_shannon_entropy_bounds(self):
        # Uniform distribution of 4 categories -> H = log2(4) = 2.0
        s_uniform = pd.Series(["A", "B", "C", "D"] * 25)
        h_uniform = calculate_shannon_entropy(s_uniform)
        assert np.isclose(h_uniform, 2.0, atol=1e-3)
        
        # Single category -> H = 0.0
        s_single = pd.Series(["A"] * 100)
        h_single = calculate_shannon_entropy(s_single)
        assert h_single == 0.0

    def test_hawkes_contagion_intensity(self):
        hawkes = HawkesProcessContagion(mu=0.1, alpha=0.8, beta=1.0)
        hawkes.events_ = [5.0]
        # At t=4 (before event), intensity should be baseline mu
        int_before = hawkes.predict_intensity([4.0])[0]
        assert np.isclose(int_before, 0.1)
        # At t=5.1 (right after event), intensity should spike
        int_after = hawkes.predict_intensity([5.1])[0]
        assert int_after > 0.5

    def test_hawkes_fit_insufficient_events_warns(self):
        hawkes = HawkesProcessContagion(mu=0.2, alpha=0.5, beta=1.0)
        with pytest.warns(UserWarning, match="requires at least 3 events"):
            hawkes.fit([1.0, 2.0])
        assert hawkes.mu == 0.2
        assert hawkes.converged_ is False

        with pytest.warns(UserWarning, match="requires at least 3 events"):
            hawkes.fit([])
        assert hawkes.converged_ is False

    def test_hawkes_fit_optimizer_failure(self, monkeypatch):
        from scipy.optimize import OptimizeResult
        import people_analytics_toolkit.chaos as chaos_module

        def fake_minimize(*args, **kwargs):
            return OptimizeResult(success=False, message="Iteration limit reached", x=np.array([0.5, 0.5, 1.0]))

        monkeypatch.setattr(chaos_module, "minimize", fake_minimize)

        hawkes = HawkesProcessContagion(mu=0.2, alpha=0.5, beta=1.0)
        events = [1.0, 3.0, 5.0, 8.0]
        with pytest.warns(UserWarning, match="MLE optimization failed to converge"):
            hawkes.fit(events)
        assert hawkes.converged_ is False
        assert hawkes.mu == 0.2  # retained original parameter

    def test_hawkes_fit_success(self):
        hawkes = HawkesProcessContagion(mu=0.2, alpha=0.5, beta=1.0)
        events = [1.0, 1.5, 2.0, 10.0, 10.5, 11.0, 20.0, 20.2, 21.0]
        hawkes.fit(events)
        assert hawkes.is_fitted_ is True
        assert hawkes.converged_ is True
        assert hawkes.mu > 0
        assert hawkes.alpha >= 0
        assert hawkes.beta > 0

    def test_hawkes_param_validation(self):
        with pytest.raises(ValueError, match="mu must be positive"):
            HawkesProcessContagion(mu=-0.1)
        with pytest.raises(ValueError, match="alpha must be non-negative"):
            HawkesProcessContagion(alpha=-0.5)
        with pytest.raises(ValueError, match="beta must be positive"):
            HawkesProcessContagion(beta=0.0)

        hawkes = HawkesProcessContagion()
        with pytest.raises(ValueError, match="max_t must be positive"):
            hawkes.fit([1.0, 2.0, 3.0], max_t=-5.0)
        with pytest.raises(ValueError, match="cannot be less than latest event"):
            hawkes.fit([1.0, 2.0, 10.0], max_t=5.0)

    def test_hawkes_branching_ratio_and_stationarity(self):
        # Subcritical (stationary)
        hawkes_sub = HawkesProcessContagion(mu=0.2, alpha=0.4, beta=1.0)
        assert np.isclose(hawkes_sub.branching_ratio, 0.4)
        assert hawkes_sub.is_stationary is True

        # Supercritical (non-stationary / explosive cascade)
        hawkes_super = HawkesProcessContagion(mu=0.2, alpha=1.5, beta=1.0)
        assert np.isclose(hawkes_super.branching_ratio, 1.5)
        assert hawkes_super.is_stationary is False

    def test_hawkes_simulate(self):
        hawkes = HawkesProcessContagion(mu=0.3, alpha=0.5, beta=1.0)
        events = hawkes.simulate(T=40.0, seed=42)

        assert isinstance(events, np.ndarray)
        assert len(events) > 0
        assert np.all(events >= 0.0)
        assert np.all(events <= 40.0)
        # Monotonically increasing
        assert np.all(np.diff(events) > 0)

        # Test max_events cap
        capped_events = hawkes.simulate(T=100.0, max_events=5, seed=42)
        assert len(capped_events) == 5

        # Error cases
        with pytest.raises(ValueError, match="horizon T must be positive"):
            hawkes.simulate(T=-10.0)
        with pytest.raises(ValueError, match="max_events must be positive"):
            hawkes.simulate(T=10.0, max_events=0)

    def test_hawkes_goodness_of_fit(self):
        hawkes = HawkesProcessContagion(mu=0.25, alpha=0.45, beta=1.1)
        sim_events = hawkes.simulate(T=80.0, seed=123)

        hawkes.fit(sim_events, max_t=80.0)
        assert hawkes.is_fitted_ is True

        gof = hawkes.goodness_of_fit()
        assert "ks_stat" in gof
        assert "ks_pvalue" in gof
        assert 0.0 <= gof["ks_pvalue"] <= 1.0
        assert "tau" in gof
        assert len(gof["tau"]) == len(sim_events)
        assert "transformed_uniforms" in gof
        assert "aic" in gof
        assert "bic" in gof
        assert "log_likelihood" in gof
        assert gof["branching_ratio"] == hawkes.branching_ratio
        assert gof["is_stationary"] == hawkes.is_stationary

        # Error case: unfitted with no events
        unfitted = HawkesProcessContagion()
        with pytest.raises(ValueError, match="No events provided"):
            unfitted.goodness_of_fit()

    def test_hawkes_serialization(self, tmp_path):
        import pickle

        hawkes = HawkesProcessContagion(mu=0.3, alpha=0.5, beta=1.2)
        sim_events = hawkes.simulate(T=30.0, seed=42)
        hawkes.fit(sim_events)

        # 1. to_dict / from_dict
        d = hawkes.to_dict()
        assert d["fitted"]["is_fitted_"] is True
        assert np.isclose(d["fitted"]["branching_ratio"], hawkes.branching_ratio)
        from_dict_model = HawkesProcessContagion.from_dict(d)
        assert np.isclose(from_dict_model.mu, hawkes.mu)
        assert np.isclose(from_dict_model.alpha, hawkes.alpha)
        assert np.isclose(from_dict_model.beta, hawkes.beta)
        assert from_dict_model.events_ == hawkes.events_

        # 2. to_json / from_json
        json_str = hawkes.to_json()
        from_json_model = HawkesProcessContagion.from_json(json_str)
        assert np.isclose(from_json_model.branching_ratio, hawkes.branching_ratio)

        # 3. to_json file
        json_file = tmp_path / "hawkes.json"
        hawkes.to_json(str(json_file))
        from_file_model = HawkesProcessContagion.from_json(str(json_file))
        assert np.isclose(from_file_model.branching_ratio, hawkes.branching_ratio)

        # 4. pickle round-trip
        pickled = pickle.dumps(hawkes)
        unpickled_model = pickle.loads(pickled)
        assert np.isclose(unpickled_model.branching_ratio, hawkes.branching_ratio)


class TestAnomalies:
    def test_kl_divergence_properties(self):
        p = np.array([0.5, 0.5])
        q = np.array([0.5, 0.5])
        assert np.isclose(calculate_kl_divergence(p, q), 0.0, atol=1e-5)
        
        p_distinct = np.array([0.9, 0.1])
        assert calculate_kl_divergence(p_distinct, q) > 0.2

    def test_dtw_distance(self):
        s1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        s2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        dist_identical, path, _ = compute_dtw_distance(s1, s2)
        assert np.isclose(dist_identical, 0.0)
        assert len(path) == 5
        
        # Shifted series has lower DTW distance than uncorrelated series
        s_shifted = np.array([1.0, 1.0, 2.0, 3.0, 4.0])
        dist_shifted, _, _ = compute_dtw_distance(s1, s_shifted)
        assert dist_shifted > 0

    def test_autoencoder_weirdness_anomaly_scoring(self):
        from people_analytics_toolkit.anomalies import AutoencoderWeirdnessDetector
        rng = np.random.default_rng(42)
        X_normal = rng.normal(0, 1, size=(60, 6))
        detector = AutoencoderWeirdnessDetector(latent_dim=2, hidden_dim=4, epochs=30, seed=42)
        detector.fit(X_normal)
        
        # Test transform
        weirdness, breakdown = detector.transform(X_normal)
        assert len(weirdness) == 60
        assert breakdown.shape == (60, 6)
        
        # Injected extreme outlier should have higher weirdness
        X_outlier = np.array([[10.0, -12.0, 15.0, 8.0, -9.0, 11.0]])
        outlier_weirdness, _ = detector.transform(X_outlier)
        assert outlier_weirdness.iloc[0] > weirdness.mean() * 3

    def test_autoencoder_fit_transform(self):
        from people_analytics_toolkit.anomalies import AutoencoderWeirdnessDetector
        rng = np.random.default_rng(42)
        df_normal = pd.DataFrame(rng.normal(0, 1, size=(50, 4)), columns=["a", "b", "c", "d"])
        detector = AutoencoderWeirdnessDetector(latent_dim=2, hidden_dim=4, epochs=20, seed=42)
        
        weirdness, breakdown = detector.fit_transform(df_normal)
        assert len(weirdness) == 50
        assert breakdown.shape == (50, 4)
        assert list(breakdown.columns) == ["a", "b", "c", "d"]
        assert detector.model_ is not None
        assert detector.threshold_ is not None

    def test_autoencoder_threshold_selection_and_predict(self):
        from people_analytics_toolkit.anomalies import AutoencoderWeirdnessDetector
        rng = np.random.default_rng(42)
        X_normal = rng.normal(0, 1, size=(80, 4))
        detector = AutoencoderWeirdnessDetector(latent_dim=2, hidden_dim=4, epochs=30, contamination=0.05, seed=42)
        detector.fit(X_normal)
        
        # Default percentile threshold
        assert detector.threshold_ is not None
        preds = detector.predict(X_normal)
        assert isinstance(preds, pd.Series)
        assert preds.dtype == bool
        assert len(preds) == 80
        # With contamination=0.05, roughly 5% (~4 items) flagged as anomalies on training set
        assert preds.sum() <= 8
        
        # Outlier prediction
        X_outlier = np.array([[15.0, -20.0, 18.0, -14.0]])
        pred_outlier = detector.predict(X_outlier)
        assert bool(pred_outlier.iloc[0]) is True
        
        # Select threshold via IQR (Tukey's fences)
        iqr_thresh = detector.select_threshold(method="iqr")
        assert isinstance(iqr_thresh, float)
        assert iqr_thresh > 0
        
        # Select threshold via Gaussian
        gauss_thresh = detector.select_threshold(method="gaussian")
        assert isinstance(gauss_thresh, float)
        assert gauss_thresh > 0
        
        # Select threshold on new data
        X_eval = rng.normal(0, 1, size=(40, 4))
        eval_thresh = detector.select_threshold(X=X_eval, method="percentile", contamination=0.10)
        assert isinstance(eval_thresh, float)
        assert eval_thresh > 0
        
        # Decision function matches transform
        scores = detector.decision_function(X_eval)
        scores_tf, _ = detector.transform(X_eval)
        assert np.allclose(scores, scores_tf)

    def test_autoencoder_serialization_json_and_pickle(self, tmp_path):
        from people_analytics_toolkit.anomalies import AutoencoderWeirdnessDetector
        rng = np.random.default_rng(42)
        df_normal = pd.DataFrame(rng.normal(0, 1, size=(60, 4)), columns=["f1", "f2", "f3", "f4"])
        detector = AutoencoderWeirdnessDetector(latent_dim=2, hidden_dim=4, epochs=25, seed=42)
        detector.fit(df_normal)
        
        scores_orig, _ = detector.transform(df_normal)
        preds_orig = detector.predict(df_normal)
        
        # Test JSON save and load
        json_path = tmp_path / "detector.json"
        detector.save(json_path)
        loaded_json = AutoencoderWeirdnessDetector.load(json_path)
        
        assert loaded_json.latent_dim == detector.latent_dim
        assert loaded_json.feature_names_ == detector.feature_names_
        assert np.isclose(loaded_json.threshold_, detector.threshold_)
        
        scores_json, _ = loaded_json.transform(df_normal)
        np.testing.assert_allclose(scores_orig, scores_json, rtol=1e-4, atol=1e-5)
        preds_json = loaded_json.predict(df_normal)
        assert (preds_orig == preds_json).all()
        
        # Test Pickle save and load
        pkl_path = tmp_path / "detector.pkl"
        detector.save(pkl_path)
        loaded_pkl = AutoencoderWeirdnessDetector.load(pkl_path)
        
        scores_pkl, _ = loaded_pkl.transform(df_normal)
        np.testing.assert_allclose(scores_orig, scores_pkl, rtol=1e-4, atol=1e-5)
        preds_pkl = loaded_pkl.predict(df_normal)
        assert (preds_orig == preds_pkl).all()
        
        # Test get_params and set_params
        params = detector.get_params()
        assert params["latent_dim"] == 2
        detector.set_params(latent_dim=3)
        assert detector.latent_dim == 3



class TestRemainingModels:
    def test_garch_volatility_fit(self):
        from people_analytics_toolkit.chaos import fit_garch_volatility
        rng = np.random.default_rng(42)
        series = pd.Series(rng.normal(100, 5, size=80))
        vol, meta = fit_garch_volatility(series)
        assert len(vol) == 80
        assert (vol > 0).all()

    def test_rolling_garch_volatility(self):
        rng = np.random.default_rng(42)
        # 1. Single series with rolling window and stride step
        series = pd.Series(rng.normal(40, 4, size=60))
        roll_vol = rolling_garch_volatility(series, window=25, min_periods=15, step=2)
        assert len(roll_vol) == 60
        # First 14 points (min_periods - 1) should be NaN
        assert roll_vol.iloc[:14].isna().all()
        # Points from index 14 onward should be valid positive volatilities
        valid_part = roll_vol.iloc[14:]
        assert valid_part.notna().all()
        assert (valid_part > 0).all()

        # Annualized scaling test
        ann_vol = rolling_garch_volatility(series, window=25, min_periods=15, step=2, annualize=True, periods_per_year=52.0)
        assert np.isclose(ann_vol.iloc[-1], roll_vol.iloc[-1] * np.sqrt(52.0))

        # 2. DataFrame level with group_col
        df = pd.DataFrame({
            "store_id": ["Store_A"] * 35 + ["Store_B"] * 35,
            "weekly_hours": rng.normal(500, 30, size=70),
        })
        group_vol = rolling_garch_volatility(df, val_col="weekly_hours", group_col="store_id")
        assert len(group_vol) == 70
        assert group_vol.notna().all()
        assert (group_vol > 0).all()

        # 3. Rolling window within groups
        group_roll = rolling_garch_volatility(df, val_col="weekly_hours", group_col="store_id", window=20, min_periods=12)
        assert len(group_roll) == 70
        # Each group should have its own initial NaN warmup
        assert group_roll.iloc[:11].isna().all()
        assert group_roll.iloc[35:46].isna().all()
        assert group_roll.iloc[20:35].notna().all()
        assert group_roll.iloc[55:70].notna().all()

    def test_hazard_embeddings(self):
        from people_analytics_toolkit.memory import fit_hazard_embeddings
        df = pd.DataFrame({
            "tenure_weeks": [5, 12, 26, 52, 100],
            "turnover_event": [1, 1, 0, 1, 0],
        })
        res_df, models = fit_hazard_embeddings(df, "tenure_weeks", "turnover_event")
        assert "survival_prob_embedding" in res_df.columns
        assert "cumulative_hazard_embedding" in res_df.columns
        assert (res_df["survival_prob_embedding"] >= 0.0).all()
        assert (res_df["survival_prob_embedding"] <= 1.0).all()


class TestNewFiveFeatures:
    def test_loo_expected_differential(self):
        from people_analytics_toolkit.attribution import LeaveOneOutExpectedDifferential
        df = pd.DataFrame({
            "shift_id": [f"S{i}" for i in range(40)],
            "manager_id": ["M1"] * 20 + ["M2"] * 20,
            "foot_traffic": [500] * 40,
            "staff_hours": [40] * 40,
            "throughput": [150] * 20 + [120] * 20,  # M1 is consistently +30 higher under identical context
        })
        loo_ed = LeaveOneOutExpectedDifferential()
        shift_df, summary = loo_ed.fit_transform(
            df,
            entity_col="manager_id",
            target_col="throughput",
            context_cols=["foot_traffic", "staff_hours"],
        )
        assert "loo_expected_differential" in shift_df.columns
        m1_alpha = summary.loc[summary["manager_id"] == "M1", "loo_expected_differential"].values[0]
        m2_alpha = summary.loc[summary["manager_id"] == "M2", "loo_expected_differential"].values[0]
        assert m1_alpha > m2_alpha

    def test_pyod_lof_anomaly_detection(self):
        from people_analytics_toolkit.anomalies import fit_pyod_local_outlier_factor
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, size=(80, 3))
        # Add 3 clear local anomalies
        X = np.vstack([X, [[8.0, 8.0, 8.0], [-7.0, -7.0, -7.0], [9.0, -9.0, 9.0]]])
        df = pd.DataFrame(X, columns=["f1", "f2", "f3"])
        scores, outliers, meta = fit_pyod_local_outlier_factor(df, ["f1", "f2", "f3"], contamination=0.05)
        assert len(scores) == 83
        assert outliers.iloc[-1] == 1  # Injected outlier flagged

    def test_hmm_regime_decoding(self):
        from people_analytics_toolkit.chaos import fit_workforce_hmm_regimes
        rng = np.random.default_rng(42)
        # 3 regimes
        s1 = rng.normal(5, 1, size=(30, 2))
        s2 = rng.normal(20, 2, size=(30, 2))
        s3 = rng.normal(40, 3, size=(30, 2))
        df = pd.DataFrame(np.vstack([s1, s2, s3]), columns=["ot", "callout"])
        res_df, profile, meta = fit_workforce_hmm_regimes(df, ["ot", "callout"], n_components=3, seed=42)
        assert "hmm_latent_state" in res_df.columns
        assert len(profile) == 3
        # Posterior probabilities sum to 1
        probs = res_df[["prob_latent_state_0", "prob_latent_state_1", "prob_latent_state_2"]].sum(axis=1)
        assert np.allclose(probs, 1.0)

    def test_lcss_and_edr_gap_tolerance(self):
        from people_analytics_toolkit.anomalies import compute_lcss_distance, compute_edr_distance
        s1 = np.array([10.0, 10.0, 10.0, 10.0, 20.0, 20.0, 20.0, 20.0])
        # s2 is identical except has a 1-point spike/gap at index 2
        s2 = np.array([10.0, 10.0, 0.0, 10.0, 20.0, 20.0, 20.0, 20.0])
        lcss_len, lcss_dist = compute_lcss_distance(s1, s2, epsilon=1.0, delta=2)
        # 7 out of 8 points match perfectly
        assert lcss_len == 7
        assert lcss_dist < 0.20
        edr_dist = compute_edr_distance(s1, s2, epsilon=1.0)
        assert edr_dist <= 0.25

    def test_shap_interaction_decomposition(self):
        try:
            import lightgbm as lgb
        except (ImportError, OSError):
            pytest.skip("lightgbm is not available or libomp is missing")
        from people_analytics_toolkit.attribution import compute_shap_interaction_matrix
        rng = np.random.default_rng(42)
        n = 120
        x1 = rng.uniform(0, 10, size=n)
        x2 = rng.uniform(0, 10, size=n)
        # Deliberate interaction: y is positive only when both x1 > 5 and x2 > 5
        y = ((x1 > 5) & (x2 > 5)).astype(int)
        df = pd.DataFrame({"x1": x1, "x2": x2})
        inter_3d, mean_abs, pairs_df = compute_shap_interaction_matrix(df, y, is_classification=True, seed=42)
        assert inter_3d.shape == (n, 2, 2)
        assert np.isclose(mean_abs[0, 1], mean_abs[1, 0])
        assert pairs_df.loc[0, "interaction_strength"] > 0.0

    def test_career_movement_embeddings(self):
        from people_analytics_toolkit.mobility import CareerMovementEmbeddings
        sequences = [
            ["Cashier", "Sales_Associate", "Merchandising_Lead", "Department_Supervisor"],
            ["Cashier", "Customer_Service_Specialist", "Department_Supervisor"],
            ["Stocker", "Inventory_Control_Specialist", "Operations_Lead", "Department_Supervisor"],
            ["Stocker", "Inventory_Control_Specialist", "Operations_Lead"],
            ["Sales_Associate", "Merchandising_Lead", "Department_Supervisor"],
        ] * 10
        cme = CareerMovementEmbeddings(embedding_dim=4, max_steps=3, seed=42)
        cme.fit(sequences)
        assert cme.role_embeddings_ is not None
        assert len(cme.role_to_idx_) >= 6
        # Check unit norm
        for i in range(len(cme.role_embeddings_)):
            assert np.isclose(np.linalg.norm(cme.role_embeddings_[i]), 1.0)
            
        candidates = pd.DataFrame([
            {"employee_id": "E1", "current_role": "Operations_Lead", "career_history": ["Stocker", "Operations_Lead"]},
            {"employee_id": "E2", "current_role": "Cashier", "career_history": ["Cashier"]},
        ])
        rec = cme.recommend_succession_candidates("Department_Supervisor", candidates)
        assert len(rec) == 2
        assert rec.iloc[0]["employee_id"] == "E1"
        row0 = rec.iloc[0]
        expected_score = round(0.5 * row0["embedding_cosine_similarity"] + 0.5 * row0["transition_reachability"], 4)
        assert np.isclose(row0["succession_similarity"], expected_score, atol=1e-4)
        # Ensure redundant columns are not present
        assert "trajectory_similarity_score" not in rec.columns
        assert "career_path_string" not in rec.columns
        assert "career_history" in rec.columns
        assert "succession_similarity" in rec.columns

        # Check custom weights
        rec_custom = cme.recommend_succession_candidates(
            "Department_Supervisor", candidates, cosine_weight=0.8, reachability_weight=0.2
        )
        row0_c = rec_custom.iloc[0]
        expected_custom = round(0.8 * row0_c["embedding_cosine_similarity"] + 0.2 * row0_c["transition_reachability"], 4)
        assert np.isclose(row0_c["succession_similarity"], expected_custom, atol=1e-4)

    def test_career_movement_embeddings_fallback_and_new_roles(self):
        from people_analytics_toolkit.mobility import CareerMovementEmbeddings
        sequences = [
            ["Cashier", "Sales_Associate", "Merchandising_Lead", "Department_Supervisor"],
            ["Cashier", "Customer_Service_Specialist", "Department_Supervisor"],
            ["Stocker", "Inventory_Control_Specialist", "Operations_Lead", "Department_Supervisor"],
        ] * 10
        cme = CareerMovementEmbeddings(embedding_dim=4, max_steps=3, seed=42)
        cme.fit(sequences)

        # 1. Fallback for unseen role
        # Default fallback="mean" returns unit-norm vector
        fb_emb = cme.get_role_embedding("Unknown_Role")
        assert fb_emb.shape == (cme.embedding_dim,)
        assert np.isclose(np.linalg.norm(fb_emb), 1.0)

        # Fallback=None raises KeyError
        with pytest.raises(KeyError, match="Role 'Unknown_Role' not found"):
            cme.get_role_embedding("Unknown_Role", fallback=None)

        # 2. get_associate_embedding with unseen roles
        assoc_emb = cme.get_associate_embedding(["Unknown_A", "Unknown_B"])
        assert np.isclose(np.linalg.norm(assoc_emb), 1.0)

        # 3. transform_new_roles with context
        new_seqs = [["Cashier", "Self_Checkout_Attendant", "Sales_Associate"]]
        new_embs = cme.transform_new_roles(new_seqs, method="context", update_vocab=True)
        assert "Self_Checkout_Attendant" in new_embs
        assert np.isclose(np.linalg.norm(new_embs["Self_Checkout_Attendant"]), 1.0)
        # Because update_vocab=True, role is now in vocabulary
        assert "Self_Checkout_Attendant" in cme.role_to_idx_
        retrieved = cme.get_role_embedding("Self_Checkout_Attendant", fallback=None)
        np.testing.assert_allclose(retrieved, new_embs["Self_Checkout_Attendant"])

        # 4. transform_new_roles with dict of neighbor roles
        dict_new = cme.transform_new_roles({"Inventory_Clerk": ["Stocker", "Operations_Lead"]})
        assert "Inventory_Clerk" in dict_new
        assert np.isclose(np.linalg.norm(dict_new["Inventory_Clerk"]), 1.0)

    def test_career_movement_embeddings_save_load(self, tmp_path):
        from people_analytics_toolkit.mobility import CareerMovementEmbeddings
        sequences = [
            ["Cashier", "Sales_Associate", "Department_Supervisor"],
            ["Stocker", "Operations_Lead", "Department_Supervisor"],
            ["Cashier", "Customer_Service_Specialist", "Department_Supervisor"],
        ] * 10
        cme = CareerMovementEmbeddings(embedding_dim=4, max_steps=3, seed=42)
        cme.fit(sequences)

        # 1. JSON save/load
        json_path = tmp_path / "cme_model.json"
        cme.save(str(json_path))
        loaded_json = CareerMovementEmbeddings.load(str(json_path))
        assert loaded_json.embedding_dim == cme.embedding_dim
        assert loaded_json.role_to_idx_ == cme.role_to_idx_
        np.testing.assert_allclose(loaded_json.role_embeddings_, cme.role_embeddings_, atol=1e-5)

        # 2. Pickle save/load
        pkl_path = tmp_path / "cme_model.pkl"
        cme.save(str(pkl_path))
        loaded_pkl = CareerMovementEmbeddings.load(str(pkl_path))
        assert loaded_pkl.embedding_dim == cme.embedding_dim
        assert loaded_pkl.role_to_idx_ == cme.role_to_idx_
        np.testing.assert_allclose(loaded_pkl.role_embeddings_, cme.role_embeddings_, atol=1e-5)

    def test_role_similarity_matrix(self):
        from people_analytics_toolkit.mobility import (
            CareerMovementEmbeddings,
            compute_role_similarity_matrix,
            get_role_similarity_matrix,
            calculate_role_similarity_matrix,
        )
        # Unfitted model raises ValueError
        unfitted = CareerMovementEmbeddings()
        with pytest.raises(ValueError, match="Model has not been fitted yet"):
            unfitted.compute_role_similarity_matrix()

        sequences = [
            ["Cashier", "Sales_Associate", "Department_Supervisor"],
            ["Stocker", "Operations_Lead", "Department_Supervisor"],
        ] * 10
        cme = CareerMovementEmbeddings(embedding_dim=4, seed=42)
        cme.fit(sequences)

        # Instance method
        mat1 = cme.compute_role_similarity_matrix()
        assert isinstance(mat1, pd.DataFrame)
        assert mat1.shape == (5, 5)
        # Cosine similarity of role with itself should be 1.0
        for col in mat1.columns:
            assert np.isclose(mat1.loc[col, col], 1.0)

        # Direct alias equivalence
        mat2 = cme.get_role_similarity_matrix()
        pd.testing.assert_frame_equal(mat1, mat2)

        # Subsetting roles
        sub_roles = ["Cashier", "Sales_Associate"]
        sub_mat = cme.compute_role_similarity_matrix(roles=sub_roles)
        assert sub_mat.shape == (2, 2)
        assert list(sub_mat.columns) == sub_roles

        # Module-level convenience functions
        mod_mat1 = compute_role_similarity_matrix(cme)
        pd.testing.assert_frame_equal(mat1, mod_mat1)
        mod_mat2 = get_role_similarity_matrix(cme)
        pd.testing.assert_frame_equal(mat1, mod_mat2)
        mod_mat3 = calculate_role_similarity_matrix(cme)
        pd.testing.assert_frame_equal(mat1, mod_mat3)

        # Passing DataFrame of embeddings
        df_embeds = pd.DataFrame(cme.role_embeddings_, index=[cme.idx_to_role_[i] for i in range(len(cme.idx_to_role_))])
        df_mat = compute_role_similarity_matrix(df_embeds)
        pd.testing.assert_frame_equal(mat1, df_mat)

    def test_shap_inflection_thresholds_for_term_rates(self):
        from people_analytics_toolkit.attribution import (
            extract_shap_inflection_thresholds,
            calculate_retained_vs_terminated_shap_contributions,
        )
        from sklearn.ensemble import RandomForestClassifier
        import shap
        
        rng = np.random.default_rng(42)
        n = 150
        ot = rng.uniform(0, 30, size=n)
        tenure = rng.uniform(1, 100, size=n)
        # Hazard: OT > 15 spikes termination risk
        prob = 1.0 / (1.0 + np.exp(-(0.2 * (ot - 15) - 0.05 * (tenure - 50))))
        y = (rng.random(n) < prob).astype(int)
        
        X = pd.DataFrame({"weekly_overtime": ot, "tenure_weeks": tenure})
        model = RandomForestClassifier(n_estimators=30, max_depth=3, random_state=42)
        model.fit(X, y)
        explainer = shap.TreeExplainer(model)
        s_vals = explainer.shap_values(X)
        if isinstance(s_vals, list):
            s_vals = s_vals[1]
        elif hasattr(s_vals, "ndim") and s_vals.ndim == 3:
            s_vals = s_vals[:, :, 1]
        
        thresholds, summary_df = extract_shap_inflection_thresholds(X, s_vals, seed=42)
        assert "weekly_overtime" in thresholds
        assert "primary_inflection_threshold" in summary_df.columns
        
        # Test side-by-side contributions
        contrib = calculate_retained_vs_terminated_shap_contributions(X, s_vals, pd.Series(y))
        assert "Retained Cohort Mean SHAP" in contrib.columns
        assert "Terminated Cohort Mean SHAP" in contrib.columns
        assert len(contrib) == 2


class TestCompensation:
    def test_calculate_salary_band_midpoint(self):
        # Scalar min and max
        assert np.isclose(calculate_salary_band_midpoint(80000, 120000), 100000.0)
        assert np.isclose(calculate_salary_band_midpoint(70000, 130000), 100000.0)

        # Scalar min and spread (e.g. 50% spread)
        assert np.isclose(calculate_salary_band_midpoint(80000, spread=0.50), 100000.0)

        # Series inputs
        mins = pd.Series([80000, 100000, 120000], index=["L1", "L2", "L3"])
        maxs = pd.Series([120000, 140000, 160000], index=["L1", "L2", "L3"])
        mids = calculate_salary_band_midpoint(mins, maxs)
        assert isinstance(mids, pd.Series)
        assert mids.name == "band_midpoint"
        assert list(mids.values) == [100000.0, 120000.0, 140000.0]

        # NumPy array inputs
        arr_mids = calculate_salary_band_midpoint(np.array([80000, 100000]), np.array([120000, 140000]))
        assert np.allclose(arr_mids, [100000.0, 120000.0])

        # Validation errors
        with pytest.raises(ValueError, match="band_min cannot be greater than band_max"):
            calculate_salary_band_midpoint(120000, 80000)
        with pytest.raises(ValueError, match="Must provide either 'band_max' or 'spread'"):
            calculate_salary_band_midpoint(80000)
        with pytest.raises(ValueError, match="spread must be non-negative"):
            calculate_salary_band_midpoint(80000, spread=-0.1)

    def test_compa_ratio_calculation(self):
        # Explicit midpoint
        assert np.isclose(calculate_compa_ratio(100000, midpoint=100000), 1.0)
        assert np.isclose(calculate_compa_ratio(80000, midpoint=100000), 0.8)
        assert np.isclose(calculate_compa_ratio(125000, midpoint=100000), 1.25)
        
        # Derived midpoint from min & max: (80k + 120k) / 2 = 100k
        assert np.isclose(calculate_compa_ratio(95000, band_min=80000, band_max=120000), 0.95)
        
        # Series handling
        salaries = pd.Series([90000, 100000, 110000])
        mids = pd.Series([100000, 100000, 100000])
        res = calculate_compa_ratio(salaries, midpoint=mids)
        assert len(res) == 3
        assert np.isclose(res.iloc[0], 0.9)

    def test_range_penetration_calculation(self):
        b_min, b_max = 80000, 120000  # Spread = 40,000
        
        # Minimum -> 0.0
        assert np.isclose(calculate_range_penetration(80000, b_min, b_max), 0.0)
        # Midpoint -> 0.5
        assert np.isclose(calculate_range_penetration(100000, b_min, b_max), 0.5)
        # Maximum -> 1.0
        assert np.isclose(calculate_range_penetration(120000, b_min, b_max), 1.0)
        # Green-circled (below min) -> negative
        assert calculate_range_penetration(75000, b_min, b_max) < 0.0
        # Red-circled (above max) -> > 1.0
        assert calculate_range_penetration(130000, b_min, b_max) > 1.0

    def test_classify_pay_band_status(self):
        rp_vals = pd.Series([-0.05, 0.10, 0.35, 0.60, 0.85, 1.15])
        statuses = classify_pay_band_status(rp_vals)
        
        assert statuses.iloc[0] == "Green-Circled (<Min)"
        assert statuses.iloc[1] == "Q1: Developing (0-25%)"
        assert statuses.iloc[2] == "Q2: Core Proficient (25-50%)"
        assert statuses.iloc[3] == "Q3: Advanced (50-75%)"
        assert statuses.iloc[4] == "Q4: Senior Cap (75-100%)"
        assert statuses.iloc[5] == "Red-Circled (>Max)"

    def test_is_red_and_green_circled(self):
        # Using pre-calculated range penetration
        rp = pd.Series([-0.1, 0.5, 1.2])
        red = is_red_circled(rp)
        green = is_green_circled(rp)
        assert not red.iloc[0] and not red.iloc[1] and red.iloc[2]
        assert green.iloc[0] and not green.iloc[1] and not green.iloc[2]

        # Using direct salary and bounds
        assert is_red_circled(130000, band_max=120000)
        assert not is_red_circled(110000, band_max=120000)
        assert is_green_circled(70000, band_min=80000)
        assert not is_green_circled(90000, band_min=80000)

    def test_calculate_pay_equity_gap_and_median_gap(self):
        # Synthetic cohort:
        # Male salaries: [90k, 100k, 110k] -> median 100k, mean 100k
        # Female salaries: [75k, 80k, 85k] -> median 80k, mean 80k
        df = pd.DataFrame({
            "salary": [90000, 100000, 110000, 75000, 80000, 85000],
            "gender": ["Male", "Male", "Male", "Female", "Female", "Female"],
            "level": ["IC2", "IC3", "IC4", "IC2", "IC3", "IC4"],
        })

        # Basic DataFrame calculation
        res = calculate_pay_equity_gap(
            df,
            salary_col="salary",
            group_col="gender",
            reference_group="Male",
            comparison_group="Female",
        )

        assert np.isclose(res["median_gap"], 0.20)
        assert np.isclose(res["pct_median_gap"], 20.0)
        assert np.isclose(res["dollar_median_gap"], 20000.0)
        assert np.isclose(res["pay_ratio_median"], 0.80)
        assert np.isclose(res["mean_gap"], 0.20)
        assert np.isclose(res["pct_mean_gap"], 20.0)
        assert res["ref_median"] == 100000.0
        assert res["comp_median"] == 80000.0
        assert res["reference_group"] == "Male"
        assert res["comparison_group"] == "Female"

        # Check group stats table
        stats = res["group_stats"]
        assert "Male" in stats.index and "Female" in stats.index
        assert stats.loc["Male", "count"] == 3
        assert stats.loc["Female", "median"] == 80000.0

        # Check quartile distribution table
        q_dist = res["quartile_distribution"]
        assert len(q_dist) > 0

        # Convenience method median_gap
        gap_dec = median_gap(df["salary"], df["gender"])
        assert np.isclose(gap_dec, 0.20)

        gap_pct = median_gap(df["salary"], df["gender"], as_percentage=True)
        assert np.isclose(gap_pct, 20.0)

        # Alias calculate_median_pay_gap
        assert np.isclose(calculate_median_pay_gap(df, "gender", salary_col="salary"), 0.20)

    def test_calculate_pay_equity_gap_controlled(self):
        # Controlled by level:
        # Level 1: Male 60k, Female 54k -> gap (60-54)/60 = 0.10 (10%)
        # Level 2: Male 100k, Female 90k -> gap (100-90)/100 = 0.10 (10%)
        df = pd.DataFrame({
            "salary": [60000, 54000, 100000, 90000],
            "gender": ["M", "F", "M", "F"],
            "job_level": ["L1", "L1", "L2", "L2"],
        })

        res = calculate_pay_equity_gap(
            df,
            salary_col="salary",
            group_col="gender",
            control_cols="job_level",
        )

        assert "controlled_median_gap" in res
        assert np.isclose(res["controlled_median_gap"], 0.10)
        assert np.isclose(res["controlled_pct_median_gap"], 10.0)
        assert "cohort_gaps" in res
        assert len(res["cohort_gaps"]) == 2

    def test_calculate_pay_equity_gap_validation_and_errors(self):
        # Single group should raise error
        with pytest.raises(ValueError, match="at least 2 distinct groups"):
            calculate_pay_equity_gap(
                salary=[100000, 110000],
                group=["Male", "Male"],
            )

        # Non-existent reference group
        df = pd.DataFrame({
            "salary": [100000, 80000],
            "group": ["A", "B"],
        })
        with pytest.raises(ValueError, match="Reference group 'NonExistent' not found"):
            calculate_pay_equity_gap(df, "salary", "group", reference_group="NonExistent")

        # Length mismatch in positional arrays
        with pytest.raises(ValueError, match="Length mismatch"):
            calculate_pay_equity_gap(salary=[100000, 120000], group=["A"])



class TestOrganizationalNetworkAnalysis:
    def test_centrality_broker_and_hub_identification(self):
        nodes_df, edges_df, G = generate_organizational_network_data(n_employees=60, seed=42)
        deg = calculate_degree_centrality(G)
        cc = calculate_closeness_centrality(G)
        bc = calculate_betweenness_centrality(G)
        ec = calculate_eigenvector_centrality(G)
        
        assert len(deg) == len(nodes_df)
        assert len(cc) == len(nodes_df)
        assert len(bc) == len(nodes_df)
        assert len(ec) == len(nodes_df)
        assert (deg >= 0.0).all() and (deg <= 1.0).all()
        assert (cc >= 0.0).all() and (cc <= 1.0).all()
        assert (bc >= 0.0).all() and (bc <= 1.0).all()
        assert (ec >= 0.0).all() and (ec <= 1.0).all()
        
        # Verify from edge DataFrame input as well
        deg_from_df = calculate_degree_centrality(edges_df)
        cc_from_df = calculate_closeness_centrality(edges_df)
        bc_from_df = calculate_betweenness_centrality(edges_df)
        assert len(deg_from_df) > 0
        assert len(cc_from_df) > 0
        assert len(bc_from_df) > 0

    def test_degree_centrality_modes_and_directed(self):
        # Directed star network: Center C, Peripheral P1, P2, P3
        # C -> P1, C -> P2, P3 -> C
        di_edges = pd.DataFrame({
            "source": ["C", "C", "P3"],
            "target": ["P1", "P2", "C"],
            "weight": [2.0, 3.0, 5.0],
        })
        G_di = nx.DiGraph()
        for _, row in di_edges.iterrows():
            G_di.add_edge(row["source"], row["target"], weight=row["weight"])

        # In-degree: C has 1 incoming (from P3), P1 and P2 have 1 incoming (from C)
        in_deg = calculate_degree_centrality(G_di, kind="in", normalized=False)
        assert in_deg["C"] == 1.0
        assert in_deg["P1"] == 1.0
        assert in_deg["P3"] == 0.0

        # Out-degree: C has 2 outgoing (to P1, P2)
        out_deg = calculate_degree_centrality(G_di, kind="out", normalized=False)
        assert out_deg["C"] == 2.0
        assert out_deg["P3"] == 1.0
        assert out_deg["P1"] == 0.0

        # Weighted in-degree
        weighted_in = calculate_degree_centrality(G_di, weight="weight", kind="in", normalized=False)
        assert weighted_in["C"] == 5.0  # weight 5.0 from P3
        assert weighted_in["P1"] == 2.0  # weight 2.0 from C

        # Undirected simple path graph: 1 - 2 - 3 (3 nodes, denom = 2)
        # 2 is center node: degree = 2, normalized = 2 / 2 = 1.0
        # 1 and 3 are endpoints: degree = 1, normalized = 1 / 2 = 0.5
        path_edges = [("A", "B"), ("B", "C")]
        deg_norm = calculate_degree_centrality(path_edges, normalized=True)
        assert np.isclose(deg_norm["B"], 1.0)
        assert np.isclose(deg_norm["A"], 0.5)
        assert np.isclose(deg_norm["C"], 0.5)

        deg_raw = calculate_degree_centrality(path_edges, normalized=False)
        assert deg_raw["B"] == 2.0
        assert deg_raw["A"] == 1.0

    def test_closeness_centrality_calculation(self):
        # 3-node path: A - B - C
        # Shortest paths:
        # B to A is 1, B to C is 1 -> sum = 2 -> closeness = (3-1)/2 = 1.0
        # A to B is 1, A to C is 2 -> sum = 3 -> closeness = (3-1)/3 = 2/3
        # C to B is 1, C to A is 2 -> sum = 3 -> closeness = (3-1)/3 = 2/3
        path_edges = [("A", "B"), ("B", "C")]
        cc = calculate_closeness_centrality(path_edges)
        assert np.isclose(cc["B"], 1.0)
        assert np.isclose(cc["A"], 2.0 / 3.0)
        assert np.isclose(cc["C"], 2.0 / 3.0)

        # Disconnected graph: A - B, and isolated C
        disc_edges = pd.DataFrame({"source": ["A"], "target": ["B"]})
        G_disc = nx.Graph()
        G_disc.add_edge("A", "B")
        G_disc.add_node("C")
        cc_disc = calculate_closeness_centrality(G_disc)
        assert cc_disc["C"] == 0.0
        assert cc_disc["A"] > 0.0

    def test_burt_constraint_structural_holes(self):
        # Barbell network: two triangles connected by a bridge node B
        # Triangle 1: (A1, A2, B), Triangle 2: (C1, C2, B)
        # B spans structural hole between A cluster and C cluster
        edges = [
            ("A1", "A2"), ("A1", "B"), ("A2", "B"),
            ("C1", "C2"), ("C1", "B"), ("C2", "B")
        ]
        constraint = calculate_burt_constraint(edges)
        
        # B connects two non-redundant clusters, so B has lower constraint than the insular perimeter nodes
        assert constraint["B"] < constraint["A1"]
        assert constraint["B"] < constraint["C1"]
        assert 0.0 < constraint["B"] < 1.0

    def test_euler_centrality_leinster_magnitude(self):
        nodes_df, edges_df, G = generate_organizational_network_data(n_employees=40, seed=42)
        euler = calculate_euler_centrality(G)
        
        assert len(euler) == 40
        assert not euler.isna().any()
        # Total graph magnitude sum should be positive
        total_magnitude = euler.sum()
        assert total_magnitude > 0.0

    def test_collaboration_overload_and_bottlenecks(self):
        nodes_df, edges_df, G = generate_organizational_network_data(n_employees=60, seed=42)
        overload_df = calculate_collaboration_overload(G, threshold_z=1.5)
        
        assert "interaction_volume" in overload_df.columns
        assert "overload_z_score" in overload_df.columns
        assert "is_overloaded" in overload_df.columns
        assert len(overload_df) == 60
        # Check Z-score standardization properties
        assert np.isclose(overload_df["overload_z_score"].mean(), 0.0, atol=1e-5)

    def test_attrition_contagion_simulation(self):
        nodes_df, edges_df, G = generate_organizational_network_data(n_employees=50, seed=42)
        departed = nodes_df["employee_id"].iloc[5]  # A designated broker node
        
        contagion_df = simulate_attrition_contagion(G, departed_node=departed, contagion_base_rate=0.50, max_hops=2)
        
        assert not contagion_df.empty
        assert "employee_id" in contagion_df.columns
        assert "contagion_risk_delta" in contagion_df.columns
        assert "hop_distance" in contagion_df.columns
        
        # Direct collaborators (hop 1) receive higher contagion shock than indirect (hop 2)
        hop1 = contagion_df[contagion_df["hop_distance"] == 1]
        hop2 = contagion_df[contagion_df["hop_distance"] == 2]
        if not hop1.empty and not hop2.empty:
            assert hop1["contagion_risk_delta"].mean() > hop2["contagion_risk_delta"].mean()

    def test_complete_ona_profile_and_archetypes(self):
        nodes_df, edges_df, G = generate_organizational_network_data(n_employees=50, seed=42)
        profile = compute_complete_ona_profile(G)
        
        assert "degree_centrality" in profile.columns
        assert "closeness_centrality" in profile.columns
        assert "betweenness_centrality" in profile.columns
        assert "eigenvector_centrality" in profile.columns
        assert "burt_constraint" in profile.columns
        assert "euler_centrality" in profile.columns
        assert "network_archetype" in profile.columns
        assert len(profile) == 50
        # Verify archetype categorization contains valid categories
        archetype_set = set(profile["network_archetype"].unique())
        assert archetype_set.issubset({
            "Broker / Bridge", "Informal Hub", "Cohesive Core", "Autonomous Specialist", "Peripheral"
        })


class TestAlgorithmicEquityAndFairness:
    def test_oaxaca_blinder_exact_decomposition(self):
        df = generate_algorithmic_equity_and_fairness_data(n_employees=600, seed=42)
        features = ["tenure_years", "education_years", "job_level", "performance_rating", "certifications"]
        
        result = oaxaca_blinder_decomposition(
            df=df,
            outcome_col="base_salary",
            group_col="demographic_group",
            feature_cols=features,
            group_majority=1,
            group_minority=0,
            reference_type="pooled",
        )
        
        raw_gap = result["raw_gap"]
        explained = result["explained_effect"]
        unexplained = result["unexplained_effect"]
        
        # Verify exact additive decomposition: Raw Gap = Explained + Unexplained
        assert np.isclose(raw_gap, explained + unexplained, atol=1e-3)
        assert np.isclose(result["explained_pct"] + result["unexplained_pct"], 100.0, atol=1e-3)
        
        # Injected structural bias was +$6,800 for majority group; unexplained effect must be positive and substantial
        assert unexplained > 3000.0
        
        # Detailed feature breakdown consistency
        breakdown = result["feature_breakdown"]
        assert len(breakdown) == len(features) + 1  # features + constant
        assert np.isclose(breakdown["explained_effect"].sum(), explained, atol=1e-3)
        assert np.isclose(breakdown["unexplained_effect"].sum(), unexplained, atol=1e-3)

    def test_group_fairness_metrics(self):
        # Perfect equality
        y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        sens = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        
        assert demographic_parity_difference(y_pred, sens) == 0.0
        assert demographic_parity_ratio(y_pred, sens) == 1.0
        assert equalized_odds_difference(y_true, y_pred, sens) == 0.0
        
        # Complete disparity
        y_pred_biased = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        assert demographic_parity_difference(y_pred_biased, sens) == 1.0
        assert demographic_parity_ratio(y_pred_biased, sens) == 0.0

    def test_exponentiated_gradient_demographic_parity(self):
        df = generate_algorithmic_equity_and_fairness_data(n_employees=500, seed=42)
        X = df[["tenure_years", "education_years", "job_level", "performance_rating", "certifications"]]
        y = df["promotion_recommendation"]
        sens = df["demographic_group"]
        
        # Unconstrained baseline Logistic Regression
        from sklearn.linear_model import LogisticRegression
        base_clf = LogisticRegression(solver="liblinear", random_state=42)
        base_clf.fit(X, y)
        base_preds = base_clf.predict(X)
        base_dp_diff = demographic_parity_difference(base_preds, sens)
        
        # Fair classifier enforcing Demographic Parity
        fair_clf = ExponentiatedGradientFairness(
            estimator=LogisticRegression(solver="liblinear", random_state=42),
            constraint="demographic_parity",
            eps=0.02,
            max_iter=15,
            random_state=42,
        )
        fair_clf.fit(X, y, sensitive_features=sens)
        fair_preds = fair_clf.predict(X)
        fair_dp_diff = demographic_parity_difference(fair_preds, sens)
        
        # Parity gap should be significantly reduced
        assert fair_dp_diff < base_dp_diff
        assert fair_dp_diff <= 0.12

    def test_exponentiated_gradient_equalized_odds(self):
        df = generate_algorithmic_equity_and_fairness_data(n_employees=500, seed=42)
        X = df[["tenure_years", "education_years", "job_level", "performance_rating", "certifications"]]
        y = df["true_merit_qualification"]
        sens = df["demographic_group"]
        
        # Fit fair classifier under Equalized Odds
        fair_clf = ExponentiatedGradientFairness(
            estimator=LogisticRegression(solver="liblinear", random_state=42),
            constraint="equalized_odds",
            eps=0.03,
            max_iter=15,
            random_state=42,
        )
        fair_clf.fit(X, y, sensitive_features=sens)
        fair_preds = fair_clf.predict(X)
        
        eo_diff = equalized_odds_difference(y, fair_preds, sens)
        assert eo_diff <= 0.20
        assert len(fair_clf.classifiers_) == 15

    def test_exponentiated_gradient_multi_group(self):
        df = generate_algorithmic_equity_and_fairness_data(n_employees=600, seed=42)
        X = df[["tenure_years", "education_years", "job_level", "performance_rating", "certifications"]]
        y = df["promotion_recommendation"]
        # Create 3 demographic groups: "0", "1", and "Group_C"
        sens = df["demographic_group"].astype(str).copy()
        sens.iloc[::3] = "Group_C"

        # Baseline unconstrained model
        base_clf = LogisticRegression(solver="liblinear", random_state=42)
        base_clf.fit(X, y)
        base_dp_diff = demographic_parity_difference(base_clf.predict(X), sens)

        # Fit multi-group ExponentiatedGradientFairness under Demographic Parity
        fair_clf = ExponentiatedGradientFairness(
            estimator=LogisticRegression(solver="liblinear", random_state=42),
            constraint="demographic_parity",
            eps=0.02,
            max_iter=15,
            random_state=42,
        )
        fair_clf.fit(X, y, sensitive_features=sens)
        assert len(fair_clf.groups_) == 3
        fair_preds = fair_clf.predict(X)
        fair_dp_diff = demographic_parity_difference(fair_preds, sens)
        assert fair_dp_diff < base_dp_diff

        # Multi-group Equalized Odds
        y_true = df["true_merit_qualification"]
        fair_eo = ExponentiatedGradientFairness(
            estimator=LogisticRegression(solver="liblinear", random_state=42),
            constraint="equalized_odds",
            eps=0.03,
            max_iter=15,
            random_state=42,
        )
        fair_eo.fit(X, y_true, sensitive_features=sens)
        assert len(fair_eo.groups_) == 3
        eo_diff = equalized_odds_difference(y_true, fair_eo.predict(X), sens)
        assert eo_diff <= 0.25

        # Single group should raise ValueError
        single_sens = np.zeros(len(X))
        with pytest.raises(ValueError, match="at least 2 sensitive groups"):
            fair_clf.fit(X, y, sensitive_features=single_sens)

    def test_individual_fairness_local_predictive_consistency(self):
        df = generate_algorithmic_equity_and_fairness_data(n_employees=400, seed=42)
        X = df[["tenure_years", "education_years", "job_level", "performance_rating", "certifications"]]
        y_prob = df["promotion_recommendation"].to_numpy(dtype=float)
        
        consistency_scores, mean_consistency, meta = calculate_individual_fairness_consistency(
            X, y_prob, n_neighbors=5
        )
        
        assert len(consistency_scores) == 400
        assert 0.0 <= mean_consistency <= 1.0
        assert "local_discrepancy" in meta
        assert "empirical_lipschitz_ratios" in meta
        
        # Inject an individual outlier whose prediction differs completely from identical peers
        y_injected = y_prob.copy()
        # Find 5 tightly clustered neighbors
        from sklearn.neighbors import NearestNeighbors
        from sklearn.preprocessing import StandardScaler
        X_scaled = StandardScaler().fit_transform(X)
        nn = NearestNeighbors(n_neighbors=6).fit(X_scaled)
        _, indices = nn.kneighbors(X_scaled)
        
        target_idx = 10
        peer_indices = indices[target_idx, 1:]
        # Set all peers to 1.0, and target to 0.0
        y_injected[peer_indices] = 1.0
        y_injected[target_idx] = 0.0
        
        _, _, meta_injected = calculate_individual_fairness_consistency(X, y_injected, n_neighbors=5)
        # Target must have high local discrepancy (= 1.0)
        assert meta_injected["local_discrepancy"].iloc[target_idx] == 1.0


class TestPrescriptiveInterventions:
    """Unit tests for prescriptive causal inference, SDiD, CausalForestDML, and uplift metrics."""

    def test_aipw_double_robustness(self):
        df = generate_causal_and_uplift_data(n_employees=800, seed=42)
        X = df[["tenure_years", "prior_performance", "flight_risk", "overtime_hours", "manager_quality"]]
        t = df["treatment_enrolled"]
        y = df["retained_1yr"]

        res = aipw_estimator(y=y, treatment=t, X=X, n_splits=3, random_state=42)

        assert 0.05 <= res["ate"] <= 0.40
        assert res["se"] > 0.0
        assert res["ci_lower"] < res["ate"] < res["ci_upper"]
        assert res["p_value"] < 0.05
        assert len(res["propensity_scores"]) == 800
        assert len(res["influence_function"]) == 800
        assert abs(np.mean(res["influence_function"])) < 1e-4

    def test_aipw_constant_treatment_validation(self):
        n = 50
        X = np.random.randn(n, 3)
        y = np.random.randn(n)

        # 1. Constant treatment across entire dataset
        t_all_zeros = np.zeros(n)
        with pytest.raises(ValueError, match="treatment must be binary"):
            aipw_estimator(y=y, treatment=t_all_zeros, X=X)

        t_all_ones = np.ones(n)
        with pytest.raises(ValueError, match="treatment must be binary"):
            aipw_estimator(y=y, treatment=t_all_ones, X=X)

        # 2. Only 1 treated sample with n_splits > 1 -> minority group cannot be CV split
        t_one_treated = np.zeros(n)
        t_one_treated[0] = 1.0
        with pytest.raises(ValueError, match="Minority treatment group has only 1 sample"):
            aipw_estimator(y=y, treatment=t_one_treated, X=X, n_splits=3)

        # 3. Minority class smaller than n_splits triggers warning and reduces splits
        t_few_treated = np.zeros(n)
        t_few_treated[:2] = 1.0  # exactly 2 treated
        with pytest.warns(UserWarning, match="exceeds minority treatment count"):
            res = aipw_estimator(y=y, treatment=t_few_treated, X=X, n_splits=5)
        assert res is not None
        assert "ate" in res
        df = generate_panel_policy_data(
            n_units=12,
            n_periods=10,
            treated_units=("Office_North",),
            post_period_start=7,
            seed=42,
        )

        res = synthetic_difference_in_differences(
            df=df,
            unit_col="location_id",
            time_col="quarter_idx",
            outcome_col="turnover_rate",
            treated_units=["Office_North"],
            post_period_start=7,
        )

        # True policy impact in synthetic generator was -3.8
        assert -6.5 <= res["att"] <= -1.5
        assert res["se"] > 0.0
        assert res["ci_lower"] < res["att"] < res["ci_upper"]
        # Unit weights should be valid probability distribution
        assert np.isclose(res["unit_weights"].sum(), 1.0, atol=1e-4)
        assert (res["unit_weights"] >= -1e-5).all()
        # Time weights should be valid probability distribution
        assert np.isclose(res["time_weights"].sum(), 1.0, atol=1e-4)
        assert (res["time_weights"] >= -1e-5).all()

    def test_causal_forest_dml_heterogeneity(self):
        df = generate_causal_and_uplift_data(n_employees=600, seed=42)
        X = df[["tenure_years", "prior_performance", "flight_risk", "overtime_hours", "manager_quality"]]
        t = df["treatment_enrolled"]
        y = df["retained_1yr"]

        forest = CausalForestDML(
            n_estimators=20,
            max_depth=3,
            min_samples_leaf=10,
            random_state=42,
        )
        forest.fit(X, t, y)

        cate_preds = forest.predict(X)
        assert len(cate_preds) == 600
        # Check that treatment effect is heterogeneous (variance > 0)
        assert np.var(cate_preds) > 0.0

        intervals = forest.predict_interval(X, alpha=0.05)
        assert len(intervals["ci_lower"]) == 600
        assert (intervals["ci_lower"] <= intervals["ci_upper"]).all()

        persuadables_df = forest.identify_persuadables(X, min_effect=0.05)
        assert "is_persuadable" in persuadables_df.columns
        assert "archetype" in persuadables_df.columns
        assert persuadables_df["is_persuadable"].sum() > 0

        # Feature importances property
        importances = forest.feature_importances_
        assert isinstance(importances, np.ndarray)
        assert importances.shape == (5,)
        assert np.isclose(importances.sum(), 1.0)
        assert (importances >= 0.0).all()
        # flight_risk (index 2) is a primary driver of true heterogeneity
        assert importances[2] > 0.1

        # get_feature_importances as pandas Series
        imp_series = forest.get_feature_importances(as_pandas=True)
        assert isinstance(imp_series, pd.Series)
        assert len(imp_series) == 5
        assert set(imp_series.index) == set(X.columns)
        assert np.isclose(imp_series.sum(), 1.0)
        assert imp_series.index[0] in ["flight_risk", "tenure_years", "prior_performance"]

        # Split count feature importances
        imp_split = forest.get_feature_importances(as_pandas=True, importance_type="split")
        assert isinstance(imp_split, pd.Series)
        assert np.isclose(imp_split.sum(), 1.0)

        # Unfitted exception
        with pytest.raises(ValueError, match="not fitted"):
            _ = CausalForestDML().feature_importances_

        # Single tree feature_importances_
        tree_imp = forest.trees[0].feature_importances_
        assert isinstance(tree_imp, np.ndarray)
        assert tree_imp.shape == (5,)
        assert np.isclose(tree_imp.sum(), 1.0)

    def test_qini_curve_and_auuc_metric(self):
        df = generate_causal_and_uplift_data(n_employees=600, seed=42)
        t = df["treatment_enrolled"]
        y = df["retained_1yr"]
        uplift_scores = df["true_cate"]

        q_df = qini_curve(y, t, uplift_scores, n_bins=10)
        assert len(q_df) == 11
        assert q_df["fraction"].iloc[0] == 0.0
        assert q_df["incremental_gain"].iloc[0] == 0.0
        assert q_df["fraction"].iloc[-1] == 1.0

        auuc_res = area_under_uplift_curve(y, t, uplift_scores, n_bins=10)
        assert "auuc" in auuc_res
        assert "auuc_random" in auuc_res
        assert "qini_score" in auuc_res
        # Prioritizing by true CATE must beat random targeting
        assert auuc_res["qini_score"] > 0.0

        X = df[["tenure_years", "prior_performance", "flight_risk", "overtime_hours", "manager_quality"]]
        dr_eval = doubly_robust_uplift_eval(y, t, uplift_scores, X)
        assert "mse_w" in dr_eval
        assert dr_eval["mse_w"] > 0.0
        assert len(dr_eval["gamma_scores"]) == 600


class TestContinuousTrajectoriesAndGraphLearning:
    """Unit tests for Continuous-Time Markov Chains, Metapath2Vec, and Self-Attention."""

    def test_ctmc_generator_and_matrix_exponential(self):
        event_log = generate_career_trajectory_event_log(n_employees=400, seed=42)
        ctmc = ContinuousTimeMarkovChain()
        ctmc.fit_from_event_log(event_log)

        Q = ctmc.generator_matrix_
        assert Q is not None
        m = len(ctmc.state_labels)
        assert Q.shape == (m, m)

        # Row sums of generator Q must be zero
        assert np.allclose(Q.sum(axis=1), 0.0, atol=1e-5)

        # Transition matrix at t=0 must be identity
        P_0 = ctmc.transition_probability_matrix(0.0)
        assert np.allclose(P_0.values, np.eye(m))

        # Transition matrix at continuous horizon t > 0
        P_2 = ctmc.transition_probability_matrix(2.0)
        assert np.allclose(P_2.sum(axis=1), 1.0, atol=1e-5)
        assert (P_2.values >= -1e-6).all()

        # Expected sojourn times
        sojourn = ctmc.expected_sojourn_times()
        assert len(sojourn) == m
        assert sojourn["L1_Associate"] > 0.0

        # Predict trajectory distribution
        traj_dist = ctmc.predict_trajectory_distribution("L1_Associate", t=3.0)
        assert np.isclose(traj_dist.sum(), 1.0, atol=1e-5)

        # Pipeline bottlenecks
        bottlenecks = ctmc.identify_pipeline_bottlenecks()
        assert "expected_sojourn_years" in bottlenecks.columns
        assert "is_mobility_bottleneck" in bottlenecks.columns

        # Stationary distribution
        pi = ctmc.stationary_distribution()
        assert isinstance(pi, pd.Series)
        assert len(pi) == m
        assert np.isclose(pi.sum(), 1.0)
        # Absorbing state receives full long-run probability
        assert np.isclose(pi["L5_Director"], 1.0)
        assert pi["L1_Associate"] == 0.0

        # Simulate continuous trajectory
        traj = ctmc.simulate_trajectory("L1_Associate", max_time=5.0, seed=42)
        assert isinstance(traj, pd.DataFrame)
        assert len(traj) > 0
        assert traj["state"].iloc[0] == "L1_Associate"
        assert traj["start_time"].iloc[0] == 0.0
        assert traj["end_time"].iloc[-1] <= 5.0
        assert (traj["duration"] > 0).all()

    def test_ctmc_ergodic_stationary_and_simulation_mechanics(self):
        # Ergodic CTMC with 3 interconnected states (no absorbing state)
        labels = ["Support", "Engineering", "Management"]
        Q = np.array([
            [-0.6, 0.4, 0.2],
            [0.2, -0.5, 0.3],
            [0.1, 0.3, -0.4]
        ])
        ctmc = ContinuousTimeMarkovChain().fit_from_matrix(Q, labels)

        # Equilibrium stationary distribution pi Q = 0
        pi = ctmc.stationary_distribution()
        assert isinstance(pi, pd.Series)
        assert np.isclose(pi.sum(), 1.0)
        assert (pi > 0).all()
        assert np.allclose(pi.values @ Q, 0.0, atol=1e-10)

        # Simulate trajectory with step chaining
        traj = ctmc.simulate_trajectory("Support", max_time=8.0, seed=123)
        assert traj["state"].iloc[0] == "Support"
        assert traj["start_time"].iloc[0] == 0.0
        # Consecutive steps must be contiguous in time
        for i in range(1, len(traj)):
            assert np.isclose(traj["start_time"].iloc[i], traj["end_time"].iloc[i - 1])
        assert np.isclose(traj["end_time"].iloc[-1], 8.0)

        # Simulate multiple trajectories
        trajs = ctmc.simulate_trajectories("Engineering", n_trajectories=15, max_time=4.0, seed=42)
        assert trajs["trajectory_id"].nunique() == 15
        assert (trajs["end_time"] <= 4.0).all()

        # Unfitted CTMC errors
        unfitted = ContinuousTimeMarkovChain()
        with pytest.raises(ValueError, match="must be fitted"):
            unfitted.stationary_distribution()
        with pytest.raises(ValueError, match="must be fitted"):
            unfitted.simulate_trajectory("Support")

    def test_metapath_biased_random_walks(self):
        G, nodes_df, edges_df = generate_heterogeneous_hcm_multigraph(n_employees=60, n_projects=10, n_skills=15, seed=42)
        node_types = dict(zip(nodes_df["node_id"], nodes_df["node_type"]))

        schema = ["Employee", "Project", "Skill", "Employee"]
        walks = generate_metapath_walks(G, meta_paths=[schema], walk_length=12, num_walks=4, random_state=42)

        assert len(walks) > 0
        for walk in walks[:10]:
            for step_idx, node in enumerate(walk):
                expected_type = schema[step_idx % (len(schema) - 1)]
                assert node_types[node] == expected_type

    def test_metapath2vec_embeddings(self):
        G, _, _ = generate_heterogeneous_hcm_multigraph(n_employees=50, n_projects=8, n_skills=12, seed=42)
        model = Metapath2Vec(embedding_dim=16, walk_length=12, num_walks=5, window_size=3, random_state=42)
        model.fit(G)

        assert model.embeddings_ is not None
        assert model.embeddings_.shape[1] == 16

        # Test node retrieval and similarity
        emb = model.get_embedding("EMP_0001")
        assert len(emb) == 16
        assert np.isclose(np.linalg.norm(emb), 1.0, atol=1e-4)

        similar_peers = model.find_similar_nodes("EMP_0001", top_k=3, target_node_type="Employee")
        assert len(similar_peers) <= 3
        assert (similar_peers["cosine_similarity"] <= 1.0001).all()

        # Team complementarity
        team_metrics = model.predict_team_complementarity(["EMP_0001", "EMP_0002", "EMP_0003"])
        assert "pairwise_cohesion" in team_metrics
        assert "diversity_spread" in team_metrics
        assert "structural_coverage" in team_metrics
        assert 0.0 <= team_metrics["pairwise_cohesion"] <= 1.0

    def test_dynamic_entity_self_attention(self):
        rng = np.random.RandomState(42)
        # 5 tokens: [Employee, TrajectoryState, Project1, Project2, Skill]
        tokens = rng.normal(size=(5, 16))
        tokens = tokens / np.linalg.norm(tokens, axis=1, keepdims=True)

        attn = DynamicEntitySelfAttention(n_heads=2, random_state=42)

        # Self-attention pooling
        res = attn.aggregate(tokens)
        assert res["context_vector"].shape == (16,)
        weights = res["attention_weights"]
        assert len(weights) == 5
        assert np.isclose(weights.sum(), 1.0, atol=1e-5)
        assert (weights >= 0.0).all()

        # Learnable projection matrices exist and have correct shapes
        assert attn.W_q_ is not None and attn.W_q_.shape == (16, 16)
        assert attn.W_k_ is not None and attn.W_k_.shape == (16, 16)
        assert attn.W_v_ is not None and attn.W_v_.shape == (16, 16)
        assert attn.W_o_ is not None and attn.W_o_.shape == (16, 16)
        assert attn.head_dim == 8

        # Multi-head attention inspection
        head_weights = res["head_attention_weights"]
        assert head_weights.shape == (2, 5)
        assert np.allclose(head_weights.sum(axis=1), 1.0)
        # Average of head weights must equal aggregate attention weights
        assert np.allclose(weights.values, head_weights.mean(axis=0))

        # Query-directed attention (e.g. project staffing profile query)
        query = rng.normal(size=16)
        query = query / np.linalg.norm(query)
        res_q = attn.aggregate(tokens, query=query)
        assert res_q["context_vector"].shape == (16,)
        assert np.isclose(res_q["attention_weights"].sum(), 1.0, atol=1e-5)

        # Invalid head divisibility check
        invalid_attn = DynamicEntitySelfAttention(n_heads=3, random_state=42)
        with pytest.raises(ValueError, match="must be divisible by n_heads"):
            invalid_attn.aggregate(tokens)

    def test_dynamic_entity_self_attention_learnable_fit_and_serialization(self, tmp_path):
        from people_analytics_toolkit.trajectories import DynamicEntitySelfAttention
        rng = np.random.RandomState(42)

        # Generate sample token sequences for 6 teams/employees
        tokens_list = [rng.normal(size=(rng.randint(3, 8), 16)) for _ in range(6)]
        targets = rng.normal(size=6)

        attn = DynamicEntitySelfAttention(n_heads=2, d_model=16, random_state=42)
        assert not attn.is_fitted_

        # Train with supervised targets
        attn.fit(tokens_list, targets=targets, epochs=25, lr=0.01)
        assert attn.is_fitted_
        assert len(attn.training_loss_history_) == 25
        # Loss must decrease during optimization
        assert attn.training_loss_history_[-1] < attn.training_loss_history_[0]

        # Serialization: JSON save and load
        json_path = tmp_path / "attention.json"
        attn.save(json_path)
        loaded_json = DynamicEntitySelfAttention.load(json_path)
        assert loaded_json.n_heads == 2
        assert loaded_json.d_model == 16
        assert loaded_json.is_fitted_
        assert np.allclose(loaded_json.W_q_, attn.W_q_)

        # Verify aggregate produces identical context vector after load
        test_tokens = tokens_list[0]
        orig_out = attn.aggregate(test_tokens)
        loaded_out = loaded_json.aggregate(test_tokens)
        assert np.allclose(orig_out["context_vector"], loaded_out["context_vector"])
        assert np.allclose(orig_out["attention_weights"].values, loaded_out["attention_weights"].values)

        # Serialization: Pickle save and load
        pkl_path = tmp_path / "attention.pkl"
        attn.save(pkl_path)
        loaded_pkl = DynamicEntitySelfAttention.load(pkl_path)
        assert np.allclose(loaded_pkl.W_o_, attn.W_o_)

        # get_params and set_params
        params = attn.get_params()
        assert params["n_heads"] == 2
        attn.set_params(n_heads=4)
        assert attn.n_heads == 4



class TestSpanOfControlAndManagerialLoad:
    """Unit tests for Span of Control, managerial load benchmarks, and reorg shock flight risk."""

    def test_span_of_control_calculation(self):
        df_pre, _ = generate_reporting_hierarchy_with_reorg(seed=42)
        metrics = calculate_span_of_control(df_pre, employee_col="employee_id", manager_col="manager_id")

        # 1. Manager verification
        assert "MGR_01" in metrics.index
        mgr_row = metrics.loc["MGR_01"]
        assert bool(mgr_row["is_manager"]) is True
        assert mgr_row["direct_reports_count"] == 8
        assert mgr_row["managerial_load_category"] == "Optimal"
        assert mgr_row["manager_id"] == "VP_ENG"

        # 2. Executive root verification
        assert "EXEC_CEO" in metrics.index
        ceo_row = metrics.loc["EXEC_CEO"]
        assert bool(ceo_row["is_manager"]) is True
        assert ceo_row["direct_reports_count"] == 3  # 3 VPs
        assert pd.isna(ceo_row["manager_span"])
        assert pd.isna(ceo_row["est_weekly_1on1_minutes"])

        # 3. Associate verification
        assert "EMP_001" in metrics.index
        assoc_row = metrics.loc["EMP_001"]
        assert bool(assoc_row["is_manager"]) is False
        assert assoc_row["direct_reports_count"] == 0
        assert assoc_row["manager_id"] == "MGR_01"
        assert assoc_row["manager_span"] == 8
        assert assoc_row["est_weekly_1on1_minutes"] == 75.0  # 600 mins / 8 reports
        assert 0.0 <= assoc_row["attention_dilution_score"] <= 0.40  # low dilution at span 8

    def test_span_reorganization_shock_contagion(self):
        df_pre, df_post = generate_reporting_hierarchy_with_reorg(seed=42)
        eval_df = evaluate_span_reorganization_shock(
            df_pre,
            df_post,
            employee_col="employee_id",
            manager_col="manager_id",
            base_flight_risk_col="baseline_flight_risk",
        )

        # In df_post, MGR_01 absorbs 17 reports, exploding from 8 to 25 reports!
        # Associates EMP_001 through EMP_008 (original reports of MGR_01)
        for emp_id in [f"EMP_{i:03d}" for i in range(1, 9)]:
            assert emp_id in eval_df.index
            row = eval_df.loc[emp_id]
            assert row["prev_manager_span"] == 8
            assert row["new_manager_span"] == 25
            assert row["span_delta"] == 17
            assert row["new_weekly_1on1_minutes"] == 24.0  # 600 mins / 25 reports
            assert row["weekly_1on1_minutes_lost"] == 51.0  # Lost 51 minutes of coaching!
            # Flight risk must experience an independent positive surge!
            assert row["reorg_shock_flight_risk_delta"] > 0.10
            assert row["post_reorg_flight_risk"] > row["base_flight_risk"]

        # Sales associates under MGR_07 experienced no reorg span shock
        for emp_id in eval_df.index:
            if eval_df.loc[emp_id, "prev_manager_id"] == "MGR_07":
                row = eval_df.loc[emp_id]
                assert row["span_delta"] == 0
                assert row["reorg_shock_flight_risk_delta"] == 0.0
                assert np.isclose(row["post_reorg_flight_risk"], row["base_flight_risk"])

    def test_span_of_control_graph_and_load_categories(self):
        import networkx as nx
        # Build DiGraph: manager -> report
        G = nx.DiGraph()
        # Manager with 18 reports -> Critical Overload
        for i in range(18):
            G.add_edge("MGR_OVERLOADED", f"EMP_OV_{i}")
        # Manager with 2 reports -> Under-leveraged
        for i in range(2):
            G.add_edge("MGR_SMALL", f"EMP_SM_{i}")

        metrics = calculate_span_of_control(G)
        assert metrics.loc["MGR_OVERLOADED", "direct_reports_count"] == 18
        assert metrics.loc["MGR_OVERLOADED", "managerial_load_category"] == "Critical Overload"
        assert metrics.loc["MGR_SMALL", "direct_reports_count"] == 2
        assert metrics.loc["MGR_SMALL", "managerial_load_category"] == "Under-leveraged"

        # Check associate under overloaded manager
        assoc = metrics.loc["EMP_OV_0"]
        assert assoc["manager_span"] == 18
        assert assoc["attention_dilution_score"] > 0.90


class TestTimeInPositionAndStagnationIndex:
    """Unit tests for Time-in-Position running duration and cohort stagnation index."""

    def test_time_in_position_calculation(self):
        df = pd.DataFrame({
            "employee_id": ["E1", "E2", "E3"],
            "last_role_change_date": ["2026-04-01", "2025-10-01", "2024-10-01"],
        })
        months = calculate_time_in_position(df, as_of_date="2026-10-01", output_unit="months")
        assert len(months) == 3
        # ~6 months for E1
        assert 5.8 <= months.iloc[0] <= 6.2
        # ~12 months for E2
        assert 11.8 <= months.iloc[1] <= 12.2
        # ~24 months for E3
        assert 23.8 <= months.iloc[2] <= 24.2

        days = calculate_time_in_position(df, as_of_date="2026-10-01", output_unit="days")
        assert 180 <= days.iloc[0] <= 185

    def test_cohort_stagnation_benchmarks(self):
        df = generate_time_in_position_workforce(n_employees=250, seed=42)
        res = calculate_stagnation_index(df, cohort_col="department")

        assert "cohort_median_months" in res.columns
        assert "cohort_p90_months" in res.columns
        assert "cohort_time_in_pos_percentile" in res.columns
        assert "is_cohort_stagnant" in res.columns
        assert "is_stagnant_top_performer" in res.columns

        # Verify that median < p90 for all cohorts
        assert (res["cohort_median_months"] < res["cohort_p90_months"]).all()
        # Percentiles bounded in [0, 1]
        assert (res["cohort_time_in_pos_percentile"] >= 0.0).all()
        assert (res["cohort_time_in_pos_percentile"] <= 1.0).all()

    def test_stagnant_top_performer_detection_and_flight_risk(self):
        df = generate_time_in_position_workforce(n_employees=300, seed=42)
        res = calculate_stagnation_index(
            df,
            time_in_pos_col="time_in_position_months",
            cohort_col="department",
            perf_col="performance_rating",
            high_perf_threshold=4.0,
            base_flight_risk_col="baseline_flight_risk",
        )

        stagnant_top = res[res["is_stagnant_top_performer"]]
        assert len(stagnant_top) > 0

        for _, row in stagnant_top.iterrows():
            assert row["performance_rating"] >= 4.0
            assert row["time_in_position_months"] >= row["cohort_p90_months"]
            assert row["career_pathing_intervention"] == "Urgent Career-Pathing Trigger"
            # Flight risk must have experienced a positive surge
            assert row["stagnation_flight_risk_delta"] > 0.05
            assert row["post_stagnation_flight_risk"] > row["baseline_flight_risk"]

        # Normal progression associates
        normal = res[res["career_pathing_intervention"] == "Normal Progression"]
        assert len(normal) > 0
        for _, row in normal.iterrows():
            assert row["stagnation_flight_risk_delta"] == 0.0


class TestBradfordFactorAndAbsenteeismFriction:
    """Unit tests for the Bradford Factor (Absenteeism Friction Score) B = S^2 * D."""

    def test_bradford_score_core_calculation(self):
        from people_analytics_toolkit.chaos import calculate_bradford_score

        # Scalar verification
        assert calculate_bradford_score(1, 10) == 10
        assert calculate_bradford_score(10, 10) == 1000
        assert calculate_bradford_score(3, 6) == 54
        assert calculate_bradford_score(0, 0) == 0

        # Vectorized verification
        spells = pd.Series([1, 10, 3, 0])
        days = pd.Series([10, 10, 6, 0])
        scores = calculate_bradford_score(spells, days)
        np.testing.assert_array_equal(scores.values, [10, 1000, 54, 0])

    def test_bradford_risk_band_classification(self):
        from people_analytics_toolkit.chaos import classify_bradford_risk

        assert classify_bradford_risk(10) == "Low Friction"
        assert classify_bradford_risk(50) == "Low Friction"
        assert classify_bradford_risk(51) == "Moderate Friction"
        assert classify_bradford_risk(200) == "Moderate Friction"
        assert classify_bradford_risk(201) == "Substantial Friction"
        assert classify_bradford_risk(500) == "Substantial Friction"
        assert classify_bradford_risk(501) == "Critical Friction"
        assert classify_bradford_risk(1000) == "Critical Friction"

        series = pd.Series([25, 120, 350, 1000])
        bands = classify_bradford_risk(series)
        assert list(bands) == ["Low Friction", "Moderate Friction", "Substantial Friction", "Critical Friction"]

    def test_bradford_factor_planned_vs_unplanned_events(self):
        from data.synthetic_generators import generate_bradford_factor_workforce
        from people_analytics_toolkit.chaos import calculate_bradford_factor

        spell_df, _ = generate_bradford_factor_workforce(n_employees=60, window_weeks=52, seed=42)
        res = calculate_bradford_factor(
            spell_df,
            employee_col="employee_id",
            start_date_col="absence_start_date",
            end_date_col="absence_end_date",
            planned_col="is_planned",
            window_weeks=52,
            as_of_date="2026-10-01",
        )

        assert "absence_spells" in res.columns
        assert "total_days_absent" in res.columns
        assert "bradford_score" in res.columns
        assert "bradford_risk_band" in res.columns
        assert "disruption_multiplier" in res.columns

        # EMP_0001: Planned 10-day leave
        emp1 = res[res["employee_id"] == "EMP_0001"].iloc[0]
        assert emp1["absence_spells"] == 1
        assert emp1["total_days_absent"] == 10
        assert emp1["bradford_score"] == 10
        assert emp1["bradford_risk_band"] == "Low Friction"
        assert emp1["disruption_multiplier"] == 1.0

        # EMP_0002: Chronic 10 separate 1-day call-outs
        emp2 = res[res["employee_id"] == "EMP_0002"].iloc[0]
        assert emp2["absence_spells"] == 10
        assert emp2["total_days_absent"] == 10
        assert emp2["bradford_score"] == 1000
        assert emp2["bradford_risk_band"] == "Critical Friction"
        assert emp2["disruption_multiplier"] == 100.0

        # Operational friction ratio: same days (10), but 100x higher friction
        assert emp2["bradford_score"] == 100 * emp1["bradford_score"]

    def test_bradford_factor_53_weeks_parameter(self):
        from data.synthetic_generators import generate_bradford_factor_workforce
        from people_analytics_toolkit.chaos import calculate_bradford_factor

        spell_df, _ = generate_bradford_factor_workforce(n_employees=40, window_weeks=53, seed=42)
        res_53 = calculate_bradford_factor(
            spell_df,
            employee_col="employee_id",
            start_date_col="absence_start_date",
            end_date_col="absence_end_date",
            window_weeks=53,
            as_of_date="2026-10-01",
        )
        assert not res_53.empty
        assert (res_53["bradford_score"] >= 0).all()

    def test_bradford_factor_daily_attendance_streaks(self):
        from people_analytics_toolkit.chaos import calculate_bradford_factor

        # Create daily attendance records for an employee
        # Streak 1: Oct 1, Oct 2, Oct 3 (3 days consecutive = 1 spell)
        # Streak 2: Oct 10 (1 day = 1 spell)
        # Total spells = 2, total days = 4 -> Bradford = 2^2 * 4 = 16
        daily_df = pd.DataFrame({
            "employee_id": ["E1"] * 10,
            "date": pd.date_range("2026-10-01", periods=10, freq="D"),
            "is_absent": [1, 1, 1, 0, 0, 0, 0, 0, 0, 1],
        })

        res = calculate_bradford_factor(
            daily_df,
            employee_col="employee_id",
            date_col="date",
            is_absent_col="is_absent",
            window_weeks=4,
            as_of_date="2026-10-15",
        )

        assert len(res) == 1
        row = res.iloc[0]
        assert row["absence_spells"] == 2
        assert row["total_days_absent"] == 4
        assert row["bradford_score"] == 16
        assert row["disruption_multiplier"] == 4.0


class TestSyntheticControlWeightsAndTwin:
    """Unit tests for Synthetic Control Weights (Synth-DiD Synthetic Twin Builder)."""

    def test_compute_synthetic_control_weights_convexity(self):
        from people_analytics_toolkit.prescriptive import compute_synthetic_control_weights

        rng = np.random.default_rng(42)
        n_donors, t_pre = 8, 15
        donor_matrix = rng.normal(20.0, 3.0, size=(n_donors, t_pre))
        # Target is a convex combination of donor 0 (60%) and donor 2 (40%) with slight noise
        target_pre = 0.60 * donor_matrix[0] + 0.40 * donor_matrix[2] + rng.normal(0, 0.05, size=t_pre)

        weights, c0, pre_rmspe = compute_synthetic_control_weights(
            donor_matrix_pre=donor_matrix,
            target_series_pre=target_pre,
            donor_names=[f"Donor_{i}" for i in range(n_donors)],
            intercept=True,
            l2_regularization=1e-5,
        )

        assert isinstance(weights, pd.Series)
        assert len(weights) == n_donors
        # Convexity constraints: non-negative and sum to 1.0
        assert (weights.values >= -1e-7).all()
        assert np.isclose(weights.sum(), 1.0, atol=1e-4)
        # Tight fit
        assert pre_rmspe < 0.20
        # Donors 0 and 2 should have the highest weights
        assert weights["Donor_0"] > 0.40
        assert weights["Donor_2"] > 0.25

    def test_build_synthetic_control_single_twin(self):
        from data.synthetic_generators import generate_retail_scheduling_pilot_data
        from people_analytics_toolkit.prescriptive import build_synthetic_control_twin

        panel_df = generate_retail_scheduling_pilot_data(
            n_stores=20,
            n_weeks=20,
            pilot_stores=("Store_01", "Store_02"),
            rollout_week=14,
            seed=42,
        )

        twin_res = build_synthetic_control_twin(
            df=panel_df,
            unit_col="store_id",
            time_col="week_idx",
            outcome_col="turnover_rate",
            pilot_unit="Store_01",
            post_period_start=14,
            intercept=True,
            l2_regularization=1e-4,
        )

        assert twin_res["pilot_unit"] == "Store_01"
        assert "weights" in twin_res
        assert "pre_rmspe" in twin_res
        assert "post_causal_lift_mean" in twin_res
        assert "trajectory_df" in twin_res

        # Synthetic twin should closely match pre-treatment trajectory
        assert twin_res["pre_rmspe"] < 1.0
        # Post-treatment: automated scheduling tool reduces turnover (negative lift)
        assert twin_res["post_causal_lift_mean"] < -2.0
        assert twin_res["post_causal_lift_pct"] < -10.0

        traj = twin_res["trajectory_df"]
        assert len(traj) == 20
        assert set(traj.columns) == {"week_idx", "actual_outcome", "synthetic_twin", "causal_lift", "is_post_treatment"}
        assert traj["is_post_treatment"].sum() == 7  # weeks 14 to 20

    def test_build_multi_pilot_synthetic_controls(self):
        from data.synthetic_generators import generate_retail_scheduling_pilot_data
        from people_analytics_toolkit.prescriptive import build_multi_pilot_synthetic_controls

        panel_df = generate_retail_scheduling_pilot_data(
            n_stores=25,
            n_weeks=22,
            pilot_stores=("Store_01", "Store_02", "Store_03", "Store_04", "Store_05"),
            rollout_week=15,
            seed=42,
        )

        multi_res = build_multi_pilot_synthetic_controls(
            df=panel_df,
            unit_col="store_id",
            time_col="week_idx",
            outcome_col="turnover_rate",
            pilot_units=["Store_01", "Store_02", "Store_03", "Store_04", "Store_05"],
            post_period_start=15,
            intercept=True,
            l2_regularization=1e-4,
        )

        summary_df = multi_res["summary_df"]
        assert len(summary_df) == 5
        assert set(summary_df["pilot_unit"]) == {"Store_01", "Store_02", "Store_03", "Store_04", "Store_05"}
        assert (summary_df["pre_rmspe"] < 1.2).all()
        # All pilot stores experience turnover reduction
        assert (summary_df["post_causal_lift"] < 0.0).all()

        weights_mat = multi_res["weights_matrix"]
        assert weights_mat.shape[0] == 5
        # Donors should sum to 1.0 for each pilot store
        assert np.allclose(weights_mat.sum(axis=1), 1.0, atol=1e-3)

        assert multi_res["aggregate_causal_lift"] < -2.5
        assert multi_res["aggregate_causal_lift_pct"] < -12.0


class TestCausalDAGDiscovery:
    """Unit tests for Directed Causal Edge Weights (DAG Discovery via NOTEARS & linear SEM)."""

    def test_notears_acyclicity_constraint(self):
        import scipy.linalg as sla
        import networkx as nx
        from people_analytics_toolkit.prescriptive import notears_linear

        rng = np.random.default_rng(42)
        n = 500
        # True DAG: X0 -> X1 -> X2 -> X3, X0 -> X2
        x0 = rng.normal(0, 1, n)
        x1 = 0.7 * x0 + rng.normal(0, 0.3, n)
        x2 = 0.5 * x1 + 0.4 * x0 + rng.normal(0, 0.3, n)
        x3 = 0.8 * x2 + rng.normal(0, 0.3, n)
        X = np.column_stack([x0, x1, x2, x3])
        X = (X - X.mean(axis=0)) / X.std(axis=0)

        W = notears_linear(X, lambda1=0.05, w_threshold=0.20)
        assert W.shape == (4, 4)
        assert np.allclose(np.diag(W), 0.0)

        # Check smooth acyclicity metric h(W)
        h_val = float(np.trace(sla.expm(W * W)) - 4)
        assert h_val <= 1e-4

        # Strictly a directed acyclic graph
        G = nx.DiGraph(W)
        assert nx.is_directed_acyclic_graph(G)

    def test_directed_edge_recovery_known_dag(self):
        from people_analytics_toolkit.prescriptive import compute_directed_causal_edge_weights

        rng = np.random.default_rng(42)
        n = 800
        s = 0.4
        # Strict linear chain with equal error variances (Peters & Buhlmann, 2014)
        A = rng.normal(0, s, n)
        B = 0.85 * A + rng.normal(0, s, n)
        C = 0.80 * B + rng.normal(0, s, n)
        df = pd.DataFrame({"A": A, "B": B, "C": C})

        res = compute_directed_causal_edge_weights(df, variables=["A", "B", "C"], lambda1=0.03, threshold=0.15)
        W_df = res["directed_weights_df"]

        # Forward causal edges should be strongly positive
        assert W_df.loc["A", "B"] > 0.4
        assert W_df.loc["B", "C"] > 0.4

        # Reverse edges must be 0
        assert W_df.loc["B", "A"] == 0.0
        assert W_df.loc["C", "B"] == 0.0
        assert W_df.loc["C", "A"] == 0.0

        # Topological sort should place A before B before C
        topo = res["topological_order"]
        assert topo.index("A") < topo.index("B") < topo.index("C")
        assert "A" in res["root_causes"]
        assert "C" in res["sink_outcomes"]

    def test_confounder_detection_overtime_turnover(self):
        from data.synthetic_generators import generate_overtime_turnover_causal_dag_data
        from people_analytics_toolkit.prescriptive import (
            compute_directed_causal_edge_weights,
            diagnose_causal_confounding,
        )

        df = generate_overtime_turnover_causal_dag_data(n_samples=1000, seed=42)
        res = compute_directed_causal_edge_weights(df, lambda1=0.03, threshold=0.15)

        W_df = res["directed_weights_df"]
        corr = res["correlation_matrix"]

        # In observational correlation, overtime and turnover are strongly linked
        assert corr.loc["overtime_hours", "turnover_risk"] > 0.60

        # Diagnose relationship between overtime_hours and turnover_risk
        diag = diagnose_causal_confounding(W_df, "overtime_hours", "turnover_risk", corr_matrix=corr)

        # Manager absence should be identified as a common upstream confounder
        assert "manager_absence" in diag["common_confounders"] or "understaffing" in diag["common_confounders"]
        assert diag["observed_correlation"] > 0.60
        assert "Confounded" in diag["verdict"] or "Mediated" in diag["verdict"] or "Spurious" in diag["verdict"]

        # Root causes should contain manager_absence or workload_surge
        assert "manager_absence" in res["root_causes"] or "workload_surge" in res["root_causes"]

    def test_total_causal_effects_calculation(self):
        from people_analytics_toolkit.prescriptive import compute_directed_causal_edge_weights

        rng = np.random.default_rng(42)
        n = 800
        s = 0.4
        # A -> B -> C: direct A->C is 0, but mediated path exists
        A = rng.normal(0, s, n)
        B = 0.80 * A + rng.normal(0, s, n)
        C = 0.70 * B + rng.normal(0, s, n)
        df = pd.DataFrame({"A": A, "B": B, "C": C})

        res = compute_directed_causal_edge_weights(df, variables=["A", "B", "C"], lambda1=0.03, threshold=0.15)
        T_df = res["total_causal_effects_df"]

        # Total causal effect from A to C should be positive through the mediated path
        assert T_df.loc["A", "C"] > 0.20
        # Reverse total causal effect must be zero
        assert np.isclose(T_df.loc["C", "A"], 0.0, atol=1e-5)


class TestRollingMetricsPipeline:
    """Unit tests for Configuration-Driven Rolling Metrics."""

    def test_compute_rolling_metrics_basic(self):
        from people_analytics_toolkit.memory import RollingMetricConfig, compute_rolling_metrics
        from data.synthetic_generators import generate_store_department_labor_panel

        df = generate_store_department_labor_panel(n_stores=2, n_depts=2, n_weeks=55, seed=42)

        configs = [
            RollingMetricConfig(
                target_column="turnover_rate",
                window=52,
                aggregation="mean",
                output_column="turnover_roll_mean_52w",
            ),
            RollingMetricConfig(
                target_column="headcount",
                window=13,
                aggregation="mean",
                output_column="headcount_roll_mean_13w",
            ),
            RollingMetricConfig(
                target_column="assigned_hours",
                window=4,
                aggregation="var",
                output_column="hours_roll_var_4w",
            ),
        ]

        res_df = compute_rolling_metrics(
            df=df,
            configs=configs,
            partition_cols=["store_id", "dept_id"],
            temporal_col="week",
        )

        assert "turnover_roll_mean_52w" in res_df.columns
        assert "headcount_roll_mean_13w" in res_df.columns
        assert "hours_roll_var_4w" in res_df.columns

        # Verify len matches input exactly
        assert len(res_df) == len(df)
        assert list(res_df.index) == list(df.index)

        # Check values
        assert (res_df["turnover_roll_mean_52w"] >= 0.0).all()
        assert (res_df["headcount_roll_mean_13w"] > 0.0).all()
        # Variance with min_periods=1 is NaN for first element, non-negative thereafter
        valid_var = res_df["hours_roll_var_4w"].dropna()
        assert len(valid_var) > 0
        assert (valid_var >= 0.0).all()

    def test_chronological_integrity_and_unsorted_index(self):
        from people_analytics_toolkit.memory import compute_rolling_metrics

        # Create sorted baseline
        df_sorted = pd.DataFrame({
            "store_id": ["S1", "S1", "S1", "S1"],
            "dept_id": ["D1", "D1", "D1", "D1"],
            "week": [1, 2, 3, 4],
            "turnover_rate": [0.02, 0.04, 0.06, 0.08],
        })

        config = [{"target_column": "turnover_rate", "window": 2, "aggregation": "mean", "output_column": "turn_mean_2w"}]

        res_sorted = compute_rolling_metrics(
            df_sorted,
            configs=config,
            partition_cols=["store_id", "dept_id"],
            temporal_col="week",
        )

        # Shuffle rows and give custom non-sequential index
        df_shuffled = df_sorted.sample(frac=1.0, random_state=42).copy()
        df_shuffled.index = [303, 101, 404, 202]

        res_shuffled = compute_rolling_metrics(
            df_shuffled,
            configs=config,
            partition_cols=["store_id", "dept_id"],
            temporal_col="week",
        )

        # Confirm exact match for each row regardless of initial order
        for idx in df_shuffled.index:
            wk = df_shuffled.loc[idx, "week"]
            expected_val = res_sorted.loc[res_sorted["week"] == wk, "turn_mean_2w"].iloc[0]
            actual_val = res_shuffled.loc[idx, "turn_mean_2w"]
            assert np.isclose(expected_val, actual_val)

        # Confirm original index and row ordering preserved
        assert list(res_shuffled.index) == [303, 101, 404, 202]

    def test_hierarchical_partitioning_isolation(self):
        from people_analytics_toolkit.memory import compute_rolling_metrics

        df = pd.DataFrame({
            "store_id": ["S1", "S1", "S2", "S2"],
            "dept_id": ["D1", "D1", "D1", "D1"],
            "week": [1, 2, 1, 2],
            "hours": [100.0, 100.0, 500.0, 500.0],
        })

        config = [{"target_column": "hours", "window": 2, "aggregation": "mean", "output_column": "mean_hours"}]

        res = compute_rolling_metrics(df, configs=config, partition_cols=["store_id", "dept_id"], temporal_col="week")

        # S1 week 1 should be 100, S1 week 2 should be 100
        # S2 week 1 should be 500 (NOT influenced by S1 100), S2 week 2 should be 500
        assert np.isclose(res.loc[0, "mean_hours"], 100.0)
        assert np.isclose(res.loc[1, "mean_hours"], 100.0)
        assert np.isclose(res.loc[2, "mean_hours"], 500.0)
        assert np.isclose(res.loc[3, "mean_hours"], 500.0)

    def test_lag_prevents_lookahead_bias(self):
        from people_analytics_toolkit.memory import compute_rolling_metrics

        df = pd.DataFrame({
            "store_id": ["S1", "S1", "S1"],
            "week": [1, 2, 3],
            "sales": [10.0, 20.0, 30.0],
        })

        configs = [
            {"target_column": "sales", "window": 2, "aggregation": "mean", "lag": 0, "output_column": "sales_lag0"},
            {"target_column": "sales", "window": 2, "aggregation": "mean", "lag": 1, "output_column": "sales_lag1"},
        ]

        res = compute_rolling_metrics(df, configs=configs, partition_cols="store_id", temporal_col="week")

        # Week 1 lag 1 should be NaN
        assert pd.isna(res.loc[0, "sales_lag1"])
        # Week 2 lag 1 should equal Week 1 lag 0 (10.0)
        assert np.isclose(res.loc[1, "sales_lag1"], res.loc[0, "sales_lag0"])
        # Week 3 lag 1 should equal Week 2 lag 0 (15.0)
        assert np.isclose(res.loc[2, "sales_lag1"], res.loc[1, "sales_lag0"])

    def test_pipeline_object_oriented_api(self):
        from people_analytics_toolkit.memory import RollingMetricsPipeline

        df = pd.DataFrame({
            "store_id": ["S1", "S1", "S1"],
            "week": [1, 2, 3],
            "metric": [1.0, 2.0, 3.0],
        })

        pipeline = (
            RollingMetricsPipeline(partition_cols="store_id", temporal_col="week")
            .add_metric("metric", window=2, aggregation="mean", output_column="metric_mean_2w")
            .add_metric("metric", window=2, aggregation="max", output_column="metric_max_2w")
        )

        assert pipeline.get_feature_names() == ["metric_mean_2w", "metric_max_2w"]

        res = pipeline.fit_transform(df)
        assert "metric_mean_2w" in res.columns
        assert "metric_max_2w" in res.columns
        assert np.isclose(res["metric_max_2w"].iloc[-1], 3.0)


class TestSingularSpectrumAnalysis:
    """Unit tests for Singular Spectrum Analysis (SVD Smoothing)."""

    def test_ssa_noise_reduction_and_signal_recovery(self):
        from people_analytics_toolkit.memory import singular_spectrum_analysis

        rng = np.random.default_rng(42)
        t = np.linspace(0, 10, 150)
        true_signal = 10.0 + 1.5 * t + 3.0 * np.sin(2 * np.pi * t / 2.5)
        noise = rng.normal(0, 1.2, size=len(t))
        noisy_series = pd.Series(true_signal + noise, name="traffic")

        res = singular_spectrum_analysis(noisy_series, window_length=25, top_k=3)

        assert "smoothed" in res
        assert "residual" in res
        assert len(res["smoothed"]) == len(noisy_series)
        assert res["n_components_used"] == 3

        # Reconstructed signal should filter out the noise significantly
        raw_rmse = np.sqrt(np.mean((noisy_series.to_numpy() - true_signal) ** 2))
        rec_rmse = np.sqrt(np.mean((res["smoothed"].to_numpy() - true_signal) ** 2))
        assert rec_rmse < raw_rmse * 0.5  # At least 50% noise reduction

        # Residuals should have mean approximately zero and standard deviation close to noise scale
        assert np.isclose(res["residual"].mean(), 0.0, atol=0.3)

    def test_ssa_zero_phase_lag_vs_rolling_mean(self):
        from people_analytics_toolkit.memory import singular_spectrum_analysis

        # Pure harmonic cycle with known peak locations
        t = np.arange(100)
        period = 14
        clean_cycle = np.sin(2 * np.pi * t / period)
        noisy_cycle = pd.Series(clean_cycle + np.random.default_rng(42).normal(0, 0.2, len(t)))

        ssa_smoothed = singular_spectrum_analysis(noisy_cycle, window_length=28, top_k=2)["smoothed"]
        causal_rolling = noisy_cycle.rolling(window=7, min_periods=1).mean()

        # Find peak indices around period 14 (should peak around t=3.5, 17.5, 31.5)
        # Check second cycle peak (between t=14 and t=28)
        cycle2_slice = slice(14, 28)
        true_peak_idx = 14 + np.argmax(clean_cycle[cycle2_slice])
        ssa_peak_idx = 14 + np.argmax(ssa_smoothed.iloc[cycle2_slice].to_numpy())
        rolling_peak_idx = 14 + np.argmax(causal_rolling.iloc[cycle2_slice].to_numpy())

        # SSA peak matches true peak within 1 step (zero phase distortion)
        assert abs(ssa_peak_idx - true_peak_idx) <= 1
        # Causal rolling peak is delayed (shifted rightward due to phase lag)
        assert rolling_peak_idx >= true_peak_idx + 2

    def test_ssa_variance_threshold_component_selection(self):
        from people_analytics_toolkit.memory import singular_spectrum_analysis

        rng = np.random.default_rng(42)
        s = pd.Series(rng.normal(100, 10, size=80))

        res_90 = singular_spectrum_analysis(s, window_length=20, variance_threshold=0.90)
        res_99 = singular_spectrum_analysis(s, window_length=20, variance_threshold=0.99)

        assert res_90["n_components_used"] <= res_99["n_components_used"]
        cum_var_90 = np.sum(res_90["explained_variance_ratio"][: res_90["n_components_used"]])
        assert cum_var_90 >= 0.90

    def test_compute_svd_smoothed_baseline_dataframe_and_partitions(self):
        from people_analytics_toolkit.memory import compute_svd_smoothed_baseline
        from data.synthetic_generators import generate_daily_store_foot_traffic

        df = generate_daily_store_foot_traffic(n_days=60, n_stores=2, seed=42)

        res_df = compute_svd_smoothed_baseline(
            df=df,
            target_col="observed_foot_traffic",
            partition_cols="store_id",
            temporal_col="date",
            top_k=3,
        )

        assert "observed_foot_traffic_svd_smoothed" in res_df.columns
        assert "observed_foot_traffic_svd_residual" in res_df.columns
        assert len(res_df) == len(df)
        assert list(res_df.index) == list(df.index)

        # Baseline should be positive and smoother than raw observed traffic
        store1 = res_df[res_df["store_id"] == "Store_01"]
        raw_diff_std = store1["observed_foot_traffic"].diff().std()
        smooth_diff_std = store1["observed_foot_traffic_svd_smoothed"].diff().std()
        assert smooth_diff_std < raw_diff_std

    def test_singular_spectrum_analysis_class_interface(self):
        from people_analytics_toolkit.memory import SingularSpectrumAnalysis

        t = np.arange(80)
        signal = 50.0 + 0.2 * t + 5.0 * np.cos(2 * np.pi * t / 7) + np.random.default_rng(42).normal(0, 1, 80)

        ssa = SingularSpectrumAnalysis(window_length=14, top_k=3)
        smoothed = ssa.fit_transform(signal)

        assert len(smoothed) == 80
        assert ssa.n_components_used_ == 3
        assert len(ssa.singular_values_) == 14
        assert np.isclose(np.sum(ssa.explained_variance_ratio_), 1.0)

        comps = ssa.decompose()
        assert comps.shape == (80, 3)
        assert "component_1" in comps.columns
        assert "component_2" in comps.columns
        assert "component_3" in comps.columns


class TestTemporalLagAndDifferencingFeatures:
    """Unit tests for Hierarchical Lag and Multi-Horizon Differencing features (Features 47 & 48)."""

    def test_compute_lag_features_basic_and_dict(self):
        from people_analytics_toolkit.memory import compute_lag_features

        df = pd.DataFrame({
            "employee_id": ["E1", "E1", "E1", "E1"],
            "month": [1, 2, 3, 4],
            "sales": [100.0, 150.0, 200.0, 250.0],
        })

        res = compute_lag_features(
            df=df,
            target_cols="sales",
            lags={"lag1": 1, "lag2": 2},
            partition_cols="employee_id",
            temporal_col="month",
        )

        assert "sales_lag1" in res.columns
        assert "sales_lag2" in res.columns
        assert pd.isna(res.loc[0, "sales_lag1"])
        assert np.isclose(res.loc[1, "sales_lag1"], 100.0)
        assert np.isclose(res.loc[2, "sales_lag1"], 150.0)
        assert pd.isna(res.loc[0, "sales_lag2"])
        assert pd.isna(res.loc[1, "sales_lag2"])
        assert np.isclose(res.loc[2, "sales_lag2"], 100.0)

    def test_lag_features_partition_isolation_and_shuffle(self):
        from people_analytics_toolkit.memory import compute_lag_features

        # Intentionally shuffled input
        df = pd.DataFrame({
            "store_id": ["S2", "S1", "S1", "S2"],
            "week": [2, 2, 1, 1],
            "turnover": [0.4, 0.2, 0.1, 0.3],
        }, index=["row_a", "row_b", "row_c", "row_d"])

        res = compute_lag_features(
            df=df,
            target_cols="turnover",
            lags=1,
            partition_cols="store_id",
            temporal_col="week",
        )

        # Output index matches original input index exactly
        assert list(res.index) == ["row_a", "row_b", "row_c", "row_d"]

        # S1 week 1 (row_c) -> lag1 is NaN
        assert pd.isna(res.loc["row_c", "turnover_lag_1"])
        # S1 week 2 (row_b) -> lag1 is 0.1 (from row_c)
        assert np.isclose(res.loc["row_b", "turnover_lag_1"], 0.1)

        # S2 week 1 (row_d) -> lag1 is NaN (no leakage from S1)
        assert pd.isna(res.loc["row_d", "turnover_lag_1"])
        # S2 week 2 (row_a) -> lag1 is 0.3 (from row_d)
        assert np.isclose(res.loc["row_a", "turnover_lag_1"], 0.3)

    def test_compute_differencing_discrete_presets(self):
        from people_analytics_toolkit.memory import compute_differencing_features

        # Monthly time series over 14 months
        months = list(range(1, 15))
        values = [float(10 * m) for m in months]  # Linear: 10, 20, 30, ...
        df = pd.DataFrame({"month": months, "revenue": values})

        res = compute_differencing_features(
            df=df,
            target_cols="revenue",
            periods="monthly",  # presets: mom=1, qoq=3, yoy=12
            temporal_col="month",
        )

        assert "revenue_mom" in res.columns
        assert "revenue_qoq" in res.columns
        assert "revenue_yoy" in res.columns

        # For linear 10*m:
        # MoM delta is 10.0 for m >= 2
        assert pd.isna(res.loc[0, "revenue_mom"])
        assert np.isclose(res.loc[1, "revenue_mom"], 10.0)
        assert np.isclose(res.loc[13, "revenue_mom"], 10.0)

        # QoQ delta is 30.0 for m >= 4
        assert pd.isna(res.loc[2, "revenue_qoq"])
        assert np.isclose(res.loc[3, "revenue_qoq"], 30.0)

        # YoY delta is 120.0 for m >= 13
        assert pd.isna(res.loc[11, "revenue_yoy"])
        assert np.isclose(res.loc[12, "revenue_yoy"], 120.0)

    def test_compute_differencing_percent_change_and_zero_division(self):
        from people_analytics_toolkit.memory import compute_differencing_features

        df = pd.DataFrame({
            "store_id": ["S1", "S1", "S1", "S1"],
            "week": [1, 2, 3, 4],
            "attrition": [0.0, 5.0, 10.0, 15.0],  # Week 1 is 0.0 -> division by zero on week 2
        })

        res = compute_differencing_features(
            df=df,
            target_cols="attrition",
            periods={"wow": 1},
            pct_change=True,
            as_percent=True,
            partition_cols="store_id",
            temporal_col="week",
        )

        assert "attrition_wow_pct" in res.columns
        # Week 1: NaN (no prior period)
        assert pd.isna(res.loc[0, "attrition_wow_pct"])
        # Week 2: (5.0 - 0.0)/0.0 -> inf replaced by NaN safely without raising error
        assert pd.isna(res.loc[1, "attrition_wow_pct"])
        # Week 3: (10.0 - 5.0)/5.0 * 100 = 100%
        assert np.isclose(res.loc[2, "attrition_wow_pct"], 100.0)
        # Week 4: (15.0 - 10.0)/10.0 * 100 = 50%
        assert np.isclose(res.loc[3, "attrition_wow_pct"], 50.0)

    def test_temporal_features_pipeline(self):
        from people_analytics_toolkit.memory import TemporalFeaturesPipeline

        df = pd.DataFrame({
            "dept": ["Engineering", "Engineering", "Engineering"],
            "period": [1, 2, 3],
            "headcount": [50.0, 55.0, 66.0],
        })

        pipeline = (
            TemporalFeaturesPipeline(partition_cols="dept", temporal_col="period")
            .add_lags("headcount", lags=[1, 2])
            .add_differencing("headcount", periods={"pop": 1}, pct_change=False)
            .add_differencing("headcount", periods={"growth": 1}, pct_change=True, as_percent=False)
        )

        res = pipeline.fit_transform(df)

        assert "headcount_lag_1" in res.columns
        assert "headcount_lag_2" in res.columns
        assert "headcount_pop" in res.columns
        assert "headcount_growth_pct" in res.columns

        # Headcount pop delta: 55 - 50 = 5.0, 66 - 55 = 11.0
        assert np.isclose(res.loc[1, "headcount_pop"], 5.0)
        assert np.isclose(res.loc[2, "headcount_pop"], 11.0)

        # Growth rate: (55 - 50)/50 = 0.10, (66 - 55)/55 = 0.20
        assert np.isclose(res.loc[1, "headcount_growth_pct"], 0.10)
        assert np.isclose(res.loc[2, "headcount_growth_pct"], 0.20)


class TestCorrelationHeatmapTable:
    """Unit tests for Feature 49: Correlation Heatmap Table (Pearson, Spearman, Kendall with CIs & p-values)."""

    def _generate_test_data(self):
        rng = np.random.default_rng(42)
        n = 50
        tenure = np.linspace(1, 10, n)
        overtime = 2.5 * tenure + rng.normal(0, 1.0, n)
        satisfaction = -1.5 * tenure + rng.normal(0, 1.0, n)
        noise = rng.normal(0, 5.0, n)
        turnover_risk = 0.6 * tenure + 0.3 * overtime - 0.4 * satisfaction + rng.normal(0, 0.5, n)

        return pd.DataFrame({
            "tenure": tenure,
            "overtime": overtime,
            "satisfaction": satisfaction,
            "noise": noise,
            "turnover_risk": turnover_risk,
        })

    def test_correlation_heatmap_df_spearman_pearson_kendall(self):
        from people_analytics_toolkit.attribution import correlation_heatmap_df

        df = self._generate_test_data()

        for method in ["spearman", "pearson", "kendall"]:
            res = correlation_heatmap_df(
                df=df,
                target="turnover_risk",
                method=method,
                p_threshold=0.05,
                viz=False,
            )

            assert isinstance(res, pd.DataFrame)
            expected_cols = [
                "METRIC",
                "CORRELATION TO turnover_risk",
                "P",
                "LB",
                "UB",
                "POWER",
                "METHOD",
                "CONTAINS_0",
                "SIGNIFICANT",
            ]
            for col in expected_cols:
                assert col in res.columns

            assert (res["METHOD"] == method).all()
            assert len(res) == 4  # 4 predictors
            # Overtime and tenure should be strongly positively correlated
            assert res[res["METRIC"] == "overtime"]["CORRELATION TO turnover_risk"].iloc[0] > 0.5
            # Satisfaction should be strongly negatively correlated
            assert res[res["METRIC"] == "satisfaction"]["CORRELATION TO turnover_risk"].iloc[0] < -0.5
            # Bounds check
            assert (res["LB"] <= res["UB"]).all()

    def test_significance_and_zero_crossing_identification(self):
        from people_analytics_toolkit.attribution import correlation_heatmap_df

        df = self._generate_test_data()
        res = correlation_heatmap_df(
            df=df,
            target="turnover_risk",
            method="spearman",
            p_threshold=0.01,
            viz=False,
        )

        # High-signal predictors should be significant and not contain 0
        tenure_row = res[res["METRIC"] == "tenure"].iloc[0]
        assert tenure_row["SIGNIFICANT"] == 1
        assert tenure_row["CONTAINS_0"] == 0

        # Noise column should have higher p-value or span zero
        noise_row = res[res["METRIC"] == "noise"].iloc[0]
        assert (noise_row["SIGNIFICANT"] == 0) or (noise_row["CONTAINS_0"] == 1)

    def test_drop_insignificant_and_drop_ci_contain_0(self):
        from people_analytics_toolkit.attribution import correlation_heatmap_df

        df = self._generate_test_data()

        res_drop_insig = correlation_heatmap_df(
            df=df,
            target="turnover_risk",
            method="spearman",
            p_threshold=0.01,
            drop_insignificant=True,
            viz=False,
        )
        assert "SIGNIFICANT" not in res_drop_insig.columns
        assert (res_drop_insig["P"] < 0.01).all()

        res_drop_ci0 = correlation_heatmap_df(
            df=df,
            target="turnover_risk",
            method="spearman",
            drop_ci_contain_0=True,
            viz=False,
        )
        assert "CONTAINS_0" not in res_drop_ci0.columns
        assert (np.sign(res_drop_ci0["LB"]) == np.sign(res_drop_ci0["UB"])).all()

    def test_styler_generation_and_return_styler(self):
        from people_analytics_toolkit.attribution import correlation_heatmap_df

        df = self._generate_test_data()

        # Attribute styler
        res_df = correlation_heatmap_df(df=df, target="turnover_risk", viz=False)
        assert "styler" in res_df.attrs
        styler = res_df.attrs["styler"]
        html = styler.to_html()
        assert "background-color" in html

        # return_styler = True
        styler_ret, df_ret = correlation_heatmap_df(
            df=df,
            target="turnover_risk",
            viz=False,
            return_styler=True,
        )
        assert styler_ret is not None
        assert isinstance(df_ret, pd.DataFrame)
        assert len(df_ret) == 4

    def test_error_handling(self):
        import pytest
        from people_analytics_toolkit.attribution import correlation_heatmap_df

        df = self._generate_test_data()

        with pytest.raises(TypeError):
            correlation_heatmap_df("not_a_df", target="turnover_risk")

        with pytest.raises(KeyError):
            correlation_heatmap_df(df, target="nonexistent_column")

        with pytest.raises(ValueError):
            df_str = df.copy()
            df_str["str_target"] = ["A"] * len(df)
            correlation_heatmap_df(df_str, target="str_target")


class TestMahalanobisDistance:
    """Test suite for calculate_mahalanobis_distance multivariate anomaly detection."""

    def test_mahalanobis_vectorized_vs_scipy(self):
        from scipy.spatial.distance import mahalanobis
        from people_analytics_toolkit.anomalies import calculate_mahalanobis_distance

        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, size=(25, 4))
        dist, is_out, meta = calculate_mahalanobis_distance(X, regularization=1e-6)

        assert len(dist) == 25
        assert isinstance(dist, pd.Series)
        assert meta["degrees_of_freedom"] == 4

        mu = meta["mean"]
        inv_cov = meta["inv_covariance"]

        # Validate row-by-row equivalence with scipy reference implementation
        for i in range(len(X)):
            scipy_d = mahalanobis(X[i], mu, inv_cov)
            assert np.isclose(dist.iloc[i], scipy_d, atol=1e-5)

    def test_mahalanobis_outlier_detection_and_p_values(self):
        from people_analytics_toolkit.anomalies import calculate_mahalanobis_distance

        rng = np.random.default_rng(42)
        N = 100
        df = pd.DataFrame({
            "overtime_hours": rng.normal(5.0, 1.5, size=N),
            "unplanned_absences": rng.normal(2.0, 0.8, size=N),
            "hourly_wage": rng.normal(30.0, 5.0, size=N),
        })

        # Inject extreme anomaly at index 0
        df.iloc[0] = [45.0, 20.0, 5.0]

        dist, is_outlier, meta = calculate_mahalanobis_distance(df, significance_level=0.01)

        assert is_outlier.iloc[0] == True
        assert meta["p_values"].iloc[0] < 0.001
        assert dist.iloc[0] > meta["threshold_distance"]
        assert meta["critical_value"] > 0
        assert meta["degrees_of_freedom"] == 3
        assert meta["n_outliers"] >= 1
        assert 0.0 < meta["outlier_fraction"] <= 0.2

    def test_mahalanobis_dataframe_and_feature_cols(self):
        from people_analytics_toolkit.anomalies import calculate_mahalanobis_distance

        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "emp_id": [f"EMP_{i:03d}" for i in range(50)],
            "perf_rating": rng.uniform(2.5, 4.8, size=50),
            "tenure_months": rng.uniform(6, 60, size=50),
            "department": ["Sales"] * 25 + ["Engineering"] * 25,
        }, index=[f"idx_{i}" for i in range(50)])

        dist, is_outlier, meta = calculate_mahalanobis_distance(
            df,
            feature_cols=["perf_rating", "tenure_months"],
            significance_level=0.05,
        )

        assert dist.index.equals(df.index)
        assert is_outlier.index.equals(df.index)
        assert meta["mean"].index.tolist() == ["perf_rating", "tenure_months"]
        assert meta["covariance"].shape == (2, 2)
        assert meta["inv_covariance"].shape == (2, 2)

    def test_mahalanobis_reference_data(self):
        from people_analytics_toolkit.anomalies import calculate_mahalanobis_distance

        rng = np.random.default_rng(42)
        # Reference baseline: standard normal distribution
        baseline_df = pd.DataFrame(rng.normal(0, 1, size=(200, 3)), columns=["a", "b", "c"])

        # Monitored test data: shifted distribution with high anomaly
        test_df = pd.DataFrame({
            "a": [0.1, 0.2, 10.0],
            "b": [-0.1, 0.0, 12.0],
            "c": [0.05, -0.2, -15.0],
        })

        dist, is_outlier, meta = calculate_mahalanobis_distance(
            data=test_df,
            reference_data=baseline_df,
            significance_level=0.01,
        )

        assert len(dist) == 3
        # First two rows are close to baseline mean
        assert dist.iloc[0] < 3.0
        assert dist.iloc[1] < 3.0
        assert not is_outlier.iloc[0]
        assert not is_outlier.iloc[1]
        # Third row is extreme shift
        assert dist.iloc[2] > 15.0
        assert is_outlier.iloc[2] == True

    def test_mahalanobis_robust_mcd(self):
        from people_analytics_toolkit.anomalies import calculate_mahalanobis_distance

        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, size=(80, 2))
        # Contaminate first 5 observations
        X[:5] = [9.0, 9.0]

        dist_emp, out_emp, meta_emp = calculate_mahalanobis_distance(X, robust=False)
        dist_rob, out_rob, meta_rob = calculate_mahalanobis_distance(X, robust=True)

        assert meta_rob["method"] == "robust_mcd"
        # Robust centroid should be closer to origin (0, 0) than empirical centroid
        assert np.linalg.norm(meta_rob["mean"]) < np.linalg.norm(meta_emp["mean"])
        # Robust distance of outliers should be higher due to unswamped centroid
        assert dist_rob.iloc[0] > dist_emp.iloc[0]

    def test_mahalanobis_grouped(self):
        from people_analytics_toolkit.anomalies import calculate_mahalanobis_distance

        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "dept": ["Retail"] * 30 + ["HQ"] * 30,
            "metric_1": np.concatenate([rng.normal(10, 2, size=30), rng.normal(100, 10, size=30)]),
            "metric_2": np.concatenate([rng.normal(5, 1, size=30), rng.normal(50, 5, size=30)]),
        })

        # Inject within-group outlier for Retail (e.g. metric_1=35 which would be low for HQ, but huge for Retail)
        df.loc[0, "metric_1"] = 35.0

        dist, is_outlier, meta = calculate_mahalanobis_distance(
            df,
            feature_cols=["metric_1", "metric_2"],
            group_col="dept",
            significance_level=0.05,
        )

        assert len(dist) == 60
        assert "group_metrics" in meta
        assert "Retail" in meta["group_metrics"]
        assert "HQ" in meta["group_metrics"]
        assert is_outlier.iloc[0] == True

    def test_mahalanobis_pairwise_vectors(self):
        from people_analytics_toolkit.anomalies import calculate_mahalanobis_distance

        u = [1.0, 2.0]
        v = [1.0, 2.0]
        cov = [[1.0, 0.0], [0.0, 1.0]]

        # Identical vectors should have distance 0.0
        dist, is_outlier, meta = calculate_mahalanobis_distance(u, v=v, cov=cov)
        assert np.isclose(dist.iloc[0], 0.0)
        assert is_outlier.iloc[0] == False
        assert meta["method"] == "pairwise"

        # Distant vector
        u_far = [10.0, 20.0]
        dist_far, is_out_far, meta_far = calculate_mahalanobis_distance(u_far, v=v, cov=cov)
        assert dist_far.iloc[0] > 10.0
        assert is_out_far.iloc[0] == True

    def test_mahalanobis_validations_and_errors(self):
        import pytest
        from people_analytics_toolkit.anomalies import calculate_mahalanobis_distance

        # Empty data
        with pytest.raises(ValueError, match="data cannot be empty"):
            calculate_mahalanobis_distance([])

        # Invalid alpha
        with pytest.raises(ValueError, match="significance_level must be in"):
            calculate_mahalanobis_distance(np.array([[1.0, 2.0]]), significance_level=1.5)

        # NaNs in data
        with pytest.raises(ValueError, match="NaNs"):
            calculate_mahalanobis_distance(pd.DataFrame({"a": [1.0, np.nan, 3.0]}))

        # Missing feature col
        with pytest.raises(KeyError, match="Feature columns not found"):
            calculate_mahalanobis_distance(pd.DataFrame({"a": [1.0, 2.0]}), feature_cols=["b"])

        # Pairwise vector dimension mismatch
        with pytest.raises(ValueError, match="Vector dimensions do not match"):
            calculate_mahalanobis_distance([1.0, 2.0], v=[1.0, 2.0, 3.0], cov=np.eye(2))

        # Pairwise without covariance
        with pytest.raises(ValueError, match="cov.*must be provided"):
            calculate_mahalanobis_distance([1.0, 2.0], v=[2.0, 3.0])


class TestClassBasedEstimatorPatterns:
    """Verifies scikit-learn standard .fit(), .transform(), .predict() consistency for estimator classes."""

    def test_local_outlier_factor_detector_class(self):
        from people_analytics_toolkit.anomalies import LocalOutlierFactorDetector, PyODLocalOutlierFactor, fit_pyod_local_outlier_factor
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "f1": rng.normal(0, 1, size=60),
            "f2": rng.normal(5, 2, size=60),
            "f3": rng.normal(-2, 0.5, size=60),
        })
        # Plant an obvious outlier
        df.iloc[0] = [10.0, 25.0, 15.0]

        detector = LocalOutlierFactorDetector(n_neighbors=15, contamination=0.05, feature_cols=["f1", "f2", "f3"])
        detector.fit(df)
        assert detector.is_fitted
        assert detector.threshold_ is not None

        # Predict
        labels = detector.predict(df)
        scores = detector.decision_function(df)
        assert len(labels) == 60
        assert len(scores) == 60
        assert labels.iloc[0] == 1  # Outlier

        # Transform returns DataFrame with scores and binary labels
        trans_df = detector.transform(df)
        assert (trans_df["lof_anomaly_score"] == scores).all()
        assert (trans_df["is_lof_outlier"] == labels).all()

        # fit_predict & alias
        detector_alias = PyODLocalOutlierFactor(n_neighbors=15, contamination=0.05)
        fp_scores, fp_labels, fp_meta = detector_alias.fit_predict(df[["f1", "f2", "f3"]])
        assert len(fp_scores) == 60
        assert fp_labels.iloc[0] == 1

        # Convenience function equivalence
        func_scores, func_labels, func_meta = fit_pyod_local_outlier_factor(df, ["f1", "f2", "f3"], contamination=0.05, n_neighbors=15)
        assert len(func_scores) == 60
        assert func_labels.iloc[0] == 1

    def test_garch_volatility_model_class(self):
        from people_analytics_toolkit.chaos import GARCHVolatilityModel, GARCHVolatility, fit_garch_volatility
        rng = np.random.default_rng(42)
        series = pd.Series(rng.normal(100, 5, size=80))

        model = GARCHVolatilityModel(p=1, q=1)
        model.fit(series)
        assert model.is_fitted_
        assert model.res_ is not None

        # Transform
        cond_vol = model.transform()
        assert len(cond_vol) == 80
        assert (cond_vol > 0).all()

        # Forecast / Predict
        forecast_df = model.predict(horizon=3)
        assert "variance" in forecast_df.columns
        assert "volatility" in forecast_df.columns
        assert len(forecast_df) == 3
        assert (forecast_df["volatility"] > 0).all()

        # fit_transform and alias
        alias_model = GARCHVolatility(p=1, q=1)
        res_vol = alias_model.fit_transform(series)
        assert len(res_vol) == 80
        assert np.allclose(res_vol.values, cond_vol.values)

        # Functional wrapper equivalence
        f_vol, f_meta = fit_garch_volatility(series, p=1, q=1)
        assert len(f_vol) == 80
        assert np.allclose(f_vol.values, cond_vol.values)

    def test_workforce_hmm_regimes_class(self):
        from people_analytics_toolkit.chaos import WorkforceHMMRegimes, WorkforceHMM, fit_workforce_hmm_regimes
        rng = np.random.default_rng(42)
        n = 80
        df = pd.DataFrame({
            "ot": np.concatenate([rng.normal(5, 1, size=n // 2), rng.normal(25, 3, size=n // 2)]),
            "callout": np.concatenate([rng.normal(1, 0.5, size=n // 2), rng.normal(8, 1.5, size=n // 2)]),
        })

        hmm = WorkforceHMMRegimes(n_components=2, seed=42)
        hmm.fit(df, emission_cols=["ot", "callout"])
        assert hmm.is_fitted_
        assert hmm.means_.shape == (2, 2)
        assert hmm.transmat_.shape == (2, 2)

        # Predict & predict_proba
        states = hmm.predict(df)
        probs = hmm.predict_proba(df)
        assert len(states) == n
        assert probs.shape == (n, 2)
        assert np.allclose(probs.sum(axis=1), 1.0)

        # Transform & fit_transform
        annotated_df = hmm.transform(df)
        assert "hmm_latent_state" in annotated_df.columns
        assert "prob_latent_state_0" in annotated_df.columns
        assert "prob_latent_state_1" in annotated_df.columns

        # Alias check
        alias_hmm = WorkforceHMM(n_components=2, seed=42)
        alias_res = alias_hmm.fit_transform(df, emission_cols=["ot", "callout"])
        assert "hmm_latent_state" in alias_res.columns

        # Functional wrapper equivalence
        res_df, profile, meta = fit_workforce_hmm_regimes(df, ["ot", "callout"], n_components=2, seed=42)
        assert len(res_df) == n
        assert len(profile) == 2

    def test_hazard_embeddings_class(self):
        from people_analytics_toolkit.memory import HazardEmbeddings, HazardEmbeddingTransformer, fit_hazard_embeddings
        df = pd.DataFrame({
            "tenure_weeks": [5, 12, 26, 52, 100],
            "turnover_event": [1, 1, 0, 1, 0],
        })

        haz = HazardEmbeddings(tenure_col="tenure_weeks", event_col="turnover_event")
        haz.fit(df)
        assert haz.is_fitted_

        res_df = haz.transform(df)
        assert "survival_prob_embedding" in res_df.columns
        assert "cumulative_hazard_embedding" in res_df.columns
        assert (res_df["survival_prob_embedding"] >= 0.0).all()
        assert (res_df["survival_prob_embedding"] <= 1.0).all()

        # Specific point predictions
        surv_preds = haz.predict_survival([10, 25])
        haz_preds = haz.predict_hazard([10, 25])
        assert len(surv_preds) == 2
        assert len(haz_preds) == 2

        # fit_transform & alias
        alias_haz = HazardEmbeddingTransformer(tenure_col="tenure_weeks", event_col="turnover_event")
        alias_res = alias_haz.fit_transform(df)
        assert "survival_prob_embedding" in alias_res.columns

        # Functional wrapper equivalence
        func_res, models = fit_hazard_embeddings(df, "tenure_weeks", "turnover_event")
        assert (func_res["survival_prob_embedding"] == res_df["survival_prob_embedding"]).all()


class TestConsistentReturnTypes:
    """Verifies that functions previously returning raw dicts now return rich dataclasses
    supporting both dot-attribute access and dictionary key indexing for 100% backward compatibility."""

    def test_fit_garch_volatility_result_type(self):
        from people_analytics_toolkit.chaos import fit_garch_volatility, GARCHResult, GARCHModelResult
        rng = np.random.default_rng(42)
        series = pd.Series(rng.normal(100, 5, size=60))
        vol, meta = fit_garch_volatility(series, p=1, q=1)

        # Type checks and aliases
        assert isinstance(meta, GARCHResult)
        assert isinstance(meta, GARCHModelResult)
        assert GARCHResult is GARCHModelResult

        # Dataclass attribute dot-access
        assert hasattr(meta, "aic")
        assert hasattr(meta, "bic")
        assert hasattr(meta, "log_likelihood")
        assert hasattr(meta, "omega")
        assert hasattr(meta, "alpha")
        assert hasattr(meta, "beta")
        assert hasattr(meta, "params")
        assert hasattr(meta, "converged")
        assert meta.p == 1
        assert meta.q == 1
        assert isinstance(meta.aic, float)
        assert isinstance(meta.converged, bool)

        # Mapping / dict-access backward compatibility
        assert meta["aic"] == meta.aic
        assert meta["bic"] == meta.bic
        assert meta["log_likelihood"] == meta.log_likelihood
        assert meta["converged"] == meta.converged
        assert "aic" in meta
        assert "nonexistent" not in meta
        assert meta.get("aic") == meta.aic
        assert meta.get("nonexistent", 999) == 999

        # Iteration, dict conversion, and unpacking
        as_dict = meta.to_dict()
        assert isinstance(as_dict, dict)
        assert as_dict["aic"] == meta.aic
        dict_cast = dict(meta)
        assert dict_cast["aic"] == meta.aic
        assert len(meta) > 5
        assert "aic" in list(meta)
        assert "aic" in meta.keys()

        # Unpacking via **meta
        def accept_kwargs(**kwargs):
            return kwargs.get("aic")
        assert accept_kwargs(**meta) == meta.aic

    def test_double_and_triple_exponential_smoothing_result_type(self):
        from people_analytics_toolkit.memory import (
            double_exponential_smoothing,
            triple_exponential_smoothing,
            ExponentialSmoothingResult,
            SmoothingResult,
        )
        t = np.arange(30, dtype=float)
        trend_series = pd.Series(20.0 + 1.5 * t)
        fitted, decomp, meta = double_exponential_smoothing(trend_series, forecast_periods=4)

        # Type checks and aliases
        assert isinstance(meta, ExponentialSmoothingResult)
        assert isinstance(meta, SmoothingResult)
        assert ExponentialSmoothingResult is SmoothingResult

        # Dataclass attribute dot-access
        assert hasattr(meta, "alpha")
        assert hasattr(meta, "beta")
        assert hasattr(meta, "forecast")
        assert hasattr(meta, "model_type")
        assert meta.model_type == "double_exponential"
        assert len(meta.forecast) == 4
        assert meta.gamma is None

        # Mapping / dict-access backward compatibility
        assert meta["alpha"] == meta.alpha
        assert meta["beta"] == meta.beta
        assert (meta["forecast"] == meta.forecast).all()
        assert "forecast" in meta
        assert meta.get("model_type") == "double_exponential"
        assert meta.get("missing_key", "default") == "default"

        # Dict conversion and keys
        as_dict = meta.to_dict()
        assert isinstance(as_dict, dict)
        assert as_dict["model_type"] == "double_exponential"
        assert dict(meta)["alpha"] == meta.alpha

        # Test triple exponential smoothing returns ExponentialSmoothingResult too
        seasonal_series = pd.Series(100.0 + 1.0 * t + 5.0 * np.sin(2 * np.pi * t / 6.0))
        _, _, triple_meta = triple_exponential_smoothing(seasonal_series, seasonal_periods=6, forecast_periods=3)
        assert isinstance(triple_meta, ExponentialSmoothingResult)
        assert triple_meta.model_type == "triple_exponential_additive"
        assert triple_meta.gamma is not None
        assert triple_meta["gamma"] == triple_meta.gamma
        assert len(triple_meta.forecast) == 3

    def test_aipw_estimator_result_type(self):
        from people_analytics_toolkit.prescriptive import aipw_estimator, AIPWResult
        df = generate_causal_and_uplift_data(n_employees=200, seed=42)
        X = df[["tenure_years", "prior_performance", "flight_risk"]]
        t = df["treatment_enrolled"]
        y = df["retained_1yr"]

        res = aipw_estimator(y=y, treatment=t, X=X, n_splits=2, random_state=42)

        # Type checks
        assert isinstance(res, AIPWResult)

        # Dataclass attribute dot-access
        assert hasattr(res, "ate")
        assert hasattr(res, "se")
        assert hasattr(res, "ci_lower")
        assert hasattr(res, "ci_upper")
        assert hasattr(res, "p_value")
        assert hasattr(res, "propensity_scores")
        assert hasattr(res, "mu0_hat")
        assert hasattr(res, "mu1_hat")
        assert hasattr(res, "influence_function")
        assert hasattr(res, "n_samples")
        assert res.n_samples == 200
        assert isinstance(res.ate, float)

        # Mapping / dict-access backward compatibility
        assert res["ate"] == res.ate
        assert res["se"] == res.se
        assert res["p_value"] == res.p_value
        assert "ate" in res
        assert "propensity_scores" in res
        assert "unknown" not in res
        assert res.get("ate") == res.ate
        assert res.get("unknown", -1) == -1

        # Dict conversion, unpacking, and len
        as_dict = res.to_dict()
        assert isinstance(as_dict, dict)
        assert as_dict["ate"] == res.ate
        assert dict(res)["ate"] == res.ate
        assert len(res) == 11

        def inspect_ate(**kwargs):
            return kwargs.get("ate")
        assert inspect_ate(**res) == res.ate

    def test_oaxaca_blinder_decomposition_result_type(self):
        from people_analytics_toolkit.equity import (
            oaxaca_blinder_decomposition,
            OaxacaBlinderResult,
            DecompositionResult,
        )
        df = generate_algorithmic_equity_and_fairness_data(n_employees=200, seed=42)
        features = ["tenure_years", "education_years", "job_level"]

        result = oaxaca_blinder_decomposition(
            df=df,
            outcome_col="base_salary",
            group_col="demographic_group",
            feature_cols=features,
            group_majority=1,
            group_minority=0,
            reference_type="pooled",
        )

        # Type checks and aliases
        assert isinstance(result, OaxacaBlinderResult)
        assert isinstance(result, DecompositionResult)
        assert OaxacaBlinderResult is DecompositionResult

        # Dataclass attribute dot-access
        assert hasattr(result, "raw_gap")
        assert hasattr(result, "explained_effect")
        assert hasattr(result, "unexplained_effect")
        assert hasattr(result, "explained_pct")
        assert hasattr(result, "unexplained_pct")
        assert hasattr(result, "reference_type")
        assert hasattr(result, "feature_breakdown")
        assert hasattr(result, "majority_model")
        assert hasattr(result, "minority_model")
        assert hasattr(result, "pooled_model")
        assert result.reference_type == "pooled"
        assert isinstance(result.raw_gap, float)
        assert isinstance(result.feature_breakdown, pd.DataFrame)
        assert result.majority_model is not None

        # Mapping / dict-access backward compatibility
        assert result["raw_gap"] == result.raw_gap
        assert result["explained_effect"] == result.explained_effect
        assert result["unexplained_effect"] == result.unexplained_effect
        assert result["explained_pct"] == result.explained_pct
        assert result["unexplained_pct"] == result.unexplained_pct
        assert (result["feature_breakdown"] == result.feature_breakdown).all().all()
        assert "raw_gap" in result
        assert "unexplained_effect" in result
        assert result.get("raw_gap") == result.raw_gap
        assert result.get("missing", 42) == 42

        # Dict conversion and iteration
        as_dict = result.to_dict()
        assert isinstance(as_dict, dict)
        assert as_dict["raw_gap"] == result.raw_gap
        assert dict(result)["raw_gap"] == result.raw_gap
        assert "raw_gap" in list(result)
        assert "raw_gap" in result.keys()













