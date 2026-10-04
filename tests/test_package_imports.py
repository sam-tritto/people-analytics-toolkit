"""Tests verifying people_analytics_toolkit top-level package and module imports."""

import people_analytics_toolkit as patk


def test_package_version_and_metadata():
    assert patk.__version__ == "0.1.0"
    assert hasattr(patk, "BayesianTargetEncoder")
    assert hasattr(patk, "GroupedLOOZScore")
    assert hasattr(patk, "fractional_difference")
    assert hasattr(patk, "calculate_ewma")
    assert hasattr(patk, "scan_ewma_spans")
    assert hasattr(patk, "calculate_ewms")
    assert hasattr(patk, "calculate_poisson_ewma")
    assert hasattr(patk, "double_exponential_smoothing")
    assert hasattr(patk, "triple_exponential_smoothing")
    assert hasattr(patk, "calculate_salary_band_midpoint")
    assert hasattr(patk, "calculate_compa_ratio")
    assert hasattr(patk, "calculate_range_penetration")
    assert hasattr(patk, "classify_pay_band_status")
    assert hasattr(patk, "is_red_circled")
    assert hasattr(patk, "is_green_circled")
    assert hasattr(patk, "calculate_pay_equity_gap")
    assert hasattr(patk, "median_gap")
    assert hasattr(patk, "calculate_median_pay_gap")
    assert hasattr(patk, "calculate_degree_centrality")
    assert hasattr(patk, "calculate_closeness_centrality")
    assert hasattr(patk, "calculate_betweenness_centrality")
    assert hasattr(patk, "calculate_eigenvector_centrality")
    assert hasattr(patk, "calculate_burt_constraint")
    assert hasattr(patk, "calculate_euler_centrality")
    assert hasattr(patk, "calculate_collaboration_overload")
    assert hasattr(patk, "simulate_attrition_contagion")
    assert hasattr(patk, "compute_complete_ona_profile")
    assert hasattr(patk, "calculate_shannon_entropy")
    assert hasattr(patk, "rolling_shannon_entropy")
    assert hasattr(patk, "fit_garch_volatility")
    assert hasattr(patk, "rolling_garch_volatility")
    assert hasattr(patk, "calculate_mahalanobis_distance")
    assert hasattr(patk, "HawkesProcessContagion")
    assert hasattr(patk, "lmdi_rate_mix_decomposition")
    assert hasattr(patk, "correlation_heatmap_df")
    assert hasattr(patk, "CareerMovementEmbeddings")
    assert hasattr(patk, "oaxaca_blinder_decomposition")
    assert hasattr(patk, "demographic_parity_difference")
    assert hasattr(patk, "demographic_parity_ratio")
    assert hasattr(patk, "equalized_odds_difference")
    assert hasattr(patk, "ExponentiatedGradientFairness")
    assert hasattr(patk, "calculate_individual_fairness_consistency")
    assert hasattr(patk, "aipw_estimator")
    assert hasattr(patk, "synthetic_difference_in_differences")
    assert hasattr(patk, "HonestCausalTree")
    assert hasattr(patk, "CausalForestDML")
    assert hasattr(patk, "qini_curve")
    assert hasattr(patk, "area_under_uplift_curve")
    assert hasattr(patk, "doubly_robust_uplift_eval")
    assert hasattr(patk, "ContinuousTimeMarkovChain")
    assert hasattr(patk, "generate_metapath_walks")
    assert hasattr(patk, "Metapath2Vec")
    assert hasattr(patk, "DynamicEntitySelfAttention")
    assert hasattr(patk, "calculate_span_of_control")
    assert hasattr(patk, "evaluate_span_reorganization_shock")
    assert hasattr(patk, "calculate_time_in_position")
    assert hasattr(patk, "calculate_stagnation_index")
    assert hasattr(patk, "compute_role_similarity_matrix")
    assert hasattr(patk, "get_role_similarity_matrix")
    assert hasattr(patk, "calculate_role_similarity_matrix")
    assert hasattr(patk, "calculate_bradford_score")
    assert hasattr(patk, "classify_bradford_risk")
    assert hasattr(patk, "calculate_bradford_factor")
    assert hasattr(patk, "compute_synthetic_control_weights")
    assert hasattr(patk, "build_synthetic_control_twin")
    assert hasattr(patk, "build_multi_pilot_synthetic_controls")
    assert hasattr(patk, "notears_linear")
    assert hasattr(patk, "compute_directed_causal_edge_weights")
    assert hasattr(patk, "diagnose_causal_confounding")
    assert hasattr(patk, "RollingMetricConfig")
    assert hasattr(patk, "compute_rolling_metrics")
    assert hasattr(patk, "RollingMetricsPipeline")
    assert hasattr(patk, "singular_spectrum_analysis")
    assert hasattr(patk, "compute_svd_smoothed_baseline")
    assert hasattr(patk, "SingularSpectrumAnalysis")
    assert hasattr(patk, "compute_lag_features")
    assert hasattr(patk, "compute_differencing_features")
    assert hasattr(patk, "TemporalFeaturesPipeline")
    # Estimator classes
    assert hasattr(patk, "LocalOutlierFactorDetector")
    assert hasattr(patk, "PyODLocalOutlierFactor")
    assert hasattr(patk, "GARCHVolatilityModel")
    assert hasattr(patk, "GARCHVolatility")
    assert hasattr(patk, "WorkforceHMMRegimes")
    assert hasattr(patk, "WorkforceHMM")
    assert hasattr(patk, "HazardEmbeddings")
    assert hasattr(patk, "HazardEmbeddingTransformer")
    # Result dataclass types
    assert hasattr(patk, "GARCHResult")
    assert hasattr(patk, "GARCHModelResult")
    assert hasattr(patk, "ExponentialSmoothingResult")
    assert hasattr(patk, "SmoothingResult")
    assert hasattr(patk, "AIPWResult")
    assert hasattr(patk, "OaxacaBlinderResult")
    assert hasattr(patk, "DecompositionResult")


def test_submodule_imports():
    from people_analytics_toolkit.cardinality import BayesianTargetEncoder
    from people_analytics_toolkit.memory import (
        fractional_difference,
        scan_ewma_spans,
        calculate_ewms,
        calculate_poisson_ewma,
        double_exponential_smoothing,
        triple_exponential_smoothing,
        ExponentialSmoothingResult,
        SmoothingResult,
        HazardEmbeddings,
        HazardEmbeddingTransformer,
        fit_hazard_embeddings,
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
    from people_analytics_toolkit.chaos import (
        calculate_shannon_entropy,
        rolling_shannon_entropy,
        GARCHVolatilityModel,
        GARCHVolatility,
        fit_garch_volatility,
        rolling_garch_volatility,
        GARCHResult,
        GARCHModelResult,
        WorkforceHMMRegimes,
        WorkforceHMM,
        fit_workforce_hmm_regimes,
        calculate_bradford_score,
        classify_bradford_risk,
        calculate_bradford_factor,
    )
    from people_analytics_toolkit.anomalies import (
        calculate_kl_divergence,
        calculate_mahalanobis_distance,
        LocalOutlierFactorDetector,
        PyODLocalOutlierFactor,
        fit_pyod_local_outlier_factor,
    )
    from people_analytics_toolkit.attribution import logarithmic_mean, correlation_heatmap_df
    from people_analytics_toolkit.mobility import (
        CareerMovementEmbeddings,
        compute_role_similarity_matrix,
        get_role_similarity_matrix,
        calculate_role_similarity_matrix,
        calculate_time_in_position,
        calculate_stagnation_index,
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
        OaxacaBlinderResult,
        DecompositionResult,
        demographic_parity_difference,
        demographic_parity_ratio,
        equalized_odds_difference,
        ExponentiatedGradientFairness,
        calculate_individual_fairness_consistency,
    )
    from people_analytics_toolkit.prescriptive import (
        aipw_estimator,
        AIPWResult,
        synthetic_difference_in_differences,
        HonestCausalTree,
        CausalForestDML,
        qini_curve,
        area_under_uplift_curve,
        doubly_robust_uplift_eval,
        compute_synthetic_control_weights,
        build_synthetic_control_twin,
        build_multi_pilot_synthetic_controls,
    )
    from people_analytics_toolkit.trajectories import (
        ContinuousTimeMarkovChain,
        generate_metapath_walks,
        Metapath2Vec,
        DynamicEntitySelfAttention,
    )

    assert ExponentialSmoothingResult is not None
    assert SmoothingResult is not None
    assert GARCHResult is not None
    assert GARCHModelResult is not None
    assert OaxacaBlinderResult is not None
    assert DecompositionResult is not None
    assert AIPWResult is not None

    assert BayesianTargetEncoder is not None
    assert fractional_difference is not None
    assert scan_ewma_spans is not None
    assert calculate_ewms is not None
    assert calculate_poisson_ewma is not None
    assert double_exponential_smoothing is not None
    assert triple_exponential_smoothing is not None
    assert calculate_salary_band_midpoint is not None
    assert calculate_compa_ratio is not None
    assert calculate_range_penetration is not None
    assert classify_pay_band_status is not None
    assert calculate_pay_equity_gap is not None
    assert median_gap is not None
    assert calculate_median_pay_gap is not None
    assert calculate_degree_centrality is not None
    assert calculate_closeness_centrality is not None
    assert calculate_betweenness_centrality is not None
    assert calculate_eigenvector_centrality is not None
    assert calculate_burt_constraint is not None
    assert calculate_euler_centrality is not None
    assert calculate_collaboration_overload is not None
    assert simulate_attrition_contagion is not None
    assert compute_complete_ona_profile is not None
    assert calculate_span_of_control is not None
    assert evaluate_span_reorganization_shock is not None
    assert oaxaca_blinder_decomposition is not None
    assert demographic_parity_difference is not None
    assert demographic_parity_ratio is not None
    assert equalized_odds_difference is not None
    assert ExponentiatedGradientFairness is not None
    assert calculate_individual_fairness_consistency is not None
    assert aipw_estimator is not None
    assert synthetic_difference_in_differences is not None
    assert HonestCausalTree is not None
    assert CausalForestDML is not None
    assert qini_curve is not None
    assert area_under_uplift_curve is not None
    assert doubly_robust_uplift_eval is not None
    assert ContinuousTimeMarkovChain is not None
    assert generate_metapath_walks is not None
    assert Metapath2Vec is not None
    assert DynamicEntitySelfAttention is not None
    assert calculate_shannon_entropy is not None
    assert rolling_shannon_entropy is not None
    assert fit_garch_volatility is not None
    assert rolling_garch_volatility is not None
    assert calculate_kl_divergence is not None
    assert calculate_mahalanobis_distance is not None
    assert logarithmic_mean is not None
    assert CareerMovementEmbeddings is not None
    assert calculate_time_in_position is not None
    assert calculate_stagnation_index is not None

    from people_analytics_toolkit.prescriptive import (
        notears_linear,
        compute_directed_causal_edge_weights,
        diagnose_causal_confounding,
    )
    assert notears_linear is not None
    assert compute_directed_causal_edge_weights is not None
    assert diagnose_causal_confounding is not None


def test_compute_calculate_prefix_consistency():
    """Verify bidirectional consistency between calculate_* and compute_* APIs."""
    pairs = [
        ("calculate_ewma", "compute_ewma"),
        ("calculate_ewms", "compute_ewms"),
        ("calculate_poisson_ewma", "compute_poisson_ewma"),
        ("calculate_rolling_metrics", "compute_rolling_metrics"),
        ("calculate_svd_smoothed_baseline", "compute_svd_smoothed_baseline"),
        ("calculate_lag_features", "compute_lag_features"),
        ("calculate_differencing_features", "compute_differencing_features"),
        ("calculate_shannon_entropy", "compute_shannon_entropy"),
        ("calculate_bradford_score", "compute_bradford_score"),
        ("calculate_bradford_factor", "compute_bradford_factor"),
        ("calculate_kl_divergence", "compute_kl_divergence"),
        ("calculate_jensen_shannon_divergence", "compute_jensen_shannon_divergence"),
        ("calculate_dtw_distance", "compute_dtw_distance"),
        ("calculate_lcss_distance", "compute_lcss_distance"),
        ("calculate_edr_distance", "compute_edr_distance"),
        ("calculate_mahalanobis_distance", "compute_mahalanobis_distance"),
        ("calculate_shap_interaction_matrix", "compute_shap_interaction_matrix"),
        ("calculate_retained_vs_terminated_shap_contributions", "compute_retained_vs_terminated_shap_contributions"),
        ("calculate_time_in_position", "compute_time_in_position"),
        ("calculate_stagnation_index", "compute_stagnation_index"),
        ("calculate_salary_band_midpoint", "compute_salary_band_midpoint"),
        ("calculate_compa_ratio", "compute_compa_ratio"),
        ("calculate_range_penetration", "compute_range_penetration"),
        ("calculate_pay_equity_gap", "compute_pay_equity_gap"),
        ("calculate_median_pay_gap", "compute_median_pay_gap"),
        ("calculate_degree_centrality", "compute_degree_centrality"),
        ("calculate_closeness_centrality", "compute_closeness_centrality"),
        ("calculate_betweenness_centrality", "compute_betweenness_centrality"),
        ("calculate_eigenvector_centrality", "compute_eigenvector_centrality"),
        ("calculate_burt_constraint", "compute_burt_constraint"),
        ("calculate_euler_centrality", "compute_euler_centrality"),
        ("calculate_collaboration_overload", "compute_collaboration_overload"),
        ("calculate_complete_ona_profile", "compute_complete_ona_profile"),
        ("calculate_span_of_control", "compute_span_of_control"),
        ("calculate_individual_fairness_consistency", "compute_individual_fairness_consistency"),
        ("calculate_synthetic_control_weights", "compute_synthetic_control_weights"),
        ("calculate_directed_causal_edge_weights", "compute_directed_causal_edge_weights"),
    ]
    for calc_name, comp_name in pairs:
        assert hasattr(patk, calc_name), f"patk missing {calc_name}"
        assert hasattr(patk, comp_name), f"patk missing {comp_name}"
        calc_fn = getattr(patk, calc_name)
        comp_fn = getattr(patk, comp_name)
        unwrapped_calc = getattr(calc_fn, "__wrapped__", calc_fn)
        unwrapped_comp = getattr(comp_fn, "__wrapped__", comp_fn)
        assert unwrapped_calc is unwrapped_comp, f"{calc_name} and {comp_name} do not point to the same callable"




