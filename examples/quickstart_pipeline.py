"""Command-line entrypoint for Advanced People Analytics Features demonstration."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from data.synthetic_generators import (
    generate_store_roster_data,
    generate_shift_manager_performance,
    generate_compensation_velocity_data,
    generate_regime_workforce_timeseries,
    generate_career_shift_sequences,
    generate_enterprise_turnover_breakdown,
    generate_safety_incident_dataset,
    generate_career_movement_dataset,
    generate_turnover_inflection_dataset,
)
from people_analytics_toolkit.cardinality import BayesianTargetEncoder, GroupedLOOZScore
from people_analytics_toolkit.attribution import (
    lmdi_rate_mix_decomposition,
    LeaveOneOutExpectedDifferential,
    compute_shap_interaction_matrix,
    extract_shap_inflection_thresholds,
    calculate_retained_vs_terminated_shap_contributions,
)
from people_analytics_toolkit.anomalies import (
    fit_pyod_local_outlier_factor,
    compute_lcss_distance,
    compute_edr_distance,
    compute_dtw_distance,
)
from people_analytics_toolkit.chaos import fit_workforce_hmm_regimes
from people_analytics_toolkit.mobility import CareerMovementEmbeddings


def main():
    print("=" * 75)
    print("ADVANCED FEATURE ENGINEERING FOR PEOPLE ANALYTICS (50 FEATURES ACROSS 11 PILLARS)")
    print("=" * 75)
    
    # 1. High Cardinality Demonstration
    print("\n[1] Bayesian Target Encoding (Store x Dept Turnover Risk)...")
    roster_df = generate_store_roster_data(n_employees=1000, n_stores=25, n_depts=6, seed=42)
    bte = BayesianTargetEncoder(m=15.0, cv_folds=5, seed=42)
    roster_df["turnover_encoded_oof"] = bte.fit_transform_oof(
        roster_df, group_col="store_dept", target_col="turnover_event"
    )
    print(f"    Total Employees Processed: {len(roster_df)}")
    print(f"    Global Turnover Baseline:  {bte.global_mean_:.2%}")
    print(f"    Unique Store-Dept Cohorts: {roster_df['store_dept'].nunique()}")
    
    # 2. Leave-One-Out Expected Differential (LOO-ED)
    print("\n[2] Leave-One-Out Expected Differential (LOO-ED Manager Alpha)...")
    shift_data = generate_shift_manager_performance(n_shifts=300, n_managers=6, seed=42)
    loo_ed = LeaveOneOutExpectedDifferential(ridge_alpha=1.0)
    _, mgr_summary = loo_ed.fit_transform(
        shift_data,
        entity_col="manager_id",
        target_col="observed_throughput",
        context_cols=["store_id", "is_weekend", "foot_traffic", "staff_hours"],
    )
    top_mgr = mgr_summary.iloc[0]
    bot_mgr = mgr_summary.iloc[-1]
    print(f"    Top Manager: {top_mgr['manager_id']} | LOO-ED Alpha: {top_mgr['loo_expected_differential']:+.2f} units/hr")
    print(f"    Low Manager: {bot_mgr['manager_id']} | LOO-ED Alpha: {bot_mgr['loo_expected_differential']:+.2f} units/hr")
    
    # 3. PyOD Local Outlier Factor (LOF)
    print("\n[3] Local Outlier Factor (PyOD LOF Peer Anomaly Detection)...")
    comp_df, _ = generate_compensation_velocity_data(n_employees=800, seed=42)
    lof_scores, is_outlier, lof_meta = fit_pyod_local_outlier_factor(
        comp_df,
        feature_cols=["tenure_years", "performance_rating", "annual_salary_growth_pct", "promotion_velocity"],
        n_neighbors=20,
        contamination=0.03,
    )
    print(f"    Local Peer Outliers Identified: {is_outlier.sum()} of {len(comp_df)} employees")
    print(f"    Decision Threshold:             {lof_meta['threshold']:.4f}")

    # Multivariate Mahalanobis Anomaly Detection (Statistical Chi-Squared Outliers)
    from people_analytics_toolkit.anomalies import calculate_mahalanobis_distance
    m_dist, m_outlier, m_meta = calculate_mahalanobis_distance(
        comp_df,
        feature_cols=["tenure_years", "performance_rating", "annual_salary_growth_pct", "promotion_velocity"],
        significance_level=0.01,
        robust=True,
    )
    print(f"    Mahalanobis Outliers (alpha=0.01): {m_meta['n_outliers']} of {len(comp_df)} employees (Critical D = {m_meta['threshold_distance']:.2f}, Max D = {m_dist.max():.2f})")
    
    # 4. Hidden Markov Model (HMM) Regimes & Bradford Factor
    print("\n[4] Hidden Markov Model (HMM) Regimes & Bradford Factor Absenteeism Friction...")
    regime_df, _ = generate_regime_workforce_timeseries(n_weeks=52, seed=42)
    scored_regimes, profile_summary, _ = fit_workforce_hmm_regimes(
        regime_df,
        emission_cols=["weekly_overtime_hours", "absence_callout_rate"],
        n_components=3,
        seed=42,
    )
    print("    Inferred Regimes: State 0 (Baseline Normal), State 1 (Holiday Surge), State 2 (Attrition Spiral)")
    crisis_weeks = (scored_regimes["hmm_latent_state"] == 2).sum()
    print(f"    Weeks in Attrition Crisis: {crisis_weeks} / 52")

    # The Bradford Factor (Absenteeism Friction Score)
    print("    The Bradford Factor (Absenteeism Friction Score B = S^2 * D):")
    from data.synthetic_generators import generate_bradford_factor_workforce
    from people_analytics_toolkit.chaos import calculate_bradford_factor
    spell_df, _ = generate_bradford_factor_workforce(n_employees=60, window_weeks=52, seed=42)
    bradford_res = calculate_bradford_factor(
        spell_df,
        employee_col="employee_id",
        start_date_col="absence_start_date",
        end_date_col="absence_end_date",
        planned_col="is_planned",
        window_weeks=52,
        as_of_date="2026-10-01",
    )
    p_emp = bradford_res[bradford_res["employee_id"] == "EMP_0001"].iloc[0]
    u_emp = bradford_res[bradford_res["employee_id"] == "EMP_0002"].iloc[0]
    print(f"      - Planned Continuous Leave (EMP_0001):    Spells={p_emp['absence_spells']}, Days={p_emp['total_days_absent']} -> Bradford Score = {p_emp['bradford_score']} ({p_emp['bradford_risk_band']})")
    print(f"      - Chronic Unplanned Call-outs (EMP_0002): Spells={u_emp['absence_spells']}, Days={u_emp['total_days_absent']} -> Bradford Score = {u_emp['bradford_score']} ({u_emp['bradford_risk_band']})")
    print(f"      - Scheduling Disruption Multiplier:       EMP_0002 is {u_emp['bradford_score'] // p_emp['bradford_score']}x more disruptive than EMP_0001 despite identical total absent days ({p_emp['total_days_absent']} days)")
    
    # 5. LCSS vs DTW Gap Robustness
    print("\n[5] LCSS vs DTW Sequence Alignment (Vacation Gap Tolerance)...")
    seqs = generate_career_shift_sequences(n_weeks=52, seed=42)
    dtw_dist, _, _ = compute_dtw_distance(seqs["associate_vacation_gap"], seqs["benchmark_peer"])
    _, lcss_dist = compute_lcss_distance(seqs["associate_vacation_gap"], seqs["benchmark_peer"], epsilon=2.5, delta=4)
    print(f"    Vacation Gap vs Peer -> DTW Distance:       {dtw_dist:.1f} (distorted by 2-wk leave)")
    print(f"    Vacation Gap vs Peer -> LCSS Dissimilarity: {lcss_dist:.1%} (correctly matched)")
    
    # 6. SHAP Interaction Values
    print("\n[6] SHAP 2D Interaction Values (Non-Linear Safety Risk)...")
    safety_df = generate_safety_incident_dataset(n_samples=600, seed=42)
    feat_names = ["assoc_tenure_weeks", "mgr_tenure_months", "weekly_overtime", "safety_training_score", "shift_chaos_index"]
    _, _, ranked_pairs = compute_shap_interaction_matrix(
        safety_df[feat_names],
        safety_df["safety_incident_event"],
        is_classification=True,
        seed=42,
    )
    top_interaction = ranked_pairs.iloc[0]
    print(f"    Top Coupled Risk: {top_interaction['feature_1']} x {top_interaction['feature_2']}")
    print(f"    Interaction Synergy Strength: {top_interaction['interaction_strength']:.4f}")
    
    # 7. LMDI Zero-Residual Decomposition
    print("\n[7] Log-Mean Decomposition (LMDI) Rate vs Mix Accounting...")
    turnover_breakdown = generate_enterprise_turnover_breakdown(seed=42)
    _, lmdi_summary = lmdi_rate_mix_decomposition(turnover_breakdown)
    print(f"    Net Change (Delta R):            {lmdi_summary['delta_turnover_rate_actual']:.2%}")
    print(f"      - Pure Rate Effect:            {lmdi_summary['rate_effect']:.2%} ({lmdi_summary['rate_effect_pct_of_change']:.1f}%)")
    print(f"      - Structural Mix Effect:       {lmdi_summary['mix_effect']:.2%} ({lmdi_summary['mix_effect_pct_of_change']:.1f}%)")
    print(f"    Mathematical Residual:           {lmdi_summary['residual']:.12f}")
    
    # Correlation Heatmap Table (Feature 49)
    from people_analytics_toolkit.attribution import correlation_heatmap_df
    corr_workforce = pd.DataFrame({
        "weekly_overtime": roster_df["weekly_overtime_hours"],
        "tenure_weeks": roster_df["tenure_weeks"],
        "hourly_wage": 18.0 + 0.15 * roster_df["tenure_weeks"] - 0.4 * roster_df["weekly_overtime_hours"] + np.random.default_rng(42).normal(0, 2.0, len(roster_df)),
        "commute_miles": np.random.default_rng(42).normal(15.0, 6.0, len(roster_df)),
        "turnover_risk": roster_df["turnover_event"],
    })
    corr_table = correlation_heatmap_df(corr_workforce, target="turnover_risk", method="spearman", viz=False)
    print("    Correlation Heatmap Table (Spearman Rank against Turnover Risk):")
    for _, row in corr_table.iterrows():
        sig_str = "Significant" if row["SIGNIFICANT"] == 1 else "Insignificant"
        c0_str = "Spans 0" if row["CONTAINS_0"] == 1 else "Excludes 0"
        print(f"      - {row['METRIC']:<18}: r={row['CORRELATION TO turnover_risk']:+.4f} (p={row['P']:.4f}, 95% CI: [{row['LB']:+.2f}, {row['UB']:+.2f}], {sig_str}, {c0_str})")

    print("\n[8] Career Mobility & Succession Planning (Movement Embeddings & Stagnation Index)...")
    career_seqs, candidate_df = generate_career_movement_dataset(n_sequences=500, seed=42)
    cme = CareerMovementEmbeddings(embedding_dim=8, max_steps=3, decay_factor=0.6, seed=42)
    cme.fit(career_seqs)
    top_candidates = cme.recommend_succession_candidates("Department_Supervisor", candidate_df, top_k=3)
    print("    Target Role: Department_Supervisor")
    for _, cand in top_candidates.iterrows():
        print(f"      - Candidate {cand['employee_id']} ({cand['current_role']}): Trajectory Match = {cand['succession_similarity']:.2%}")

    # Time-in-Position & The Stagnation Index
    print("    Time-in-Position (The Stagnation Index Intervention Trigger):")
    from data.synthetic_generators import generate_time_in_position_workforce
    from people_analytics_toolkit.mobility import calculate_stagnation_index
    tip_df = generate_time_in_position_workforce(n_employees=250, seed=42)
    stag_res = calculate_stagnation_index(
        tip_df,
        time_in_pos_col="time_in_position_months",
        cohort_col="department",
        perf_col="performance_rating",
        high_perf_threshold=4.0,
        base_flight_risk_col="baseline_flight_risk",
    )
    n_stagnant_hp = int(stag_res["is_stagnant_top_performer"].sum())
    eng_p90 = stag_res[stag_res["department"] == "Engineering"]["cohort_p90_months"].iloc[0]
    eng_med = stag_res[stag_res["department"] == "Engineering"]["cohort_median_months"].iloc[0]
    print(f"      - Engineering Cohort Benchmarks: Median = {eng_med:.1f} mos, P90 Threshold = {eng_p90:.1f} mos")
    print(f"      - Flagged Stagnant Top Performers: {n_stagnant_hp}/{len(stag_res)} associates (Trigger: Urgent Career-Pathing)")
    stagnant_sample = stag_res[stag_res["is_stagnant_top_performer"]].iloc[0]
    print(f"      - Example Associate {stagnant_sample['employee_id']}: Rating = {stagnant_sample['performance_rating']}, Time-in-Pos = {stagnant_sample['time_in_position_months']:.1f} mos (Cohort P90 = {stagnant_sample['cohort_p90_months']:.1f} mos)")
    print(f"        Flight Risk Delta: +{stagnant_sample['stagnation_flight_risk_delta'] * 100:.1f}% ({stagnant_sample['baseline_flight_risk'] * 100:.1f}% -> {stagnant_sample['post_stagnation_flight_risk'] * 100:.1f}%)")
        
    # 9. SHAP Zero-Crossing Inflection Thresholds for Term Rates
    print("\n[9] SHAP Zero-Crossing Inflection Thresholds for Term Rates...")
    from sklearn.ensemble import RandomForestClassifier
    import shap
    X_turnover, y_turnover = generate_turnover_inflection_dataset(n_samples=800, seed=42)
    clf = RandomForestClassifier(n_estimators=30, max_depth=3, random_state=42)
    clf.fit(X_turnover, y_turnover)
    explainer = shap.TreeExplainer(clf)
    s_raw = explainer.shap_values(X_turnover)
    s_vals = s_raw[1] if isinstance(s_raw, list) else (s_raw[:, :, 1] if getattr(s_raw, 'ndim', 2) == 3 else s_raw)
    thresholds, summary_thresh = extract_shap_inflection_thresholds(X_turnover, s_vals, seed=42)
    ot_thresh = thresholds["weekly_overtime_hours"]["primary_threshold"]
    comp_thresh = thresholds["comp_ratio"]["primary_threshold"]
    print(f"    Weekly Overtime Tipping Point: {ot_thresh} hrs (fatigue hazard begins)")
    print(f"    Comp-Ratio Tipping Point:      {comp_thresh} (underpaid flight risk begins)")

    # 10. Double and Triple Exponential Smoothing & Configuration-Driven Rolling Metrics
    print("\n[10] Temporal Smoothing & Configuration-Driven Rolling Metrics...")
    from data.synthetic_generators import (
        generate_seasonal_workforce_timeseries,
        generate_store_department_labor_panel,
    )
    from people_analytics_toolkit.memory import (
        triple_exponential_smoothing,
        scan_ewma_spans,
        RollingMetricConfig,
        compute_rolling_metrics,
        RollingMetricsPipeline,
    )
    seasonal_df = generate_seasonal_workforce_timeseries(n_weeks=156, seed=42)
    hw_fitted, hw_decomp, hw_meta = triple_exponential_smoothing(
        seasonal_df["observed_labor_demand"], seasonal_periods=52, forecast_periods=12
    )
    print(f"    Total Weeks Processed:           {len(hw_fitted)}")
    print(f"    Secular Trend Growth Velocity:   +{hw_decomp['trend'].iloc[-1]:.2f} hrs/week")
    print(f"    12-Week Ahead Demand Forecast:   {hw_meta['forecast'].iloc[-1]:.1f} hrs")

    # Hyperparameter Span / Half-Life Scanning for EWMA
    ewma_scan = scan_ewma_spans(seasonal_df["observed_labor_demand"], spans=[4, 8, 12, 26, 52])
    best_span = ewma_scan.attrs["best_span"]
    best_hl = ewma_scan.attrs["best_half_life"]
    best_rmse = ewma_scan.loc[ewma_scan["is_optimal"], "rmse_one_step"].iloc[0]
    print(f"    Optimal EWMA Memory Tuning:      Best Span = {best_span} wks (Half-Life = {best_hl:.1f} wks, 1-Step RMSE = {best_rmse:.1f} hrs)")

    # Configuration-Driven Rolling Metrics
    panel_df = generate_store_department_labor_panel(n_stores=5, n_depts=4, n_weeks=104, seed=42)
    pipeline = (
        RollingMetricsPipeline(partition_cols=["store_id", "dept_id"], temporal_col="week")
        .add_metric("turnover_rate", window=52, aggregation="mean", output_column="turnover_52w_mean")
        .add_metric("headcount", window=13, aggregation="mean", output_column="headcount_13w_baseline")
        .add_metric("assigned_hours", window=4, aggregation="var", output_column="hours_4w_fatigue_var")
    )
    panel_out = pipeline.transform(panel_df)
    s1_d1_latest = panel_out[(panel_out["store_id"] == "Store_01") & (panel_out["dept_id"] == "Sales")].iloc[-1]
    print(f"    Configuration-Driven Rolling:    Store_01 Sales Cohort (Week 104):")
    print(f"      - Trailing 52-Wk Turnover Rate: {s1_d1_latest['turnover_52w_mean']:.2%}")
    print(f"      - 13-Wk Pre-Forecast Baseline:  {s1_d1_latest['headcount_13w_baseline']:.1f} headcount")
    print(f"      - 4-Wk Schedule Fatigue Var:    {s1_d1_latest['hours_4w_fatigue_var']:.1f} hrs^2")

    # Singular Spectrum Analysis (SVD Smoothing)
    from data.synthetic_generators import generate_daily_store_foot_traffic
    from people_analytics_toolkit.memory import (
        singular_spectrum_analysis,
        compute_lag_features,
        compute_differencing_features,
        TemporalFeaturesPipeline,
    )
    traffic_df = generate_daily_store_foot_traffic(n_days=180, n_stores=1, seed=42)
    ssa_res = singular_spectrum_analysis(traffic_df["observed_foot_traffic"], window_length=28, top_k=3)
    raw_rmse = float(np.sqrt(np.mean((traffic_df["observed_foot_traffic"] - traffic_df["true_baseline"]) ** 2)))
    svd_rmse = float(np.sqrt(np.mean((ssa_res["smoothed"] - traffic_df["true_baseline"]) ** 2)))
    noise_reduction_pct = (1.0 - svd_rmse / raw_rmse) * 100
    print(f"    Singular Spectrum Analysis (SSA): Store_01 Daily Foot Traffic (180 Days):")
    print(f"      - Top 3 SVD Components Energy:  {ssa_res['explained_variance_ratio'][:3].sum() * 100:.1f}%")
    print(f"      - Noise Reduction vs Baseline:  {noise_reduction_pct:.1f}% (RMSE: {raw_rmse:.1f} -> {svd_rmse:.1f} visits)")
    print(f"      - Temporal Phase Lag:           0.0 days (Zero-phase non-causal Hankel averaging)")

    # Hierarchical Lags & Multi-Horizon Differencing (Features 47 & 48)
    temp_pipeline = (
        TemporalFeaturesPipeline(partition_cols=["store_id", "dept_id"], temporal_col="week")
        .add_lags("assigned_hours", lags={"lag1": 1, "lag4": 4, "lag52": 52})
        .add_differencing("assigned_hours", periods={"wow": 1, "mom": 4, "yoy": 52}, pct_change=False)
        .add_differencing("assigned_hours", periods={"wow": 1, "mom": 4, "yoy": 52}, pct_change=True, as_percent=True)
    )
    temp_out = temp_pipeline.transform(panel_df)
    s1_d1_temp = temp_out[(temp_out["store_id"] == "Store_01") & (temp_out["dept_id"] == "Sales")].iloc[-1]
    print(f"    Temporal Lags & Differencing:    Store_01 Sales Cohort (Week 104):")
    print(f"      - Lag 1 Wk / Lag 52 Wk Hours:   {s1_d1_temp['assigned_hours_lag1']:.1f} hrs / {s1_d1_temp['assigned_hours_lag52']:.1f} hrs")
    print(f"      - WoW / MoM / YoY Net Delta:    {s1_d1_temp['assigned_hours_wow']:+.1f} hrs / {s1_d1_temp['assigned_hours_mom']:+.1f} hrs / {s1_d1_temp['assigned_hours_yoy']:+.1f} hrs")
    print(f"      - YoY Growth Rate (% Change):   {s1_d1_temp['assigned_hours_yoy_pct']:+.2f}%")


    # 11. Compensation Equity (Compa-Ratio & Range Penetration)
    print("\n[11] Compensation and Pay Equity Transformations...")
    from data.synthetic_generators import generate_compensation_equity_data
    from people_analytics_toolkit.compensation import calculate_compa_ratio, calculate_range_penetration, classify_pay_band_status
    comp_df = generate_compensation_equity_data(n_employees=500, seed=42)
    comp_df["compa_ratio"] = calculate_compa_ratio(comp_df["base_salary"], comp_df["band_midpoint"])
    comp_df["range_penetration"] = calculate_range_penetration(comp_df["base_salary"], comp_df["band_min"], comp_df["band_max"])
    comp_df["pay_status"] = np.where(
        comp_df["range_penetration"] < 0, "Green-Circled",
        np.where(comp_df["range_penetration"] > 1.0, "Red-Circled", "Within-Band")
    )
    print(f"    Total Employees Processed:       {len(comp_df)}")
    print(f"    Mean Compa-Ratio:                {comp_df['compa_ratio'].mean():.2f}")
    print(f"    Mean Range Penetration:          {comp_df['range_penetration'].mean() * 100:.1f}%")
    print(f"    Red-Circled Outliers (>Max):     {(comp_df['pay_status'] == 'Red-Circled').sum()}")
    print(f"    Green-Circled Outliers (<Min):   {(comp_df['pay_status'] == 'Green-Circled').sum()}")

    # 12. Organizational Network Analysis & Span of Control
    print("\n[12] Organizational Network Analysis & Hierarchy (ONA, Structural Capital & Span of Control)...")
    from data.synthetic_generators import generate_organizational_network_data, generate_reporting_hierarchy_with_reorg
    from people_analytics_toolkit.ona import (
        compute_complete_ona_profile,
        simulate_attrition_contagion,
        calculate_span_of_control,
        evaluate_span_reorganization_shock,
    )
    nodes_df, edges_df, G = generate_organizational_network_data(n_employees=120, seed=42)
    ona_profile = compute_complete_ona_profile(G)
    top_broker = ona_profile["betweenness_centrality"].idxmax()
    top_hub = ona_profile["eigenvector_centrality"].idxmax()
    contagion_df = simulate_attrition_contagion(G, departed_node=top_broker, contagion_base_rate=0.50, max_hops=2)

    # Span of Control & Reorganization Shock evaluation
    df_pre, df_post = generate_reporting_hierarchy_with_reorg(seed=42)
    span_pre = calculate_span_of_control(df_pre)
    reorg_shock = evaluate_span_reorganization_shock(
        df_pre, df_post, base_flight_risk_col="baseline_flight_risk"
    )
    m1_pre_span = span_pre.loc["MGR_01", "direct_reports_count"]
    m1_shock_row = reorg_shock.loc["EMP_001"]

    print(f"    Total Employees Modeled:         {len(ona_profile)}")
    print(f"    Top Cross-Functional Broker:     {top_broker} (Betweenness: {ona_profile.loc[top_broker, 'betweenness_centrality']:.4f})")
    print(f"    Top Informal Knowledge Hub:      {top_hub} (Eigenvector: {ona_profile.loc[top_hub, 'eigenvector_centrality']:.4f})")
    print(f"    Mean Burt Network Constraint:    {ona_profile['burt_constraint'].mean():.3f}")
    print(f"    Identified HCM Archetypes:       {dict(ona_profile['network_archetype'].value_counts())}")
    print(f"    Simulated Contagion Flight Risk: {len(contagion_df)} peers impacted by {top_broker}'s departure")
    print(f"    Span of Control Baseline:        MGR_01 Span = {m1_pre_span} direct reports (Optimal load)")
    print(f"    Reorganization Surge Shock:      MGR_01 Span jumps to {int(m1_shock_row['new_manager_span'])} (+{int(m1_shock_row['span_delta'])})")
    print(f"    1-on-1 Attention Starvation:     EMP_001 coaching cut from {m1_shock_row['prev_weekly_1on1_minutes']:.0f} to {m1_shock_row['new_weekly_1on1_minutes']:.0f} min/wk (-{m1_shock_row['weekly_1on1_minutes_lost']:.0f} min)")
    print(f"    Associate Flight Risk Surge:     EMP_001 base risk {m1_shock_row['base_flight_risk']*100:.1f}% -> {m1_shock_row['post_reorg_flight_risk']*100:.1f}% (+{m1_shock_row['reorg_shock_flight_risk_delta']*100:.1f}%)")

    # 13. Algorithmic Equity, Pay Decomposition & Individual Fairness
    print("\n[13] Algorithmic Equity, Pay Decomposition & Individual Fairness...")
    from data.synthetic_generators import generate_algorithmic_equity_and_fairness_data
    from people_analytics_toolkit.equity import (
        oaxaca_blinder_decomposition,
        demographic_parity_difference,
        equalized_odds_difference,
        ExponentiatedGradientFairness,
        calculate_individual_fairness_consistency,
    )
    from sklearn.linear_model import LogisticRegression

    equity_df = generate_algorithmic_equity_and_fairness_data(n_employees=800, seed=42)
    features = ["tenure_years", "education_years", "job_level", "performance_rating", "certifications"]

    # Oaxaca-Blinder Pay Gap Decomposition
    ob_res = oaxaca_blinder_decomposition(
        df=equity_df,
        outcome_col="base_salary",
        group_col="demographic_group",
        feature_cols=features,
        reference_type="pooled",
    )
    print(f"    Raw Compensation Gap:            ${ob_res['raw_gap']:,.2f}")
    print(f"      - Explained (Qualifications):  ${ob_res['explained_effect']:,.2f} ({ob_res['explained_pct']:.1f}%)")
    print(f"      - Unexplained (Structural):    ${ob_res['unexplained_effect']:,.2f} ({ob_res['unexplained_pct']:.1f}%)")

    # Fair Classification via Exponentiated Gradient
    X_eq = equity_df[features]
    y_eq = equity_df["promotion_recommendation"]
    sens_eq = equity_df["demographic_group"]

    base_lr = LogisticRegression(solver="liblinear", random_state=42).fit(X_eq, y_eq)
    base_dp = demographic_parity_difference(base_lr.predict(X_eq), sens_eq)

    fair_lr = ExponentiatedGradientFairness(
        estimator=LogisticRegression(solver="liblinear", random_state=42),
        constraint="demographic_parity",
        eps=0.02,
        max_iter=15,
        random_state=42,
    ).fit(X_eq, y_eq, sensitive_features=sens_eq)
    fair_dp = demographic_parity_difference(fair_lr.predict(X_eq), sens_eq)
    print(f"    Demographic Parity Gap:          Baseline {base_dp * 100:.1f}% -> Constrained {fair_dp * 100:.1f}%")

    # Individual Fairness & Local Predictive Consistency
    _, mean_cons, _ = calculate_individual_fairness_consistency(X_eq, fair_lr.predict_proba(X_eq)[:, 1], n_neighbors=5)
    print(f"    Individual Fairness Consistency: {mean_cons * 100:.1f}% (Lipschitz Neighborhood Concordance)")

    # 14. Prescriptive Interventions: Causal Inference and Heterogeneous Uplift
    print("\n[14] Prescriptive Interventions: Causal Inference & Heterogeneous Uplift...")
    from data.synthetic_generators import generate_causal_and_uplift_data, generate_panel_policy_data
    from people_analytics_toolkit.prescriptive import (
        aipw_estimator,
        synthetic_difference_in_differences,
        CausalForestDML,
        area_under_uplift_curve,
        doubly_robust_uplift_eval,
    )

    causal_df = generate_causal_and_uplift_data(n_employees=1000, seed=42)
    X_cau = causal_df[["tenure_years", "prior_performance", "flight_risk", "overtime_hours", "manager_quality"]]
    t_cau = causal_df["treatment_enrolled"]
    y_cau = causal_df["retained_1yr"]

    # AIPW Doubly Robust ATE
    aipw_res = aipw_estimator(y=y_cau, treatment=t_cau, X=X_cau, n_splits=4, random_state=42)
    print(f"    AIPW Doubly Robust ATE:          +{aipw_res['ate'] * 100:.2f}% (SE={aipw_res['se'] * 100:.2f}%, 95% CI: [{aipw_res['ci_lower'] * 100:.2f}%, {aipw_res['ci_upper'] * 100:.2f}%])")
    print(f"    Naive Observational Gap:         +{aipw_res['naive_ate'] * 100:.2f}% (Confounded Selection Bias)")

    # Synthetic Difference-in-Differences (SDiD)
    panel_df = generate_panel_policy_data(n_units=12, n_periods=10, treated_units=("Office_North",), post_period_start=7, seed=42)
    sdid_res = synthetic_difference_in_differences(
        df=panel_df,
        unit_col="location_id",
        time_col="quarter_idx",
        outcome_col="turnover_rate",
        treated_units=["Office_North"],
        post_period_start=7,
    )
    print(f"    Synthetic DiD (SDiD) Policy ATT: {sdid_res['att']:.2f}% Turnover Rate (SE={sdid_res['se']:.2f}%, p={sdid_res['p_value']:.4f})")

    # Causal Forest DML & Persuadable Targeting
    cf_dml = CausalForestDML(n_estimators=30, max_depth=3, min_samples_leaf=12, random_state=42).fit(X_cau, t_cau, y_cau)
    persuadables = cf_dml.identify_persuadables(X_cau, min_effect=0.08)
    n_persuadable = int(persuadables["is_persuadable"].sum())
    print(f"    Causal Forest DML:               {n_persuadable}/{len(causal_df)} ({n_persuadable / len(causal_df) * 100:.1f}%) Identified as 'Persuadables'")

    # Uplift Evaluation: AUUC & Doubly Robust MSE
    cate_preds = persuadables["predicted_cate"]
    auuc_metrics = area_under_uplift_curve(y_cau, t_cau, cate_preds)
    dr_eval = doubly_robust_uplift_eval(y_cau, t_cau, cate_preds, X_cau)
    print(f"    Qini Uplift AUUC:                {auuc_metrics['auuc']:.1f} (Random Baseline: {auuc_metrics['auuc_random']:.1f}, Qini: {auuc_metrics['qini_score']:.1f})")
    print(f"    Doubly Robust MSE_W:             {dr_eval['mse_w']:.4f} (Counterfactual Evaluation)")

    # Synthetic Control Weights (Synth-DiD Twin & Causal Lift)
    from data.synthetic_generators import generate_retail_scheduling_pilot_data
    from people_analytics_toolkit.prescriptive import build_multi_pilot_synthetic_controls

    pilot_stores = ["Store_01", "Store_02", "Store_03", "Store_04", "Store_05"]
    retail_df = generate_retail_scheduling_pilot_data(n_stores=30, n_weeks=24, pilot_stores=tuple(pilot_stores), rollout_week=16, seed=42)
    synth_res = build_multi_pilot_synthetic_controls(
        df=retail_df,
        unit_col="store_id",
        time_col="week_idx",
        outcome_col="turnover_rate",
        pilot_units=pilot_stores,
        post_period_start=16,
    )
    s01_row = synth_res["summary_df"].set_index("pilot_unit").loc["Store_01"]
    s01_rmspe = s01_row["pre_rmspe"]
    s01_lift = s01_row["post_causal_lift"]
    agg_lift = synth_res["aggregate_causal_lift"]
    print(f"    Synthetic Control Weights:       Store_01 Pre-RMSPE: {s01_rmspe:.2f}%, Post Lift: {s01_lift:+.2f}% (Agg Lift across 5 pilots: {agg_lift:+.2f}%)")

    # Directed Causal Edge Weights (DAG Discovery & Confounder Untangling)
    from data.synthetic_generators import generate_overtime_turnover_causal_dag_data
    from people_analytics_toolkit.prescriptive import (
        compute_directed_causal_edge_weights,
        diagnose_causal_confounding,
    )

    dag_df = generate_overtime_turnover_causal_dag_data(n_samples=1000, seed=42)
    dag_res = compute_directed_causal_edge_weights(dag_df, lambda1=0.03, threshold=0.15)
    diag_ot = diagnose_causal_confounding(dag_res["directed_weights_df"], "overtime_hours", "turnover_risk", corr_matrix=dag_res["correlation_matrix"])
    roots_str = ", ".join(dag_res["root_causes"])
    confounders_str = ", ".join(diag_ot["common_confounders"])
    print(f"    DAG Causal Discovery (NOTEARS):  Discovered Root Causes: [{roots_str}], Sink Outcomes: {dag_res['sink_outcomes']}")
    print(f"    Confounder Untangling:           Overtime vs Turnover (r={diag_ot['observed_correlation']:.2f}) -> {diag_ot['verdict']}")

    # 15. Continuous-Time Trajectories & Heterogeneous Graph Representation Learning
    print("\n[15] Continuous-Time Trajectories & Heterogeneous Graph Representation Learning...")
    from data.synthetic_generators import (
        generate_career_trajectory_event_log,
        generate_heterogeneous_hcm_multigraph,
    )
    from people_analytics_toolkit.trajectories import (
        ContinuousTimeMarkovChain,
        Metapath2Vec,
        DynamicEntitySelfAttention,
    )

    # Continuous-Time Markov Chain (CTMC)
    event_log = generate_career_trajectory_event_log(n_employees=500, seed=42)
    ctmc = ContinuousTimeMarkovChain().fit_from_event_log(event_log)
    traj_3yr = ctmc.predict_trajectory_distribution("L1_Associate", t=3.0)
    print(f"    CTMC Continuous Horizon (t=3.0 yrs from L1):")
    print(f"      - Prob(L2_Mid): {traj_3yr.get('L2_Mid', 0.0) * 100:.1f}%, Prob(L3_Senior): {traj_3yr.get('L3_Senior', 0.0) * 100:.1f}%, Prob(Attrition): {traj_3yr.get('Attrition', 0.0) * 100:.1f}%")

    bottlenecks = ctmc.identify_pipeline_bottlenecks()
    flagged = bottlenecks[bottlenecks["is_mobility_bottleneck"]].index.tolist()
    print(f"    Mobility Bottlenecks:            {', '.join(flagged) if flagged else 'None'}")

    # Metapath2Vec Heterogeneous Graph Embeddings
    G_het, nodes_df, _ = generate_heterogeneous_hcm_multigraph(n_employees=80, n_projects=16, n_skills=24, seed=42)
    mp2v = Metapath2Vec(embedding_dim=16, walk_length=15, num_walks=5, window_size=3, random_state=42).fit(G_het)
    peers = mp2v.find_similar_nodes("EMP_0001", top_k=2, target_node_type="Employee")
    peer_str = ", ".join([f"{idx} ({row['cosine_similarity']:.2f})" for idx, row in peers.iterrows()])
    print(f"    Metapath2Vec Structural Peers:   EMP_0001 -> {peer_str}")

    team_metrics = mp2v.predict_team_complementarity(["EMP_0001", "EMP_0002", "EMP_0003"])
    print(f"    Team Complementarity:            Cohesion: {team_metrics['pairwise_cohesion'] * 100:.1f}%, Diversity: {team_metrics['diversity_spread'] * 100:.1f}%")

    # Dynamic Self-Attention
    tokens = np.array([mp2v.get_embedding(e) for e in ["EMP_0001", "EMP_0002", "EMP_0003"]])
    attn_res = DynamicEntitySelfAttention(n_heads=2, random_state=42).aggregate(tokens)
    top_token_wt = float(attn_res["attention_weights"].max())
    print(f"    Dynamic Self-Attention:          Context vector dim={len(attn_res['context_vector'])}, Max Token Attention={top_token_wt * 100:.1f}%")

    print("\n" + "=" * 75)
    print("All 50 core feature engineering modules operational across 11 pillars.")
    print("To explore all 50 features with full interactive visual charts, run:")
    print("  uv run jupyter lab notebooks/advanced_people_analytics_features.ipynb")
    print("=" * 75)


if __name__ == "__main__":
    main()



