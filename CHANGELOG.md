# Changelog

All notable changes to the `people-analytics-toolkit` package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- Extended support for Python 3.10 environments with `typing_extensions` fallback.
- Shared session-scoped pytest fixtures in `tests/conftest.py`.
- Parametrized tests validating polymorphic inputs across `pd.Series`, `np.ndarray`, `list`, and scalars.
- Hypothesis property-based testing suites for compensation parity, shift invariance, and Bradford factor mathematical invariants.
- Automated CI/CD matrix testing pipeline across Python 3.10, 3.11, and 3.12.

---

## [0.1.0] - 2026-10-03

### Added
- Initial open-source release of **`people-analytics-toolkit`**.
- **50 production-grade feature engineering methodologies** across 11 core analytical pillars:
  1. **High Cardinality & Hierarchical Shrinkage**: `BayesianTargetEncoder` (empirical Bayes m-estimate smoothing, OOF cross-validation), `GroupedLOOZScore` (leave-one-out cohort normalization).
  2. **Non-linear Memory & Fractional Integration**: `fractional_difference` (Hosking-Sowell weights preserving fractional memory while enforcing stationarity), `calculate_ewma`, `calculate_ewms` (discrete count leaky integrator), `double_exponential_smoothing` (Holt's linear trend decomposition), `SingularSpectrumAnalysis` (Hankel trajectory embedding).
  3. **Point Processes, Continuous-Time Trajectories & Systemic Friction**: `calculate_bradford_score` ($B = S^2 \times D$), `classify_bradford_risk`, `HawkesProcessIntensity` (self-exciting contagion kernel), `ContinuousTimeMarkovChain` (generator matrix $Q$ and sojourn bottleneck analysis).
  4. **Graph Representation Learning & Organizational Network Analysis (ONA)**: `Metapath2Vec` (heterogeneous HCM multigraph random walk embeddings), `calculate_burt_constraint` (structural hole identification), `calculate_degree_centrality`, `calculate_betweenness_centrality`.
  5. **Causal Uplift & Doubly Robust Estimation**: `aipw_estimator` (Augmented Inverse Probability Weighting doubly robust ATE), `compute_directed_causal_edge_weights` (NOTEARS continuous acyclic DAG formulation), `diagnose_causal_confounding`.
  6. **Time-to-Event & Competing Risks Survival Modeling**: `calculate_time_in_position`, `calculate_stagnation_index`, cumulative incidence function estimators with cause-specific hazards.
  7. **Dynamic Time Warping & Sequence Trajectory Alignment**: `compute_dtw_distance`, `compute_lcss_distance`, `compute_edr_distance` for multi-dimensional career sequences.
  8. **Structural Decomposition & Rate/Mix LMDI Attribution**: `lmdi_rate_mix_decomposition` (Logarithmic Mean Divisia Index index-number decomposition for workforce turnover and payroll), `LeaveOneOutExpectedDifferential` (manager value-added alpha).
  9. **Explainable AI & Non-linear Feature Attribution (SHAP)**: `compute_shap_interaction_matrix`, `extract_shap_inflection_thresholds`, `calculate_retained_vs_terminated_shap_contributions`.
  10. **Algorithmic Equity & Fairness**: `oaxaca_blinder_decomposition` (threefold wage gap decomposition), `DemographicParityConstraint`, `EqualizedOddsConstraint`, `ExponentiatedGradientFairness` (constrained optimization via saddle-point dynamics).
  11. **Compensation & Pay Equity**: `calculate_compa_ratio`, `calculate_range_penetration`, `classify_pay_band_status`, `is_red_circled`, `is_green_circled`, `calculate_pay_equity_gap`.
- Interactive visualization suite: Seaborn/Matplotlib publication-quality charts for all 50 features.
- Full PEP 561 type annotation compliance (`py.typed`).
- Standardized synthetic data generation module packaged in wheel distribution.
