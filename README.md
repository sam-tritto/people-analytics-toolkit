<p align="center">
  <img src="https://raw.githubusercontent.com/sam-tritto/people-analytics-toolkit/main/logos/logo.png" alt="people-analytics-toolkit" width="600" />
</p>

# Advanced Feature Engineering for People Analytics

A production-grade, mathematically rigorous framework and interactive tutorial project implementing 50 advanced feature engineering techniques designed specifically for workforce intelligence, human capital management (HCM), and People Analytics.

**Interactive Documentation & Live Walkthrough**: [https://sams-data-portfolio.com/projects/people-analytics-toolkit](https://sams-data-portfolio.com/projects/people-analytics-toolkit)
---

## Quick Start

### Prerequisites
- Python 3.10+
- `uv` package manager (version 0.8+)

### Installation

#### From PyPI
Install the core framework or include optional feature bundles in square brackets:

```bash
# Core package (base statistical and mathematical features)
pip install people-analytics-toolkit

# All optional feature engineering bundles
pip install "people-analytics-toolkit[all]"

# Or select specific domain bundles
pip install "people-analytics-toolkit[timeseries,explainability]"
```

#### Optional Dependency Bundles

| Extra | Included Libraries | Features & Capabilities Enabled |
| :--- | :--- | :--- |
| `[timeseries]` | `arch`, `hmmlearn` | GARCH(1,1) dynamic volatility indices, Hidden Markov Model (HMM) latent regimes |
| `[anomalies]` | `pyod` | Local Outlier Factor (LOF) and density-based anomaly detectors |
| `[deeplearning]` | `torch`, `pyod` | Autoencoder reconstruction error, deep trajectory embeddings |
| `[survival]` | `lifelines` | Kaplan-Meier & Nelson-Aalen hazard rate embeddings, survival curves |
| `[explainability]` | `shap`, `lightgbm` | 2D SHAP interaction matrices, turnover risk inflection thresholds |
| `[all]` | *All of the above* | Complete 50-feature analytical and modeling suite |
| `[dev]` | `pytest`, `hypothesis`, `jupyterlab`, `mypy`, `matplotlib`, `seaborn` | Development, property testing, CI validation, and interactive notebooks |

#### Local Development with `uv`
Clone or navigate to the repository directory and synchronize dependencies with desired extras:

```bash
cd people-analytics-toolkit

# Sync with all 50-feature mathematical dependencies:
uv sync --extra all

# Or sync all dependencies including testing, JupyterLab, and dev tooling:
uv sync --all-extras
```

### Running the CLI Verification Script
Execute the command-line demonstration runner across the 50-feature suite:

```bash
uv run python main.py
# Or directly run the pipeline example:
uv run python examples/quickstart_pipeline.py
```

### Running the Test Suite
Execute the automated unit, parametrized, and property-based test suite:

```bash
uv run pytest tests/ -v
```

### Launching the Interactive Master Tutorial Notebook
Launch JupyterLab to interact with all 50 features, data generators, and visualization panels:

```bash
uv run jupyter lab notebooks/advanced_people_analytics_features.ipynb
```

---

<p align="center">
  <img src="https://raw.githubusercontent.com/sam-tritto/people-analytics-toolkit/main/logos/toby_2.jpg" alt="Architectural Problem Pillars" width="800" />
</p>

## Architectural Problem Pillars

The 50 feature engineering techniques are structured across 11 fundamental operational pillars. Click any feature to jump directly to its problem formulation and enterprise use cases:

### 1. High Cardinality & Hierarchies
- [Bayesian Target Encoding](#feature-1)
- [Grouped Leave-One-Out (LOO) Z-Scores](#feature-2)
- [Leave-One-Out Expected Differential (LOO-ED)](#feature-3)

### 2. Preserving Memory & Time
- [Fractional Differencing](#feature-4)
- [Weighted Weeks Since (Tiered Recognition)](#feature-5)
- [EWMA & EWMS (Count Event Filtering)](#feature-6)
- [Double & Triple Exponential Smoothing (HW)](#feature-7)
- [Kaplan-Meier & Nelson-Aalen Embeddings](#feature-8)
- [Rolling Metrics (Configuration Pipeline)](#feature-45)
- [Singular Spectrum Analysis (SVD Smoothing)](#feature-46)
- [Hierarchical Lags (Autoregressive Shifts)](#feature-47)
- [Differencing & % Change (YoY, MoM, WoW)](#feature-48)

### 3. Capturing Chaos, Volatility & Regimes
- [Rolling Shannon Entropy (Chaos Index)](#feature-9)
- [GARCH(1,1) Dynamic Volatility Index](#feature-10)
- [Hawkes Process Intensity (Absence Contagion)](#feature-11)
- [Hidden Markov Model (HMM) Latent Regimes](#feature-12)
- [Bradford Factor (Absenteeism Friction Score)](#feature-42)

### 4. Structural Anomalies, Density & Sequences
- [Autoencoder Reconstruction Error (Weirdness)](#feature-13)
- [Local Outlier Factor (LOF via PyOD)](#feature-14)
- [Kullback-Leibler (KL) Divergence (Drift)](#feature-15)
- [Dynamic Time Warping (DTW) Distance](#feature-16)
- [Longest Common Subsequence (LCSS) & EDR](#feature-17)
- [Mahalanobis Distance (Covariance Outliers)](#feature-50)

### 5. Attribution, Interactions & Accounting
- [Log-Mean Decomposition (LMDI-I)](#feature-18)
- [SHAP Interaction Values (2D Risk Synergies)](#feature-19)
- [Correlation Heatmap Table (p & CIs)](#feature-49)

### 6. Succession Planning, Mobility & Behavioral Tipping Points
- [Movement Embeddings (Career Trajectory)](#feature-20)
- [SHAP Inflection Thresholds for Term Rates](#feature-21)
- [Time-in-Position (The Stagnation Index)](#feature-41)

### 7. Compensation & Pay Equity Transformations
- [Compa-Ratio (Midpoint & Market Parity)](#feature-22)
- [Range Penetration (Band Progression Audit)](#feature-23)

### 8. Relational Dynamics & ONA (Organizational Network Analysis)
- [Betweenness Centrality (Cross-Dept Brokers)](#feature-24)
- [Eigenvector Centrality (Informal Hubs)](#feature-25)
- [Ronald Burt's Network Constraint Index](#feature-26)
- [Euler Centrality & Leinster Magnitude](#feature-27)
- [Span of Control (Managerial Load & Reorg)](#feature-40)

### 9. Algorithmic Equity, Pay Decomposition & Individual Fairness
- [Oaxaca-Blinder Pay Gap Decomposition](#feature-28)
- [Demographic Parity (Statistical Parity)](#feature-29)
- [Equalized Odds (TPR & FPR Parity)](#feature-30)
- [Individual Fairness & Local Consistency](#feature-31)

### 10. Prescriptive Interventions: Causal Inference & Uplift
- [Augmented IPW (AIPW Doubly Robust ATE)](#feature-32)
- [Synthetic Difference-in-Differences (SDiD)](#feature-33)
- [Causal Forest with Double ML (CATE)](#feature-34)
- [Uplift Evaluation: Qini Curve & AUUC](#feature-35)
- [Synthetic Control Weights (Synth-DiD Twin)](#feature-43)
- [Directed Causal Edge Weights (DAG NOTEARS)](#feature-44)

### 11. Continuous Trajectories & Graph Representation Learning
- [Continuous-Time Markov Chains (CTMC Rates)](#feature-36)
- [Meta-Path Biased Random Walks (HCM Schema)](#feature-37)
- [Metapath2Vec (SPPMI Factorization & Cohesion)](#feature-38)
- [Dynamic Entity & Trajectory Self-Attention](#feature-39)

---

## Feature Catalog & Enterprise Use Cases

<a id="feature-1"></a>
### 1. Bayesian Target Encoding (Empirical Bayes Shrinkage)
* **Problem**: In retail, healthcare, and distributed operations, crossing categorical dimensions like `STORE_ID x DEPT_CODE` creates thousands of high-cardinality cohorts. Small cohorts ($N=2$ to $5$) suffer extreme sampling variance, causing standard one-hot or naive target encoding to overfit to 0% or 100% turnover rates and leak target distributions.
* **Use Cases**:
  - **New Store & Branch Baselines**: Establishing credible attrition and sales baselines for newly opened locations before longitudinal records accumulate.
  - **High-Cardinality Tree Features**: Encoding granular combinations of store location, shift code, and job family for gradient boosted models without dimensional explosion.
  - **Sparse Role Flight Risk**: Modeling turnover propensities for specialized, low-headcount positions across distributed business units.

<a id="feature-2"></a>
### 2. Grouped Leave-One-Out (LOO) Z-Scores
* **Problem**: Standard Z-score calculations within localized peer groups suffer from self-masking. When an employee logs massive overtime (e.g. 50 hours in a 5-person team), their own outlier value pulls the cohort mean upward and inflates the standard deviation, concealing the severity of their own anomaly.
* **Use Cases**:
  - **Overtime Burnout Detection**: Flagging associates whose workload dramatically exceeds localized peer norms without the calculation being distorted by their own hours.
  - **Incentive & Bonus Anomaly Audits**: Identifying disproportionate incentive allocations within small branch offices or regional sales squads.
  - **Localized Absenteeism Spikes**: Uncovering acute attendance disruptions in small retail department cohorts.

<a id="feature-3"></a>
### 3. Leave-One-Out Expected Differential (LOO-ED)
* **Problem**: Evaluating store or department managers solely on raw turnover or gross throughput conflates external operational tailwinds (e.g. flagship foot traffic, prime weekend footfall) with true managerial leadership skill and coaching impact.
* **Use Cases**:
  - **True Managerial Alpha**: Disentangling shift manager operational skill from external store foot traffic, seasonal rush, and staffing volume.
  - **Equitable Leadership Benchmarking**: Evaluating supervisory effectiveness across fundamentally different retail or hospital branches fairly.
  - **Promotion Readiness Scoring**: Isolating leaders who consistently generate positive productivity differentials regardless of assignment difficulty.

<a id="feature-4"></a>
### 4. Fractional Differencing
* **Problem**: Standard integer differencing ($d=1$) removes non-stationarity at the cost of obliterating multi-year organizational memory, reducing multi-year headcount and retention histories to white noise. Conversely, leaving series undifferenced ($d=0$) produces spurious regressions.
* **Use Cases**:
  - **Long-Term Headcount Forecasting**: Maintaining historical organizational cycle memory while achieving statistical stationarity for machine learning regressors.
  - **Macro Attrition Trend Modeling**: Preserving multi-year retention cultural persistence across seasonal hiring waves.
  - **Budget & Labor Demand Planning**: Forecasting capacity across multi-year horizon horizons with optimal memory preservation.

<a id="feature-5"></a>
### 5. Weighted Weeks Since (Continuous Event-Decay Features)
* **Problem**: Binary milestone flags (e.g. `Received_Recognition_This_Year = 1`) treat an award received yesterday identically to one won 50 weeks ago, completely ignoring temporal decay and the relative prestige of tiered recognitions.
* **Use Cases**:
  - **Continuous Engagement & Recognition Pulse**: Tracking the decaying psychological boost of spot bonuses, peer recognition, and leadership awards over time.
  - **Disciplinary Action Cooling**: Modeling the decaying behavioral impact of written warnings and formal performance reviews.
  - **Milestone Recency Scoring**: Quantifying recent career positive reinforcement pulses to predict short-term flight risk.

<a id="feature-6"></a>
### 6. Exponentially Weighted Moving Average & Sum (EWMA & EWMS)
* **Problem**: Simple rolling averages create artificial step changes ("ghost drops") when historical spikes roll out of the calculation window. Furthermore, discrete event counts (e.g. sick call-outs) require intensity accumulation filters rather than simple smoothed means.
* **Use Cases**:
  - **Dynamic Workload Tracking**: Smoothing weekly scheduled hours to capture acute operational strain while prioritizing recent trend changes.
  - **Absence Rate Accumulation**: Computing event intensity rates from discrete daily attendance logs for predictive workforce models.
  - **Recent Sentiment Trajectory**: Weighting pulse survey responses to reflect current organizational morale over distant historical feedback.

<a id="feature-7"></a>
### 7. Double and Triple Exponential Smoothing (Holt-Winters)
* **Problem**: Standard smoothing models lag behind strong directional expansion or contraction trends and fail to anticipate recurring multi-period seasonal waves (e.g. annual summer surges or Q4 retail holiday peaks).
* **Use Cases**:
  - **Seasonal Staffing Forecasts**: Generating multi-horizon headcount projections that account for both annual retail seasonality and underlying workforce growth velocity.
  - **Overtime Budgeting**: Disentangling permanent structural workload expansion from temporary holiday seasonal surges.
  - **Recruitment Pipeline Sizing**: Forecasting future hiring demand curves based on historical seasonal applicant velocity.

<a id="feature-8"></a>
### 8. Kaplan-Meier & Nelson-Aalen Hazard Embeddings
* **Problem**: Feeding employee tenure linearly (e.g. `tenure = 14 months`) into attrition models assumes uniform departure risk over time, ignoring steep non-linear behavioral cliff points such as the initial 90-day onboarding cliff and annual vesting anniversaries.
* **Use Cases**:
  - **Empirical Hazard Coordinates**: Converting raw tenure integers into calibrated survival probabilities $\hat{S}(t)$ and cumulative hazard rates $\hat{H}(t)$.
  - **Onboarding Flight Risk Watch**: Identifying associates approaching high-risk tenure milestones (e.g. 90-day onboarding mark, 1-year cliff).
  - **Equity Vesting Retention Audits**: Calibrating flight risk escalation curves around equity vesting and bonus payout schedules.

<a id="feature-9"></a>
### 9. Rolling Shannon Entropy (The "Chaos Index")
* **Problem**: Two departments may average the exact same weekly hours per associate (e.g. 38 hrs), but one maintains predictable, stable shifts while the other forces erratic, fragmented split shifts, short call-ins, and chaotic scheduling.
* **Use Cases**:
  - **Schedule Volatility & Friction Index**: Measuring operational shift unpredictability to predict associate burnout and voluntary resignations.
  - **Departmental Workload Stability Audits**: Benchmarking schedule predictability across store networks to identify poor scheduling practices.
  - **Part-Time Schedule Optimization**: Ensuring fair, consistent shift distribution across hourly workforce pools.

<a id="feature-10"></a>
### 10. GARCH(1,1) Dynamic Volatility Index
* **Problem**: Workforce demand volatility is not constant over time; it clusters into distinct temporal regimes where an erratic week predictably generates unstable labor demands for subsequent weeks.
* **Use Cases**:
  - **Labor Demand Turbulence Tracking**: Quantifying dynamic conditional volatility in labor hours to identify branches experiencing operational instability.
  - **Contingent Staffing Buffer Planning**: Adjusting flex-staffing and temp agency buffer capacities based on real-time volatility clustering.
  - **Managerial Stress Forecasting**: Identifying locations where ongoing schedule unpredictability is likely to burn out frontline supervisors.

<a id="feature-11"></a>
### 11. Hawkes Process Intensity (The "Contagion" Feature)
* **Problem**: Absenteeism, sudden call-outs, and resignations exhibit social contagion dynamics. When one associate abruptly leaves or calls out, team strain increases the conditional likelihood of subsequent departures.
* **Use Cases**:
  - **Real-Time Absenteeism Spreading**: Modeling the self-exciting cascade of call-outs within tightly coupled shift crews.
  - **Resignation Contagion Early Warning**: Detecting the initial pulse of turnover contagion before a localized team experiences mass departures.
  - **Overtime Strain Cascade Mitigation**: Deploying floating staff to intercept teams entering a self-reinforcing understaffing spiral.

<a id="feature-12"></a>
### 12. Hidden Markov Model (HMM) Latent Regimes
* **Problem**: High weekly overtime can indicate two diametrically opposed realities: a healthy, planned seasonal surge with high morale, or an acute understaffed attrition crisis. Standard models fail to differentiate these operational states.
* **Use Cases**:
  - **Operational State Classification**: Automatically classifying store status into 'Normal Baseline', 'Planned Seasonal Peak', or 'Understaffed Crisis'.
  - **Crisis Posterior Probability**: Feeding downstream retention models the probability of being in an acute crisis regime rather than raw overtime hours.
  - **Automated Intervention Dispatch**: Triggering executive staffing interventions only when a location shifts into the unmanaged distress state.

<a id="feature-13"></a>
### 13. Autoencoder Reconstruction Error (The "Weirdness" Index)
* **Problem**: Rigid business rules (e.g. `overtime > 20% AND turnover > 15%`) fail to detect complex organizational dysfunction spanning dozens of subtle, correlated operational signals.
* **Use Cases**:
  - **Store Operational Decoupling**: Identifying locations whose multidimensional operational signature diverges sharply from healthy peer store norms.
  - **Root Cause Attribution**: Decomposing autoencoder reconstruction error per feature to reveal specific operational vulnerabilities (e.g. training deficits, inventory misalignment).
  - **Holistic Branch Auditing**: Prioritizing field visits and operational audits based on global anomaly severity scores.

<a id="feature-14"></a>
### 14. Local Outlier Factor (LOF via PyOD)
* **Problem**: Global anomaly models compare every employee against the entire enterprise, failing to recognize that what is normal for an entry-level cohort (e.g. 15% raise, frequent job change) is an extreme anomaly for senior, tenured specialists.
* **Use Cases**:
  - **Localized Compensation Audits**: Detecting salary, equity, or bonus anomalies relative to immediate tenure, job level, and rating peer clusters.
  - **Off-Cycle Promotion Screening**: Identifying promotion velocity anomalies within localized professional bands.
  - **Fairness & Equity Monitoring**: Flagging localized compensation discrepancies that escape global corporate band reviews.

<a id="feature-15"></a>
### 15. Kullback-Leibler (KL) Divergence (The "Drift" Index)
* **Problem**: Workforce planners lack a single unified statistical metric to monitor whether a store or branch's departmental labor distribution is drifting away from corporate benchmark staffing models.
* **Use Cases**:
  - **Staffing Allocation Drift**: Measuring the divergence between actual store departmental headcount distributions and target engineering models.
  - **Reorganization Compliance**: Tracking how closely restructured divisions adhere to intended target functional allocations.
  - **Skill Portfolio Realignment**: Quantifying structural drift in enterprise skill distributions across technology and operational units.

<a id="feature-16"></a>
### 16. Dynamic Time Warping (DTW) Distance
* **Problem**: Euclidean distance between annual demand curves falsely flags branches as mismatched simply because seasonal demand peaks arrive earlier in warmer climates or different regional markets.
* **Use Cases**:
  - **Climate-Adjusted Peer Store Clustering**: Grouping retail stores by underlying seasonal demand curve shapes regardless of calendar phase shifts.
  - **Career Trajectory Alignment**: Matching multi-year career progression velocity curves across accelerated and standard development tracks.
  - **Demand Profile Benchmarking**: Transferring successful staffing models between stores with temporally shifted but structurally identical operational cycles.

<a id="feature-17"></a>
### 17. Longest Common Subsequence (LCSS) & Edit Distance on Real Sequences (EDR)
* **Problem**: DTW matches every single timestamp and is severely distorted by temporal gaps or localized noise (such as a 2-week leave of absence, vacation, or temporary sabbatical).
* **Use Cases**:
  - **Gap-Tolerant Career Pathing**: Clustering employee promotion and mobility sequences while ignoring temporary leaves or lateral pauses.
  - **Shift Pattern Matching**: Identifying recurring weekly schedule profiles across employees despite occasional holiday disruptions.
  - **Talent Benchmark Pathways**: Mapping non-linear career trajectories against benchmark paths taken by enterprise executive leaders.

<a id="feature-18"></a>
### 18. Log-Mean Decomposition (LMDI-I)
* **Problem**: When company-wide turnover decreases, executives cannot determine whether retention initiatives actually worked (Rate Effect) or whether high-turnover departments merely downsized relative to the company (Structural Mix Effect).
* **Use Cases**:
  - **Executive Turnover Attribution**: Mathematically decomposing annual turnover shifts into pure retention performance vs structural organizational reallocation with zero residual error.
  - **Diversity & Representation Accounting**: Partitioning shifts in diversity metrics into genuine hiring/promotion rate progress vs business unit expansion/contraction.
  - **Labor Cost Variance Analysis**: Isolating whether wage growth was driven by wage increases (rate) or shifts toward higher-compensated job categories (mix).

<a id="feature-19"></a>
### 19. SHAP Interaction Values (2D Non-Linear Feature Interactions)
* **Problem**: Standard feature importance evaluates variables in isolation, blinding leadership to dangerous multi-factor synergies where two moderate metrics interact to produce catastrophic workforce risks.
* **Use Cases**:
  - **Compounding Flight Risk Discovery**: Identifying high-risk factor combinations (e.g. `New Manager < 6 Months` combined with `Tenure < 90 Days` driving 50%+ of early attrition).
  - **Workforce Safety Incident Prevention**: Uncovering non-linear interactions between shift length and consecutive days worked that spike injury rates.
  - **Targeted HR Interventions**: Designing specific dual-factor mitigation policies rather than blunt single-variable corporate mandates.

<a id="feature-20"></a>
### 20. Movement Embeddings (Career Trajectory & Graph Embeddings)
* **Problem**: Succession planning typically views job roles as rigid vertical silos, ignoring valuable lateral experiences and cross-functional skills acquired through non-traditional career paths.
* **Use Cases**:
  - **Cross-Functional Talent Discovery**: Calculating mathematical similarity between employee career trajectories and target roles to identify non-obvious, highly qualified internal successors.
  - **Internal Mobility Recommendations**: Suggesting lateral moves that build critical bridge skills required for future executive leadership positions.
  - **Hidden Talent Sourcing**: Surfacing high-potential candidates across divergent departments who share latent competencies with top performers.

<a id="feature-21"></a>
### 21. SHAP Inflection Thresholds for Term Rates (Zero-Crossing Behavioral Tipping Points)
* **Problem**: Business leaders need clear, operational thresholds (e.g. "At exactly what overtime hour does retention flip into flight risk?") rather than complex, uninterpretable machine learning probability outputs.
* **Use Cases**:
  - **Actionable Overtime Caps**: Extracting the exact mathematical tipping point where additional overtime shifts from positive engagement to severe flight risk.
  - **Pay Compression Tipping Points**: Identifying the precise compa-ratio threshold below which resignation rates accelerate non-linearly.
  - **Targeted Retention Threshold Policies**: Establishing evidence-based operational policy guardrails across distinct department and tenure cohorts.

<a id="feature-22"></a>
### 22. Compa-Ratio (Comparative Ratio)
* **Problem**: Comparing raw base salaries confounds job grade levels, seniority, and regional market cost differences. Total rewards teams require a normalized metric to gauge competitiveness against market midpoints.
* **Use Cases**:
  - **Market Competitiveness Auditing**: Standardizing compensation analysis across diverse job grades and geographical salary structures.
  - **Flight Risk Compensation Screening**: Flagging associates compensated below 80% of market midpoint as high voluntary departure risks.
  - **Annual Merit Review Calibration**: Allocating salary increase budgets to bring deserving performers toward target market midpoint parity.

<a id="feature-23"></a>
### 23. Range Penetration
* **Problem**: Compa-Ratio references only the salary midpoint and ignores pay band width (spread), leaving compensation teams unable to evaluate where an associate stands between the minimum entry point and maximum ceiling of their grade.
* **Use Cases**:
  - **Pay Band Progression Tracking**: Measuring employee career advancement across the complete salary range (0% to 100%).
  - **Red-Circle / Green-Circle Audits**: Automatically flagging employees paid above grade maximums (requiring salary freezes) or below grade minimums (requiring statutory equity adjustments).
  - **Promotion Increment Sizing**: Structuring promotional salary adjustments to place newly promoted individuals appropriately within their new pay band.

<a id="feature-24"></a>
### 24. Betweenness Centrality (Cross-Functional Information Brokers)
* **Problem**: Standard organizational charts depict only formal hierarchical lines, failing to identify informal intermediaries whose cross-departmental relationships keep siloed business units aligned.
* **Use Cases**:
  - **Key Person Retention Risk**: Identifying critical structural brokers whose departure would sever informal communication pathways between decoupled teams.
  - **Cross-Silo Collaboration Audits**: Pinpointing employees who facilitate knowledge transfer across product, engineering, and commercial functions.
  - **Post-Merger Integration Tracking**: Monitoring the emergence of cross-entity brokers uniting acquired companies with the parent enterprise.

<a id="feature-25"></a>
### 25. Eigenvector Centrality (Informal Enterprise Hubs)
* **Problem**: Simple degree centrality counts connections equally, failing to distinguish between an employee connected to junior team members and an individual with direct access to senior executive decision-makers.
* **Use Cases**:
  - **Informal Influence Mapping**: Identifying cultural keystones and informal opinion leaders to champion change management initiatives.
  - **Succession Pipeline Validation**: Verifying that leadership candidates possess authentic informal enterprise influence rather than merely formal administrative authority.
  - **Executive Communication Audits**: Evaluating the informal reach and network power of key management personnel.

<a id="feature-26"></a>
### 26. Ronald Burt's Network Constraint Index (Structural Holes & Network Redundancy)
* **Problem**: Insular, tightly knit teams foster echo chambers and groupthink, while associates who span structural holes gain access to novel ideas, drive innovation, and exhibit faster career mobility.
* **Use Cases**:
  - **Innovation Capacity Scoring**: Identifying individuals with low network constraint who bridge decoupled social clusters and generate innovative ideas.
  - **Team Silo Diagnosis**: Flagging departments with high average network constraint as redundant, insular units prone to groupthink.
  - **High-Potential Leadership Screening**: Using network autonomy and structural hole access as predictive indicators of executive leadership agility.

<a id="feature-27"></a>
### 27. Euler Centrality & Leinster Metric Space Magnitude (Global Structural Distinctiveness)
* **Problem**: Standard network metrics reward density and sheer connection volume, systematically overlooking isolated domain specialists who occupy structurally distinct positions in the global enterprise topology.
* **Use Cases**:
  - **Enterprise Structural Diversity**: Quantifying the effective number of structurally unique contributors across an organizational network using metric space magnitude.
  - **Niche Specialist Identification**: Recognizing employees who occupy irreplaceable topological positions despite low daily interaction volume.
  - **Reorganization Topology Assessment**: Evaluating whether planned structural realignments increase or decrease overall organizational network complexity.

<a id="feature-28"></a>
### 28. Oaxaca-Blinder Pay Gap Decomposition (Econometric Disparity Accounting)
* **Problem**: Raw wage comparisons between demographic groups fail to separate legitimate differences in human capital endowments (tenure, education, job level) from unexplained structural wage gaps and systemic market bias.
* **Use Cases**:
  - **Statutory Pay Equity Audits**: Decomposing enterprise wage gaps into explained human capital factors vs unexplained residual disparities for regulatory compliance.
  - **Remediation Budget Allocation**: Sizing targeted pay adjustments to eliminate unexplained wage differentials across gender and racial cohorts.
  - **Hiring Offer Calibration**: Ensuring starting salary formulas do not propagate historical market inequities into new employee offers.

<a id="feature-29"></a>
### 29. Demographic Parity (Statistical Parity) Constraint
* **Problem**: Unconstrained machine learning models trained on historical hiring or promotion data learn past institutional selection biases, resulting in unequal positive selection rates across sensitive demographic groups.
* **Use Cases**:
  - **Automated Resume Screening Audits**: Enforcing equal positive selection rates across protected demographic groups in candidate shortlisting algorithms.
  - **Four-Fifths Rule Compliance**: Monitoring AI-driven recruitment and promotion models to satisfy EEOC 80% adverse impact criteria.
  - **Fairness-Constrained Model Training**: Optimizing predictive classifiers with strict group parity bounds using fair reduction techniques.

<a id="feature-30"></a>
### 30. Equalized Odds Constraint via Exponentiated Gradient Minimax Reduction
* **Problem**: Demographic Parity ignores qualifications and can incentivize blunt quotas. High-stakes promotion, performance, and succession decisions require that equally qualified candidates have identical true positive rates regardless of demographic group.
* **Use Cases**:
  - **Merit-Preserving Promotion Algorithms**: Guaranteeing that qualified candidates across all demographic groups have equal opportunity of receiving advancement recommendations.
  - **Performance Evaluation Calibration**: Eliminating disparate false positive and false negative error rates in automated talent management models.
  - **Regulatory Algorithm Auditing**: Meeting state and federal standards for non-discriminatory algorithmic talent scoring.

<a id="feature-31"></a>
### 31. Individual Fairness & Local Predictive Consistency (Lipschitz Concordance)
* **Problem**: Group fairness constraints guarantee macro-level balance across demographics but permit arbitrary, unfair decisions between nearly identical individuals (e.g. flipping promotion outcomes between candidates with identical credentials).
* **Use Cases**:
  - **Predictive Consistency Verification**: Measuring local consistency across $k$-nearest neighbors in standardized human capital credential space.
  - **Model Memorization & Glitch Auditing**: Flagging individual predictions that deviate significantly from those of closely matched peer profiles.
  - **Arbitrary Decision Prevention**: Ensuring similar employees receive similar talent assessment scores across hiring, comp, and retention models.

<a id="feature-32"></a>
### 32. Augmented Inverse Propensity Weighting (AIPW Doubly Robust ATE)
* **Problem**: Estimating the true effectiveness of historical HR interventions (e.g. mentorship initiatives, leadership training, retention bonuses) from observational HRIS data is severely distorted by confounding and non-random selection bias.
* **Use Cases**:
  - **HR Program ROI Evaluation**: Accurately quantifying the average treatment effect (ATE) of leadership academies, coaching programs, and wellness benefits on retention.
  - **Doubly Robust Policy Assessment**: Obtaining unbiased causal effect estimates and valid 95% confidence intervals even if either the propensity or outcome model is mis-specified.
  - **Observational Data De-biasing**: Removing self-selection bias from employee program participation data.

<a id="feature-33"></a>
### 33. Synthetic Difference-in-Differences (SDiD Panel Policy Evaluation)
* **Problem**: Enterprise-wide policy rollouts (e.g. pilot 4-day workweeks or remote-work policies in specific offices) violate the parallel trends assumption required by classical Difference-in-Differences.
* **Use Cases**:
  - **Regional Workplace Policy Pilots**: Estimating the causal impact of new flexible work arrangements, compensation structures, or shift guidelines across pilot locations.
  - **Overcoming Non-Parallel Trends**: Finding optimal unit and time weights to construct valid pre-treatment baseline trajectories for treated branches.
  - **Executive Policy Reporting**: Delivering credible, policy-grade causal impact figures with robust standard errors to C-suite stakeholders.

<a id="feature-34"></a>
### 34. Causal Forest with Double Machine Learning (CausalForestDML)
* **Problem**: Blanket HR interventions waste budget on employees who would stay regardless ("Sure Things") or leave anyway ("Lost Causes"). Organizations need to estimate individual-level Conditional Average Treatment Effects (CATE) to identify "Persuadables".
* **Use Cases**:
  - **Targeted Retention Bonus Allocation**: Identifying specific employees whose retention probability will dramatically improve when offered retention incentives.
  - **Personalized Leadership Coaching**: Targeting expensive professional development programs to associates with the highest predicted causal uplift.
  - **Intervention Optimization**: Maximizing overall workforce program impact while slashing budget waste on non-responsive cohorts.

<a id="feature-35"></a>
### 35. Prescriptive Uplift Evaluation (Qini Curve, AUUC, and Doubly Robust MSE)
* **Problem**: In prescriptive modeling, the counterfactual unchosen state is never observed, making standard supervised metrics (such as ROC-AUC or RMSE) completely incapable of evaluating uplift targeting quality.
* **Use Cases**:
  - **Uplift Model Selection & Benchmarking**: Evaluating prescriptive retention and intervention targeting models using Qini curves and Area Under the Uplift Curve (AUUC).
  - **Cumulative Gain Analysis**: Quantifying the incremental workforce retentions achieved per dollar of intervention budget relative to random outreach.
  - **Doubly Robust Model Evaluation**: Evaluating uplift model predictions against doubly robust pseudo-ground-truth scores without observational bias.

<a id="feature-36"></a>
### 36. Continuous-Time Markov Chains (CTMC Career Trajectories)
* **Problem**: Discrete-time Markov models assume promotions and role transitions happen on rigid annual calendar cycles, misrepresenting the continuous, stochastic nature of modern career mobility and voluntary departures.
* **Use Cases**:
  - **Continuous Promotion Velocity Modeling**: Estimating continuous transition rate generator matrices ($Q$) and multi-year career progression probabilities.
  - **Career Path Bottleneck Diagnosis**: Computing expected sojourn times in each job grade to pinpoint organizational levels where talent stagnates.
  - **Long-Term Succession Capacity**: Forecasting the probability of talent absorption into voluntary attrition versus progression into senior executive tiers over arbitrary time horizons.

<a id="feature-37"></a>
### 37. Meta-Path Biased Random Walks on Heterogeneous HCM Multigraphs
* **Problem**: Modern human capital multigraphs contain multiple node types (Employees, Teams, Projects, Skills) and relationship types. Simple random walks get trapped in high-density social cliques without traversing meaningful semantic career paths.
* **Use Cases**:
  - **Semantic Career Path Modeling**: Guiding random walks along domain-governed paths (e.g. `Employee -> Project -> Skill -> Employee`) to discover latent professional relationships.
  - **Internal Cross-Functional Sourcing**: Uncovering employees who share specific project and skill pathways despite belonging to separate reporting hierarchies.
  - **Workforce Knowledge Graph Mining**: Structuring complex enterprise relational data for graph representation learning and talent recommendations.

<a id="feature-38"></a>
### 38. Metapath2Vec (SPPMI Matrix Factorization & Team Complementarity)
* **Problem**: Graph neural network pipelines often require complex runtime dependencies. Workforce intelligence systems require lightweight, scalable entity embeddings to benchmark employee similarity and quantify team skill diversity.
* **Use Cases**:
  - **Project Squad Assembly**: Quantifying team complementarity, cohesion, and skill diversity spread bounded in $[0, 1]$ for new initiative staffing.
  - **Structural Peer Benchmarking**: Mapping employees into dense embedding spaces to discover true structural peers for compensation and performance calibration.
  - **Cross-Departmental Skill Clustering**: Identifying hidden talent clusters with complementary skill sets across decentralized business units.

<a id="feature-39"></a>
### 39. Dynamic Entity & Trajectory Self-Attention
* **Problem**: Employee career histories contain varying sequences of role assignments, certifications, and project deliveries. Static average pooling fails to dynamically weight breakthrough assignments while downweighting obsolete early career roles.
* **Use Cases**:
  - **Contextual Career History Representation**: Using multi-head self-attention to dynamically weight significant career milestones and projects for attrition and promotion prediction.
  - **Trajectory Embedding Generation**: Pooling variable-length career transition histories into unified, dense vectors for downstream talent intelligence models.
  - **Leadership Readiness Assessment**: Evaluating how historical sequence patterns of responsibility and scope translate into future executive performance.

<a id="feature-40"></a>
### 40. Span of Control (Managerial Load & Reorganization Flight Risk)
* **Problem**: Corporate restructuring often surges managerial spans (e.g. from 8 to 25 reports), sharply diluting 1-on-1 coaching time and triggering severe flight risk surges among direct reports independent of compensation or individual performance.
* **Use Cases**:
  - **Reorganization Impact Simulation**: Modeling the hidden attrition liability and associate attention dilution resulting from management consolidation plans.
  - **Managerial Load Benchmarking**: Classifying leaders into Under-leveraged, Optimal, Stretched, and Critical Overload tiers based on span of control.
  - **Coaching Attention Audits**: Estimating dedicated weekly 1-on-1 coaching minutes per associate to maintain operational standards and prevent turnover.

<a id="feature-41"></a>
### 41. Time-in-Position (The Stagnation Index & Flight Risk Surge)
* **Problem**: High-performing associates who remain in the same job code far longer than cohort norms experience severe stagnation, causing sudden voluntary resignations that blindside leadership.
* **Use Cases**:
  - **Stagnant High-Performer Early Warning**: Flagging associates with top performance ratings whose time-in-position exceeds the 90th percentile of their department peer group.
  - **Proactive Career Interventions**: Triggering automated alerts for internal mobility discussions, strategic project rotations, or promotion reviews before flight risk peaks.
  - **Tenure Stagnation Audits**: Benchmarking career progression velocities across business units to identify organizational bottlenecks and talent logjams.

<a id="feature-42"></a>
### 42. The Bradford Factor (Absenteeism Friction Score & Schedule Disruption)
* **Problem**: Standard attendance tracking counts only total days absent ($D$), treating a single 10-day medical leave identically to 10 sporadic 1-day absences, even though frequent short absences cause vastly worse scheduling chaos and overtime scramble.
* **Use Cases**:
  - **Absence Friction Triage**: Identifying chronic short-term absence patterns ($B = S^2 \times D$) that severely disrupt operational shift schedules.
  - **Automated HR Trigger Tiers**: Routing cases to standard operational tolerance, informal manager check-ins, or formal medical review based on friction tiers.
  - **Fairness in Attendance Management**: Protecting employees taking legitimate continuous medical leave while addressing volatile short-term call-out friction.

<a id="feature-43"></a>
### 43. Synthetic Control Weights (Synth-DiD Synthetic Twin & Causal Lift)
* **Problem**: Standard pilot comparisons attempt to match a pilot store to a single "similar" store, but complex local market variations make finding an identical 1-to-1 twin virtually impossible.
* **Use Cases**:
  - **Pilot Program Evaluation**: Blending a donor pool of untreated locations to synthesize a counterfactual "twin" that matches a pilot store's pre-treatment trajectory.
  - **Automated Scheduling Rollouts**: Measuring the true causal lift of automated shift-scheduling tools on turnover and labor efficiency across pilot branches.
  - **Capital Investment ROI**: Evaluating the true causal impact of branch facility upgrades, breakroom enhancements, or localized technology rollouts.

<a id="feature-44"></a>
### 44. Directed Causal Edge Weights (DAG Discovery & Confounder Untangling)
* **Problem**: Correlation matrices show only that variables co-vary. In complex workforce feedback loops (e.g. "Overtime vs Turnover"), correlation cannot distinguish whether overtime causes turnover or whether an upstream driver (e.g. Manager Absence) causes both.
* **Use Cases**:
  - **Root Cause vs Symptom Diagnosis**: Learning directed causal graphs (DAGs) to identify whether to intervene on supervisor absenteeism, scheduling chaos, or overtime hours.
  - **Confounder Untangling**: Distinguishing true direct causal drivers from spurious correlations caused by shared upstream organizational factors.
  - **Total Path Influence Calculation**: Computing total mediated causal impact across the enterprise to prioritize high-leverage HR policy interventions.

<a id="feature-45"></a>
### 45. Configuration-Driven Rolling Metrics (Multi-Horizon Cohort Pipelines)
* **Problem**: Hardcoding separate rolling aggregations for dozens of operational metrics across hierarchical partitions (such as `Store x Department`) leads to brittle pipelines, code duplication, and subtle temporal lookahead leakage.
* **Use Cases**:
  - **Multi-Horizon Workforce Dashboards**: Automatically constructing trailing 4-week, 13-week, and 52-week metrics (headcount, overtime, turnover) with strict chronological integrity.
  - **Feature Store Engineering**: Generating standardized temporal aggregations across disparate workforce data streams without index permutation errors.
  - **Schedule Stability Auditing**: Measuring rolling variance in assigned hours across retail store departments to identify burnout-inducing schedule turbulence.

<a id="feature-46"></a>
### 46. Singular Spectrum Analysis (SVD Smoothing & Lag-Free Baselines)
* **Problem**: Moving averages introduce severe phase-lag distortion ($W/2$ delays) that shifts peak timings and distorts seasonal patterns in daily foot traffic and workforce demand curves.
* **Use Cases**:
  - **Zero Phase-Lag Baseline Extraction**: Decomposing noisy daily store traffic and labor hours into smooth, lag-free secular trends and cyclical components.
  - **Downstream Model Covariates**: Supplying clean, non-lagged baseline features to gradient-boosted trees and neural forecasting models.
  - **Anomaly Detection**: Separating underlying operational demand signals from high-frequency daily noise to flag genuine attendance anomalies.

<a id="feature-47"></a>
### 47. Hierarchical Lag & Autoregressive Features
* **Problem**: Applying standard shift operations to unpartitioned workforce panels causes catastrophic cross-cohort target leakage (e.g. shifting the bottom row of Store 1 into the top row of Store 2) and lookahead bias.
* **Use Cases**:
  - **Safe Autoregressive Modeling**: Generating historical lagged observations ($t-1$, $t-4$, $t-52$) strictly bounded within each distinct store and department partition.
  - **Inertia Feature Generation**: Feeding prior-period baseline metrics into turnover and labor demand forecasting models without data corruption.
  - **Panel Data Preprocessing**: Guaranteeing strict chronological sorting and index preservation across complex enterprise organizational structures.

<a id="feature-48"></a>
### 48. Multi-Horizon Differencing & Growth Rates (YoY, MoM, WoW & % Change)
* **Problem**: Raw metric changes across different calendar cadences are distorted by seasonal cycles, scale disparities between departments, and potential division-by-zero crashes on zero-baseline periods.
* **Use Cases**:
  - **Seasonally Deseasonalized Growth**: Calculating year-over-year (YoY) differences to isolate true organic workforce growth from holiday expansion.
  - **Acute Shift Turbulence**: Computing week-over-week (WoW) percentage changes in scheduled hours to detect sudden schedule disruptions triggering turnover.
  - **Departmental Scale Parity**: Standardizing growth metrics across small specialty teams and massive operational departments using protected percentage rate formulas.

<a id="feature-49"></a>
### 49. Correlation Heatmap Table (Pearson, Spearman, Kendall with CIs & p-values)
* **Problem**: Raw correlation matrices display uncalibrated point estimates without statistical significance, confidence intervals, or multiple testing corrections, leading analysts to base policy on noisy or directionally ambiguous relationships.
* **Use Cases**:
  - **Calibrated Feature Selection**: Screening candidate workforce predictors against retention, engagement, and productivity targets using rigorous $p$-value adjustments (`fdr_bh`, `bonf`).
  - **Directional Ambiguity Filtering**: Identifying predictors whose 95% confidence intervals cross zero ($LB < 0 < UB$), eliminating false-positive feature candidates.
  - **Executive-Ready Statistical Reporting**: Generating conditionally styled correlation heatmaps that clearly distinguish statistically proven drivers from statistical noise.

<a id="feature-50"></a>
### 50. Multivariate Mahalanobis Distance (Covariance-Scaled Outliers)
* **Problem**: Standard Euclidean distance assumes features are orthogonal and uncorrelated. In workforce datasets where metrics are heavily correlated (tenure vs comp, hours vs productivity), Euclidean distance generates false alarms while missing true multi-metric decouplings.
* **Use Cases**:
  - **Covariance-Aware Outlier Detection**: Identifying associates whose multidimensional performance, pay, and tenure combination breaks natural correlation patterns.
  - **Robust Compensation Auditing**: Using Minimum Covariance Determinant (MCD) estimators to flag executive compensation anomalies without distortion from historical outliers.
  - **Chi-Squared Anomaly Flagging**: Converting multi-attribute distances into rigorous $p$-values to prioritize HR governance reviews.

---

<p align="center">
  <img src="https://raw.githubusercontent.com/sam-tritto/people-analytics-toolkit/main/logos/toby_1.jpg" alt="Directory Structure" width="800" />
</p>

## Directory Structure

```
people-analytics-toolkit/
├── pyproject.toml              # Package specification (people-analytics-toolkit), dependencies, mypy, and pytest configuration
├── README.md                   # Authoritative reference and technical guide (zero emojis)
├── CHANGELOG.md                # Version release history (Keep a Changelog format)
├── CONTRIBUTING.md             # Contributor guidelines and development workflow
├── tox.ini                     # Multi-environment test configuration (py310, py311, py312, typecheck)
├── .github/workflows/ci.yml    # GitHub Actions automated matrix CI/CD workflow
├── main.py                     # CLI verification entrypoint (delegates to examples/quickstart_pipeline.py)
├── examples/
│   └── quickstart_pipeline.py  # End-to-end pipeline demonstration (50 features across 11 pillars)
├── data/
│   ├── __init__.py             # Packaged data generator distribution module
│   └── synthetic_generators.py # Reproducible enterprise People Analytics data generators
├── src/
│   └── people_analytics_toolkit/  # Core package distribution modules (PEP 517/518/621 src-layout)
│       ├── __init__.py            # Package root exposing public API and __all__
│       ├── _typing.py             # Backwards-compatibility typing fallbacks (<3.11)
│       ├── py.typed               # PEP 561 typing marker
│       ├── cardinality.py         # Bayesian Target Encoder & Grouped LOO Z-Scores
│       ├── memory.py              # Frac Diff, Weighted Weeks, EWMA/EWMS, Holt-Winters, Hazards
│       ├── chaos.py               # Shannon Entropy, GARCH(1,1), Hawkes Contagion, HMM Regimes, Bradford Factor
│       ├── anomalies.py           # Autoencoder, PyOD LOF, KL Divergence, DTW, LCSS/EDR
│       ├── attribution.py         # LMDI Decomposition, LOO-ED Alpha, SHAP Interactions & Thresholds
│       ├── mobility.py            # Career Movement Embeddings, Time-in-Position & Stagnation Index
│       ├── compensation.py        # Compa-Ratio, Range Penetration, & Pay Band Classification
│       ├── ona.py                 # Betweenness, Eigenvector, Burt Constraint, Euler Centrality, Overload, Contagion, Span of Control
│       ├── equity.py              # Oaxaca-Blinder, Demographic Parity, Equalized Odds, Individual Fairness
│       ├── prescriptive.py        # AIPW, Synthetic DiD, CausalForestDML, Qini Curve, AUUC, Doubly Robust MSE, Synthetic Control Weights, DAG Discovery
│       └── trajectories.py        # Continuous-Time Markov Chains, Meta-Path Walks, Metapath2Vec, Self-Attention
├── notebooks/
│   └── advanced_people_analytics_features.ipynb  # Interactive tutorial with 50 publication-grade charts
├── scripts/
│   └── build_notebook.py       # Notebook generator script (11 problem-driven parts)
└── tests/
    ├── conftest.py                             # Standardized session-scoped synthetic data fixtures (19 fixtures)
    ├── test_features.py                        # Mathematical unit and integration tests
    ├── test_package_imports.py                 # Package re-export and import integrity tests
    ├── test_polymorphism_and_properties.py     # Parametrized container polymorphism and Hypothesis invariant tests
    └── test_testing_gaps.py                    # Edge cases, fluent pipelines, and extended coverage tests
```
