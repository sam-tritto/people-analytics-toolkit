"""Synthetic data generators for Advanced People Analytics Feature Engineering.

Generates realistic workforce datasets for:
- High cardinality store-department hierarchies and localized overtime outliers.
- Longitudinal cohort time-series.
- Macro headcount time-series with memory and trends.
- Tiered recognition events.
- Schedule shifts and fragmented labor logs.
- Self-exciting absence point-process events.
- Multi-dimensional store operational profiles.
- Enterprise departmental role mixes.
- Seasonal labor demand curves.
- Multi-period enterprise turnover accounting data.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import networkx as nx


def generate_store_roster_data(
    n_employees: int = 2500,
    n_stores: int = 50,
    n_depts: int = 8,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate cross-sectional enterprise employee roster with hierarchical structure.
    
    Includes store-department combinations, tenure with hazard cliffs, overtime hours
    with localized outliers, and binary turnover events.
    """
    rng = np.random.default_rng(seed)
    
    # Stores with varying sizes (some new/small, some large flagship)
    store_weights = rng.dirichlet(np.ones(n_stores) * 0.8)
    store_ids = [f"STORE_{i+1:03d}" for i in range(n_stores)]
    assigned_stores = rng.choice(store_ids, size=n_employees, p=store_weights)
    
    dept_codes = [f"DEPT_{d+1:02d}" for d in range(n_depts)]
    raw_weights = np.exp(-0.2 * np.arange(n_depts))
    dept_weights = raw_weights / raw_weights.sum()
    assigned_depts = rng.choice(dept_codes, size=n_employees, p=dept_weights)
    
    # Non-linear tenure distribution with early career concentration
    # Mixture of Weibull distributions representing early turnover cliffs
    tenure_raw = np.where(
        rng.random(n_employees) < 0.45,
        rng.weibull(0.9, size=n_employees) * 16,     # Early-tenure cohort (< 6 months)
        rng.weibull(1.8, size=n_employees) * 65 + 16 # Established cohort
    )
    tenure_weeks = np.clip(np.round(tenure_raw), 1, 260).astype(int)
    
    # Overtime hours: baseline log-normal with deliberate outliers per cohort
    base_ot = rng.lognormal(mean=1.2, sigma=0.6, size=n_employees)
    base_ot = np.clip(base_ot, 0.0, 35.0)
    
    df = pd.DataFrame({
        "employee_id": [f"EMP_{i+1:05d}" for i in range(n_employees)],
        "store_id": assigned_stores,
        "dept_code": assigned_depts,
        "tenure_weeks": tenure_weeks,
        "weekly_overtime_hours": np.round(base_ot, 1),
    })
    
    df["store_dept"] = df["store_id"] + "_" + df["dept_code"]
    
    # Inject localized severe overtime outliers in small cohorts to test LOO Z-Score
    small_cohorts = df["store_dept"].value_counts()[lambda x: (x >= 4) & (x <= 10)].index.tolist()
    if small_cohorts:
        outlier_indices = []
        for cohort in small_cohorts[:5]:
            idx = df[df["store_dept"] == cohort].index[0]
            outlier_indices.append(idx)
        df.loc[outlier_indices, "weekly_overtime_hours"] = 42.5
    
    # Target variable: turnover event within next 90 days
    # Probability driven by store-department base rate, overtime burnout, and tenure hazard
    dept_risk = {f"DEPT_{d+1:02d}": 0.10 + 0.03 * d for d in range(n_depts)}
    store_risk = {s: rng.uniform(0.08, 0.32) for s in store_ids}
    
    logits = (
        np.array([store_risk[s] for s in df["store_id"]]) * 3.0
        + np.array([dept_risk[d] for d in df["dept_code"]]) * 2.0
        + (df["weekly_overtime_hours"] > 20).astype(float) * 0.8
        + np.exp(-df["tenure_weeks"] / 14.0) * 1.5  # 90-day cliff
        - 2.8
    )
    prob_turnover = 1.0 / (1.0 + np.exp(-logits))
    df["turnover_event"] = (rng.random(n_employees) < prob_turnover).astype(int)
    
    return df


def generate_cohort_timeseries_overtime(
    n_weeks: int = 52,
    cohort_size: int = 12,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate longitudinal overtime hours for a cohort and an exemplar associate.
    
    Demonstrates masking in standard Z-scores versus sensitivity in LOO Z-scores.
    """
    rng = np.random.default_rng(seed)
    records = []
    
    for t in range(n_weeks):
        # Baseline cohort behavior (mean ~ 5 hrs, std ~ 2 hrs, mild seasonal wave)
        cohort_base = 5.0 + 1.0 * np.sin(2 * np.pi * t / 52.0)
        cohort_ot = rng.normal(loc=cohort_base, scale=2.0, size=cohort_size - 1)
        cohort_ot = np.clip(cohort_ot, 0.0, 18.0)
        
        # Exemplar associate: realistic multi-wave project & crunch dynamics
        # Pops up during crunches, dips below cohort average during comp-time recovery, returns to baseline
        if t < 12:
            # Baseline phase: fluctuating closely around cohort mean
            exemplar_ot = rng.normal(loc=5.4, scale=1.5)
        elif t < 18:
            # Wave 1: Q1 project crunch spike (pops well above cohort)
            exemplar_ot = rng.normal(loc=19.5, scale=2.5)
        elif t < 25:
            # Recovery phase: compensatory time off (dips below cohort average)
            exemplar_ot = rng.normal(loc=2.8, scale=1.2)
        elif t < 32:
            # Wave 2: Mid-year moderate surge (pops moderately above cohort)
            exemplar_ot = rng.normal(loc=13.5, scale=2.0)
        elif t < 37:
            # Summer equilibrium: returns to normal baseline
            exemplar_ot = rng.normal(loc=5.0, scale=1.4)
        elif t < 46:
            # Wave 3: Severe Q4 operational staffing crisis (major sustained spike)
            exemplar_ot = rng.normal(loc=29.0, scale=3.5)
        else:
            # Year-end holiday cooldown
            exemplar_ot = rng.normal(loc=6.8, scale=1.8)
        
        exemplar_ot = max(0.0, exemplar_ot)
        
        records.append({
            "week": t + 1,
            "employee_id": "EMP_EXEMPLAR",
            "overtime_hours": round(float(exemplar_ot), 2),
            "is_exemplar": 1,
        })
        
        for i, val in enumerate(cohort_ot):
            records.append({
                "week": t + 1,
                "employee_id": f"EMP_COHORT_{i+1:02d}",
                "overtime_hours": round(float(val), 2),
                "is_exemplar": 0,
            })
            
    return pd.DataFrame(records)


def generate_headcount_timeseries(
    n_weeks: int = 156,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate long-memory, non-stationary enterprise headcount time-series.
    
    Has an overarching linear trend, annual seasonality, and persistent AR(1) memory shocks.
    Ideal for Fractional Differencing analysis.
    """
    rng = np.random.default_rng(seed)
    
    t = np.arange(n_weeks)
    trend = 1200 + 4.5 * t
    seasonality = 80 * np.sin(2 * np.pi * t / 52.0) - 40 * np.cos(4 * np.pi * t / 52.0)
    
    # Highly persistent autoregressive noise (rho = 0.96)
    noise = np.zeros(n_weeks)
    for i in range(1, n_weeks):
        noise[i] = 0.96 * noise[i - 1] + rng.normal(0, 15)
        
    headcount = trend + seasonality + noise
    
    dates = pd.date_range(start="2023-01-01", periods=n_weeks, freq="W-MON")
    return pd.DataFrame({
        "date": dates,
        "week_index": t,
        "headcount": np.round(headcount).astype(int),
    })


def generate_rewards_timeseries(
    n_weeks: int = 52,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate event history for recognition awards across associates.
    
    Tiered values: Gold ($100), Silver ($50), Bronze ($25).
    """
    rng = np.random.default_rng(seed)
    associates = [f"EMP_{i+1:03d}" for i in range(10)]
    records = []
    
    for emp in associates:
        # Each associate receives 1 to 4 awards randomly distributed
        n_awards = rng.integers(1, 5)
        weeks = np.sort(rng.choice(np.arange(1, n_weeks), size=n_awards, replace=False))
        tiers = rng.choice(["Gold", "Silver", "Bronze"], size=n_awards, p=[0.2, 0.35, 0.45])
        
        for w, tier in zip(weeks, tiers):
            amount = 100.0 if tier == "Gold" else (50.0 if tier == "Silver" else 25.0)
            records.append({
                "employee_id": emp,
                "award_week": int(w),
                "award_tier": tier,
                "award_amount": amount,
            })
            
    return pd.DataFrame(records).sort_values(["employee_id", "award_week"]).reset_index(drop=True)


def generate_labor_hours_traffic(
    n_weeks: int = 104,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate weekly labor hours with realistic multiple demand surges and volatility clustering.
    
    Demonstrates EWMA smoothing without ghost drops and fuels GARCH volatility modeling.
    """
    rng = np.random.default_rng(seed)
    
    t = np.arange(n_weeks)
    # Smooth baseline with gentle annual seasonality and expansion trend
    baseline = 1500.0 + 1.2 * t + 35.0 * np.sin(2 * np.pi * t / 52.0)
    
    # Multiple distinct operational demand surges that EWMA smooths out:
    # 1. Q1 annual warehouse audit surge (weeks 12-14, 64-66)
    # 2. Mid-year summer promotion / flash sale (weeks 26-28, 78-80)
    # 3. Fall logistics ramp-up (weeks 38-40, 90-92)
    # 4. Q4 peak holiday retail rush (weeks 48-51, 100-103)
    spikes = np.zeros(n_weeks)
    for yr in [0, 52]:
        spikes[yr + 12 : yr + 15] += 220.0
        spikes[yr + 26 : yr + 29] += 280.0
        spikes[yr + 38 : yr + 41] += 190.0
        spikes[yr + 48 : min(n_weeks, yr + 52)] += 340.0
        
    scheduled_hours = baseline + spikes
    
    # Realistic volatility clustering (tranquil periods alternating with turbulent shock regimes)
    volatility = np.ones(n_weeks) * 16.0
    volatility[20:34] = 65.0   # Summer staffing volatility regime
    volatility[72:86] = 75.0   # Year 2 operational supply disruption regime
    
    shocks = rng.normal(0, 1, size=n_weeks) * volatility
    observed_hours = scheduled_hours + shocks
    
    dates = pd.date_range(start="2024-01-01", periods=n_weeks, freq="W-MON")
    return pd.DataFrame({
        "date": dates,
        "week": np.arange(1, n_weeks + 1),
        "labor_hours": np.round(observed_hours, 1),
        "scheduled_hours": np.round(scheduled_hours, 1),
        "true_baseline": np.round(baseline, 1),
        "labor_shock": np.round(shocks, 1),
    })


def generate_shift_schedules(
    n_records: int = 600,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate shift logs for two departments: predictable vs chaotic.
    
    Dept A has standardized 8-hour shifts.
    Dept B has unpredictable durations (4h, 6h, 8h, 10h, split shifts).
    """
    rng = np.random.default_rng(seed)
    records = []
    
    # Dept A: Predictable schedules (90% standard 8h day, 10% standard 8h evening)
    for i in range(n_records // 2):
        shift_type = rng.choice(["Standard_Day_8h", "Standard_Eve_8h"], p=[0.75, 0.25])
        hours = 8.0
        records.append({
            "department": "Dept_Predictable",
            "shift_id": f"S_PRED_{i:04d}",
            "shift_type": shift_type,
            "duration_hours": hours,
            "split_shift": 0,
        })
        
    # Dept B: Chaotic schedules (erratic shift types and lengths)
    chaotic_types = ["Split_4h", "Short_4h", "Standard_8h", "Long_10h", "Weekend_Closer_6h", "Call_In_3h"]
    chaotic_probs = [0.18, 0.22, 0.25, 0.15, 0.12, 0.08]
    hours_map = {
        "Split_4h": 4.0, "Short_4h": 4.0, "Standard_8h": 8.0,
        "Long_10h": 10.0, "Weekend_Closer_6h": 6.0, "Call_In_3h": 3.0,
    }
    
    for i in range(n_records // 2):
        st = rng.choice(chaotic_types, p=chaotic_probs)
        records.append({
            "department": "Dept_Chaotic",
            "shift_id": f"S_CHAO_{i:04d}",
            "shift_type": st,
            "duration_hours": hours_map[st],
            "split_shift": 1 if "Split" in st else 0,
        })
        
    return pd.DataFrame(records)


def generate_callout_events(
    days: int = 120,
    seed: int = 42,
) -> List[float]:
    """Generate synthetic event arrival times for unscheduled absence call-outs.
    
    Simulates a self-exciting Hawkes process: when an event occurs,
    future events become temporarily more likely before decaying back to baseline rate.
    """
    rng = np.random.default_rng(seed)
    
    mu = 0.25      # Baseline call-outs per day
    alpha = 0.65   # Contagion shock magnitude per event
    beta = 0.85    # Decay rate of shock
    
    # Ogata thinning algorithm for Hawkes process simulation
    events: List[float] = []
    t = 0.0
    
    while t < days:
        # Upper bound on intensity
        current_intensity = mu + sum(alpha * np.exp(-beta * (t - s)) for s in events if t > s)
        lambda_bar = current_intensity + 0.1
        
        # Candidate inter-arrival time from homogeneous Poisson process
        u = rng.random()
        step = -np.log(u) / lambda_bar
        t += step
        if t >= days:
            break
            
        # Re-evaluate intensity at candidate time t
        lambda_t = mu + sum(alpha * np.exp(-beta * (t - s)) for s in events if t > s)
        
        # Accept/reject
        d = rng.random()
        if d <= (lambda_t / lambda_bar):
            events.append(round(t, 4))
            
    return events


def generate_store_profiles_multivariate(
    n_stores: int = 250,
    n_features: int = 20,
    n_anomalies: int = 15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Generate multi-dimensional store operational metrics.
    
    Normal stores adhere to a correlated latent manifold.
    Anomalous stores structurally break cross-metric relationships.
    """
    rng = np.random.default_rng(seed)
    
    feature_names = [
        "overtime_ratio", "unplanned_absenteeism", "turnover_rate", "tenure_median",
        "training_completion", "break_compliance", "customer_satisfaction", "shrink_rate",
        "sales_per_labor_hour", "manager_turnover", "promotion_rate", "eNPS_score",
        "schedule_adherence", "open_requisition_days", "grievance_rate", "cross_training_index",
        "peer_recognition_rate", "safety_incident_rate", "overstaffing_hours", "understaffing_hours"
    ]
    
    # Generate 4-dimensional latent factors
    latent = rng.normal(size=(n_stores, 4))
    # Factor loadings
    loadings = rng.normal(size=(4, n_features)) * 0.7
    
    # Baseline normal data
    data = latent @ loadings + rng.normal(scale=0.3, size=(n_stores, n_features))
    
    labels = np.zeros(n_stores, dtype=int)
    
    # Inject severe structural anomalies into the last n_anomalies stores
    # (e.g. High training completion coupled with high grievance and rocketing turnover)
    anomaly_indices = np.arange(n_stores - n_anomalies, n_stores)
    labels[anomaly_indices] = 1
    
    for idx in anomaly_indices:
        # Flip correlation relationships
        data[idx, 0] += rng.uniform(3.0, 5.0)  # Extreme overtime
        data[idx, 2] += rng.uniform(3.5, 6.0)  # Extreme turnover
        data[idx, 4] += rng.uniform(3.0, 4.5)  # Artificially high training (paper compliance)
        data[idx, 11] -= rng.uniform(4.0, 6.0) # Plummeting eNPS
        data[idx, 17] += rng.uniform(3.0, 5.0) # High safety incident rate
        
    df = pd.DataFrame(data, columns=feature_names)
    df.insert(0, "store_id", [f"STORE_{i+1:03d}" for i in range(n_stores)])
    
    return df, pd.Series(labels, name="is_anomalous")


def generate_role_distributions(seed: int = 42) -> Tuple[np.ndarray, Dict[str, np.ndarray], List[str]]:
    """Generate enterprise benchmark role distribution and individual store distributions.
    
    Used to calculate Kullback-Leibler (KL) Divergence drift.
    """
    rng = np.random.default_rng(seed)
    
    roles = ["Cashier", "Stocker", "Dept_Lead", "Specialist", "Asst_Manager", "Store_Manager"]
    # Corporate target allocation
    q_benchmark = np.array([0.40, 0.25, 0.15, 0.10, 0.07, 0.03])
    
    # Store 1: Compliant store (minor sampling noise)
    p_compliant = q_benchmark + rng.normal(0, 0.015, size=len(roles))
    p_compliant = np.clip(p_compliant, 0.01, 1.0)
    p_compliant /= p_compliant.sum()
    
    # Store 2: Top-heavy management drift (too many leads/managers, too few frontline)
    p_top_heavy = np.array([0.20, 0.15, 0.28, 0.18, 0.12, 0.07])
    p_top_heavy /= p_top_heavy.sum()
    
    # Store 3: Frontline hollowed-out drift (massive cashier proportion, zero specialists)
    p_hollowed = np.array([0.62, 0.24, 0.08, 0.01, 0.03, 0.02])
    p_hollowed /= p_hollowed.sum()
    
    stores_p = {
        "STORE_COMPLIANT": p_compliant,
        "STORE_TOP_HEAVY": p_top_heavy,
        "STORE_HOLLOWED_OUT": p_hollowed,
    }
    
    return q_benchmark, stores_p, roles


def generate_seasonal_curves(seed: int = 42) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Generate 52-week seasonal foot-traffic / staffing curves.
    
    Demonstrates phase shifts where Euclidean distance fails but DTW succeeds.
    """
    rng = np.random.default_rng(seed)
    weeks = np.arange(52)
    
    # Corporate benchmark: Spring surge at week 18, Summer peak at week 26, Holiday surge at week 48
    benchmark = (
        100
        + 25 * np.exp(-0.5 * ((weeks - 18) / 3.0) ** 2)
        + 40 * np.exp(-0.5 * ((weeks - 26) / 4.0) ** 2)
        + 55 * np.exp(-0.5 * ((weeks - 48) / 3.0) ** 2)
    )
    
    # Store A: In-sync store with minor random noise
    store_in_sync = benchmark + rng.normal(0, 2.5, size=52)
    
    # Store B: Southern market store with phase-shifted peak (Spring surge happens 4 weeks earlier)
    store_phase_shifted = (
        100
        + 27 * np.exp(-0.5 * ((weeks - 14) / 3.0) ** 2)
        + 38 * np.exp(-0.5 * ((weeks - 22) / 4.0) ** 2)
        + 52 * np.exp(-0.5 * ((weeks - 48) / 3.0) ** 2)
        + rng.normal(0, 2.0, size=52)
    )
    
    # Store C: Structurally anomalous curve (misses summer peak entirely, mid-year collapse)
    store_abnormal = (
        100
        + 20 * np.exp(-0.5 * ((weeks - 18) / 3.0) ** 2)
        - 30 * np.exp(-0.5 * ((weeks - 28) / 5.0) ** 2)
        + 50 * np.exp(-0.5 * ((weeks - 48) / 3.0) ** 2)
        + rng.normal(0, 3.0, size=52)
    )
    
    stores = {
        "STORE_IN_SYNC": store_in_sync,
        "STORE_PHASE_SHIFTED": store_phase_shifted,
        "STORE_ABNORMAL": store_abnormal,
    }
    
    return benchmark, stores


def generate_enterprise_turnover_breakdown(seed: int = 42) -> pd.DataFrame:
    """Generate Year-over-Year headcount and turnover by department for LMDI.
    
    Designed to highlight the contrast between Rate Effect and Structural Mix Effect.
    Total turnover rate dropped, but LMDI proves how much was real retention improvement
    versus departmental headcount reallocation.
    """
    rng = np.random.default_rng(seed)
    
    departments = [
        "Retail_Store_Operations",
        "Supply_Chain_Logistics",
        "Customer_Care_Contact_Center",
        "Corporate_Headquarters",
        "Digital_ECommerce",
        "Information_Technology",
        "Merchandising",
        "Human_Resources",
    ]
    
    # Year 0: Contact Center and Retail have large headcount and high turnover
    h_y0 = np.array([5200, 3100, 2400, 850, 600, 950, 500, 400])
    r_y0 = np.array([0.34, 0.28, 0.45, 0.12, 0.16, 0.15, 0.10, 0.11])
    
    # Year 1: High turnover department (Contact Center) downsized by 35% via automation;
    # Corporate & Tech grew. Some department rates improved slightly.
    h_y1 = np.array([5100, 3050, 1550, 920, 850, 1200, 530, 420])
    r_y1 = np.array([0.32, 0.26, 0.44, 0.11, 0.15, 0.14, 0.10, 0.10])
    
    df = pd.DataFrame({
        "department": departments,
        "headcount_y0": h_y0,
        "turnover_rate_y0": r_y0,
        "headcount_y1": h_y1,
        "turnover_rate_y1": r_y1,
    })
    
    df["turnover_count_y0"] = np.round(df["headcount_y0"] * df["turnover_rate_y0"]).astype(int)
    df["turnover_count_y1"] = np.round(df["headcount_y1"] * df["turnover_rate_y1"]).astype(int)
    
    return df


def generate_shift_manager_performance(
    n_shifts: int = 400,
    n_managers: int = 8,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate shift-level operational data with assigned managers to evaluate LOO-ED.
    
    Contains store foot traffic, scheduled labor, weekend indicator, and observed throughput.
    Managers possess true latent marginal value (manager alpha) that gets disentangled from
    systemic shift-level advantages.
    """
    rng = np.random.default_rng(seed)
    
    manager_ids = [f"MGR_{i+1:02d}" for i in range(n_managers)]
    # True latent marginal impact on throughput (units/hour)
    true_alpha = {
        "MGR_01": +14.5,  # Top performer
        "MGR_02": +7.2,
        "MGR_03": +3.0,
        "MGR_04": 0.0,
        "MGR_05": -2.5,
        "MGR_06": -6.0,
        "MGR_07": -12.0,  # Negative drag
        "MGR_08": +5.5,
    }
    
    stores = ["STORE_DOWNTOWN", "STORE_SUBURB_NORTH", "STORE_AIRPORT", "STORE_MALL"]
    store_base = {"STORE_DOWNTOWN": 180.0, "STORE_SUBURB_NORTH": 140.0, "STORE_AIRPORT": 210.0, "STORE_MALL": 160.0}
    
    records = []
    for s_idx in range(n_shifts):
        store = rng.choice(stores)
        is_weekend = int(rng.random() < 0.28)
        foot_traffic = rng.normal(550, 60) + (180 if is_weekend else 0) + (50 if "AIRPORT" in store else 0)
        staff_hours = rng.normal(45, 4) + (10 if is_weekend else 0)
        
        # MGR_07 unfortunately often gets assigned to high-volume airport weekend shifts,
        # which masks their poor performance in raw numbers!
        if is_weekend and "AIRPORT" in store and rng.random() < 0.6:
            mgr = "MGR_07"
        elif not is_weekend and "SUBURB" in store and rng.random() < 0.5:
            mgr = "MGR_01"  # MGR_01 gets quiet suburban shifts, making raw numbers look modest!
        else:
            mgr = rng.choice(manager_ids)
            
        expected_throughput = store_base[store] + 0.15 * foot_traffic + 1.2 * staff_hours + (25.0 if is_weekend else 0)
        observed_throughput = expected_throughput + true_alpha[mgr] + rng.normal(0, 8.0)
        
        records.append({
            "shift_id": f"SHIFT_{s_idx+1:04d}",
            "store_id": store,
            "manager_id": mgr,
            "is_weekend": is_weekend,
            "foot_traffic": round(float(foot_traffic), 1),
            "staff_hours": round(float(staff_hours), 1),
            "observed_throughput": round(float(observed_throughput), 1),
            "true_manager_alpha": true_alpha[mgr],
        })
        
    return pd.DataFrame(records)


def generate_compensation_velocity_data(
    n_employees: int = 1200,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Generate employee career velocity records for Local Outlier Factor (LOF).
    
    Includes tenure, job level, performance rating, promotion speed, and salary growth rate.
    Injects subtle local density anomalies (e.g. associates receiving extreme raises compared
    to peers within their exact tenure/rating pocket, while looking unremarkable globally).
    """
    rng = np.random.default_rng(seed)
    
    tenure_years = rng.uniform(0.5, 10.0, size=n_employees)
    ratings = rng.choice([2, 3, 4, 5], p=[0.10, 0.55, 0.25, 0.10], size=n_employees)
    
    # Baseline expected annual salary growth rate (%)
    base_growth = 2.5 + (ratings - 2) * 2.2 - 0.25 * tenure_years
    base_growth = np.clip(base_growth, 1.0, 16.0)
    salary_growth = base_growth + rng.normal(0, 1.2, size=n_employees)
    
    # Promotion velocity (promotions per year of tenure)
    promo_velocity = (ratings / 5.0) * 0.45 + rng.normal(0, 0.08, size=n_employees)
    promo_velocity = np.clip(promo_velocity, 0.0, 1.2)
    
    df = pd.DataFrame({
        "employee_id": [f"EMP_{i+1:05d}" for i in range(n_employees)],
        "tenure_years": np.round(tenure_years, 2),
        "performance_rating": ratings,
        "annual_salary_growth_pct": np.round(salary_growth, 2),
        "promotion_velocity": np.round(promo_velocity, 3),
    })
    
    labels = np.zeros(n_employees, dtype=int)
    # Inject 25 local density anomalies: rating=3 associates with 8-year tenure receiving 14% raises!
    # In a global distribution, 14% raise exists (common for new high-performing hires),
    # but in the local neighborhood of 8-year Rating 3 associates (where average is 3%), it is an extreme local anomaly.
    local_target = df[(df["performance_rating"] == 3) & (df["tenure_years"] > 6.0)].index
    inject_idx = rng.choice(local_target, size=min(25, len(local_target)), replace=False)
    
    df.loc[inject_idx, "annual_salary_growth_pct"] += 9.5
    df.loc[inject_idx, "promotion_velocity"] += 0.45
    labels[inject_idx] = 1
    
    return df, pd.Series(labels, name="is_local_anomaly")


def generate_regime_workforce_timeseries(
    n_weeks: int = 104,
    seed: int = 42,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Generate 2-year weekly workforce metrics driven by Hidden Markov Model (HMM) latent regimes.
    
    Regimes:
        State 0: Normal Stable Operations (overtime ~ 4h, call-out rate ~ 2%)
        State 1: Planned Holiday Surge (overtime ~ 18h, call-out rate ~ 3%, high scheduled throughput)
        State 2: Understaffed Attrition Spiral (overtime ~ 19h, call-out rate ~ 9%, high volatility)
    """
    rng = np.random.default_rng(seed)
    
    # Latent state transition sequence
    states = np.zeros(n_weeks, dtype=int)
    current_state = 0
    
    # Transition matrix
    trans_matrix = np.array([
        [0.90, 0.07, 0.03],  # From Normal
        [0.15, 0.80, 0.05],  # From Holiday Surge
        [0.05, 0.02, 0.93],  # From Attrition Spiral (sticky crisis)
    ])
    
    for t in range(1, n_weeks):
        # Force a planned holiday surge around weeks 44-50
        if 44 <= t <= 50:
            current_state = 1
        elif 75 <= t <= 90:
            current_state = 2  # Attrition crisis
        else:
            current_state = rng.choice([0, 1, 2], p=trans_matrix[current_state])
        states[t] = current_state
        
    ot_means = [4.0, 18.0, 19.5]
    ot_stds = [1.2, 2.5, 4.0]
    
    callout_means = [0.02, 0.03, 0.095]
    callout_stds = [0.005, 0.008, 0.02]
    
    ot_series = np.array([rng.normal(ot_means[s], ot_stds[s]) for s in states])
    callout_series = np.array([rng.normal(callout_means[s], callout_stds[s]) for s in states])
    
    df = pd.DataFrame({
        "week": np.arange(1, n_weeks + 1),
        "weekly_overtime_hours": np.clip(np.round(ot_series, 1), 0.0, 45.0),
        "absence_callout_rate": np.clip(np.round(callout_series, 3), 0.001, 0.25),
        "true_latent_state": states,
    })
    
    return df, states


def generate_career_shift_sequences(
    n_weeks: int = 52,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Generate 52-week shift patterns comparing standard peers vs associate with vacation gaps.
    
    Used to demonstrate EDR and LCSS tolerance to gaps compared to DTW failure.
    """
    rng = np.random.default_rng(seed)
    
    weeks = np.arange(n_weeks)
    # Peer standard seasonal shift pattern: 38 hrs baseline, ramping up to 48 hrs in Q4
    baseline = 38 + 10 * np.exp(-0.5 * ((weeks - 46) / 4.0) ** 2)
    
    peer_sequence = baseline + rng.normal(0, 1.0, size=n_weeks)
    
    # Associate A: Same pattern, but took 2 weeks of unpaid leave in summer (weeks 24 and 25 are 0 hrs)
    assoc_gap = peer_sequence.copy()
    assoc_gap[24:26] = 0.0
    
    # Associate B: Completely different, erratic split pattern
    assoc_erratic = 25 + 15 * np.sin(2 * np.pi * weeks / 8.0) + rng.normal(0, 2.0, size=n_weeks)
    
    return {
        "benchmark_peer": np.round(peer_sequence, 1),
        "associate_vacation_gap": np.round(assoc_gap, 1),
        "associate_erratic_schedule": np.round(assoc_erratic, 1),
    }


def generate_safety_incident_dataset(
    n_samples: int = 2500,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate workforce safety and operational incident risk dataset for SHAP Interactions.
    
    Crucial feature: Severe non-linear 2D interaction between Associate Early Tenure (< 90 days)
    and Manager Inexperience (< 6 months tenure), which drastically spikes incident probability.
    """
    rng = np.random.default_rng(seed)
    
    assoc_tenure_weeks = rng.weibull(1.2, size=n_samples) * 35 + 2
    mgr_tenure_months = rng.gamma(2.0, 8.0, size=n_samples) + 1
    weekly_overtime = rng.lognormal(mean=1.5, sigma=0.6, size=n_samples)
    safety_training_score = rng.uniform(50, 100, size=n_samples)
    shift_chaos_index = rng.uniform(0.2, 2.2, size=n_samples)
    
    # Non-linear interaction risk formulation
    # Base risk:
    log_odds = (
        -3.5
        + 0.04 * weekly_overtime
        - 0.03 * (safety_training_score - 70)
        + 0.5 * (shift_chaos_index - 1.0)
    )
    
    # Main effects for tenure:
    is_new_assoc = (assoc_tenure_weeks < 13).astype(float)
    is_new_mgr = (mgr_tenure_months < 6).astype(float)
    
    log_odds += 0.4 * is_new_assoc
    log_odds += 0.3 * is_new_mgr
    
    # POWERFUL 2D NON-LINEAR INTERACTION:
    # When both associate and manager are inexperienced on the same shift, risk skyrockets!
    interaction_synergy = is_new_assoc * is_new_mgr * 2.4
    log_odds += interaction_synergy
    
    prob = 1.0 / (1.0 + np.exp(-log_odds))
    incidents = (rng.random(n_samples) < prob).astype(int)
    
    df = pd.DataFrame({
        "assoc_tenure_weeks": np.round(assoc_tenure_weeks, 1),
        "mgr_tenure_months": np.round(mgr_tenure_months, 1),
        "weekly_overtime": np.round(weekly_overtime, 1),
        "safety_training_score": np.round(safety_training_score, 1),
        "shift_chaos_index": np.round(shift_chaos_index, 2),
        "safety_incident_event": incidents,
    })
    
    return df


def generate_career_movement_dataset(
    n_sequences: int = 500,
    seed: int = 42,
) -> Tuple[List[List[str]], pd.DataFrame]:
    """Generate organizational career progression histories and succession candidates.
    
    Transitions model realistic pathways across 10 retail/operations roles.
    Includes traditional feeder pathways and lateral skill-building pathways.
    """
    rng = np.random.default_rng(seed)
    
    roles = [
        "Cashier",
        "Stocker",
        "Sales_Associate",
        "Customer_Service_Specialist",
        "Inventory_Control_Specialist",
        "Merchandising_Lead",
        "Operations_Lead",
        "Department_Supervisor",
        "Assistant_Store_Manager",
        "Store_Manager",
    ]
    
    # Transition probability graph
    # Key = current role, Value = list of (next_role, probability)
    transition_map = {
        "Cashier": [("Sales_Associate", 0.5), ("Customer_Service_Specialist", 0.35), ("Cashier", 0.15)],
        "Stocker": [("Inventory_Control_Specialist", 0.45), ("Operations_Lead", 0.35), ("Merchandising_Lead", 0.20)],
        "Sales_Associate": [("Customer_Service_Specialist", 0.40), ("Merchandising_Lead", 0.45), ("Department_Supervisor", 0.15)],
        "Customer_Service_Specialist": [("Department_Supervisor", 0.55), ("Operations_Lead", 0.30), ("Merchandising_Lead", 0.15)],
        "Inventory_Control_Specialist": [("Operations_Lead", 0.65), ("Department_Supervisor", 0.25), ("Merchandising_Lead", 0.10)],
        "Merchandising_Lead": [("Department_Supervisor", 0.60), ("Assistant_Store_Manager", 0.30), ("Operations_Lead", 0.10)],
        "Operations_Lead": [("Department_Supervisor", 0.55), ("Assistant_Store_Manager", 0.35), ("Inventory_Control_Specialist", 0.10)],
        "Department_Supervisor": [("Assistant_Store_Manager", 0.80), ("Department_Supervisor", 0.20)],
        "Assistant_Store_Manager": [("Store_Manager", 0.85), ("Assistant_Store_Manager", 0.15)],
        "Store_Manager": [("Store_Manager", 1.0)],
    }
    
    sequences = []
    starter_roles = ["Cashier", "Stocker", "Sales_Associate"]
    
    for _ in range(n_sequences):
        curr = rng.choice(starter_roles)
        seq = [curr]
        path_len = rng.integers(2, 6)
        for _ in range(path_len):
            options, probs = zip(*transition_map[curr])
            nxt = rng.choice(options, p=probs)
            seq.append(nxt)
            curr = nxt
            if curr == "Store_Manager":
                break
        sequences.append(seq)
        
    # Generate a pool of 20 current candidate associates eligible for Department_Supervisor succession
    candidates = []
    candidate_profiles = [
        ("EMP_101", ["Cashier", "Customer_Service_Specialist", "Department_Supervisor"], "Department_Supervisor"),
        ("EMP_102", ["Stocker", "Inventory_Control_Specialist", "Operations_Lead"], "Operations_Lead"),
        ("EMP_103", ["Sales_Associate", "Merchandising_Lead"], "Merchandising_Lead"),
        ("EMP_104", ["Stocker", "Inventory_Control_Specialist"], "Inventory_Control_Specialist"),
        ("EMP_105", ["Cashier", "Customer_Service_Specialist"], "Customer_Service_Specialist"),
        ("EMP_106", ["Sales_Associate", "Cashier"], "Cashier"),
        ("EMP_107", ["Stocker", "Operations_Lead"], "Operations_Lead"),
        ("EMP_108", ["Cashier", "Sales_Associate", "Merchandising_Lead"], "Merchandising_Lead"),
    ]
    
    for emp_id, hist, curr in candidate_profiles:
        candidates.append({
            "employee_id": emp_id,
            "current_role": curr,
            "career_history": hist,
        })
        
    candidate_df = pd.DataFrame(candidates)
    return sequences, candidate_df


def generate_turnover_inflection_dataset(
    n_samples: int = 1800,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Generate realistic employee operational dataset for SHAP zero-crossing inflection analysis.
    
    Contains features with distinct non-linear tipping points where retention flips to turnover hazard:
    - weekly_overtime_hours: Tipping point ~ 15.0 hours (fatigue threshold)
    - comp_ratio: Tipping point ~ 0.93 (underpaid market dissatisfaction)
    - commute_minutes: Tipping point ~ 30-34 minutes (commute burnout)
    - manager_1on1_frequency: Tipping point ~ 1.8 monthly check-ins (manager detachment)
    - peer_turnover_rate: Tipping point ~ 15% (contagion tipping point)
    """
    rng = np.random.default_rng(seed)
    
    weekly_overtime = rng.gamma(2.0, 5.0, size=n_samples)  # mean ~10, range 0-40
    comp_ratio = rng.normal(0.96, 0.12, size=n_samples)     # market ratio 0.70-1.30
    # Commute calibrated so comfortable commutes are 10-25 mins, and long commutes escalate above 30 mins
    commute_minutes = np.clip(rng.normal(25.0, 8.5, size=n_samples) + rng.exponential(6.0, size=n_samples), 8.0, 85.0)
    manager_1on1 = rng.poisson(lam=2.2, size=n_samples) + rng.uniform(0, 0.9, size=n_samples)
    peer_turnover_rate = rng.beta(2, 8, size=n_samples) * 0.45  # 0 to 40%
    
    # Ground truth non-linear log-odds of voluntary termination
    log_odds = (
        -2.2
        + 0.22 * (weekly_overtime - 15.0)    # Shifts positive when OT > 15
        - 4.5 * (comp_ratio - 0.93)          # Protective when comp_ratio > 0.93
        + 0.12 * (commute_minutes - 30.0)    # Tipping point right around 30-34 minutes!
        - 0.9 * (manager_1on1 - 1.8)         # Protective when 1on1 > 1.8
        + 8.0 * (peer_turnover_rate - 0.15)  # Contagion spike above 15%
    )
    
    prob = 1.0 / (1.0 + np.exp(-log_odds))
    is_terminated = (rng.random(n_samples) < prob).astype(int)
    
    df = pd.DataFrame({
        "weekly_overtime_hours": np.round(weekly_overtime, 1),
        "comp_ratio": np.round(comp_ratio, 3),
        "commute_minutes": np.round(commute_minutes, 1),
        "manager_1on1_frequency": np.round(manager_1on1, 1),
        "peer_turnover_rate": np.round(peer_turnover_rate, 3),
    })
    
    return df, pd.Series(is_terminated, name="is_terminated")


def generate_compensation_equity_data(
    n_employees: int = 1200,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate structured workforce salary and compensation grade data.
    
    Creates job grades (IC1 to IC5), established salary band parameters (min, mid, max),
    and employee base salaries with natural tenure and performance progression,
    including edge-case green-circled (<min) and red-circled (>max) employees.
    """
    rng = np.random.default_rng(seed)
    
    grades = ["IC1", "IC2", "IC3", "IC4", "IC5"]
    departments = ["Engineering", "Product", "Operations", "Sales", "Customer_Success"]
    
    band_defs = {
        "IC1": (50000, 62500, 75000),
        "IC2": (68000, 85000, 102000),
        "IC3": (95000, 120000, 145000),
        "IC4": (135000, 170000, 205000),
        "IC5": (185000, 235000, 285000),
    }
    
    records = []
    for i in range(n_employees):
        emp_id = f"EMP_{i+1:05d}"
        grade = rng.choice(grades, p=[0.20, 0.30, 0.28, 0.16, 0.06])
        dept = rng.choice(departments)
        tenure = rng.uniform(0.5, 9.0)
        perf = rng.choice([2, 3, 4, 5], p=[0.10, 0.50, 0.30, 0.10])
        
        b_min, b_mid, b_max = band_defs[grade]
        spread = b_max - b_min
        
        target_rp = 0.15 + (tenure / 8.0) * 0.45 + (perf - 3) * 0.15 + rng.normal(0, 0.08)
        
        # Inject occasional green-circled (<0) or red-circled (>1)
        r_val = rng.random()
        if r_val < 0.03:
            target_rp = rng.uniform(-0.15, -0.01)
        elif r_val < 0.07:
            target_rp = rng.uniform(1.02, 1.18)
            
        salary = b_min + target_rp * spread
        
        records.append({
            "employee_id": emp_id,
            "department": dept,
            "job_grade": grade,
            "job_level": grade,
            "tenure_years": round(tenure, 2),
            "performance_rating": perf,
            "band_min": b_min,
            "band_mid": b_mid,
            "band_midpoint": b_mid,
            "band_max": b_max,
            "base_salary": round(salary, 2),
        })
        
    return pd.DataFrame(records)


def generate_seasonal_workforce_timeseries(
    n_weeks: int = 156,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate 3-year weekly workforce operational timeseries with growth trend and annual seasonality.
    
    Models multi-year expansion alongside summer retail surges and Q4 holiday peaks for Holt-Winters modeling.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_weeks)
    
    trend = 1200.0 + 3.2 * t
    annual_cycle = 2 * np.pi * t / 52.0
    seasonality = (
        120.0 * np.sin(annual_cycle - np.pi / 2.0)
        + 85.0 * np.sin(2.0 * annual_cycle)
    )
    
    noise = rng.normal(0, 25.0, size=n_weeks)
    observed = trend + seasonality + noise
    
    return pd.DataFrame({
        "week": t + 1,
        "observed_labor_demand": np.round(observed, 1),
        "true_trend": np.round(trend, 1),
        "true_seasonality": np.round(seasonality, 1),
    })


def generate_organizational_network_data(
    n_employees: int = 120,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, nx.Graph]:
    """Generate multi-department organizational collaboration network data.
    
    Creates employee nodes across 5 departments (Engineering, Product, Operations,
    Sales, People/HR) and a digital interaction graph (meetings, emails, Slack messages).
    Injects cross-functional brokers, high-degree informal hubs, and collaboration-overloaded
    bottlenecks.
    
    Returns
    -------
    nodes_df : pd.DataFrame
        Columns: [employee_id, department, job_level, performance_rating, intended_archetype]
    edges_df : pd.DataFrame
        Columns: [source, target, meeting_hours, email_count, slack_messages, weight]
    G : nx.Graph
        Pre-built NetworkX weighted collaboration graph.
    """
    rng = np.random.default_rng(seed)
    
    departments = ["Engineering", "Product", "Operations", "Sales", "HR"]
    dept_weights = [0.35, 0.20, 0.20, 0.15, 0.10]
    
    # 1. Create nodes
    emp_ids = [f"EMP_{i+1:04d}" for i in range(n_employees)]
    assigned_depts = rng.choice(departments, p=dept_weights, size=n_employees)
    job_levels = rng.choice(["IC1", "IC2", "IC3", "IC4", "Lead", "Manager"], p=[0.20, 0.30, 0.25, 0.12, 0.08, 0.05], size=n_employees)
    ratings = rng.choice([2, 3, 4, 5], p=[0.10, 0.55, 0.25, 0.10], size=n_employees)
    
    nodes_df = pd.DataFrame({
        "employee_id": emp_ids,
        "department": assigned_depts,
        "job_level": job_levels,
        "performance_rating": ratings,
        "intended_archetype": "General",
    })
    
    # Designate specialized ground-truth test nodes
    # Brokers: span cross-departmental holes
    n_brokers = max(2, int(0.04 * n_employees))
    broker_indices = [int(i * (n_employees - 1) / (n_brokers + 1)) for i in range(1, n_brokers + 1)]
    for idx in broker_indices:
        nodes_df.loc[idx, "intended_archetype"] = "Broker"
        
    # Informal Hubs: core connectors inside departments
    n_hubs = max(2, int(0.04 * n_employees))
    hub_indices = [int((i + 0.5) * (n_employees - 1) / (n_hubs + 1)) for i in range(n_hubs)]
    for idx in hub_indices:
        nodes_df.loc[idx, "intended_archetype"] = "Hub"
        
    # Overloaded Bottlenecks: extreme inbound volume
    n_overloaded = max(1, int(0.02 * n_employees))
    overloaded_indices = [int((i + 0.25) * (n_employees - 1) / (n_overloaded + 1)) for i in range(n_overloaded)]
    for idx in overloaded_indices:
        nodes_df.loc[idx, "intended_archetype"] = "Overloaded"
        
    # 2. Build graph edges
    G = nx.Graph()
    for _, row in nodes_df.iterrows():
        G.add_node(row["employee_id"], department=row["department"], job_level=row["job_level"])
        
    edges = []
    
    # Intra-department edges (dense clusters)
    for dept in departments:
        dept_nodes = nodes_df[nodes_df["department"] == dept]["employee_id"].tolist()
        k_dept = len(dept_nodes)
        for i in range(k_dept):
            for j in range(i + 1, k_dept):
                u, v = dept_nodes[i], dept_nodes[j]
                # Hub nodes connect to almost everyone in their department
                u_is_hub = u in [emp_ids[idx] for idx in hub_indices]
                v_is_hub = v in [emp_ids[idx] for idx in hub_indices]
                
                prob = 0.70 if (u_is_hub or v_is_hub) else 0.25
                if rng.random() < prob:
                    meetings = rng.uniform(2.0, 14.0)
                    emails = rng.integers(10, 90)
                    slack = rng.integers(25, 250)
                    weight = round(0.3 * meetings + 0.05 * emails + 0.02 * slack, 2)
                    G.add_edge(u, v, weight=weight, meetings=meetings, emails=emails, slack=slack)
                    edges.append({
                        "source": u, "target": v,
                        "meeting_hours": round(meetings, 1),
                        "email_count": emails,
                        "slack_messages": slack,
                        "weight": weight,
                    })
                    
    # Cross-department edges (sparse, concentrated through brokers)
    broker_ids = [emp_ids[idx] for idx in broker_indices]
    for b_id in broker_ids:
        b_dept = nodes_df.loc[nodes_df["employee_id"] == b_id, "department"].iloc[0]
        other_depts = [d for d in departments if d != b_dept]
        for od in other_depts:
            target_peers = nodes_df[nodes_df["department"] == od]["employee_id"].tolist()
            chosen_peers = rng.choice(target_peers, size=min(3, len(target_peers)), replace=False)
            for p_id in chosen_peers:
                if not G.has_edge(b_id, p_id):
                    meetings = rng.uniform(4.0, 16.0)
                    emails = rng.integers(20, 110)
                    slack = rng.integers(40, 300)
                    weight = round(0.35 * meetings + 0.06 * emails + 0.025 * slack, 2)
                    G.add_edge(b_id, p_id, weight=weight, meetings=meetings, emails=emails, slack=slack)
                    edges.append({
                        "source": b_id, "target": p_id,
                        "meeting_hours": round(meetings, 1),
                        "email_count": emails,
                        "slack_messages": slack,
                        "weight": weight,
                    })
                    
    # Random sporadic cross-department edges (weak ties)
    for _ in range(35):
        u, v = rng.choice(emp_ids, size=2, replace=False)
        u_dept = nodes_df.loc[nodes_df["employee_id"] == u, "department"].iloc[0]
        v_dept = nodes_df.loc[nodes_df["employee_id"] == v, "department"].iloc[0]
        if u_dept != v_dept and not G.has_edge(u, v):
            meetings = rng.uniform(1.0, 4.0)
            emails = rng.integers(5, 25)
            slack = rng.integers(5, 50)
            weight = round(0.2 * meetings + 0.03 * emails + 0.01 * slack, 2)
            G.add_edge(u, v, weight=weight, meetings=meetings, emails=emails, slack=slack)
            edges.append({
                "source": u, "target": v,
                "meeting_hours": round(meetings, 1),
                "email_count": emails,
                "slack_messages": slack,
                "weight": weight,
            })
            
    # Inject heavy collaboration overload on bottleneck nodes
    for o_idx in overloaded_indices:
        o_id = emp_ids[o_idx]
        avail = [eid for eid in emp_ids if eid != o_id]
        targets = rng.choice(avail, size=min(18, len(avail)), replace=False)
        for t_id in targets:
            if not G.has_edge(o_id, t_id):
                meetings = rng.uniform(8.0, 22.0)
                emails = rng.integers(50, 180)
                slack = rng.integers(100, 500)
                weight = round(0.4 * meetings + 0.08 * emails + 0.03 * slack, 2)
                G.add_edge(o_id, t_id, weight=weight, meetings=meetings, emails=emails, slack=slack)
                edges.append({
                    "source": o_id, "target": t_id,
                    "meeting_hours": round(meetings, 1),
                    "email_count": emails,
                    "slack_messages": slack,
                    "weight": weight,
                })
                
    edges_df = pd.DataFrame(edges)
    return nodes_df, edges_df, G


def generate_algorithmic_equity_and_fairness_data(
    n_employees: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate workforce data for Oaxaca-Blinder pay decomposition and fair ML classification.

    Includes demographic attributes (majority/minority status), observable human capital
    credentials (tenure, education, job level, performance ratings, certifications),
    continuous compensation with known structural bias, and promotion eligibility labels.

    Parameters:
        n_employees: Number of employee records to generate.
        seed: Random seed for reproducibility.

    Returns:
        pd.DataFrame containing demographic indicators, qualifications, salary, and promotion labels.
    """
    rng = np.random.default_rng(seed)

    # Demographic indicators (1 = Majority Group A, 0 = Minority Group B)
    group_a = rng.choice([1, 0], p=[0.55, 0.45], size=n_employees)

    # Human capital and job characteristics
    # Notice: Group A has slight differences in mean tenure/level, but also experiences structural wage premium
    tenure_years = np.clip(rng.gamma(2.5, 2.0, size=n_employees) + group_a * 0.8, 0.5, 18.0)
    education_years = rng.choice([2, 4, 6, 8], p=[0.20, 0.50, 0.20, 0.10], size=n_employees)
    job_level = rng.choice([1, 2, 3, 4, 5], p=[0.25, 0.30, 0.25, 0.14, 0.06], size=n_employees)
    perf_rating = rng.choice([2, 3, 4, 5], p=[0.10, 0.55, 0.25, 0.10], size=n_employees)
    certifications = rng.poisson(lam=1.2, size=n_employees)
    hours_worked = rng.normal(40.0, 3.5, size=n_employees) + (job_level * 0.8)

    # True wage generating process:
    # Baseline human capital returns:
    explained_salary = (
        48000.0
        + 3800.0 * tenure_years
        + 4200.0 * (education_years - 2)
        + 14000.0 * (job_level - 1)
        + 3200.0 * (perf_rating - 3)
        + 1800.0 * certifications
    )

    # Injected structural bias: Majority Group A receives an unexplained structural premium
    structural_premium = 6800.0 * group_a
    noise = rng.normal(0, 2800.0, size=n_employees)

    base_salary = np.round(explained_salary + structural_premium + noise, 2)

    # Promotion recommendation process:
    # Ground truth qualification latent index
    qualification_index = (
        -3.2
        + 0.25 * tenure_years
        + 0.30 * (education_years - 2)
        + 0.55 * (job_level - 1)
        + 0.85 * (perf_rating - 3)
        + 0.40 * certifications
    )

    # Historical selection process with systemic demographic disparity:
    # In unconstrained historical decisions, Group A was favored by +0.8 in log-odds
    historical_log_odds = qualification_index + 0.80 * group_a
    selection_prob = 1.0 / (1.0 + np.exp(-historical_log_odds))
    promotion_label = (rng.random(n_employees) < selection_prob).astype(int)

    # True merit qualification label (objective, group-blind qualification)
    objective_prob = 1.0 / (1.0 + np.exp(-qualification_index))
    true_merit_label = (rng.random(n_employees) < objective_prob).astype(int)

    df = pd.DataFrame({
        "employee_id": [f"EMP_{i+1:05d}" for i in range(n_employees)],
        "demographic_group": group_a,
        "tenure_years": np.round(tenure_years, 2),
        "education_years": education_years,
        "job_level": job_level,
        "performance_rating": perf_rating,
        "certifications": certifications,
        "hours_worked": np.round(hours_worked, 1),
        "base_salary": base_salary,
        "promotion_recommendation": promotion_label,
        "true_merit_qualification": true_merit_label,
    })

    return df


def generate_causal_and_uplift_data(
    n_employees: int = 1200,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate observational HR cohort with confounding, treatment assignment, and heterogeneous uplift.

    Simulates the evaluation of a targeted retention bonus or leadership accelerator intervention
    where assignment is observational (confounded by flight risk and workload) and treatment effect
    varies strongly across human capital dimensions (persuadables vs lost causes).
    """
    rng = np.random.default_rng(seed)

    tenure_years = rng.uniform(0.5, 9.5, size=n_employees)
    prior_performance = np.clip(rng.normal(3.3, 0.6, size=n_employees), 1.0, 5.0)
    flight_risk = rng.beta(2.5, 4.0, size=n_employees)
    overtime_hours = rng.exponential(14.0, size=n_employees) + 4.0
    manager_quality = np.clip(rng.normal(3.4, 0.7, size=n_employees), 1.0, 5.0)

    # Confounded treatment assignment: managers offer retention bonus to high flight risk and high overtime
    confounding_index = (
        -1.5
        + 0.22 * tenure_years
        + 2.1 * flight_risk
        + 0.035 * overtime_hours
        - 0.35 * manager_quality
    )
    propensity_true = 1.0 / (1.0 + np.exp(-confounding_index))
    treatment = (rng.random(n_employees) < propensity_true).astype(int)

    # Heterogeneous Treatment Effect (CATE):
    # Uplift is highest for high flight-risk employees with moderate tenure (persuadables)
    true_cate = (
        0.08
        + 0.40 * flight_risk
        - 0.03 * tenure_years
        + 0.05 * (prior_performance - 3.0)
    )

    # Baseline counterfactual retention without intervention
    base_retention_prob = 1.0 / (
        1.0 + np.exp(1.2 - 3.2 * flight_risk - 0.03 * overtime_hours + 0.45 * manager_quality)
    )

    # Treated counterfactual retention
    treated_retention_prob = np.clip(base_retention_prob + true_cate, 0.02, 0.98)

    # Realized retention outcome
    realized_prob = np.where(treatment == 1, treated_retention_prob, base_retention_prob)
    retained_outcome = (rng.random(n_employees) < realized_prob).astype(int)

    # Continuous productivity / performance boost outcome
    baseline_prod = 70.0 + 4.0 * prior_performance + 1.8 * tenure_years + rng.normal(0, 3.5, size=n_employees)
    treatment_boost = 7.5 + 10.0 * flight_risk - 0.8 * tenure_years + rng.normal(0, 1.5, size=n_employees)
    productivity_outcome = baseline_prod + treatment * treatment_boost

    return pd.DataFrame({
        "employee_id": [f"EMP_{i+1:05d}" for i in range(n_employees)],
        "tenure_years": np.round(tenure_years, 2),
        "prior_performance": np.round(prior_performance, 2),
        "flight_risk": np.round(flight_risk, 3),
        "overtime_hours": np.round(overtime_hours, 1),
        "manager_quality": np.round(manager_quality, 2),
        "treatment_enrolled": treatment,
        "true_propensity": np.round(propensity_true, 3),
        "true_cate": np.round(true_cate, 3),
        "retained_1yr": retained_outcome,
        "productivity_score": np.round(productivity_outcome, 2),
    })


def generate_panel_policy_data(
    n_units: int = 15,
    n_periods: int = 12,
    treated_units: Tuple[str, ...] = ("Office_North",),
    post_period_start: int = 8,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate balanced panel data of organizational units over time with non-parallel trends.

    Simulates the rollout of a workplace flexibility macro-policy in designated regional offices.
    """
    rng = np.random.default_rng(seed)

    unit_names = [f"Office_{name}" for name in [
        "North", "South", "East", "West", "Central", "Metro", "Bay", "Lakes",
        "Plains", "Coast", "Valley", "Piedmont", "Highland", "Harbor", "Desert"
    ][:n_units]]

    rows = []
    for u_idx, u_name in enumerate(unit_names):
        is_treated_unit = u_name in treated_units
        unit_base = 18.0 + rng.uniform(-3.0, 3.0)
        # Heterogeneous unit growth rate violating parallel trends
        unit_slope = rng.uniform(0.1, 0.45)

        for t in range(1, n_periods + 1):
            is_post = t >= post_period_start
            macro_trend = 0.15 * t + 0.8 * np.sin(2 * np.pi * t / 4.0)

            # True causal policy impact: -3.8% voluntary attrition reduction
            treatment_effect = -3.8 if (is_treated_unit and is_post) else 0.0

            noise = rng.normal(0, 0.45)
            turnover_rate = unit_base + unit_slope * t + macro_trend + treatment_effect + noise

            rows.append({
                "location_id": u_name,
                "quarter_idx": t,
                "quarter_name": f"Q{((t - 1) % 4) + 1}_Y{((t - 1) // 4) + 1}",
                "turnover_rate": float(np.round(np.clip(turnover_rate, 5.0, 40.0), 2)),
                "is_treated_unit": int(is_treated_unit),
                "is_post_period": int(is_post),
            })

    return pd.DataFrame(rows)


def generate_career_trajectory_event_log(
    n_employees: int = 500,
    max_years: float = 7.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate irregular continuous-time career event log across job architecture levels.

    Simulates stochastic promotions, lateral transfers, and departures over continuous time.
    """
    rng = np.random.default_rng(seed)

    states = ["L1_Associate", "L2_Mid", "L3_Senior", "L4_Lead", "L5_Director", "Attrition"]

    rows = []
    for i in range(1, n_employees + 1):
        emp_id = f"EMP_{i:04d}"
        curr_time = rng.uniform(0.0, 1.5)  # initial hiring continuous timestamp
        curr_state = "L1_Associate"

        while curr_time < max_years and curr_state != "Attrition":
            # State-dependent sojourn rate (1/lambda)
            if curr_state == "L1_Associate":
                duration = rng.exponential(1.4) + 0.3
                next_prob = [0.0, 0.78, 0.0, 0.0, 0.0, 0.22]
            elif curr_state == "L2_Mid":
                duration = rng.exponential(2.0) + 0.4
                next_prob = [0.0, 0.0, 0.72, 0.08, 0.0, 0.20]
            elif curr_state == "L3_Senior":
                # Structural promotion bottleneck: longer sojourn time
                duration = rng.exponential(2.8) + 0.5
                next_prob = [0.0, 0.0, 0.0, 0.65, 0.05, 0.30]
            elif curr_state == "L4_Lead":
                duration = rng.exponential(2.5) + 0.6
                next_prob = [0.0, 0.0, 0.0, 0.0, 0.60, 0.40]
            elif curr_state == "L5_Director":
                duration = rng.exponential(4.0) + 1.0
                next_prob = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
            else:
                break

            end_time = min(max_years, curr_time + duration)
            rows.append({
                "employee_id": emp_id,
                "role_level": curr_state,
                "start_year": float(np.round(curr_time, 2)),
                "end_year": float(np.round(end_time, 2)),
                "duration_years": float(np.round(end_time - curr_time, 2)),
            })

            if end_time >= max_years:
                break

            curr_time = end_time
            curr_state = rng.choice(states, p=next_prob)

    return pd.DataFrame(rows)


def generate_heterogeneous_hcm_multigraph(
    n_employees: int = 80,
    n_projects: int = 16,
    n_skills: int = 24,
    seed: int = 42,
) -> Tuple[nx.Graph, pd.DataFrame, pd.DataFrame]:
    """Generate heterogeneous organizational graph with Employees, Projects, Skills, and Departments."""
    import networkx as nx
    rng = np.random.default_rng(seed)

    G = nx.Graph()
    departments = ["Engineering", "Product", "Analytics", "Design", "Operations"]

    # 1. Add Department nodes
    for dept in departments:
        G.add_node(dept, node_type="Department", name=dept)

    # 2. Add Skill nodes
    skills = [f"Skill_{idx:02d}" for idx in range(1, n_skills + 1)]
    for skl in skills:
        G.add_node(skl, node_type="Skill", name=skl)

    # 3. Add Project nodes with required skills
    projects = [f"Project_{idx:02d}" for idx in range(1, n_projects + 1)]
    for prj in projects:
        G.add_node(prj, node_type="Project", name=prj)
        # Link project to 3-5 required skills
        req_skills = rng.choice(skills, size=rng.integers(3, 6), replace=False)
        for skl in req_skills:
            G.add_edge(prj, skl, relation="requires")

    # 4. Add Employee nodes with skills, departments, and project assignments
    employees = [f"EMP_{idx:04d}" for idx in range(1, n_employees + 1)]
    for emp in employees:
        emp_dept = rng.choice(departments)
        G.add_node(emp, node_type="Employee", department=emp_dept)
        G.add_edge(emp, emp_dept, relation="member_of")

        # Assign 2-5 skills
        emp_skills = rng.choice(skills, size=rng.integers(2, 6), replace=False)
        for skl in emp_skills:
            G.add_edge(emp, skl, relation="possesses")

        # Assign 1-3 projects
        emp_prjs = rng.choice(projects, size=rng.integers(1, 4), replace=False)
        for prj in emp_prjs:
            G.add_edge(emp, prj, relation="assigned_to")

    # Node and Edge DataFrames
    nodes_data = [
        {"node_id": n, "node_type": attrs.get("node_type", "Unknown")}
        for n, attrs in G.nodes(data=True)
    ]
    edges_data = [
        {"source": u, "target": v, "relation": attrs.get("relation", "connected")}
        for u, v, attrs in G.edges(data=True)
    ]

    return G, pd.DataFrame(nodes_data), pd.DataFrame(edges_data)


def generate_reporting_hierarchy_with_reorg(
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate enterprise reporting hierarchy snapshots before and after a major reorganization.

    Pre-reorganization:
        Managers maintain healthy, optimal spans (6 to 9 direct reports). Specifically,
        MGR_01 oversees 8 high-performing associates (EMP_001 through EMP_008) with low baseline flight risk.

    Post-reorganization:
        A corporate restructuring consolidates three engineering teams. MGR_02 and MGR_03's teams
        are reassigned into MGR_01, causing MGR_01's direct span of control to suddenly surge
        from 8 direct reports to 25 direct reports!

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (df_pre, df_post) DataFrames containing employee_id, manager_id, department,
        job_level, performance_rating, and baseline_flight_risk.
    """
    rng = np.random.default_rng(seed)

    # 1. Pre-reorganization structure
    # Executive leadership
    ceo = "EXEC_CEO"
    vps = ["VP_ENG", "VP_PROD", "VP_SALES"]

    managers_eng = ["MGR_01", "MGR_02", "MGR_03", "MGR_04"]
    managers_prod = ["MGR_05", "MGR_06"]
    managers_sales = ["MGR_07", "MGR_08", "MGR_09"]

    rows_pre = [
        {"employee_id": ceo, "manager_id": None, "department": "Executive", "job_level": "C-Suite", "performance_rating": 5, "baseline_flight_risk": 0.05},
        {"employee_id": "VP_ENG", "manager_id": ceo, "department": "Engineering", "job_level": "VP", "performance_rating": 5, "baseline_flight_risk": 0.08},
        {"employee_id": "VP_PROD", "manager_id": ceo, "department": "Product", "job_level": "VP", "performance_rating": 4, "baseline_flight_risk": 0.10},
        {"employee_id": "VP_SALES", "manager_id": ceo, "department": "Sales", "job_level": "VP", "performance_rating": 4, "baseline_flight_risk": 0.12},
    ]

    for m in managers_eng:
        rows_pre.append({"employee_id": m, "manager_id": "VP_ENG", "department": "Engineering", "job_level": "Manager", "performance_rating": 4, "baseline_flight_risk": 0.12})
    for m in managers_prod:
        rows_pre.append({"employee_id": m, "manager_id": "VP_PROD", "department": "Product", "job_level": "Manager", "performance_rating": 4, "baseline_flight_risk": 0.11})
    for m in managers_sales:
        rows_pre.append({"employee_id": m, "manager_id": "VP_SALES", "department": "Sales", "job_level": "Manager", "performance_rating": 4, "baseline_flight_risk": 0.14})

    # Individual contributors:
    # MGR_01 has 8 direct reports (EMP_001 to EMP_008)
    for i in range(1, 9):
        emp_id = f"EMP_{i:03d}"
        rows_pre.append({
            "employee_id": emp_id,
            "manager_id": "MGR_01",
            "department": "Engineering",
            "job_level": f"IC{rng.choice([2, 3, 4])}",
            "performance_rating": rng.choice([4, 5]),
            "baseline_flight_risk": round(float(rng.uniform(0.08, 0.14)), 3),
        })

    # MGR_02 has 9 direct reports (EMP_009 to EMP_017)
    for i in range(9, 18):
        emp_id = f"EMP_{i:03d}"
        rows_pre.append({
            "employee_id": emp_id,
            "manager_id": "MGR_02",
            "department": "Engineering",
            "job_level": f"IC{rng.choice([2, 3])}",
            "performance_rating": rng.choice([3, 4]),
            "baseline_flight_risk": round(float(rng.uniform(0.12, 0.18)), 3),
        })

    # MGR_03 has 8 direct reports (EMP_018 to EMP_025)
    for i in range(18, 26):
        emp_id = f"EMP_{i:03d}"
        rows_pre.append({
            "employee_id": emp_id,
            "manager_id": "MGR_03",
            "department": "Engineering",
            "job_level": f"IC{rng.choice([1, 2, 3])}",
            "performance_rating": rng.choice([3, 4]),
            "baseline_flight_risk": round(float(rng.uniform(0.10, 0.16)), 3),
        })

    # MGR_04 has 7 direct reports (EMP_026 to EMP_032)
    for i in range(26, 33):
        emp_id = f"EMP_{i:03d}"
        rows_pre.append({
            "employee_id": emp_id,
            "manager_id": "MGR_04",
            "department": "Engineering",
            "job_level": f"IC{rng.choice([2, 3])}",
            "performance_rating": rng.choice([3, 4]),
            "baseline_flight_risk": round(float(rng.uniform(0.11, 0.17)), 3),
        })

    # Product and Sales associates (EMP_033 to EMP_070)
    curr_emp = 33
    for mgr, dept in [("MGR_05", "Product"), ("MGR_06", "Product"), ("MGR_07", "Sales"), ("MGR_08", "Sales"), ("MGR_09", "Sales")]:
        team_size = rng.integers(6, 9)
        for _ in range(team_size):
            emp_id = f"EMP_{curr_emp:03d}"
            rows_pre.append({
                "employee_id": emp_id,
                "manager_id": mgr,
                "department": dept,
                "job_level": f"IC{rng.choice([1, 2, 3])}",
                "performance_rating": rng.choice([3, 4]),
                "baseline_flight_risk": round(float(rng.uniform(0.10, 0.18)), 3),
            })
            curr_emp += 1

    df_pre = pd.DataFrame(rows_pre)

    # 2. Post-reorganization structure:
    # MGR_02 and MGR_03 step into Principal IC roles; their 17 reports (EMP_009 through EMP_025)
    # are re-assigned directly to MGR_01!
    # MGR_01's span explodes from 8 to 8 + 9 + 8 = 25 direct reports!
    df_post = df_pre.copy()
    reassigned_reports = [f"EMP_{i:03d}" for i in range(9, 26)]
    df_post.loc[df_post["employee_id"].isin(reassigned_reports), "manager_id"] = "MGR_01"

    # MGR_02 and MGR_03 become Staff/Principal ICs reporting to VP_ENG with 0 direct reports
    df_post.loc[df_post["employee_id"].isin(["MGR_02", "MGR_03"]), "job_level"] = "Staff IC"

    return df_pre, df_post


def generate_time_in_position_workforce(
    n_employees: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate enterprise workforce dataset with time-in-position, performance ratings, and flight risk.

    Models employee tenures across departments (Engineering, Product, Analytics, Sales, Operations).
    Includes last role change dates, running months in current position, performance ratings (1 to 5),
    and baseline flight risk probabilities. Injects a cohort of stagnant top performers whose
    time-in-position exceeds the 90th percentile of their department cohort, triggering an acute voluntary
    attrition spike.

    Returns
    -------
    pd.DataFrame
        Columns: [employee_id, department, job_level, performance_rating,
                  last_role_change_date, time_in_position_months, baseline_flight_risk]
    """
    rng = np.random.default_rng(seed)
    as_of = pd.Timestamp("2026-10-01")

    departments = ["Engineering", "Product", "Analytics", "Sales", "Operations"]
    dept_weights = [0.30, 0.20, 0.15, 0.20, 0.15]
    levels = ["IC1", "IC2", "IC3", "Senior", "Lead"]

    rows = []
    dept_scale = {"Engineering": 16.0, "Product": 14.0, "Analytics": 15.0, "Sales": 11.0, "Operations": 13.0}

    for i in range(1, n_employees + 1):
        emp_id = f"EMP_{i:04d}"
        dept = rng.choice(departments, p=dept_weights)
        lvl = rng.choice(levels, p=[0.20, 0.30, 0.25, 0.15, 0.10])
        perf = rng.choice([2, 3, 4, 5], p=[0.12, 0.50, 0.26, 0.12])

        scale = dept_scale[dept]
        months = rng.gamma(shape=2.2, scale=scale / 2.2)
        months = np.clip(months, 2.0, 72.0)

        base_risk = 0.08 + (5 - perf) * 0.015 + rng.normal(0, 0.02)
        base_risk = float(np.clip(base_risk, 0.04, 0.30))

        rows.append({
            "employee_id": emp_id,
            "department": dept,
            "job_level": lvl,
            "performance_rating": perf,
            "time_in_position_months": round(float(months), 1),
            "baseline_flight_risk": round(base_risk, 3),
        })

    df = pd.DataFrame(rows)

    df["last_role_change_date"] = [
        (as_of - pd.Timedelta(days=int(m * 30.4375))).strftime("%Y-%m-%d")
        for m in df["time_in_position_months"]
    ]

    for dept in departments:
        dept_mask = df["department"] == dept
        p90 = df.loc[dept_mask, "time_in_position_months"].quantile(0.90)
        target_idx = df[dept_mask & (df["performance_rating"] >= 4)].index[:2]
        for idx in target_idx:
            stagnant_months = round(float(p90 + rng.uniform(6.0, 18.0)), 1)
            df.loc[idx, "time_in_position_months"] = stagnant_months
            df.loc[idx, "last_role_change_date"] = (as_of - pd.Timedelta(days=int(stagnant_months * 30.4375))).strftime("%Y-%m-%d")

    return df


def generate_bradford_factor_workforce(
    n_employees: int = 120,
    window_weeks: int = 52,
    as_of_date: str = "2026-10-01",
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate realistic absence event logs and daily attendance records to demonstrate the Bradford Factor.

    Simulates four distinct behavioral archetypes:
    1. Planned Continuous Leave: Associate taking a single planned 10-day medical/parental leave
       (S=1, D=10 -> Bradford Score: 1^2 * 10 = 10).
    2. Chronic Unplanned Call-outs: Associate taking 10 separate 1-day unplanned call-outs
       (S=10, D=10 -> Bradford Score: 10^2 * 10 = 1000).
    3. Moderate Occasional Sickness: Associate taking 3 separate 2-day spells
       (S=3, D=6 -> Bradford Score: 3^2 * 6 = 54).
    4. Steady / Perfect Attendance: Associate with 0 absence spells.

    Parameters
    ----------
    n_employees : int, default=120
        Total headcount to simulate across departments.
    window_weeks : int, default=52
        Evaluation window in weeks (supports 52 or 53 weeks).
    as_of_date : str, default='2026-10-01'
        Evaluation snapshot date.
    seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        - spell_events_df: Event-level records with start_date, end_date, duration_days, is_planned.
        - employees_summary_df: Associate-level ground-truth metadata (archetype, department, role).
    """
    rng = np.random.default_rng(seed)
    as_of = pd.to_datetime(as_of_date)
    window_days = window_weeks * 7
    window_start = as_of - pd.Timedelta(days=window_days)

    departments = ["Logistics", "Customer Support", "Retail Store", "Field Engineering", "Manufacturing"]
    archetypes = [
        "Planned Medical Leave",
        "Chronic Unplanned Call-outs",
        "Moderate Occasional Sickness",
        "Standard Attendance",
    ]
    archetype_weights = [0.15, 0.20, 0.35, 0.30]

    emp_records = []
    spell_events = []

    for i in range(1, n_employees + 1):
        emp_id = f"EMP_{i:04d}"
        dept = rng.choice(departments)
        arch = rng.choice(archetypes, p=archetype_weights)

        # Force EMP_0001 to be Planned Medical Leave (S=1, D=10)
        # and EMP_0002 to be Chronic Unplanned Call-outs (S=10, D=10)
        if i == 1:
            arch = "Planned Medical Leave"
        elif i == 2:
            arch = "Chronic Unplanned Call-outs"

        emp_records.append({
            "employee_id": emp_id,
            "department": dept,
            "archetype": arch,
        })

        if arch == "Standard Attendance":
            # 0 or 1 short spell
            if rng.random() < 0.25:
                rand_day = rng.integers(10, window_days - 5)
                s_date = window_start + pd.Timedelta(days=int(rand_day))
                e_date = s_date + pd.Timedelta(days=1)
                spell_events.append({
                    "employee_id": emp_id,
                    "department": dept,
                    "absence_start_date": s_date.strftime("%Y-%m-%d"),
                    "absence_end_date": e_date.strftime("%Y-%m-%d"),
                    "duration_days": 2,
                    "absence_reason": "Mild Cold",
                    "is_planned": False,
                })

        elif arch == "Planned Medical Leave":
            # Exactly 1 long planned block of 10-14 days
            dur = 10 if i == 1 else int(rng.integers(10, 15))
            rand_day = rng.integers(30, window_days - dur - 10)
            s_date = window_start + pd.Timedelta(days=int(rand_day))
            e_date = s_date + pd.Timedelta(days=dur - 1)
            spell_events.append({
                "employee_id": emp_id,
                "department": dept,
                "absence_start_date": s_date.strftime("%Y-%m-%d"),
                "absence_end_date": e_date.strftime("%Y-%m-%d"),
                "duration_days": dur,
                "absence_reason": "Scheduled Surgery & Recovery",
                "is_planned": True,
            })

        elif arch == "Chronic Unplanned Call-outs":
            # 8 to 12 separate 1-day or 2-day unplanned callouts
            n_spells = 10 if i == 2 else int(rng.integers(8, 13))
            used_days = set()
            for _ in range(n_spells):
                rand_day = rng.integers(5, window_days - 3)
                while rand_day in used_days:
                    rand_day = rng.integers(5, window_days - 3)
                used_days.add(rand_day)
                used_days.add(rand_day + 1)

                dur = 1 if i == 2 else int(rng.choice([1, 1, 1, 2]))
                s_date = window_start + pd.Timedelta(days=int(rand_day))
                e_date = s_date + pd.Timedelta(days=dur - 1)
                spell_events.append({
                    "employee_id": emp_id,
                    "department": dept,
                    "absence_start_date": s_date.strftime("%Y-%m-%d"),
                    "absence_end_date": e_date.strftime("%Y-%m-%d"),
                    "duration_days": dur,
                    "absence_reason": rng.choice(["Migraine", "Unscheduled Personal", "Stomach Flu", "Vehicle Issue"]),
                    "is_planned": False,
                })

        elif arch == "Moderate Occasional Sickness":
            # 2 to 4 spells of 1-3 days
            n_spells = rng.integers(2, 5)
            used_days = set()
            for _ in range(n_spells):
                rand_day = rng.integers(5, window_days - 5)
                while rand_day in used_days:
                    rand_day = rng.integers(5, window_days - 5)
                used_days.add(rand_day)
                used_days.add(rand_day + 1)
                used_days.add(rand_day + 2)

                dur = int(rng.choice([1, 2, 3]))
                s_date = window_start + pd.Timedelta(days=int(rand_day))
                e_date = s_date + pd.Timedelta(days=dur - 1)
                spell_events.append({
                    "employee_id": emp_id,
                    "department": dept,
                    "absence_start_date": s_date.strftime("%Y-%m-%d"),
                    "absence_end_date": e_date.strftime("%Y-%m-%d"),
                    "duration_days": dur,
                    "absence_reason": rng.choice(["Flu", "Dental Procedure", "Fever"]),
                    "is_planned": False,
                })

    spell_df = pd.DataFrame(spell_events)
    emp_df = pd.DataFrame(emp_records)

    return spell_df, emp_df


def generate_retail_scheduling_pilot_data(
    n_stores: int = 30,
    n_weeks: int = 24,
    pilot_stores: Tuple[str, ...] = ("Store_01", "Store_02", "Store_03", "Store_04", "Store_05"),
    rollout_week: int = 16,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate longitudinal panel data of retail stores evaluating an automated scheduling tool pilot.

    Simulates realistic store-level turnover trajectories where pilot stores implement
    an automated scheduling tool at `rollout_week`, reducing turnover friction by ~3.8%
    relative to their synthetic control twins.

    Parameters
    ----------
    n_stores : int, default=30
        Total number of stores in the retail footprint.
    n_weeks : int, default=24
        Total evaluation weeks in the panel (e.g., 15 pre-treatment, 9 post-treatment).
    pilot_stores : Tuple[str, ...], default=('Store_01', 'Store_02', 'Store_03', 'Store_04', 'Store_05')
        Designated pilot store identifiers.
    rollout_week : int, default=16
        Week index when the automated scheduling tool is deployed in pilot stores.
    seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - 'store_id': Store identifier (e.g. Store_01 .. Store_30)
        - 'week_idx': Week index 1..n_weeks
        - 'turnover_rate': Observed annualized voluntary turnover rate (%)
        - 'is_pilot_store': 1 if in pilot group, 0 otherwise
        - 'is_post_rollout': 1 if week_idx >= rollout_week, 0 otherwise
    """
    rng = np.random.default_rng(seed)
    pilot_set = set(pilot_stores)

    rows = []
    # Distinct baseline profiles for each store
    for s_idx in range(1, n_stores + 1):
        store_id = f"Store_{s_idx:02d}"
        is_pilot = int(store_id in pilot_set)

        # Baseline turnover: 16% to 26%
        base_rate = 18.0 + rng.uniform(-3.5, 4.5)
        # Store secular trend
        trend_slope = rng.uniform(0.04, 0.16)
        # Store seasonal phase
        phase_shift = rng.uniform(0, 2 * np.pi)

        for w in range(1, n_weeks + 1):
            is_post = int(w >= rollout_week)
            macro_season = 0.8 * np.sin(2 * np.pi * w / 12.0 + phase_shift)

            # True causal effect of scheduling tool in pilot stores: -3.8% turnover
            causal_lift = -3.8 * is_pilot * is_post

            noise = rng.normal(0, 0.35)
            turnover = base_rate + trend_slope * w + macro_season + causal_lift + noise
            turnover = float(np.round(np.clip(turnover, 6.0, 38.0), 2))

            rows.append({
                "store_id": store_id,
                "week_idx": w,
                "turnover_rate": turnover,
                "is_pilot_store": is_pilot,
                "is_post_rollout": is_post,
            })

    return pd.DataFrame(rows)


def generate_overtime_turnover_causal_dag_data(
    n_samples: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    r"""Generate synthetic workforce observational data governed by a known ground-truth Directed Acyclic Graph (DAG).

    Models the classic 'Overtime vs Turnover' death spiral where high observed correlation
    is partially driven by a mutual upstream confounder (Manager Absence) and mediated by Burnout:
    - Root causes (in-degree 0): 'manager_absence', 'workload_surge'
    - 'understaffing' <- 0.65 * manager_absence + 0.50 * workload_surge
    - 'overtime_hours' <- 0.60 * understaffing + 0.40 * manager_absence
    - 'burnout_index' <- 0.70 * overtime_hours + 0.35 * understaffing
    - 'turnover_risk' <- 0.65 * burnout_index + 0.45 * manager_absence

    Parameters
    ----------
    n_samples : int, default=1000
        Number of employee or departmental observations.
    seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with 6 standardized workforce variables.
    """
    rng = np.random.default_rng(seed)
    s = 0.5

    # Exogenous root causes with equal baseline variance
    manager_absence = rng.normal(0, s, size=n_samples)
    workload_surge = rng.normal(0, s, size=n_samples)

    # Downstream Structural Equations with equal error variances (Peters & Buhlmann, 2014)
    understaffing = (
        0.65 * manager_absence
        + 0.50 * workload_surge
        + rng.normal(0, s, size=n_samples)
    )

    overtime_hours = (
        0.60 * understaffing
        + 0.40 * manager_absence
        + rng.normal(0, s, size=n_samples)
    )

    burnout_index = (
        0.70 * overtime_hours
        + 0.35 * understaffing
        + rng.normal(0, s, size=n_samples)
    )

    turnover_risk = (
        0.65 * burnout_index
        + 0.45 * manager_absence
        + rng.normal(0, s, size=n_samples)
    )

    df = pd.DataFrame({
        "manager_absence": manager_absence,
        "workload_surge": workload_surge,
        "understaffing": understaffing,
        "overtime_hours": overtime_hours,
        "burnout_index": burnout_index,
        "turnover_risk": turnover_risk,
    })

    return df


def generate_store_department_labor_panel(
    n_stores: int = 10,
    n_depts: int = 4,
    n_weeks: int = 104,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a multi-year panel of store and department weekly operational metrics.

    Simulates realistic workforce dynamics across hierarchical partitions:
    - 52-week seasonal turnover cycles (summer ramp and Q1 post-holiday turnover surge).
    - Long-term baseline labor trends and headcount.
    - Assigned hours with schedule-induced volatility and fatigue surges.

    Parameters
    ----------
    n_stores : int, default=10
        Number of retail/operational store locations.
    n_depts : int, default=4
        Number of distinct departments per store.
    n_weeks : int, default=104
        Number of weekly temporal observations (e.g., 2 years = 104 weeks).
    seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with hierarchical partitions (store_id, dept_id), temporal index (week),
        and operational metrics (headcount, assigned_hours, turnover_count, turnover_rate).
    """
    rng = np.random.default_rng(seed)

    store_names = [f"Store_{i+1:02d}" for i in range(n_stores)]
    dept_names = ["Sales", "Logistics", "Cashier", "CustomerService"][:n_depts]
    if len(dept_names) < n_depts:
        dept_names += [f"Dept_{j+1}" for j in range(len(dept_names), n_depts)]

    records = []

    for s_idx, store in enumerate(store_names):
        store_mult = 0.8 + 0.05 * s_idx  # Variation in store scale
        for d_idx, dept in enumerate(dept_names):
            dept_base_hc = (25 + 15 * (d_idx % 3)) * store_mult
            hourly_rate_per_emp = 34.0 + 2.0 * d_idx

            for w in range(1, n_weeks + 1):
                woy = ((w - 1) % 52) + 1  # Week of year (1-52)

                # Seasonal holiday ramp in weeks 44-51
                holiday_factor = 1.35 if 44 <= woy <= 51 else 1.0
                # Post-holiday turnover surge in weeks 2-8
                q1_surge = 0.035 if 2 <= woy <= 8 else 0.015

                # Headcount evolves smoothly
                hc = max(8, int(round(dept_base_hc * holiday_factor + rng.normal(0, 1.5))))

                # Hours: baseline plus occasional schedule turbulence shocks
                turbulence = rng.choice([0.0, 0.0, 0.0, 120.0, 250.0], p=[0.70, 0.15, 0.08, 0.05, 0.02])
                hours = max(100.0, hc * hourly_rate_per_emp * holiday_factor + turbulence + rng.normal(0, 20.0))

                # Turnover: base rate + seasonal shock + random Poisson departure count
                base_turn_prob = max(0.005, min(0.15, q1_surge + (0.01 * (turbulence > 50)) + rng.normal(0, 0.005)))
                turn_count = int(rng.binomial(hc, base_turn_prob))
                turn_rate = turn_count / max(hc, 1)

                records.append({
                    "store_id": store,
                    "dept_id": dept,
                    "week": w,
                    "year": ((w - 1) // 52) + 1,
                    "week_of_year": woy,
                    "headcount": hc,
                    "assigned_hours": round(float(hours), 1),
                    "turnover_count": turn_count,
                    "turnover_rate": round(float(turn_rate), 4),
                })

    df = pd.DataFrame(records)
    return df


def generate_daily_store_foot_traffic(
    n_days: int = 180,
    n_stores: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate daily store foot traffic and scheduled labor hours with chaotic noise.

    Simulates realistic retail dynamics:
    - Smooth secular traffic growth trend.
    - Strong weekly seasonality (Friday-Sunday foot traffic surges).
    - Bi-weekly/monthly payday surges (1st and 15th of each month).
    - Erratic day-to-day weather shocks and temporary disruptions.
    - Ground truth unobserved baseline vs noisy observed foot traffic and labor hours.

    Parameters
    ----------
    n_days : int, default=180
        Number of consecutive daily observations (approx 6 months).
    n_stores : int, default=3
        Number of retail stores.
    seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with store_id, date, day_index, day_of_week, true_baseline,
        observed_foot_traffic, and observed_labor_hours.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-01-01", periods=n_days, freq="D")
    records = []

    store_names = [f"Store_{i+1:02d}" for i in range(n_stores)]

    for s_idx, store in enumerate(store_names):
        base_traffic = 1200.0 + 300.0 * s_idx
        trend_slope = 1.2 + 0.4 * s_idx

        for d_idx, dt in enumerate(dates):
            dow = dt.dayofweek  # 0=Mon, 6=Sun
            dom = dt.day        # Day of month

            # 1. Secular growth trend
            trend_val = base_traffic + trend_slope * d_idx

            # 2. Weekly seasonal cycle (Mon-Thu lower, Fri-Sun surge)
            dow_effects = {0: -180.0, 1: -220.0, 2: -150.0, 3: -80.0, 4: +200.0, 5: +450.0, 6: +280.0}
            weekly_cycle = dow_effects[dow]

            # 3. Payday surge (1st and 15th)
            payday_effect = 220.0 if dom in (1, 2, 15, 16) else 0.0

            # True structural baseline curve (trend + harmonic seasonality + payday)
            true_baseline = trend_val + weekly_cycle + payday_effect

            # 4. Chaotic daily operational noise (weather, inventory stockouts, sudden spikes)
            noise = float(rng.normal(0, 160.0))
            if rng.random() < 0.05:  # Occasional severe storm or promotion spike
                noise += float(rng.choice([-400.0, +500.0]))

            observed_traffic = max(200.0, true_baseline + noise)

            # Scheduled labor hours responding to traffic with human scheduling noise
            labor_ratio = 0.085  # ~8.5 staff hours per 100 customer visits
            labor_hours = max(25.0, observed_traffic * labor_ratio + rng.normal(0, 8.0))

            records.append({
                "store_id": store,
                "date": dt,
                "day_index": d_idx,
                "day_of_week": dow,
                "day_name": dt.strftime("%A"),
                "is_weekend": int(dow in (5, 6)),
                "true_baseline": round(float(true_baseline), 2),
                "observed_foot_traffic": round(float(observed_traffic), 2),
                "observed_labor_hours": round(float(labor_hours), 2),
                "daily_noise": round(float(noise), 2),
            })

    return pd.DataFrame(records)








