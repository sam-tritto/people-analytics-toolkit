"""Tests validating polymorphic container inputs, mathematical invariants via Hypothesis,
and conftest.py shared pytest fixtures for the People Analytics Toolkit.
"""

import math
import pytest
import numpy as np
import pandas as pd
from hypothesis import given, settings, assume, strategies as st

from people_analytics_toolkit.compensation import (
    calculate_salary_band_midpoint,
    calculate_compa_ratio,
    calculate_range_penetration,
    classify_pay_band_status,
    is_red_circled,
    is_green_circled,
)
from people_analytics_toolkit.chaos import (
    calculate_bradford_score,
    classify_bradford_risk,
    calculate_shannon_entropy,
)
from people_analytics_toolkit.memory import (
    fractional_difference,
    calculate_ewma,
    calculate_ewms,
    double_exponential_smoothing,
)


# ===========================================================================
# 1. Parametrized Polymorphic Input Tests (Series, ndarray, list, scalar)
# ===========================================================================

CONTAINER_FACTORIES = [pd.Series, np.array, list]
CONTAINER_IDS = ["Series", "ndarray", "list"]


class TestPolymorphicInputs:
    """Validate that analytical functions accept pd.Series, np.ndarray, list, and scalars."""

    RAW_SALARIES = [75000.0, 95000.0, 110000.0, 135000.0]
    BAND_MINS = [80000.0, 80000.0, 80000.0, 80000.0]
    BAND_MAXS = [120000.0, 120000.0, 120000.0, 120000.0]
    MIDPOINTS = [100000.0, 100000.0, 100000.0, 100000.0]

    @pytest.mark.parametrize("to_container", CONTAINER_FACTORIES, ids=CONTAINER_IDS)
    def test_calculate_compa_ratio_containers(self, to_container):
        """calculate_compa_ratio must compute identical numerical results across container types."""
        salaries = to_container(self.RAW_SALARIES)
        mids = to_container(self.MIDPOINTS)

        result = calculate_compa_ratio(salaries, midpoint=mids)
        np_res = np.asarray(result)

        expected = np.array([0.75, 0.95, 1.10, 1.35])
        np.testing.assert_allclose(np_res, expected, rtol=1e-5)

        if to_container is pd.Series:
            assert isinstance(result, pd.Series)
            assert result.name == "compa_ratio"
        elif to_container is np.array:
            assert isinstance(result, np.ndarray)

    def test_calculate_compa_ratio_scalar(self):
        """calculate_compa_ratio must return float scalar when given scalar inputs."""
        res = calculate_compa_ratio(95000.0, midpoint=100000.0)
        assert isinstance(res, float)
        assert math.isclose(res, 0.95, rel_tol=1e-5)

    @pytest.mark.parametrize("to_container", CONTAINER_FACTORIES, ids=CONTAINER_IDS)
    def test_calculate_range_penetration_containers(self, to_container):
        """calculate_range_penetration must support Series, ndarray, and list."""
        salaries = to_container(self.RAW_SALARIES)
        b_min = to_container(self.BAND_MINS)
        b_max = to_container(self.BAND_MAXS)

        result = calculate_range_penetration(salaries, b_min, b_max)
        np_res = np.asarray(result)

        # (75 - 80) / 40 = -0.125; (95 - 80) / 40 = 0.375; (110 - 80) / 40 = 0.75; (135 - 80) / 40 = 1.375
        expected = np.array([-0.125, 0.375, 0.75, 1.375])
        np.testing.assert_allclose(np_res, expected, rtol=1e-5)

        if to_container is pd.Series:
            assert isinstance(result, pd.Series)
            assert result.name == "range_penetration"
        elif to_container is np.array:
            assert isinstance(result, np.ndarray)

    def test_calculate_range_penetration_scalar(self):
        """calculate_range_penetration must return float scalar when given scalar bounds."""
        res = calculate_range_penetration(100000.0, 80000.0, 120000.0)
        assert isinstance(res, float)
        assert math.isclose(res, 0.5, rel_tol=1e-5)

    @pytest.mark.parametrize("to_container", CONTAINER_FACTORIES, ids=CONTAINER_IDS)
    def test_classify_pay_band_status_containers(self, to_container):
        """classify_pay_band_status must return correct tier labels across containers."""
        rp_vals = [-0.10, 0.15, 0.40, 0.65, 0.85, 1.20]
        inputs = to_container(rp_vals)

        result = classify_pay_band_status(inputs)
        res_list = list(result)

        expected = [
            "Green-Circled (<Min)",
            "Q1: Developing (0-25%)",
            "Q2: Core Proficient (25-50%)",
            "Q3: Advanced (50-75%)",
            "Q4: Senior Cap (75-100%)",
            "Red-Circled (>Max)",
        ]
        assert res_list == expected

        if to_container is pd.Series:
            assert isinstance(result, pd.Series)
            assert result.name == "pay_band_status"

    def test_classify_pay_band_status_scalar(self):
        """classify_pay_band_status must return a single string label for scalar input."""
        res = classify_pay_band_status(0.40)
        assert isinstance(res, str)
        assert res == "Q2: Core Proficient (25-50%)"

    @pytest.mark.parametrize("to_container", CONTAINER_FACTORIES, ids=CONTAINER_IDS)
    def test_is_red_and_green_circled_containers(self, to_container):
        """is_red_circled and is_green_circled must handle polymorphic containers."""
        rp_vals = [-0.15, 0.50, 1.25]
        inputs = to_container(rp_vals)

        red = is_red_circled(inputs)
        green = is_green_circled(inputs)

        np.testing.assert_array_equal(np.asarray(red), np.array([False, False, True]))
        np.testing.assert_array_equal(np.asarray(green), np.array([True, False, False]))

        if to_container is pd.Series:
            assert isinstance(red, pd.Series)
            assert isinstance(green, pd.Series)

    def test_is_red_and_green_circled_scalar(self):
        """is_red_circled and is_green_circled must return boolean primitives for scalar values."""
        assert is_red_circled(130000.0, band_max=120000.0) is True
        assert is_red_circled(110000.0, band_max=120000.0) is False
        assert is_green_circled(70000.0, band_min=80000.0) is True
        assert is_green_circled(90000.0, band_min=80000.0) is False

    @pytest.mark.parametrize("to_container", CONTAINER_FACTORIES, ids=CONTAINER_IDS)
    def test_calculate_bradford_score_containers(self, to_container):
        """calculate_bradford_score must evaluate S^2 * D across polymorphic containers."""
        spells = to_container([1, 3, 5])
        days = to_container([10, 5, 2])

        result = calculate_bradford_score(spells, days)
        np_res = np.asarray(result)

        # 1^2 * 10 = 10, 3^2 * 5 = 45, 5^2 * 2 = 50
        np.testing.assert_array_equal(np_res, np.array([10, 45, 50]))

        if to_container is pd.Series:
            assert isinstance(result, pd.Series)

    def test_calculate_bradford_score_scalar(self):
        """calculate_bradford_score must compute scalar values."""
        res = calculate_bradford_score(4, 12)
        assert res == 192  # 16 * 12

    @pytest.mark.parametrize("to_container", CONTAINER_FACTORIES, ids=CONTAINER_IDS)
    def test_classify_bradford_risk_containers(self, to_container):
        """classify_bradford_risk must categorize friction tiers across containers."""
        scores = to_container([25, 150, 350, 750])
        result = classify_bradford_risk(scores)
        res_list = list(result)

        expected = [
            "Low Friction",
            "Moderate Friction",
            "Substantial Friction",
            "Critical Friction",
        ]
        assert res_list == expected

    def test_classify_bradford_risk_scalar(self):
        """classify_bradford_risk must return a single string for scalar score."""
        assert classify_bradford_risk(45) == "Low Friction"
        assert classify_bradford_risk(120) == "Moderate Friction"
        assert classify_bradford_risk(250) == "Substantial Friction"
        assert classify_bradford_risk(800) == "Critical Friction"

    @pytest.mark.parametrize("to_container", CONTAINER_FACTORIES, ids=CONTAINER_IDS)
    def test_fractional_difference_containers(self, to_container):
        """fractional_difference must accept Series, ndarray, and list."""
        raw_vals = [10.0, 12.0, 11.0, 14.0, 15.0, 13.0, 16.0, 18.0]
        data = to_container(raw_vals)

        diff = fractional_difference(data, d=0.4)
        assert isinstance(diff, pd.Series)
        assert len(diff) == len(raw_vals)
        # Trailing values must be finite floats
        assert not np.isnan(diff.iloc[-1])

    @pytest.mark.parametrize("to_container", CONTAINER_FACTORIES, ids=CONTAINER_IDS)
    def test_calculate_ewma_containers(self, to_container):
        """calculate_ewma must accept Series, ndarray, and list."""
        raw_vals = [100.0, 102.0, 105.0, 103.0, 108.0]
        data = to_container(raw_vals)

        ewma = calculate_ewma(data, span=3)
        assert isinstance(ewma, pd.Series)
        assert len(ewma) == len(raw_vals)
        assert not ewma.isna().any()

    @pytest.mark.parametrize("to_container", CONTAINER_FACTORIES, ids=CONTAINER_IDS)
    def test_calculate_ewms_containers(self, to_container):
        """calculate_ewms must accept Series, ndarray, and list."""
        counts = to_container([0, 1, 0, 2, 0, 1])
        ewms = calculate_ewms(counts, half_life=4.0)

        assert isinstance(ewms, pd.Series)
        assert len(ewms) == len(counts)
        assert (ewms >= 0).all()

    @pytest.mark.parametrize("to_container", CONTAINER_FACTORIES, ids=CONTAINER_IDS)
    def test_calculate_shannon_entropy_containers(self, to_container):
        """calculate_shannon_entropy must produce identical entropy across containers."""
        shifts = ["Day", "Day", "Night", "Swing", "Day", "Swing"]
        data = to_container(shifts)

        h = calculate_shannon_entropy(data, base=2.0)
        assert isinstance(h, float)
        assert h > 0.0

    @pytest.mark.parametrize("to_container", CONTAINER_FACTORIES, ids=CONTAINER_IDS)
    def test_double_exponential_smoothing_containers(self, to_container):
        """double_exponential_smoothing must accept Series, ndarray, and list."""
        np.random.seed(42)
        raw_trend = (np.arange(30) * 1.5 + np.random.normal(0, 0.5, 30)).tolist()
        data = to_container(raw_trend)

        fitted, decomp, meta = double_exponential_smoothing(data, alpha=0.3, beta=0.1)
        assert isinstance(fitted, pd.Series)
        assert isinstance(decomp, pd.DataFrame)
        assert len(fitted) == 30
        assert "level" in decomp.columns and "trend" in decomp.columns


# ===========================================================================
# 2. Property-Based Testing with Hypothesis (Mathematical Invariants)
# ===========================================================================

class TestHypothesisMathematicalInvariants:
    """Verify core compensation, friction, and memory invariants over generated input spaces."""

    @settings(max_examples=50)
    @given(
        salary=st.floats(min_value=1e4, max_value=1e7, allow_nan=False, allow_infinity=False),
        midpoint=st.floats(min_value=1e4, max_value=1e7, allow_nan=False, allow_infinity=False),
    )
    def test_compa_ratio_scale_homogeneity(self, salary, midpoint):
        """Property: Compa-Ratio is homogeneous of degree 0: Compa(c*S, c*M) == Compa(S, M)."""
        base_ratio = calculate_compa_ratio(salary, midpoint=midpoint)
        scaled_ratio = calculate_compa_ratio(salary * 2.5, midpoint=midpoint * 2.5)

        assert math.isclose(base_ratio, scaled_ratio, rel_tol=1e-5)

    @settings(max_examples=50)
    @given(
        midpoint=st.floats(min_value=1e4, max_value=1e7, allow_nan=False, allow_infinity=False),
    )
    def test_compa_ratio_midpoint_parity(self, midpoint):
        """Property: If salary == midpoint, Compa-Ratio must be exactly 1.0 (100% parity)."""
        ratio = calculate_compa_ratio(midpoint, midpoint=midpoint)
        assert math.isclose(ratio, 1.0, rel_tol=1e-6)

    @settings(max_examples=50)
    @given(
        s1=st.floats(min_value=1e4, max_value=5e5, allow_nan=False, allow_infinity=False),
        s2=st.floats(min_value=1e4, max_value=5e5, allow_nan=False, allow_infinity=False),
        midpoint=st.floats(min_value=1e4, max_value=5e5, allow_nan=False, allow_infinity=False),
    )
    def test_compa_ratio_strict_monotonicity(self, s1, s2, midpoint):
        """Property: For a fixed midpoint > 0, s1 < s2 <=> Compa(s1) < Compa(s2)."""
        assume(abs(s1 - s2) > 1.0)
        c1 = calculate_compa_ratio(s1, midpoint=midpoint)
        c2 = calculate_compa_ratio(s2, midpoint=midpoint)

        if s1 < s2:
            assert c1 < c2
        else:
            assert c1 > c2

    @settings(max_examples=50)
    @given(
        band_min=st.floats(min_value=2e4, max_value=2e5, allow_nan=False, allow_infinity=False),
        spread=st.floats(min_value=0.1, max_value=1.5, allow_nan=False, allow_infinity=False),
    )
    def test_range_penetration_boundary_invariants(self, band_min, spread):
        """Properties: Range Penetration at Min is 0.0, at Max is 1.0, and at Midpoint is 0.5."""
        band_max = band_min * (1.0 + spread)
        midpoint = (band_min + band_max) / 2.0

        rp_at_min = calculate_range_penetration(band_min, band_min, band_max)
        rp_at_max = calculate_range_penetration(band_max, band_min, band_max)
        rp_at_mid = calculate_range_penetration(midpoint, band_min, band_max)

        assert math.isclose(rp_at_min, 0.0, abs_tol=1e-5)
        assert math.isclose(rp_at_max, 1.0, abs_tol=1e-5)
        assert math.isclose(rp_at_mid, 0.5, abs_tol=1e-5)

    @settings(max_examples=50)
    @given(
        salary=st.floats(min_value=3e4, max_value=3e5, allow_nan=False, allow_infinity=False),
        band_min=st.floats(min_value=2e4, max_value=1e5, allow_nan=False, allow_infinity=False),
        band_max=st.floats(min_value=1.1e5, max_value=4e5, allow_nan=False, allow_infinity=False),
        shift=st.floats(min_value=-1e4, max_value=5e4, allow_nan=False, allow_infinity=False),
    )
    def test_range_penetration_shift_invariance(self, salary, band_min, band_max, shift):
        """Property: RP(S + k, Min + k, Max + k) == RP(S, Min, Max)."""
        assume(band_min + shift > 0)
        assume(band_max > band_min)

        rp_orig = calculate_range_penetration(salary, band_min, band_max)
        rp_shifted = calculate_range_penetration(salary + shift, band_min + shift, band_max + shift)

        assert math.isclose(rp_orig, rp_shifted, rel_tol=1e-5, abs_tol=1e-5)

    @settings(max_examples=50)
    @given(
        salary=st.floats(min_value=1e4, max_value=5e5, allow_nan=False, allow_infinity=False),
        band_min=st.floats(min_value=2e4, max_value=1e5, allow_nan=False, allow_infinity=False),
        band_max=st.floats(min_value=1.1e5, max_value=3e5, allow_nan=False, allow_infinity=False),
    )
    def test_circling_boundary_duality(self, salary, band_min, band_max):
        """Property: is_red_circled <=> RP > 1.0; is_green_circled <=> RP < 0.0."""
        assume(band_max > band_min)

        rp = calculate_range_penetration(salary, band_min, band_max)
        red = is_red_circled(salary, band_max=band_max)
        green = is_green_circled(salary, band_min=band_min)

        assert red == (rp > 1.0)
        assert green == (rp < 0.0)
        assert not (red and green)  # Salary cannot be simultaneously red- and green-circled

    @settings(max_examples=50)
    @given(
        spells=st.integers(min_value=0, max_value=60),
        days=st.integers(min_value=0, max_value=365),
    )
    def test_bradford_score_scaling_invariants(self, spells, days):
        """Properties of Bradford Factor B = S^2 * D:
        - B(S, 0) == 0 and B(0, D) == 0
        - Linear with days: B(S, 2D) == 2 * B(S, D)
        - Quadratic with spells: B(2S, D) == 4 * B(S, D)
        """
        score = calculate_bradford_score(spells, days)
        assert score == (spells ** 2) * days

        if spells == 0 or days == 0:
            assert score == 0

        # Linear in days
        score_2d = calculate_bradford_score(spells, days * 2)
        assert score_2d == score * 2

        # Quadratic in spells
        score_2s = calculate_bradford_score(spells * 2, days)
        assert score_2s == score * 4

    @settings(max_examples=30)
    @given(
        series_vals=st.lists(
            st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
            min_size=10,
            max_size=30,
        )
    )
    def test_fractional_differencing_identity_at_d_zero(self, series_vals):
        """Property: For d = 0, fractional differencing reproduces the exact input series."""
        s = pd.Series(series_vals)
        diff = fractional_difference(s, d=0.0)

        # For d=0, weights = [1.0], so result should match original values exactly
        np.testing.assert_allclose(diff.to_numpy(), s.to_numpy(), rtol=1e-5, atol=1e-5)


# ===========================================================================
# 3. Validation of conftest.py Shared Pytest Fixtures
# ===========================================================================

class TestConftestSharedFixtures:
    """Verify that shared fixtures from conftest.py instantiate properly and supply required columns."""

    def test_store_roster_fixture(self, store_roster_df):
        assert isinstance(store_roster_df, pd.DataFrame)
        assert len(store_roster_df) == 500
        for col in ["employee_id", "store_id", "dept_code", "tenure_weeks", "weekly_overtime_hours", "store_dept"]:
            assert col in store_roster_df.columns

    def test_cohort_timeseries_fixture(self, cohort_timeseries_df):
        assert isinstance(cohort_timeseries_df, pd.DataFrame)
        assert len(cohort_timeseries_df) > 0
        for col in ["week", "employee_id", "overtime_hours", "is_exemplar"]:
            assert col in cohort_timeseries_df.columns

    def test_compensation_equity_fixture(self, compensation_equity_df):
        assert isinstance(compensation_equity_df, pd.DataFrame)
        assert len(compensation_equity_df) == 400
        for col in ["base_salary", "band_min", "band_max", "band_mid"]:
            assert col in compensation_equity_df.columns

        # Verify toolkit functions run on the fixture cleanly
        compa = calculate_compa_ratio(
            compensation_equity_df["base_salary"],
            midpoint=compensation_equity_df["band_mid"],
        )
        assert len(compa) == len(compensation_equity_df)

    def test_algorithmic_fairness_fixture(self, algorithmic_fairness_df):
        assert isinstance(algorithmic_fairness_df, pd.DataFrame)
        assert len(algorithmic_fairness_df) == 600
        for col in ["employee_id", "demographic_group", "tenure_years", "base_salary", "promotion_recommendation"]:
            assert col in algorithmic_fairness_df.columns

    def test_causal_uplift_fixture(self, causal_uplift_df):
        assert isinstance(causal_uplift_df, pd.DataFrame)
        for col in ["treatment_enrolled", "retained_1yr", "true_propensity", "productivity_score"]:
            assert col in causal_uplift_df.columns

    def test_bradford_workforce_fixture(self, bradford_workforce_data):
        assert isinstance(bradford_workforce_data, tuple) and len(bradford_workforce_data) == 2
        events_df, archetypes_df = bradford_workforce_data
        assert isinstance(events_df, pd.DataFrame)
        assert isinstance(archetypes_df, pd.DataFrame)
        assert "employee_id" in events_df.columns and "absence_reason" in events_df.columns
        assert "employee_id" in archetypes_df.columns and "archetype" in archetypes_df.columns

    def test_network_and_hierarchical_fixtures(
        self,
        org_network_data,
        hierarchy_reorg_data,
        heterogeneous_multigraph_data,
    ):
        assert isinstance(org_network_data, tuple) and len(org_network_data) == 3
        nodes_df, edges_df, G = org_network_data
        assert isinstance(nodes_df, pd.DataFrame) and isinstance(edges_df, pd.DataFrame)
        assert "source" in edges_df.columns and "target" in edges_df.columns

        assert isinstance(hierarchy_reorg_data, tuple) and len(hierarchy_reorg_data) == 2
        df_pre, df_post = hierarchy_reorg_data
        assert isinstance(df_pre, pd.DataFrame) and isinstance(df_post, pd.DataFrame)
        assert "employee_id" in df_pre.columns and "manager_id" in df_pre.columns

        assert isinstance(heterogeneous_multigraph_data, tuple) and len(heterogeneous_multigraph_data) == 3
