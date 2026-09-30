"""
Tests for the lifecycle preference moment and calibration helpers.
"""

import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ogusa import estimate_lifecycle_params as elp

LAMBDAS = np.array(
    [0.25, 0.25, 0.2, 0.1, 0.1, 0.09, 0.005, 0.004, 0.0009, 0.0001]
)


class MockParams:
    """
    Minimal parameter object for lifecycle calibration helper tests.
    """

    S = 80
    J = 10
    starting_age = 20
    ending_age = 100
    lambdas = LAMBDAS.reshape(10, 1)
    omega_SS = np.ones(80) / 80
    rho = np.linspace(0.001, 1.0, 80)
    g_n_ss = 0.01
    g_y = 0.02
    delta = 0.05
    beta_annual = np.linspace(0.91, 0.995, 10)
    chi_b = np.ones(10) * 80
    chi_n = np.linspace(20, 80, 80)
    baseline_spending = False
    _data = {
        "beta_annual": {"validators": {"range": {"min": 0.0, "max": 0.9999}}},
        "chi_b": {"validators": {"range": {"min": 0.0, "max": 10000.0}}},
        "chi_n": {"validators": {"range": {"min": 0.0, "max": 10000.0}}},
    }


class MockParamsJointOmega(MockParams):
    """
    Parameter object with an (S, J) population distribution.
    """

    omega_SS = (np.ones(80) / 80).reshape(80, 1) * LAMBDAS.reshape(1, 10)


def _mock_ss_output(p, scale=1.0):
    """
    Build the subset of SS output used to warm-start repeated solves.
    """
    return {
        "b_sp1": np.ones((p.S, p.J)) * scale,
        "n": np.ones((p.S, p.J)) * 0.4 * scale,
        "r_p": 0.04 * scale,
        "r": 0.04 * scale,
        "w": 1.2 * scale,
        "p_m": np.ones(1) * scale,
        "B": 2.0 * scale,
        "K_d": 3.0 * scale,
        "Y": 10.0 * scale,
        "BQ": np.ones(p.J) * scale,
        "TR": 0.2 * scale,
        "G": 0.5 * scale,
        "factor": scale,
    }


def _type_constant_wealth(p):
    """
    Savings that rise with lifetime-income type and are flat in age.
    """
    return np.tile(np.arange(1, p.J + 1, dtype=float), (p.S, 1))


# ---------------------------------------------------------------------------
# Configuration and moment dimensions
# ---------------------------------------------------------------------------


def test_default_config_dimensions():
    """
    Default targets: 60 hours moments, one share per type, old-age ratio.
    """
    p = MockParams()
    config = elp.LifecycleCalibrationConfig()
    assert config.moment_ages.size == 60
    assert not config.include_wealth_profile
    assert not config.include_income_gini
    assert not config.include_savings_rate
    assert config.include_wealth_distribution
    assert config.include_wealth_income_ratio
    assert config.include_bequest_flow_ratio
    assert not config.include_old_age_wealth_ratio
    assert config.include_old_age_ratio_by_type

    ss_output = {
        "n": np.ones((p.S, p.J)) * 0.35,
        "b_sp1": _type_constant_wealth(p)
        * np.linspace(1, 3, p.S).reshape(p.S, 1),
        "before_tax_income": np.ones((p.S, p.J)) * 0.5,
        "BQ": np.ones(p.J),
        "Y": 10.0,
        "factor": 2.0,
    }
    moments = elp.compute_model_moments(ss_output, p, config)

    assert len(moments.names) == 60 + p.J + 2 + 6
    assert moments.names[0] == "labor_supply_age_20"
    assert moments.names[59] == "labor_supply_age_79"
    assert moments.names[60] == "wealth_share_0_25"
    assert moments.names[69] == "wealth_share_99p99_100"
    assert moments.names[70] == "wealth_income_ratio"
    assert moments.names[71] == "bequest_flow_ratio"
    assert moments.names[72] == "tilt_0_50_80_89_over_60_64"
    assert moments.names[77] == "tilt_99_100_80_89_over_60_64"
    assert np.allclose(moments.values[:60], 0.35)
    assert np.isclose(moments.values[60:70].sum(), 1.0)
    assert moments.values[70] > 0
    assert 0 < moments.values[71] < 1
    assert np.all(moments.values[72:78] > 1.0)


def test_config_validate_rejects_wealth_ages_at_starting_age():
    """
    Wealth at the starting age has no model counterpart in b_sp1.
    """
    p = MockParams()
    config = elp.LifecycleCalibrationConfig(
        include_wealth_profile=True, wealth_profile_min_age=20
    )
    with pytest.raises(ValueError, match="exceed the model starting age"):
        config.validate(p)


def test_config_validate_requires_bequest_data_when_included():
    """
    The bequest-to-output data value must accompany the flag.
    """
    p = MockParams()
    config = elp.LifecycleCalibrationConfig(include_bequest_to_output=True)
    with pytest.raises(ValueError, match="bequest_to_output_data"):
        config.validate(p)


# ---------------------------------------------------------------------------
# Wealth distribution moments with bins from lambdas
# ---------------------------------------------------------------------------


def test_wealth_share_bin_names_follow_lambdas():
    """
    Bin labels are cumulative population percentiles of lambdas.
    """
    names = elp.wealth_share_bin_names(LAMBDAS)
    assert names == (
        "wealth_share_0_25",
        "wealth_share_25_50",
        "wealth_share_50_70",
        "wealth_share_70_80",
        "wealth_share_80_90",
        "wealth_share_90_99",
        "wealth_share_99_99p5",
        "wealth_share_99p5_99p9",
        "wealth_share_99p9_99p99",
        "wealth_share_99p99_100",
    )
    assert elp.wealth_share_bin_names([1.0]) == ("wealth_share_0_100",)


def test_model_wealth_shares_match_type_shares():
    """
    With wealth flat in age and rising in type, bin shares are type shares.
    """
    p = MockParams()
    b_sp1 = _type_constant_wealth(p)
    shares = elp.model_wealth_shares({"b_sp1": b_sp1}, p)
    expected = LAMBDAS * np.arange(1, p.J + 1)
    expected = expected / expected.sum()

    assert shares.shape == (p.J,)
    assert np.isclose(shares.sum(), 1.0)
    # With wealth constant within type, sorting by wealth sorts by type and
    # each percentile bin is exactly one type, so shares match tightly even
    # for the smallest top bins.
    assert np.allclose(shares, expected, rtol=1e-8)
    assert np.all(shares[6:] > 0)


def test_percentile_bin_shares_split_straddling_cells():
    """
    A cutoff inside a cell allocates that cell's wealth proportionally.
    """
    dist = np.array([1.0, 2.0, 4.0])
    weights = np.array([0.5, 0.25, 0.25])
    shares = elp.percentile_bin_shares(dist, weights, np.array([0.6, 0.4]))
    # Bottom 60%: all of cell 1 (0.5) plus 0.1/0.25 of cell 2 (value 2).
    total = 1.0 * 0.5 + 2.0 * 0.25 + 4.0 * 0.25
    bottom = (1.0 * 0.5 + 2.0 * 0.1) / total
    assert np.allclose(shares, [bottom, 1.0 - bottom])
    assert np.isclose(shares.sum(), 1.0)
    # Weights need not be normalized on input.
    scaled = elp.percentile_bin_shares(dist, weights * 7, np.array([0.6, 0.4]))
    assert np.allclose(scaled, shares)


def test_model_wealth_shares_use_joint_population_weights():
    """
    An (S, J) omega_SS gives the same shares as the lambdas outer product.
    """
    b_sp1 = _type_constant_wealth(MockParams()) * np.linspace(
        1, 2, 80
    ).reshape(80, 1)
    shares_vec = elp.model_wealth_shares({"b_sp1": b_sp1}, MockParams())
    shares_joint = elp.model_wealth_shares(
        {"b_sp1": b_sp1}, MockParamsJointOmega()
    )
    assert np.allclose(shares_vec, shares_joint)


def test_model_wealth_distribution_optional_gini_and_var_log():
    """
    Gini and variance of logs append after the shares when requested.
    """
    p = MockParams()
    config = elp.LifecycleCalibrationConfig(
        include_wealth_gini=True, include_wealth_var_log=True
    )
    names = elp._wealth_distribution_moment_names(
        LAMBDAS, config.include_wealth_gini, config.include_wealth_var_log
    )
    values = elp._model_wealth_distribution_moments(
        {"b_sp1": _type_constant_wealth(p)}, p, config
    )
    assert names[-2:] == ("wealth_gini", "wealth_var_log")
    assert values.shape == (p.J + 2,)
    assert 0 < values[-2] < 1
    assert values[-1] > 0


def test_data_wealth_shares_use_lambdas_bins():
    """
    SCF shares are computed on bins equal to p.lambdas.
    """
    p = MockParams()
    rng = np.random.default_rng(1)
    scf = pd.DataFrame(
        {
            "age": rng.integers(21, 80, size=5000),
            "networth_infadj": np.exp(rng.normal(11, 1.5, size=5000)),
            "wgt": np.ones(5000),
        }
    )
    config = elp.LifecycleCalibrationConfig()
    shares = elp._data_wealth_distribution_moments(scf, p, config)
    assert shares.shape == (p.J,)
    assert np.isclose(shares.sum(), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Age indexing, profiles, and the old-age wealth ratio
# ---------------------------------------------------------------------------


def test_wealth_age_indices_shift_by_one_year():
    """
    Wealth held at age a is savings chosen at age a - 1.
    """
    p = MockParams()
    assert np.array_equal(
        elp._wealth_age_indices(np.array([21, 22, 79]), p),
        np.array([0, 1, 58]),
    )
    with pytest.raises(ValueError):
        elp._wealth_age_indices(np.array([20]), p)


def test_wealth_profile_model_moment_uses_shifted_index():
    """
    The level wealth profile at age 21 equals b_sp1[0] times factor.
    """
    p = MockParams()
    config = elp.LifecycleCalibrationConfig(
        include_labor_profile=False,
        include_wealth_distribution=False,
        include_wealth_income_ratio=False,
        include_bequest_flow_ratio=False,
        include_old_age_ratio_by_type=False,
        include_wealth_profile=True,
        wealth_profile_moment="level",
    )
    b_sp1 = np.tile(np.arange(1, p.S + 1, dtype=float).reshape(p.S, 1), p.J)
    moments = elp.compute_model_moments(
        {"b_sp1": b_sp1, "factor": 2.0}, p, config
    )
    assert moments.names[0] == "net_wealth_age_21"
    assert np.isclose(moments.values[0], 2.0 * b_sp1[0, 0])
    assert np.isclose(moments.values[-1], 2.0 * b_sp1[58, 0])


def test_model_old_age_wealth_ratio():
    """
    The ratio compares population-weighted wealth across two age windows.
    """
    p = MockParams()
    config = elp.LifecycleCalibrationConfig()
    b_sp1 = np.tile(np.arange(1, p.S + 1, dtype=float).reshape(p.S, 1), p.J)
    ratio = elp.model_old_age_wealth_ratio({"b_sp1": b_sp1}, p, config)
    # Ages 75-79 hold b_sp1[54:59] (55..59); ages 60-64 hold b_sp1[39:44].
    assert np.isclose(
        ratio, np.mean(np.arange(55, 60)) / np.mean(np.arange(40, 45))
    )


def test_old_age_wealth_ratio_from_scf():
    """
    Data ratio is a weighted mean over all households in each age window.
    """
    scf = pd.DataFrame(
        {
            "age": [60, 61, 64, 75, 79, 79],
            "networth_infadj": [100.0, 100.0, 100.0, 50.0, 50.0, 200.0],
            "wgt": [1.0, 1.0, 1.0, 1.0, 1.0, 2.0],
        }
    )
    config = elp.LifecycleCalibrationConfig()
    ratio = elp.old_age_wealth_ratio_from_scf(scf, config)
    assert np.isclose(ratio, ((50 + 50 + 400) / 4) / 100.0)


def test_smooth_profile_centered_with_shrinking_edges():
    """
    Smoothing averages neighbours and leaves the length unchanged.
    """
    values = np.array([0.0, 3.0, 6.0, 9.0])
    smoothed = elp._smooth_profile(values, 3)
    assert np.allclose(smoothed, [1.5, 3.0, 6.0, 7.5])
    assert np.allclose(elp._smooth_profile(values, 1), values)


def test_labor_profile_from_cps_includes_zero_hours_and_smooths():
    """
    Non-workers count as zero hours; the profile is a share of 112 hours.
    """
    p = MockParams()
    config = elp.LifecycleCalibrationConfig(labor_smoothing_window=1)
    cps = pd.DataFrame(
        {
            "age": np.repeat(np.arange(20, 80), 2),
            "hours_per_week": np.tile([56.0, np.nan], 60),
            "weight": np.ones(120),
        }
    )
    labor = elp.labor_profile_from_cps(cps, config)
    assert labor.shape == (60,)
    assert np.allclose(labor, 28.0 / 112.0)

    smoothed_config = elp.LifecycleCalibrationConfig(labor_smoothing_window=3)
    cps_trend = pd.DataFrame(
        {
            "age": np.arange(20, 80),
            "hours_per_week": np.arange(60, dtype=float),
            "weight": np.ones(60),
        }
    )
    smoothed = elp.labor_profile_from_cps(cps_trend, smoothed_config)
    assert np.isclose(smoothed[0], 0.5 / 112.0)
    assert np.isclose(smoothed[10], 10.0 / 112.0)
    config.validate(p)


def test_type_weights_by_age_reduce_to_lambdas_with_vector_omega():
    """
    With a length-S omega_SS, within-age type weights are lambdas.
    """
    weights = elp._type_weights_by_age(MockParams())
    assert weights.shape == (80, 10)
    assert np.allclose(weights, LAMBDAS.reshape(1, 10))
    joint = elp._joint_pop_weights(MockParamsJointOmega())
    assert np.isclose(joint.sum(), 1.0)


# ---------------------------------------------------------------------------
# Data moments end to end on synthetic microdata
# ---------------------------------------------------------------------------


def test_compute_data_moments_with_synthetic_microdata():
    """
    Default data moments need only CPS hours and SCF wealth microdata.
    """
    p = MockParams()
    rng = np.random.default_rng(0)
    ages = np.arange(20, 80)
    cps = pd.DataFrame(
        {
            "age": ages,
            "hours_per_week": np.linspace(20, 40, ages.size),
            "weight": np.ones(ages.size),
        }
    )
    scf_ages = rng.integers(18, 96, size=4000)
    wealth_values = np.exp(rng.normal(11, 1.5, size=4000)) * (scf_ages / 40.0)
    scf = pd.DataFrame(
        {
            "age": scf_ages,
            "networth_infadj": wealth_values,
            "networth": 1.0,
            "wgt": np.ones(4000),
            "income": 60000.0 + wealth_values * 0.02,
            "ssretinc": np.where(scf_ages >= 65, 15000.0, 0.0),
            "transfothinc": 1000.0,
        }
    )
    config = elp.LifecycleCalibrationConfig()
    moments = elp.compute_data_moments(p, config, cps=cps, scf=scf)

    assert len(moments.names) == 60 + p.J + 2 + 6
    assert moments.names[-6:] == elp.tilt_moment_names(config, p)
    assert np.all(moments.values[-6:] > 0)
    assert moments.names[60:70] == elp.wealth_share_bin_names(LAMBDAS)
    assert np.isclose(moments.values[60:70].sum(), 1.0, atol=1e-6)
    assert moments.names[70] == "wealth_income_ratio"
    in_model_ages = scf[scf["age"] >= 20]
    expected_ratio = (
        in_model_ages["networth_infadj"].mean()
        / (
            in_model_ages["income"]
            - in_model_ages["ssretinc"]
            - in_model_ages["transfothinc"]
        ).mean()
    )
    assert np.isclose(moments.values[70], expected_ratio)
    assert moments.names[71] == "bequest_flow_ratio"
    assert 0 < moments.values[71] < 1

    model_like = elp.compute_model_moments(
        {
            "n": np.ones((p.S, p.J)) * 0.3,
            "b_sp1": _type_constant_wealth(p),
            "before_tax_income": np.ones((p.S, p.J)),
            "BQ": np.ones(p.J),
            "Y": 10.0,
        },
        p,
        config,
    )
    assert model_like.names == moments.names
    frame = moments.to_frame(model_like)
    assert list(frame.columns) == ["moment", "data", "model"]


def test_compute_data_moments_optional_wealth_profile_and_bequests():
    """
    Optional moments append in the documented order.
    """
    p = MockParams()
    ages = np.arange(20, 80)
    cps = pd.DataFrame({"age": ages, "hours_per_week": 30.0, "weight": 1.0})
    scf = pd.DataFrame(
        {
            "age": np.repeat(np.arange(21, 80), 3),
            "networth_infadj": np.repeat(np.linspace(1000, 100000, 59), 3),
            "wgt": 1.0,
        }
    )
    config = elp.LifecycleCalibrationConfig(
        include_wealth_profile=True,
        include_wealth_income_ratio=False,
        include_bequest_flow_ratio=False,
        include_old_age_ratio_by_type=False,
        include_bequest_to_output=True,
        bequest_to_output_data=0.03,
    )
    moments = elp.compute_data_moments(p, config, cps=cps, scf=scf)
    assert moments.names[60] == "net_wealth_age_21"
    assert moments.names[118] == "net_wealth_age_79"
    profile = moments.values[60:119]
    assert np.isclose(profile.mean(), 1.0)
    assert moments.names[-1] == "bequest_to_output"
    assert np.isclose(moments.values[-1], 0.03)
    model = elp.compute_model_moments(
        {
            "n": np.ones((p.S, p.J)) * 0.3,
            "b_sp1": _type_constant_wealth(p),
            "BQ": np.ones(p.J) * 0.05,
            "Y": 10.0,
            "factor": 3.0,
        },
        p,
        config,
    )
    assert np.isclose(model.values[-1], 0.5 / 10.0)


def test_model_wealth_income_ratio_and_bequest_flow_ratio():
    """
    Level and bequest-flow moments follow their population-weighted formulas.
    """
    p = MockParams()
    b_sp1 = np.tile(np.arange(1, p.S + 1, dtype=float).reshape(p.S, 1), p.J)
    income = np.ones((p.S, p.J)) * 4.0
    ss_output = {"b_sp1": b_sp1, "before_tax_income": income}
    joint = elp._joint_pop_weights(p)

    ratio = elp.model_wealth_income_ratio(ss_output, p)
    # Wealth held by the living excludes the last row of b_sp1 and is
    # averaged over the whole population, including the starting age.
    expected = (b_sp1[:-1] * joint[1:]).sum() / joint.sum() / 4.0
    assert np.isclose(ratio, expected)

    flow = elp.model_bequest_flow_ratio(ss_output, p)
    rho = p.rho.reshape(-1, 1)
    assert np.isclose(
        flow, (rho * joint * b_sp1).sum() / (joint * b_sp1).sum()
    )
    assert 0 < flow < 1


def test_scf_income_series_and_bequest_flow_from_scf():
    """
    SCF income concepts and the mortality-weighted bequest flow.
    """
    p = MockParams()
    scf = pd.DataFrame(
        {
            "age": [30, 70, 99, 120, 15],
            "networth_infadj": [100.0, 300.0, 500.0, 700.0, 900.0],
            "wgt": [1.0, 1.0, 1.0, 1.0, 1.0],
            "income": [50.0, 40.0, 30.0, 20.0, 10.0],
            "ssretinc": [0.0, 20.0, 20.0, 10.0, 0.0],
            "transfothinc": [1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )
    assert np.allclose(
        elp.scf_income_series(scf, "pre_transfer"), [49, 19, 9, 9, 9]
    )
    assert np.allclose(elp.scf_income_series(scf, "total"), scf["income"])
    with pytest.raises(ValueError):
        elp.scf_income_series(scf.drop(columns="income"), "total")

    flow = elp.bequest_flow_ratio_from_scf(scf, p)
    # Age 15 is dropped; age 120 is clipped to 99, the last model age.
    rho = p.rho
    kept = np.array([100.0, 300.0, 500.0, 700.0])
    rhos = np.array([rho[10], rho[50], rho[79], rho[79]])
    assert np.isclose(flow, (rhos * kept).sum() / kept.sum())

    config = elp.LifecycleCalibrationConfig()
    ratio = elp.wealth_income_ratio_from_scf(scf, p, config)
    assert np.isclose(ratio, kept.mean() / np.mean([49, 19, 9, 9]))


def test_tilt_bins_merge_top_one_percent():
    """
    Default tilt bins are lambdas with the top one percent merged.
    """
    p = MockParams()
    config = elp.LifecycleCalibrationConfig()
    bins = config.tilt_bins(p)
    assert np.allclose(bins, [0.5, 0.2, 0.1, 0.1, 0.09, 0.01])
    names = elp.tilt_moment_names(config, p)
    assert len(names) == 6
    assert names[0] == "tilt_0_50_80_89_over_60_64"
    assert names[-1] == "tilt_99_100_80_89_over_60_64"
    assert elp.merged_type_groups(LAMBDAS) == [
        [0, 1],
        [2],
        [3],
        [4],
        [5],
        [6, 7, 8, 9],
    ]
    assert elp.merged_type_groups([0.6, 0.4]) == [[0], [1]]


def test_old_age_ratio_by_type_model_and_data():
    """
    With wealth scaled by age, every bin's tilt equals the age scaling.
    """
    p = MockParams()
    config = elp.LifecycleCalibrationConfig(include_old_age_ratio_by_type=True)
    age_scale = np.linspace(1.0, 3.0, p.S).reshape(p.S, 1)
    b_sp1 = _type_constant_wealth(p) * age_scale
    model = elp.model_old_age_ratio_by_type({"b_sp1": b_sp1}, p, config)
    hold_num = elp._age_indices(config.tilt_numerator_age_labels, p)
    hold_den = elp._age_indices(config.tilt_denominator_age_labels, p)
    expected = age_scale[hold_num - 1].mean() / age_scale[hold_den - 1].mean()
    assert model.shape == (6,)
    assert np.allclose(model, expected, rtol=1e-6)

    rng = np.random.default_rng(3)
    ages = np.concatenate(
        [rng.integers(60, 65, 3000), rng.integers(80, 90, 3000)]
    )
    base = np.exp(rng.normal(11, 1.0, size=6000))
    scf = pd.DataFrame(
        {
            "age": ages,
            "networth_infadj": base * np.where(ages >= 80, 0.8, 1.0),
            "wgt": 1.0,
        }
    )
    data = elp.old_age_ratio_by_type_from_scf(scf, p, config)
    assert data.shape == (6,)
    assert np.all(data > 0)


def test_savings_rate_data_moment_uses_macro_moment(monkeypatch):
    """
    Test that the savings-rate data moment comes from macro moments.
    """

    def fake_get_macro_moments(year):
        assert year == 2030
        return {elp.SAVINGS_RATE_DATA_LABEL: 0.18}

    monkeypatch.setattr(
        elp.compute_moments,
        "get_macro_moments",
        fake_get_macro_moments,
    )

    assert np.allclose(elp.savings_rate_data_moment(macro_year=2030), 0.18)


def test_compute_inheritance_moments_from_scf():
    """
    Test optional inherited-transfer moments from full SCF-like data.
    """
    scf = pd.DataFrame(
        {
            "inheritance": [0.0, 100.0, 200.0],
            "received": [0, 1, 1],
            "networth_infadj": [50.0, 1000.0, 2000.0],
            "wgt": [1.0, 1.0, 2.0],
        }
    )

    moments = elp.compute_inheritance_moments_from_scf(
        scf,
        amount_col="inheritance",
        received_col="received",
    )

    assert moments.names == (
        "inheritance_received_rate",
        "inheritance_amount_conditional_mean",
        "inheritance_to_networth_mean",
    )
    assert np.allclose(moments.values[0], 0.75)
    assert np.allclose(moments.values[1], (100 + 2 * 200) / 3)


# ---------------------------------------------------------------------------
# Parameter packing and SMM helpers
# ---------------------------------------------------------------------------


def test_pack_unpack_lifecycle_params_roundtrip():
    """
    Test transformed packing and unpacking of lifecycle parameters.
    """
    p = MockParams()
    config = elp.LifecycleCalibrationConfig()
    beta = np.linspace(0.92, 0.99, p.J)
    chi_b = np.linspace(50, 100, p.J)
    ages = np.arange(p.starting_age, p.starting_age + p.S)
    basis, _ = elp._build_chi_n_spline_basis(
        ages,
        config.chi_n_n_spline_knots,
        config.chi_n_spline_degree,
    )
    gamma = np.linspace(np.log(10), np.log(70), config.chi_n_n_spline_knots)
    chi_n = np.exp(basis @ gamma)

    theta = elp.pack_lifecycle_params(beta, chi_b, chi_n, p, config)
    unpacked = elp.unpack_lifecycle_params(
        theta,
        p,
        config,
        base_chi_n=p.chi_n,
    )

    assert theta.size == 30
    assert np.allclose(unpacked["beta_annual"], beta)
    assert np.allclose(unpacked["chi_b"], chi_b)
    assert np.allclose(unpacked["chi_n"], chi_n)


def test_build_chi_n_profile_scales_default_tail():
    """
    Test that the default chi_n tail shape is preserved and scaled.
    """
    p = MockParams()
    config = elp.LifecycleCalibrationConfig()
    base_chi_n = np.arange(1, p.S + 1, dtype=float)
    estimated = np.ones(config.estimated_chi_n_ages.size) * 10

    full = elp.build_chi_n_profile(estimated, base_chi_n, p, config)

    assert np.allclose(full[:60], 10)
    expected_scale = 10 / base_chi_n[59]
    assert np.allclose(full[60:], base_chi_n[60:] * expected_scale)


def test_extract_dfols_bounds_in_transformed_space():
    """
    Bounds intersect validators with config and live in logit/log space.
    """
    p = MockParams()
    config = elp.LifecycleCalibrationConfig()
    lower, upper = elp._extract_dfols_bounds(p, config)

    n_params = 2 * p.J + config.chi_n_n_spline_knots
    assert lower.shape == (n_params,)
    assert upper.shape == (n_params,)
    assert np.all(upper > lower)
    assert np.isclose(lower[0], np.log(0.8 / 0.2))
    assert np.isclose(upper[0], np.log(0.999 / 0.001))
    assert np.isclose(lower[p.J], np.log(0.1))
    assert np.isclose(upper[p.J], np.log(200.0))
    assert np.isclose(lower[2 * p.J], np.log(config.bound_epsilon))
    assert np.isclose(upper[-1], np.log(10000.0))

    theta0 = elp.pack_lifecycle_params(
        p.beta_annual, p.chi_b, p.chi_n, p, config
    )
    assert np.all(theta0 >= lower) and np.all(theta0 <= upper)


def test_weighting_matrix_handles_singular_bootstrap_vcv():
    """
    Test pseudo-inverse/ridge handling for singular bootstrap VCV matrices.
    """
    boot = np.ones((5, 3))

    W = elp.weighting_matrix(
        3,
        method="optimal",
        bootstrap_moments=boot,
        ridge=1e-6,
    )

    assert W.shape == (3, 3)
    assert np.all(np.isfinite(W))
    assert np.allclose(W, W.T)


def test_smm_distance_supports_relative_and_absolute_residuals():
    """
    Test percentage-deviation and level-deviation distance calculations.
    """
    model = elp.MomentSet(("a", "b", "c"), np.array([2.0, 6.0, 2e-9]))
    data = elp.MomentSet(("a", "b", "c"), np.array([1.0, 3.0, 0.0]))
    W = np.eye(3)

    absolute = elp.smm_distance(model, data, W, method="absolute")
    relative = elp.smm_distance(
        model,
        data,
        W,
        method="relative",
        floor=1e-8,
    )

    assert np.allclose(absolute, 1.0**2 + 3.0**2 + (2e-9) ** 2)
    assert np.allclose(relative, 1.0**2 + 1.0**2 + 0.2**2)


def test_smm_residual_and_objective_use_bounded_failure_penalty(
    monkeypatch, caplog
):
    """
    Solver failures return a finite, bounded penalty rather than 1e15.
    """
    p = MockParams()
    config = elp.LifecycleCalibrationConfig(failure_residual=5.0)
    data = elp.MomentSet(("a", "b"), np.array([1.0, 2.0]))
    W = np.eye(2)

    def failing(*args, **kwargs):
        raise ValueError("no steady state")

    monkeypatch.setattr(elp, "_evaluate_model_moments", failing)
    caplog.set_level(logging.WARNING, logger=elp.logger.name)

    residual = elp.smm_residual(np.zeros(30), data, W, p, config)
    objective = elp.smm_objective(np.zeros(30), data, W, p, config)

    assert np.allclose(residual, 5.0)
    assert np.isclose(objective, 2 * 25.0)
    assert "evaluation failed" in caplog.text

    def huge(*args, **kwargs):
        return elp.MomentSet(("a", "b"), np.array([1e6, 2.0]))

    monkeypatch.setattr(elp, "_evaluate_model_moments", huge)
    clipped = elp.smm_residual(np.zeros(30), data, W, p, config)
    assert np.isclose(clipped[0], 5.0)
    assert np.isclose(clipped[1], 0.0)


def test_estimate_lifecycle_params_runs_dfols_from_each_start(
    monkeypatch, caplog
):
    """
    The SMM driver calls DFO-LS once per start and keeps the best result.
    """
    p = MockParams()
    config = elp.LifecycleCalibrationConfig(
        include_labor_profile=False,
        include_wealth_distribution=False,
        include_old_age_wealth_ratio=False,
        n_starts=2,
        log_optimizer_progress=True,
    )
    data_moments = elp.MomentSet(("moment",), np.array([0.0]))
    theta0 = elp.pack_lifecycle_params(
        p.beta_annual, p.chi_b, p.chi_n, p, config
    )
    objectives = iter([4.0, 3.0])
    calls = []

    def fake_solve(residual_fn, x0, bounds=None, **kwargs):
        calls.append(x0.copy())
        residual = residual_fn(x0)
        assert residual.shape == (1,)
        return SimpleNamespace(
            x=x0, obj=next(objectives), nf=7, msg="ok", success=True
        )

    monkeypatch.setattr(elp, "dfols", SimpleNamespace(solve=fake_solve))
    monkeypatch.setattr(elp, "apply_lifecycle_params", lambda *a, **k: None)
    monkeypatch.setattr(
        elp,
        "solve_ss_with_cache",
        lambda *args, **kwargs: _mock_ss_output(p),
    )
    monkeypatch.setattr(
        elp, "compute_model_moments", lambda *a, **k: data_moments
    )
    caplog.set_level(logging.INFO, logger=elp.logger.name)

    result = elp.estimate_lifecycle_params(
        p, config=config, theta0=theta0, data_moments=data_moments
    )

    assert len(calls) == 2
    assert result.objective_value == 3.0
    assert result.best_start_index == 1
    assert len(result.all_start_results) == 2
    assert result.beta_annual.shape == (p.J,)
    assert result.chi_n.shape == (p.S,)
    assert "DFO-LS start 1/2" in caplog.text
    assert "objective=3.000000e+00" in caplog.text


def test_estimate_lifecycle_params_requires_dfols(monkeypatch):
    """
    A clear ImportError is raised when dfo-ls is unavailable.
    """
    monkeypatch.setattr(elp, "dfols", None)
    with pytest.raises(ImportError, match="dfo-ls"):
        elp.estimate_lifecycle_params(MockParams())


# ---------------------------------------------------------------------------
# Steady-state warm starts
# ---------------------------------------------------------------------------


def test_solve_ss_with_cache_uses_run_ss_then_ss_solver(monkeypatch):
    """
    Repeated SS solves warm start SS_solver by keyword from the last output.
    """
    p = MockParams()
    calls = []

    def fake_run_ss(params, client=None):
        calls.append(("run", client))
        return _mock_ss_output(params, scale=1.0)

    def fake_ss_solver(
        bmat,
        nmat,
        r_p,
        r,
        w,
        p_m,
        Y,
        BQ,
        G,
        TR,
        Ig_baseline,
        factor,
        p,
        client,
        fsolve_flag=False,
    ):
        calls.append(("solver", factor, client))
        assert np.allclose(bmat, np.ones((p.S, p.J)))
        assert np.allclose(nmat, np.ones((p.S, p.J)) * 0.4)
        assert np.isclose(r_p, 0.04)
        assert np.isclose(r, 0.04)
        assert np.isclose(w, 1.2)
        assert np.allclose(p_m, np.ones(1))
        assert np.isclose(Y, 10.0)
        assert np.allclose(BQ, np.ones(p.J))
        assert np.isclose(G, 0.5)
        assert np.isclose(TR, 0.2)
        assert Ig_baseline is None
        return _mock_ss_output(p, scale=2.0)

    monkeypatch.setattr(elp.SS, "run_SS", fake_run_ss)
    monkeypatch.setattr(elp.SS, "SS_solver", fake_ss_solver)

    cache = elp.SSSolutionCache(use_ss_solver=True)
    first = elp.solve_ss_with_cache(p, client="client", ss_cache=cache)
    second = elp.solve_ss_with_cache(p, client="client", ss_cache=cache)

    assert [call[0] for call in calls] == ["run", "solver"]
    assert first["factor"] == 1.0
    assert second["factor"] == 2.0
    assert cache.previous_output is second


def test_solve_ss_with_cache_matches_older_solver_signature(monkeypatch):
    """
    A solver without the G argument is still called correctly.
    """
    p = MockParams()

    def fake_ss_solver(
        bmat, nmat, r_p, r, w, p_m, Y, BQ, TR, Ig_baseline, factor, p, client
    ):
        return _mock_ss_output(p, scale=6.0)

    monkeypatch.setattr(elp.SS, "SS_solver", fake_ss_solver)
    monkeypatch.setattr(
        elp.SS, "run_SS", lambda *a, **k: pytest.fail("cold solve used")
    )
    cache = elp.SSSolutionCache(
        previous_output=_mock_ss_output(p), use_ss_solver=True
    )
    output = elp.solve_ss_with_cache(p, ss_cache=cache)
    assert output["factor"] == 6.0


def test_solve_ss_with_cache_falls_back_to_run_ss_and_warns(
    monkeypatch, caplog
):
    """
    Failed warm starts log a warning and fall back to SS.run_SS.
    """
    p = MockParams()
    calls = []

    def fake_run_ss(params, client=None):
        calls.append("run")
        return _mock_ss_output(params, scale=3.0)

    def fake_ss_solver(**kwargs):
        calls.append("solver")
        raise RuntimeError("failed warm start")

    monkeypatch.setattr(elp.SS, "run_SS", fake_run_ss)
    monkeypatch.setattr(elp.SS, "SS_solver", fake_ss_solver)
    caplog.set_level(logging.WARNING, logger=elp.logger.name)

    cache = elp.SSSolutionCache(
        previous_output=_mock_ss_output(p, scale=1.0),
        use_ss_solver=True,
    )
    output = elp.solve_ss_with_cache(p, ss_cache=cache)

    assert calls == ["solver", "run"]
    assert output["factor"] == 3.0
    assert cache.previous_output is output
    assert "SS warm start failed" in caplog.text
    assert "failed warm start" in caplog.text


def test_solve_ss_with_cache_warns_on_unknown_required_argument(
    monkeypatch, caplog
):
    """
    A solver signature with an unexpected required argument is not guessed.
    """
    p = MockParams()
    calls = []

    def fake_ss_solver(bmat, nmat, brand_new_arg, p, client):
        calls.append("solver")
        return _mock_ss_output(p)

    monkeypatch.setattr(elp.SS, "SS_solver", fake_ss_solver)
    monkeypatch.setattr(
        elp.SS,
        "run_SS",
        lambda *a, **k: calls.append("run") or _mock_ss_output(p, 9.0),
    )
    caplog.set_level(logging.WARNING, logger=elp.logger.name)
    cache = elp.SSSolutionCache(
        previous_output=_mock_ss_output(p), use_ss_solver=True
    )
    output = elp.solve_ss_with_cache(p, ss_cache=cache)

    assert calls == ["run"]
    assert output["factor"] == 9.0
    assert "brand_new_arg" in caplog.text


def test_solve_ss_with_cache_can_disable_ss_solver(monkeypatch):
    """
    Test that the restart path can be disabled for robustness checks.
    """
    p = MockParams()
    calls = []

    def fake_run_ss(params, client=None):
        calls.append("run")
        return _mock_ss_output(params, scale=4.0)

    def fake_ss_solver(*args, **kwargs):
        calls.append("solver")
        return _mock_ss_output(p, scale=5.0)

    monkeypatch.setattr(elp.SS, "run_SS", fake_run_ss)
    monkeypatch.setattr(elp.SS, "SS_solver", fake_ss_solver)

    cache = elp.SSSolutionCache(
        previous_output=_mock_ss_output(p, scale=1.0),
        use_ss_solver=False,
    )
    output = elp.solve_ss_with_cache(p, ss_cache=cache)

    assert calls == ["run"]
    assert output["factor"] == 4.0
