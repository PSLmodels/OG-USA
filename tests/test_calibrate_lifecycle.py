"""
Tests for the household-only steady-state solve used in calibration.
"""

import json
import logging
from importlib import resources
from types import SimpleNamespace

import numpy as np
from dataclasses import replace
import pytest

from ogusa import calibrate_lifecycle as cl


class MockParams:
    """
    Minimal parameter object for the household solver wrapper.
    """

    S = 4
    J = 2
    FOC_root_method = "hybr"
    e = np.ones((2, 4, 2)) * 1.5


def _ss_output(p, scale=1.0):
    """
    Steady-state output with the household-level inputs already present.
    """
    return {
        "r_p": 0.04,
        "w": 1.2,
        "factor": 100000.0,
        "p_m": np.array([1.0]),
        "p_i": np.array([1.0]),
        "p_tilde": np.array(1.0),
        "BQ": np.array([0.1, 0.2]),
        "RM": 0.0,
        "TR": 0.3,
        "Y": 10.0,
        "bq": np.ones((p.S, p.J)) * 0.01,
        "rm": np.zeros((p.S, p.J)),
        "tr": np.ones((p.S, p.J)) * 0.02,
        "ubi": np.zeros((p.S, p.J)),
        "b_sp1": np.ones((p.S, p.J)) * scale,
        "n": np.ones((p.S, p.J)) * 0.3 * scale,
    }


def _fake_solve_for_j(offset):
    """
    Return a fake SS.solve_for_j whose root is the guess plus an offset.
    """

    def fake(
        guesses, r_p, w, p_tilde, p_i, bq_j, rm_j, tr_j, ubi_j, factor, j, p
    ):
        assert bq_j.shape == (p.S,)
        assert tr_j.shape == (p.S,)
        assert np.isclose(r_p, 0.04)
        return SimpleNamespace(
            x=np.asarray(guesses) + offset + j,
            fun=np.full(2 * p.S, 1e-12 * (j + 1)),
            success=j == 0,
        )

    return fake


def test_household_environment_from_stored_household_inputs():
    """
    Household-level arrays in the SS output are used directly.
    """
    p = MockParams()
    env = cl.HouseholdEnvironment.from_ss_output(_ss_output(p), p)
    assert env.r_p == 0.04
    assert env.w == 1.2
    assert env.p_tilde == 1.0
    assert env.p_i.shape == (1,)
    assert env.bq.shape == (p.S, p.J)
    assert np.allclose(env.tr, 0.02)
    assert env.factor == 100000.0


def test_solve_households_serial_collects_results(monkeypatch, caplog):
    """
    Serial solves return savings, labor, Euler errors, and success flags.
    """
    p = MockParams()
    monkeypatch.setattr(cl.SS, "solve_for_j", _fake_solve_for_j(0.5))
    caplog.set_level(logging.WARNING, logger=cl.logger.name)
    env = cl.HouseholdEnvironment.from_ss_output(_ss_output(p), p)

    solution = cl.solve_households(
        env, p, np.ones((p.S, p.J)), np.ones((p.S, p.J)) * 0.3
    )

    assert solution.b_sp1.shape == (p.S, p.J)
    assert np.allclose(solution.b_sp1[:, 0], 1.5)
    assert np.allclose(solution.b_sp1[:, 1], 2.5)
    assert np.allclose(solution.n[:, 0], 0.8)
    assert solution.euler_errors.shape == (2 * p.S, p.J)
    assert np.isclose(solution.max_abs_euler_error, 2e-12)
    assert solution.success.tolist() == [True, False]
    assert not solution.all_converged
    assert "types [1]" in caplog.text


def test_solve_households_rejects_bad_guess_shapes():
    """
    Guess arrays must be (S, J).
    """
    p = MockParams()
    env = cl.HouseholdEnvironment.from_ss_output(_ss_output(p), p)
    with pytest.raises(ValueError, match=r"\(S, J\)"):
        cl.solve_households(env, p, np.ones(p.S), np.ones((p.S, p.J)))


def test_solve_households_parallel_falls_back_to_serial(monkeypatch, caplog):
    """
    A failing Dask client triggers a warning and a serial solve.
    """
    p = MockParams()
    monkeypatch.setattr(cl.SS, "solve_for_j", _fake_solve_for_j(0.0))
    monkeypatch.setattr(cl, "_scatter_params", lambda p, client: "future")

    class FailingClient:
        def submit(self, *args, **kwargs):
            raise RuntimeError("no workers")

        def gather(self, futures):  # pragma: no cover - never reached
            return futures

    caplog.set_level(logging.WARNING, logger=cl.logger.name)
    env = cl.HouseholdEnvironment.from_ss_output(_ss_output(p), p)
    solution = cl.solve_households(
        env,
        p,
        np.ones((p.S, p.J)),
        np.ones((p.S, p.J)),
        client=FailingClient(),
    )
    assert np.allclose(solution.b_sp1[:, 0], 1.0)
    assert "solving types serially" in caplog.text


def test_solve_households_uses_client_when_available(monkeypatch):
    """
    With a working client, each type is submitted once and gathered.
    """
    p = MockParams()
    fake = _fake_solve_for_j(0.0)
    monkeypatch.setattr(cl.SS, "solve_for_j", fake)
    monkeypatch.setattr(cl, "_scatter_params", lambda p, client: p)

    class Client:
        def __init__(self):
            self.submitted = []

        def submit(self, fn, *args):
            self.submitted.append(args[-1])
            return fn(*args)

        def gather(self, futures):
            return futures

    client = Client()
    env = cl.HouseholdEnvironment.from_ss_output(_ss_output(p), p)
    solution = cl.solve_households(
        env, p, np.ones((p.S, p.J)), np.ones((p.S, p.J)), client=client
    )
    assert client.submitted == [0, 1]
    assert np.allclose(solution.b_sp1[:, 1], 2.0)


def test_partial_equilibrium_ss_updates_household_arrays_only(monkeypatch):
    """
    The returned dict swaps b_sp1, b_s, and n and keeps aggregates.
    """
    p = MockParams()
    monkeypatch.setattr(cl.SS, "solve_for_j", _fake_solve_for_j(1.0))
    ss_output = _ss_output(p)

    updated, solution = cl.partial_equilibrium_ss(ss_output, p)

    assert updated is not ss_output
    assert np.allclose(updated["b_sp1"][:, 0], 2.0)
    assert np.allclose(updated["b_s"][0, :], 0.0)
    assert np.allclose(updated["b_s"][1:, 0], 2.0)
    assert np.allclose(updated["n"][:, 1], 0.3 + 2.0)
    assert updated["Y"] == 10.0
    # Before-tax income is recomputed as r_p * b_s + w * e * n.
    expected_income = 0.04 * updated["b_s"] + 1.2 * 1.5 * updated["n"]
    assert np.allclose(updated["before_tax_income"], expected_income)
    assert np.allclose(ss_output["b_sp1"], 1.0)
    assert solution.all_converged is False


@pytest.mark.local
def test_partial_equilibrium_reproduces_general_equilibrium_households():
    """
    At equilibrium prices, the household-only solve returns the GE solution.

    This solves the full OG-USA steady state, so it is local only.
    """
    import ogcore
    from ogcore.parameters import Specifications
    from ogusa import estimate_lifecycle_params as elp

    ogcore.config.VERBOSE = False
    p = Specifications(baseline=True, num_workers=1)
    with (
        resources.files("ogusa")
        .joinpath("ogusa_default_parameters.json")
        .open() as file
    ):
        p.update_specifications(json.load(file))

    ss_output = cl.SS.run_SS(p, client=None)
    updated, solution = cl.partial_equilibrium_ss(ss_output, p)

    assert solution.all_converged
    assert solution.max_abs_euler_error < 1e-6
    assert np.allclose(updated["b_sp1"], ss_output["b_sp1"], rtol=1e-5)
    assert np.allclose(updated["n"], ss_output["n"], rtol=1e-5)

    config = elp.LifecycleCalibrationConfig()
    ge_moments = elp.compute_model_moments(ss_output, p, config)
    pe_moments = elp.compute_model_moments(updated, p, config)
    assert np.allclose(ge_moments.values, pe_moments.values, rtol=1e-5)

    # Rebuilding the environment from aggregates matches the stored arrays.
    trimmed = {
        k: v for k, v in ss_output.items() if k not in cl._HOUSEHOLD_INPUT_KEYS
    }
    env_stored = cl.HouseholdEnvironment.from_ss_output(ss_output, p)
    env_rebuilt = cl.HouseholdEnvironment.from_ss_output(trimmed, p)
    assert np.allclose(env_stored.p_i, env_rebuilt.p_i)
    assert np.isclose(env_stored.p_tilde, env_rebuilt.p_tilde)
    assert np.allclose(env_stored.bq, env_rebuilt.bq)
    assert np.allclose(env_stored.tr, env_rebuilt.tr)
    assert np.allclose(env_stored.ubi, env_rebuilt.ubi)


# ---------------------------------------------------------------------------
# Phase 3: chi_n inversion
# ---------------------------------------------------------------------------


class MockLaborParams:
    """
    Parameter object with the elliptical utility fields chi_n updates need.
    """

    S = 80
    J = 2
    starting_age = 20
    ending_age = 100
    ltilde = 1.0
    b_ellipse = 0.573
    upsilon = 2.856
    lambdas = np.array([0.6, 0.4]).reshape(2, 1)
    omega_SS = np.ones(80) / 80
    FOC_root_method = "hybr"
    _data = {"chi_n": {"validators": {"range": {"min": 0.0, "max": 1e4}}}}

    def __init__(self):
        self.chi_n = np.tile(np.linspace(20.0, 80.0, self.S), (2, 1))

    def update_specifications(self, revision):
        chi_n = np.asarray(revision["chi_n"], dtype=float)
        if np.any(chi_n > 1e4) or np.any(chi_n < 0):
            raise ValueError("chi_n out of range")
        self.chi_n = np.tile(chi_n, (2, 1))


def _hours_from_foc(chi_n, lhs, p):
    """
    Solve chi_n * MDU(n) = lhs for n, age by age, by bisection.
    """
    from ogcore import household

    n = np.zeros_like(chi_n)
    for s in range(chi_n.size):
        lo, hi = 1e-6, p.ltilde - 1e-6
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            value = float(household.marg_ut_labor(np.array([mid]), 1.0, p))
            if chi_n[s] * value > lhs[s]:
                hi = mid
            else:
                lo = mid
        n[s] = 0.5 * (lo + hi)
    return n


def test_aggregate_labor_by_age_uses_type_weights():
    """
    Hours are averaged over types with within-age population weights.
    """
    p = MockLaborParams()
    n = np.zeros((p.S, p.J))
    n[:, 0] = 0.2
    n[:, 1] = 0.5
    ages = np.array([20, 50, 79])
    labor = cl.aggregate_labor_by_age(n, p, ages)
    assert np.allclose(labor, 0.6 * 0.2 + 0.4 * 0.5)


def test_chi_n_update_is_exact_with_fixed_lhs():
    """
    One full step hits the target when the FOC left-hand side is fixed.
    """
    p = MockLaborParams()
    chi_n = np.array([20.0, 40.0, 60.0])
    lhs = np.array([1.5, 1.5, 1.5])
    n_model = _hours_from_foc(chi_n, lhs, p)
    n_target = np.array([0.3, 0.2, 0.05])
    new_chi_n = cl.chi_n_update(chi_n, n_model, n_target, p)
    assert np.allclose(_hours_from_foc(new_chi_n, lhs, p), n_target, atol=1e-6)
    # Higher target hours require lower chi_n and vice versa.
    assert np.all((new_chi_n > chi_n) == (n_target < n_model))
    # No step with zero damping.
    assert np.allclose(
        cl.chi_n_update(chi_n, n_model, n_target, p, 0.0), chi_n
    )


def test_invert_chi_n_converges_on_fixed_lhs_household_block(monkeypatch):
    """
    With a fixed FOC left-hand side the inversion converges in a few passes.
    """
    from ogusa import estimate_lifecycle_params as elp

    p = MockLaborParams()
    lhs = np.linspace(2.0, 1.0, p.S)  # falls with age like w*e*(1-mtr)*c^-s
    calls = []

    def fake_partial_equilibrium_ss(
        ss_output, params, client=None, b_guess=None, n_guess=None
    ):
        calls.append(1)
        chi_n = params.chi_n[-1, :]
        n_one = _hours_from_foc(chi_n, lhs, params)
        n = np.tile(n_one.reshape(-1, 1), (1, params.J))
        updated = dict(ss_output)
        updated["n"] = n
        updated["b_sp1"] = ss_output["b_sp1"]
        return updated, cl.HouseholdSolution(
            b_sp1=updated["b_sp1"],
            n=n,
            euler_errors=np.zeros((2 * params.S, params.J)),
            success=np.ones(params.J, dtype=bool),
        )

    monkeypatch.setattr(
        cl, "partial_equilibrium_ss", fake_partial_equilibrium_ss
    )
    config = elp.LifecycleCalibrationConfig(chi_n_tail_method="flat")
    ages = config.moment_ages
    target = np.interp(ages, [20, 30, 60, 79], [0.18, 0.30, 0.25, 0.04])
    ss_output = {
        "b_sp1": np.ones((p.S, p.J)),
        "n": np.ones((p.S, p.J)) * 0.3,
    }

    result = cl.invert_chi_n(ss_output, p, target, config=config, tol=1e-6)

    assert result.converged
    assert result.iterations <= 3
    assert len(calls) == result.iterations
    assert np.allclose(result.labor_model, target, rtol=1e-5)
    assert result.chi_n.shape == (p.S,)
    assert np.allclose(p.chi_n[-1, :], result.chi_n)
    # Flat tail in levels beyond the last target age.
    assert np.allclose(result.chi_n[60:], result.chi_n[59])
    assert result.capped_ages.size == 0
    # chi_n rises steeply where target hours are tiny.
    assert result.chi_n[59] > 10 * result.chi_n[30]


def test_invert_chi_n_reports_capped_ages(monkeypatch):
    """
    Ages whose required chi_n exceeds the bound are clipped and reported.
    """
    from ogusa import estimate_lifecycle_params as elp

    p = MockLaborParams()
    lhs = np.full(p.S, 3.0)

    def fake_partial_equilibrium_ss(
        ss_output, params, client=None, b_guess=None, n_guess=None
    ):
        n_one = _hours_from_foc(params.chi_n[-1, :], lhs, params)
        n = np.tile(n_one.reshape(-1, 1), (1, params.J))
        updated = dict(ss_output)
        updated["n"] = n
        return updated, cl.HouseholdSolution(
            b_sp1=ss_output["b_sp1"],
            n=n,
            euler_errors=np.zeros((2 * params.S, params.J)),
            success=np.ones(params.J, dtype=bool),
        )

    monkeypatch.setattr(
        cl, "partial_equilibrium_ss", fake_partial_equilibrium_ss
    )
    config = elp.LifecycleCalibrationConfig()
    target = np.full(config.moment_ages.size, 0.3)
    target[-5:] = 0.001  # needs chi_n far above the cap
    ss_output = {"b_sp1": np.ones((p.S, p.J)), "n": np.ones((p.S, p.J)) * 0.3}

    result = cl.invert_chi_n(
        ss_output, p, target, config=config, chi_n_max=500.0, max_iter=5
    )

    assert not result.converged
    assert result.iterations == 5
    assert np.array_equal(result.capped_ages, config.moment_ages[-5:])
    assert np.all(result.chi_n <= 500.0)
    assert np.allclose(result.labor_model[:-5], 0.3, rtol=1e-3)


def test_invert_chi_n_validates_targets():
    """
    Targets must be one per age and strictly inside the time endowment.
    """
    from ogusa import estimate_lifecycle_params as elp

    p = MockLaborParams()
    config = elp.LifecycleCalibrationConfig()
    ss_output = {"b_sp1": np.ones((p.S, p.J)), "n": np.ones((p.S, p.J))}
    with pytest.raises(ValueError, match="one value per target age"):
        cl.invert_chi_n(ss_output, p, np.ones(5) * 0.3, config=config)
    bad = np.full(config.moment_ages.size, 0.3)
    bad[0] = 0.0
    with pytest.raises(ValueError, match="strictly inside"):
        cl.invert_chi_n(ss_output, p, bad, config=config)


# ---------------------------------------------------------------------------
# Phase 4: beta and chi_b calibration
# ---------------------------------------------------------------------------


class MockPrefParams(MockLaborParams):
    """
    Parameter object for the beta / chi_b calibration tests.
    """

    J = 10
    lambdas = np.array(
        [0.25, 0.25, 0.2, 0.1, 0.1, 0.09, 0.005, 0.004, 0.0009, 0.0001]
    ).reshape(10, 1)
    rho = np.concatenate([np.full(40, 0.002), np.linspace(0.01, 1.0, 40)])
    _data = {
        "beta_annual": {"validators": {"range": {"min": 0.0, "max": 0.9999}}},
        "chi_b": {"validators": {"range": {"min": 0.0, "max": 1e4}}},
        "chi_n": {"validators": {"range": {"min": 0.0, "max": 1e4}}},
    }

    def __init__(self, beta_annual, chi_b):
        super().__init__()
        self.beta_annual = np.asarray(beta_annual, dtype=float)
        self.chi_b = np.asarray(chi_b, dtype=float)

    def update_specifications(self, revision):
        if "beta_annual" in revision:
            self.beta_annual = np.asarray(revision["beta_annual"], dtype=float)
        if "chi_b" in revision:
            self.chi_b = np.asarray(revision["chi_b"], dtype=float)
        if "chi_n" in revision:
            super().update_specifications({"chi_n": revision["chi_n"]})


def _synthetic_household_block(params):
    """
    Wealth that rises with beta at every age and with chi_b at old ages.
    """
    S, J = params.S, params.J
    age = np.arange(S)
    profile = np.sin(np.pi * (age + 1) / (S + 1)) + 0.05
    old = np.clip((age - 40) / 39.0, 0.0, 1.0).reshape(S, 1)
    level = np.exp(40.0 * (params.beta_annual - 0.95)) * (
        1.0 + np.arange(1, J + 1) ** 2
    )
    b = profile.reshape(S, 1) * level.reshape(1, J)
    b = b * (1.0 + 0.02 * params.chi_b.reshape(1, J) * old)
    income = np.ones((S, J)) * (1.0 + 0.5 * np.arange(J))
    return b, income


def _fake_pe_from_synthetic():
    def fake(ss_output, params, client=None, b_guess=None, n_guess=None):
        b, income = _synthetic_household_block(params)
        updated = dict(ss_output)
        updated["b_sp1"] = b
        updated["n"] = np.ones((params.S, params.J)) * 0.3
        updated["before_tax_income"] = income
        return updated, cl.HouseholdSolution(
            b_sp1=b,
            n=updated["n"],
            euler_errors=np.zeros((2 * params.S, params.J)),
            success=np.ones(params.J, dtype=bool),
        )

    return fake


def _synthetic_targets(p_true, config, options):
    """
    Data moments implied by the synthetic block at the true parameters.
    """
    from ogusa import estimate_lifecycle_params as elp

    b, income = _synthetic_household_block(p_true)
    ss = {"b_sp1": b, "before_tax_income": income}
    names = list(elp.wealth_share_bin_names(p_true.lambdas.ravel()))
    values = list(elp.model_wealth_shares(ss, p_true))
    names.append("wealth_income_ratio")
    values.append(elp.model_wealth_income_ratio(ss, p_true))
    names.append("bequest_flow_ratio")
    values.append(elp.model_bequest_flow_ratio(ss, p_true))
    if options.chi_b_mode == "by_type":
        names.extend(elp.tilt_moment_names(config, p_true))
        values.extend(elp.model_old_age_ratio_by_type(ss, p_true, config))
    return elp.MomentSet(tuple(names), np.array(values))


def test_preference_parameterization_groups_and_bounds():
    """
    Bottom types share a beta factor; chi_b groups follow the mode.
    """
    p = MockPrefParams(np.linspace(0.91, 0.995, 10), np.full(10, 80.0))
    common = cl._PreferenceParameterization(
        p,
        cl.PreferenceCalibrationOptions(
            chi_b_mode="common_scale", exclude_bottom=False
        ),
    )
    assert common.beta_groups[0] == [0, 1]
    assert common.n_beta == 9
    assert common.chi_b_groups == [list(range(10))]
    assert common.size == 10
    assert np.all(common.upper > common.lower)
    beta, chi_b = common.unpack(np.zeros(10))
    assert np.allclose(beta, p.beta_annual)
    assert np.allclose(chi_b, p.chi_b)
    theta = np.zeros(10)
    theta[-1] = np.log(0.5)
    _, chi_b_half = common.unpack(theta)
    assert np.allclose(chi_b_half, 40.0)

    by_type = cl._PreferenceParameterization(
        p, cl.PreferenceCalibrationOptions(chi_b_mode="by_type")
    )
    # Default excludes the bottom bin: bottom types join type 3's group.
    assert by_type.beta_groups[0] == [0, 1, 2]
    assert by_type.n_beta == 8
    assert by_type.chi_b_groups == [[0, 1, 2], [3], [4], [5], [6, 7, 8, 9]]
    assert by_type.size == 8 + 5


@pytest.mark.parametrize("mode", ["common_scale", "by_type"])
def test_calibrate_beta_chi_b_recovers_synthetic_parameters(monkeypatch, mode):
    """
    The least-squares calibration recovers the parameters that generated
    the targets on a synthetic household block.
    """
    from ogusa import estimate_lifecycle_params as elp

    monkeypatch.setattr(
        cl, "partial_equilibrium_ss", _fake_pe_from_synthetic()
    )
    options = cl.PreferenceCalibrationOptions(chi_b_mode=mode, max_nfev=2000)
    config = elp.LifecycleCalibrationConfig(
        include_old_age_ratio_by_type=(mode == "by_type")
    )
    # With the bottom bin excluded, types 1 to 3 share one factor that
    # preserves their starting pattern, so the truth keeps them equal.
    beta_true = np.array(
        [0.93, 0.93, 0.93, 0.95, 0.955, 0.96, 0.97, 0.975, 0.98, 0.985]
    )
    if mode == "common_scale":
        chi_b_true = np.full(10, 60.0)
        chi_b_start = np.full(10, 80.0)
    else:
        chi_b_true = np.array([30.0, 30.0, 30.0, 70, 90, 110, 60, 60, 60, 60])
        chi_b_start = np.full(10, 80.0)
    p_true = MockPrefParams(beta_true, chi_b_true)
    data_moments = _synthetic_targets(p_true, config, options)

    beta_start = np.linspace(0.92, 0.99, 10)
    beta_start[:3] = beta_start[2]
    p = MockPrefParams(beta_start, chi_b_start)
    ss_output = {
        "b_sp1": np.ones((p.S, p.J)),
        "n": np.ones((p.S, p.J)) * 0.3,
        "before_tax_income": np.ones((p.S, p.J)),
    }
    result = cl.calibrate_beta_chi_b(
        ss_output, p, data_moments, config=config, options=options
    )

    assert result.success
    assert result.cost < 1e-8
    assert np.allclose(result.beta_annual, beta_true, atol=5e-4)
    assert np.allclose(result.chi_b, chi_b_true, rtol=1e-2)
    assert np.allclose(p.beta_annual, result.beta_annual)
    assert np.allclose(result.residuals, 0.0, atol=1e-5)
    frame = result.to_frame()
    assert list(frame.columns) == ["target", "data", "model", "log_residual"]
    assert frame["target"].iloc[0] == "wealth_share_50_70"
    assert "wealth_share_0_50" not in frame["target"].tolist()
    if mode == "by_type":
        assert "tilt_0_50_80_89_over_60_64" not in frame["target"].tolist()
        assert "tilt_50_70_80_89_over_60_64" in frame["target"].tolist()


def test_preference_targets_keep_bottom_bin_when_requested():
    """
    exclude_bottom=False merges the bottom bins into one share target.
    """
    from ogusa import estimate_lifecycle_params as elp

    p = MockPrefParams(np.linspace(0.92, 0.99, 10), np.full(10, 80.0))
    options = cl.PreferenceCalibrationOptions(
        chi_b_mode="common_scale", exclude_bottom=False
    )
    config = elp.LifecycleCalibrationConfig(
        include_old_age_ratio_by_type=False
    )
    names = list(elp.wealth_share_bin_names(p.lambdas.ravel())) + [
        "wealth_income_ratio",
        "bequest_flow_ratio",
    ]
    values = np.concatenate([np.full(10, 0.1), [7.0, 0.02]])
    moments = elp.MomentSet(tuple(names), values)
    target_names, target_values, selection = cl.preference_targets(
        moments, p, config, options
    )
    assert target_names[0] == "wealth_share_0_50"
    assert np.isclose(target_values[0], 0.2)
    assert selection["share_bins"][0] == [0, 1]
    assert len(target_names) == 9 + 2


def test_calibrate_beta_chi_b_requires_needed_moments():
    """
    Missing level or bequest moments raise a clear error.
    """
    from ogusa import estimate_lifecycle_params as elp

    p = MockPrefParams(np.linspace(0.92, 0.99, 10), np.full(10, 80.0))
    moments = elp.MomentSet(("wealth_share_0_25",), np.array([0.1]))
    with pytest.raises(ValueError, match="wealth_income_ratio"):
        cl.calibrate_beta_chi_b(
            {"b_sp1": np.ones((80, 10)), "n": np.ones((80, 10))}, p, moments
        )


# ---------------------------------------------------------------------------
# Phase 5: outer general-equilibrium loop
# ---------------------------------------------------------------------------


class MockGEParams(MockPrefParams):
    """
    Parameter object with the fields the warm-started GE solve reads.
    """

    baseline = True
    baseline_spending = False
    budget_balance = False
    use_zeta = False
    M = 1
    alpha_T = np.array([0.1])
    SS_root_method = "hybr"
    mindist_SS = 1e-9


def _ge_output(p, r_p=0.04, scale=1.0):
    return {
        "r_p": r_p,
        "r": r_p + 0.01,
        "w": 1.3,
        "p_m": np.array([1.0]),
        "Y": 2.0,
        "BQ": np.linspace(0.01, 0.1, p.J),
        "G": 0.3,
        "TR": 0.2,
        "factor": 1e5 * scale,
        "b_sp1": np.ones((p.S, p.J)) * scale,
        "n": np.ones((p.S, p.J)) * 0.3,
        "before_tax_income": np.ones((p.S, p.J)),
    }


def test_ss_guesses_layout_follows_installed_solver(monkeypatch):
    """
    Guess vector: r_p, r, w, p_m, Y, BQ items, [G], TR, factor.
    """
    p = MockGEParams(np.linspace(0.92, 0.99, 10), np.full(10, 80.0))
    prev = _ge_output(p)
    monkeypatch.setattr(cl, "_ss_solver_has_G", lambda: False)
    guesses = cl._ss_guesses_from_solution(prev, p)
    assert guesses[:3] == [0.04, 0.05, 1.3]
    assert guesses[3] == 1.0
    assert guesses[4] == 2.0
    assert len(guesses) == 3 + 1 + 1 + p.J + 2
    assert guesses[-2:] == [0.2, 1e5]
    vals = cl._unpack_ss_solution(np.array(guesses), p)
    assert vals["r_p"] == 0.04 and vals["factor"] == 1e5 and vals["TR"] == 0.2
    assert np.allclose(vals["BQ"], prev["BQ"])
    assert np.isclose(vals["Y"], 0.2 / 0.1)

    monkeypatch.setattr(cl, "_ss_solver_has_G", lambda: True)
    guesses_g = cl._ss_guesses_from_solution(prev, p)
    assert len(guesses_g) == len(guesses) + 1
    assert guesses_g[-3:] == [0.3, 0.2, 1e5]
    vals_g = cl._unpack_ss_solution(np.array(guesses_g), p)
    assert vals_g["G"] == 0.3
    assert np.allclose(vals_g["BQ"], prev["BQ"])


def test_solve_ge_steady_state_warm_starts_and_falls_back(monkeypatch):
    """
    A converged warm start assembles output through SS_solver; a failed
    one falls back to run_SS.
    """
    from types import SimpleNamespace

    p = MockGEParams(np.linspace(0.92, 0.99, 10), np.full(10, 80.0))
    prev = _ge_output(p)
    calls = []
    monkeypatch.setattr(cl, "_ss_solver_has_G", lambda: False)

    def fake_root(fun, x0, args=None, method=None, tol=None):
        calls.append(("root", list(x0)))
        assert fun is cl.SS.SS_fsolve
        assert len(args) == 7
        return SimpleNamespace(
            success=True, x=np.asarray(x0) * 1.1, message="ok"
        )

    def fake_ss_solver(**kwargs):
        calls.append(("solver", kwargs["fsolve_flag"], kwargs["factor"]))
        return _ge_output(p, scale=2.0)

    import scipy.optimize

    monkeypatch.setattr(scipy.optimize, "root", fake_root)
    monkeypatch.setattr(cl.SS, "SS_solver", fake_ss_solver)
    monkeypatch.setattr(
        cl.SS, "run_SS", lambda p, client=None: pytest.fail("cold solve")
    )
    out = cl.solve_ge_steady_state(p, previous=prev)
    assert out["factor"] == 2e5
    assert calls[0][0] == "root"
    assert calls[1] == ("solver", True, pytest.approx(1.1e5))

    def failing_root(fun, x0, args=None, method=None, tol=None):
        return SimpleNamespace(success=False, x=np.asarray(x0), message="no")

    monkeypatch.setattr(scipy.optimize, "root", failing_root)
    monkeypatch.setattr(
        cl.SS, "run_SS", lambda p, client=None: _ge_output(p, scale=3.0)
    )
    out = cl.solve_ge_steady_state(p, previous=prev)
    assert out["factor"] == 3e5
    # No previous solution: straight to run_SS.
    assert cl.solve_ge_steady_state(p)["factor"] == 3e5


def test_calibrate_lifecycle_preferences_converges_with_fakes(monkeypatch):
    """
    The outer loop alternates the inner steps and stops when parameters and
    prices settle; damping halves when the parameter change grows.
    """
    from ogusa import estimate_lifecycle_params as elp

    p = MockGEParams(np.linspace(0.92, 0.99, 10), np.full(10, 80.0))
    config = elp.LifecycleCalibrationConfig(
        include_wealth_distribution=False,
        include_wealth_income_ratio=False,
        include_bequest_flow_ratio=False,
        include_old_age_ratio_by_type=False,
    )
    ages = config.moment_ages
    data = elp.MomentSet(
        tuple(f"labor_supply_age_{a}" for a in ages), np.full(ages.size, 0.3)
    )
    state = {"r_p": 0.04}

    def fake_invert(ss, params, target, config=None, client=None):
        chi_n = np.full(params.S, 30.0 * (1 + 10 * state["r_p"]))
        params.update_specifications({"chi_n": chi_n.tolist()})
        return cl.ChiNInversionResult(
            chi_n=chi_n,
            ages=ages,
            labor_model=target,
            labor_target=target,
            iterations=3,
            converged=True,
            max_abs_log_gap=1e-4,
            history=[],
            capped_ages=np.array([]),
            ss_output=ss,
            solution=None,
        )

    def fake_pref(
        ss, params, data_moments, config=None, options=None, client=None
    ):
        # beta responds to the current interest rate; chi_b fixed.
        beta = np.clip(0.9 + 2.0 * state["r_p"], 0.5, 0.999) * np.ones(
            params.J
        )
        params.update_specifications({"beta_annual": beta.tolist()})
        return cl.PreferenceCalibrationResult(
            beta_annual=beta,
            chi_b=params.chi_b,
            theta=np.zeros(1),
            residuals=np.zeros(1),
            residual_names=("x",),
            data_values=np.ones(1),
            model_values=np.ones(1),
            cost=1e-6,
            nfev=5,
            success=True,
            message="ok",
            ss_output=ss,
            solution=None,
        )

    def fake_ge(params, previous=None, client=None):
        # Interest rate falls toward 0.03 as beta rises: a contraction.
        state["r_p"] = 0.03 + 0.5 * (0.97 - params.beta_annual.mean())
        return _ge_output(params, r_p=state["r_p"])

    def fake_model_moments(ss, params, config):
        return data

    monkeypatch.setattr(cl, "invert_chi_n", fake_invert)
    monkeypatch.setattr(cl, "calibrate_beta_chi_b", fake_pref)
    monkeypatch.setattr(cl, "solve_ge_steady_state", fake_ge)
    monkeypatch.setattr(elp, "compute_model_moments", fake_model_moments)

    outcome = cl.calibrate_lifecycle_preferences(
        p,
        config=config,
        data_moments=data,
        max_outer=30,
        param_tol=1e-6,
        price_tol=1e-6,
    )
    assert outcome.converged
    assert 2 < outcome.iterations < 30
    assert np.allclose(outcome.beta_annual, 0.9 + 2.0 * state["r_p"])
    assert outcome.chi_n.shape == (p.S,)
    assert outcome.history[-1].param_change < 1e-6
    assert set(outcome.parameter_dict) == {"beta_annual", "chi_b", "chi_n"}
    assert list(outcome.to_frame().columns) == ["moment", "data", "model"]

    # Non-convergence within the cap is reported, not raised.
    p2 = MockGEParams(np.linspace(0.92, 0.99, 10), np.full(10, 80.0))
    state["r_p"] = 0.04
    short = cl.calibrate_lifecycle_preferences(
        p2,
        config=config,
        data_moments=data,
        max_outer=1,
        param_tol=1e-12,
        price_tol=1e-12,
    )
    assert not short.converged
    assert short.iterations == 1


def test_preference_parameterization_respects_bounds_at_ceiling():
    """
    Parameters already at a ceiling stay within it after a round trip.
    """
    beta = np.linspace(0.92, 0.99, 10)
    beta[5:9] = 0.9999
    p = MockPrefParams(beta, np.full(10, 80.0))
    param = cl._PreferenceParameterization(
        p, cl.PreferenceCalibrationOptions()
    )
    beta_out, chi_b_out = param.unpack(param.upper)
    assert np.all(beta_out <= 0.9999)
    assert np.all(chi_b_out <= 1e4)
    beta_lo, _ = param.unpack(param.lower)
    assert np.all(beta_lo >= 1e-4)

    capped = cl._PreferenceParameterization(
        p,
        cl.PreferenceCalibrationOptions(
            beta_annual_max=0.995, chi_b_max=100.0
        ),
    )
    beta_cap, chi_b_cap = capped.unpack(np.zeros(capped.size))
    assert np.all(beta_cap <= 0.995)
    assert np.isclose(beta_cap[5], 0.995)
    beta_up, chi_b_up = capped.unpack(capped.upper)
    assert np.all(beta_up <= 0.995 + 1e-12)
    assert np.all(chi_b_up <= 100.0 + 1e-9)


def test_preference_target_selection_reproduces_targets():
    """
    The selection matrix maps the full moment vector onto the calibration
    targets, summing merged share bins.
    """
    from ogusa import estimate_lifecycle_params as elp

    options = cl.PreferenceCalibrationOptions(
        chi_b_mode="by_type", exclude_bottom=False
    )
    config = elp.LifecycleCalibrationConfig()
    p = MockPrefParams(np.linspace(0.92, 0.99, 10), np.full(10, 80.0))
    data = _synthetic_targets(p, config, options)
    A = cl.preference_target_selection(data, p, config, options)
    names, values, _ = cl.preference_targets(data, p, config, options)
    assert A.shape == (len(names), len(data.names))
    assert np.allclose(A @ data.values, values)
    # Bottom two types merged into one share target.
    assert A[0].sum() == 2 and names[0] == "wealth_share_0_50"
    assert np.all(A[1:].sum(axis=1) == 1)
    assert set(np.unique(A)) <= {0.0, 1.0}


def test_preference_inference_shapes_and_methods(monkeypatch):
    """
    Standard errors come from the least-squares Jacobian; a moment
    covariance switches on the sandwich form and the J test.
    """
    from ogusa import estimate_lifecycle_params as elp

    monkeypatch.setattr(
        cl, "partial_equilibrium_ss", _fake_pe_from_synthetic()
    )
    options = cl.PreferenceCalibrationOptions(chi_b_mode="by_type")
    config = elp.LifecycleCalibrationConfig()
    beta_true = np.array(
        [0.93, 0.93, 0.93, 0.95, 0.955, 0.96, 0.97, 0.975, 0.98, 0.985]
    )
    chi_b_true = np.array([30.0, 30.0, 30.0, 70, 90, 110, 60, 60, 60, 60])
    p_true = MockPrefParams(beta_true, chi_b_true)
    data = _synthetic_targets(p_true, config, options)
    # Perturb the data so the fit is not exact.
    noisy = elp.MomentSet(
        data.names,
        data.values * np.exp(0.02 * np.cos(np.arange(len(data.names)))),
    )
    p = MockPrefParams(beta_true.copy(), chi_b_true.copy())
    ss_output = {
        "b_sp1": np.ones((p.S, p.J)),
        "n": np.ones((p.S, p.J)) * 0.3,
        "before_tax_income": np.ones((p.S, p.J)),
    }
    result = cl.calibrate_beta_chi_b(
        ss_output, p, noisy, config=config, options=options
    )
    m = len(result.residual_names)
    k = result.theta.size
    assert result.jacobian.shape == (m, k)
    assert result.weights.shape == (m,)

    nls = cl.preference_inference(result, p, options=options)
    assert nls.method == "nls"
    assert nls.n_moments == m and nls.n_params == k
    assert nls.theta_se.shape == (k,)
    assert nls.beta_se.shape == (p.J,) and nls.chi_b_se.shape == (p.J,)
    assert np.all(np.isfinite(nls.theta_se)) and np.all(nls.theta_se >= 0)
    assert np.isnan(nls.j_stat)
    # Tied bottom types share a standard error (delta method by group).
    assert np.isclose(nls.beta_se[0], nls.beta_se[1])
    assert np.allclose(nls.chi_b_se[6:], nls.chi_b_se[6])
    frame = nls.to_frame(p)
    assert list(frame.columns) == [
        "type",
        "beta_annual",
        "beta_se",
        "beta_at_bound",
        "chi_b",
        "chi_b_se",
        "chi_b_at_bound",
    ]

    A = cl.preference_target_selection(noisy, p, config, options)
    full_vcv = np.diag((0.05 * noisy.values) ** 2)
    sandwich = cl.preference_inference(
        result, p, options=options, moment_vcv=A @ full_vcv @ A.T
    )
    assert sandwich.method == "sandwich"
    assert sandwich.j_df == m - k > 0
    assert np.isfinite(sandwich.j_stat) and sandwich.j_stat >= 0
    assert 0.0 <= sandwich.j_pvalue <= 1.0
    assert np.all(sandwich.theta_se > 0)

    with pytest.raises(ValueError):
        cl.preference_inference(
            result, p, options=options, moment_vcv=np.eye(m + 1)
        )

    # A parameter on its bound is treated as fixed: NaN standard error,
    # one fewer free parameter, one more degree of freedom.
    param = cl._PreferenceParameterization(p, options)
    pinned = replace(result, theta=result.theta.copy())
    pinned.theta[3] = param.upper[3]
    fixed = cl.preference_inference(pinned, p, options=options)
    assert fixed.at_bound[3] and fixed.at_bound.sum() == 1
    assert fixed.n_params == k - 1 and fixed.j_df == m - k + 1
    assert np.isnan(fixed.theta_se[3]) and np.isnan(fixed.beta_se[5])
    assert np.all(np.isfinite(np.delete(fixed.theta_se, 3)))
    frame = fixed.to_frame(p)
    assert frame["beta_at_bound"].tolist().count(True) == 1


def test_calibrate_beta_chi_b_jacobian_matches_manual_differences(
    monkeypatch,
):
    """
    The Jacobian stored on the result is a forward difference with the
    absolute step in options.diff_step, not SciPy's relative step.
    """
    from ogusa import estimate_lifecycle_params as elp

    fake_pe = _fake_pe_from_synthetic()
    monkeypatch.setattr(cl, "partial_equilibrium_ss", fake_pe)
    options = cl.PreferenceCalibrationOptions(
        chi_b_mode="by_type", max_nfev=1, diff_step=1e-3
    )
    config = elp.LifecycleCalibrationConfig()
    beta = np.array(
        [0.93, 0.93, 0.93, 0.95, 0.955, 0.96, 0.97, 0.975, 0.98, 0.985]
    )
    chi_b = np.array([30.0, 30.0, 30.0, 70, 90, 110, 60, 60, 60, 60])
    data = _synthetic_targets(MockPrefParams(beta, chi_b), config, options)
    noisy = elp.MomentSet(data.names, data.values * 1.05)
    p = MockPrefParams(beta.copy(), chi_b.copy())
    ss_output = {
        "b_sp1": np.ones((p.S, p.J)),
        "n": np.ones((p.S, p.J)) * 0.3,
        "before_tax_income": np.ones((p.S, p.J)),
    }
    result = cl.calibrate_beta_chi_b(
        ss_output, p, noisy, config=config, options=options
    )
    # One iteration only: the solution is the starting point.
    assert np.allclose(result.theta, 0.0, atol=1e-8)

    names, dvals, selection = cl.preference_targets(noisy, p, config, options)
    param = cl._PreferenceParameterization(
        MockPrefParams(beta, chi_b), options
    )
    weights = cl._preference_weights(len(names), options)

    def resid(theta):
        b, c = param.unpack(theta)
        q = MockPrefParams(b, c)
        updated, _ = fake_pe(ss_output, q)
        model = cl._preference_model_values(
            updated, q, config, options, selection
        )
        return weights * np.log(model / dvals)

    r0 = resid(np.zeros(param.size))
    manual = np.empty_like(result.jacobian)
    for k in range(param.size):
        h = 1e-3 if param.upper[k] > 1e-3 else -1e-3
        shifted = np.zeros(param.size)
        shifted[k] = h
        manual[:, k] = (resid(shifted) - r0) / h
    assert result.jacobian.shape == manual.shape
    assert np.allclose(result.jacobian, manual, rtol=1e-6, atol=1e-9)
    assert np.linalg.norm(manual) > 0.1


def test_theta_roundtrip_price_change_and_chi_n_bounds():
    """
    Transformed-parameter stacking inverts, price changes are relative,
    and chi_n bounds come from the validators unless overridden.
    """
    p = MockGEParams(np.linspace(0.92, 0.99, 10), np.full(10, 80.0))
    theta = cl._theta_from_p(p)
    assert theta.shape == (2 * p.J + p.S,)
    cl._apply_theta(theta + 0.1, p)
    assert np.allclose(cl._theta_from_p(p), theta + 0.1)
    assert np.all(p.beta_annual < 1.0)

    old = _ge_output(p, r_p=0.04)
    new = dict(old, r_p=0.044, BQ=old["BQ"] * 1.5)
    biggest, changes = cl._price_change(new, old)
    assert changes["r_p"] == pytest.approx(0.1)
    assert changes["BQ"] == pytest.approx(0.5)
    assert changes["w"] == 0.0
    assert biggest == pytest.approx(0.5)

    lower, upper = cl._chi_n_bounds(p, None, None)
    assert lower == pytest.approx(0.0, abs=1e-6) and upper == 1e4
    assert cl._chi_n_bounds(p, 1.0, 500.0) == (1.0, 500.0)
    cl.apply_chi_n(p, np.full(p.S, 3.0))
    assert np.allclose(p.chi_n[-1, :], 3.0)
