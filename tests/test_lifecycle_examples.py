"""
Tests of the helper functions in the lifecycle calibration example
scripts (``examples/run_lifecycle_calibration.py`` and
``examples/validate_lifecycle_time_path.py``).
"""

import importlib.util
import json
import os
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from ogusa import calibrate_lifecycle as cl
from ogusa import estimate_lifecycle_params as elp

CUR_PATH = os.path.abspath(os.path.dirname(__file__))
EXAMPLES = os.path.join(CUR_PATH, "..", "examples")


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(EXAMPLES, name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def run_mod():
    return _load("run_lifecycle_calibration")


@pytest.fixture(scope="module")
def validate_mod():
    return _load("validate_lifecycle_time_path")


def test_history_frame_one_row_per_pass(run_mod):
    records = [
        cl.OuterIterationRecord(
            iteration=i,
            param_change=1.0 / i,
            price_change=0.1 / i,
            damping=1.0,
            chi_n_iterations=3,
            pref_nfev=10 * i,
            pref_cost=0.5,
            ge_seconds=12.0,
            prices={
                "r_p": 0.04,
                "r": 0.05,
                "w": 1.3,
                "factor": 1e5,
                "TR": 0.2,
            },
            residuals={},
        )
        for i in (1, 2, 3)
    ]
    outcome = SimpleNamespace(history=records)
    frame = run_mod.history_frame(outcome)
    assert list(frame["iteration"]) == [1, 2, 3]
    assert {"param_change", "price_change", "r_p", "factor", "TR"} <= set(
        frame.columns
    )
    assert frame["pref_nfev"].tolist() == [10, 20, 30]


def test_moment_frame_log_gap_nan_for_nonpositive_data(run_mod):
    data = elp.MomentSet(("a", "b", "c"), np.array([1.0, -0.5, 2.0]))
    model = elp.MomentSet(("a", "b", "c"), np.array([2.0, 0.1, 2.0]))
    frame = run_mod.moment_frame(data, model)
    assert list(frame.columns) == ["moment", "data", "model", "log_gap"]
    assert np.isclose(frame["log_gap"].iloc[0], np.log(2.0))
    assert np.isnan(frame["log_gap"].iloc[1])
    assert frame["log_gap"].iloc[2] == 0.0


def test_load_plot_module_exposes_helpers(run_mod):
    plots = run_mod._load_plot_module()
    for name in (
        "load_default_spec",
        "load_or_solve_ss",
        "cps_hours_profile",
        "plot_labor_profiles",
        "plot_wealth_profiles",
    ):
        assert callable(getattr(plots, name))


def test_standard_errors_writes_table(run_mod, monkeypatch, tmp_path):
    """
    The example's inference step evaluates one least-squares iteration on
    a copy of p, maps a bootstrap covariance to the targets, and writes
    the table.
    """
    J = 3
    p = SimpleNamespace(
        J=J, beta_annual=np.array([0.95, 0.96, 0.97]), chi_b=np.ones(J)
    )
    names = ("m1", "m2", "m3", "m4")
    data = elp.MomentSet(names, np.ones(4))
    config = elp.LifecycleCalibrationConfig()
    options = cl.PreferenceCalibrationOptions()
    seen = {}

    def fake_calibrate(ss, p_copy, data_moments, config=None, options=None):
        seen["p_is_copy"] = p_copy is not p
        seen["max_nfev"] = options.max_nfev
        return "pref-result"

    def fake_bootstrap(p_, cfg, seed=None):
        seen["iterations"] = cfg.bootstrap_iterations
        return np.random.default_rng(0).normal(
            size=(cfg.bootstrap_iterations, 4)
        )

    def fake_selection(data_moments, p_, cfg, opts):
        return np.eye(4)

    def fake_inference(pref, p_, options=None, moment_vcv=None):
        seen["vcv_shape"] = None if moment_vcv is None else moment_vcv.shape
        return cl.PreferenceInference(
            theta_se=np.zeros(2),
            beta_se=np.full(J, 0.01),
            chi_b_se=np.full(J, 0.5),
            vcv_theta=np.eye(2),
            j_stat=1.0,
            j_df=2,
            j_pvalue=0.6,
            method="sandwich",
            n_moments=4,
            n_params=2,
            at_bound=np.zeros(2, dtype=bool),
        )

    monkeypatch.setattr(run_mod.cl, "calibrate_beta_chi_b", fake_calibrate)
    monkeypatch.setattr(run_mod.elp, "bootstrap_data_moments", fake_bootstrap)
    monkeypatch.setattr(
        run_mod.cl, "preference_target_selection", fake_selection
    )
    monkeypatch.setattr(run_mod.cl, "preference_inference", fake_inference)
    args = SimpleNamespace(bootstrap_iterations=7)
    path = run_mod.standard_errors(
        p, {}, data, config, options, args, str(tmp_path)
    )
    assert os.path.exists(path)
    assert seen == {
        "p_is_copy": True,
        "max_nfev": 1,
        "iterations": 7,
        "vcv_shape": (4, 4),
    }
    with open(path, "r", encoding="utf-8") as file:
        header = file.readline().strip().split(",")
    assert header[:3] == ["type", "beta_annual", "beta_se"]


def test_validate_load_default_spec(validate_mod):
    p = validate_mod.load_default_spec(baseline=True, num_workers=1)
    assert p.S == 80 and p.J == 10 and p.baseline


def test_tpi_summary_missing_and_present(validate_mod, tmp_path):
    assert validate_mod.tpi_summary(str(tmp_path)) == {"tpi_saved": False}
    tpi_dir = os.path.join(tmp_path, "TPI")
    os.makedirs(tpi_dir)
    tpi = {
        "euler_savings": np.array([[1e-9, -2e-9]]),
        "euler_labor_leisure": np.array([[3e-9]]),
        "RC_error": np.array([1e-4]),
        "r": np.linspace(0.05, 0.04, 10),
        "Y": np.ones(10),
    }
    with open(os.path.join(tpi_dir, "TPI_vars.pkl"), "wb") as file:
        pickle.dump(tpi, file)
    summary = validate_mod.tpi_summary(str(tmp_path))
    assert summary["tpi_saved"]
    assert summary["max_abs_euler_savings"] == pytest.approx(2e-9)
    assert summary["max_abs_euler_labor_leisure"] == pytest.approx(3e-9)
    assert summary["r_first"] == pytest.approx(0.05)
    assert summary["r_last_T"] == pytest.approx(tpi["r"][5])
    assert "K_first" not in summary


class _Params:
    def __init__(self, output_base):
        self.output_base = output_base


def test_run_one_solves_saves_and_records(validate_mod, monkeypatch, tmp_path):
    calls = []
    ss = {"r_p": 0.04, "w": 1.3, "factor": 1e5}
    tpi = {"euler_savings": np.zeros((2, 2)), "r": np.array([0.04, 0.05])}
    monkeypatch.setattr(
        validate_mod.SS,
        "run_SS",
        lambda p, client=None: calls.append("ss") or ss,
    )
    monkeypatch.setattr(
        validate_mod.TPI,
        "run_TPI",
        lambda p, client=None: calls.append("tpi") or tpi,
    )
    p = _Params(str(tmp_path))
    record = validate_mod.run_one(p, None, "baseline")
    assert calls == ["ss", "tpi"]
    assert record["converged"] and record["error"] is None
    assert record["ss_r_p"] == 0.04 and record["tpi_saved"]
    for rel in ("SS/SS_vars.pkl", "model_params.pkl", "TPI/TPI_vars.pkl"):
        assert os.path.exists(os.path.join(tmp_path, rel))
    with open(os.path.join(tmp_path, "run_record.json")) as file:
        assert json.load(file)["label"] == "baseline"

    # A saved steady state skips the solve; a TPI failure is recorded.
    calls.clear()
    ss_path = os.path.join(tmp_path, "saved_ss.pkl")
    with open(ss_path, "wb") as file:
        pickle.dump(ss, file)

    def failing_tpi(p, client=None):
        raise RuntimeError("Transition path equlibrium not found")

    monkeypatch.setattr(validate_mod.TPI, "run_TPI", failing_tpi)
    record = validate_mod.run_one(p, None, "reform", ss_pickle=ss_path)
    assert calls == []
    assert not record["converged"]
    assert "RuntimeError" in record["error"]
