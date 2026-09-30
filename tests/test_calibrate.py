"""
Tests of calibrate.py module
"""

import pytest
import numpy as np
import os
import ogcore
from ogusa.calibrate import Calibration

CUR_PATH = os.path.abspath(os.path.dirname(__file__))


def test_calibrate():
    p = ogcore.Specifications()
    _ = Calibration(p)


def test_read_tax_func_estimate_error():
    with pytest.raises(RuntimeError):
        p = ogcore.Specifications()
        p.tax_func_type = "mono"  # this is NOT the tax func type in the pickle
        tax_func_path = os.path.join(
            CUR_PATH, "test_io_data", "TxFuncEst_policy.pkl"
        )
        c = Calibration(p)
        _, _ = c.read_tax_func_estimate(p, tax_func_path)


def test_read_tax_func_estimate():
    p = ogcore.Specifications()
    p.BW = 11
    p.tax_func_type = "DEP"
    p.start_year = 2021
    tax_func_path = os.path.join(
        CUR_PATH, "test_io_data", "TxFuncEst_policy.pkl"
    )
    c = Calibration(p)
    dict_params, _ = c.read_tax_func_estimate(p, tax_func_path)
    print("Dict keys = ", dict_params.keys())

    assert isinstance(dict_params["tfunc_etr_params_S"], np.ndarray)


def test_get_dict():
    p = ogcore.Specifications()
    c = Calibration(p)
    c_dict = c.get_dict()

    assert isinstance(c_dict, dict)


@pytest.mark.needs_fred
def test_get_macro_params():
    p = ogcore.Specifications()
    c = Calibration(p, get_macro_params=True)
    c_dict = c.get_dict()

    assert isinstance(c_dict, dict)


@pytest.mark.local  # this test requires api access
def test_get_pop():
    p = ogcore.Specifications()
    c = Calibration(p, estimate_pop=True)
    c_dict = c.get_dict()

    assert isinstance(c_dict, dict)


@pytest.mark.local  # this test is slow and requires a lot of memory, so we mark it as local only
def test_estimate_taxes():
    p = ogcore.Specifications()
    p.tax_func_type = "HSV"
    p.age_specific = True
    c = Calibration(p, estimate_tax_functions=True)
    c_dict = c.get_dict()

    assert isinstance(c_dict, dict)


def _fake_lifecycle_outcome(p_calib, converged=True):
    from ogusa import calibrate_lifecycle as cl

    beta = np.linspace(0.95, 0.99, p_calib.J)
    chi_b = np.linspace(10.0, 80.0, p_calib.J)
    chi_n = np.linspace(50.0, 5000.0, p_calib.S)
    return cl.LifecycleCalibrationOutcome(
        beta_annual=beta,
        chi_b=chi_b,
        chi_n=chi_n,
        ss_output={},
        iterations=3,
        converged=converged,
        history=[],
        data_moments=None,
        model_moments=None,
        chi_n_result=None,
        pref_result=None,
    )


def test_calibration_lifecycle_prefs_wiring(monkeypatch, tmp_path):
    """
    The lifecycle calibration runs on a copy of p carrying the other
    calibrated parameters, its result lands in get_dict, and a saved JSON
    is reused on the next construction.
    """
    from ogusa import calibrate_lifecycle as cl

    calls = []

    def fake_calibrate(p_calib, config=None, options=None, **kwargs):
        calls.append((p_calib, kwargs))
        return _fake_lifecycle_outcome(p_calib)

    monkeypatch.setattr(cl, "calibrate_lifecycle_preferences", fake_calibrate)
    p = ogcore.Specifications()
    beta_before = np.array(p.beta_annual, copy=True)
    path = os.path.join(tmp_path, "prefs.json")
    c = Calibration(
        p,
        estimate_lifecycle_prefs=True,
        lifecycle_params_path=path,
        lifecycle_kwargs={"max_outer": 4},
    )
    assert len(calls) == 1
    p_calib, kwargs = calls[0]
    assert p_calib is not p
    assert kwargs["max_outer"] == 4
    # The copy carries the earnings profile and transfer matrices already
    # produced by the class; the original is untouched.
    assert np.allclose(p_calib.e, c.e)
    assert np.allclose(p_calib.eta, c.eta)
    assert np.allclose(p.beta_annual, beta_before)
    d = c.get_dict()
    assert set(("beta_annual", "chi_b", "chi_n", "e", "eta", "zeta")) <= set(d)
    assert len(d["beta_annual"]) == p.J
    assert len(d["chi_n"]) == p.S
    assert c.lifecycle_outcome.converged
    assert os.path.exists(path)

    # Second construction reads the file instead of calibrating.
    c2 = Calibration(
        p, estimate_lifecycle_prefs=True, lifecycle_params_path=path
    )
    assert len(calls) == 1
    assert c2.lifecycle_outcome is None
    assert c2.get_dict()["chi_b"] == d["chi_b"]

    # A file for different dimensions triggers re-calibration.
    with open(path, "r", encoding="utf-8") as file:
        record = __import__("json").load(file)
    record["chi_n"] = record["chi_n"][:-1]
    with open(path, "w", encoding="utf-8") as file:
        __import__("json").dump(record, file)
    Calibration(p, estimate_lifecycle_prefs=True, lifecycle_params_path=path)
    assert len(calls) == 2

    # The deprecated chi_n flag routes to the joint calibration.
    with pytest.warns(DeprecationWarning):
        c3 = Calibration(p, estimate_chi_n=True, lifecycle_params_path=path)
    assert "beta_annual" in c3.get_dict()


def test_get_dict_without_lifecycle_prefs_has_no_preference_keys():
    p = ogcore.Specifications()
    c = Calibration(p)
    d = c.get_dict()
    assert "beta_annual" not in d and "chi_n" not in d


def test_read_lifecycle_parameters_branches(tmp_path):
    """
    Missing file and dimension mismatch ask for a new calibration; a file
    without the three parameters is an error; a consistent file is read.
    """
    import json

    p = ogcore.Specifications()
    c = Calibration(p)
    path = os.path.join(tmp_path, "prefs.json")
    assert c.read_lifecycle_parameters(p, path) == (None, True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump({"beta_annual": [0.9] * p.J}, file)
    with pytest.raises(RuntimeError):
        c.read_lifecycle_parameters(p, path)

    good = {
        "beta_annual": [0.9] * p.J,
        "chi_b": [1.0] * p.J,
        "chi_n": [1.0] * p.S,
        "_meta": {"converged": True},
    }
    with open(path, "w", encoding="utf-8") as file:
        json.dump(good, file)
    params, run = c.read_lifecycle_parameters(p, path)
    assert not run
    assert set(params) == {"beta_annual", "chi_b", "chi_n"}

    good["chi_b"] = [1.0] * (p.J + 1)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(good, file)
    assert c.read_lifecycle_parameters(p, path) == (None, True)


def test_parameter_updates_excludes_lifecycle_prefs():
    p = ogcore.Specifications()
    c = Calibration(p)
    updates = c._parameter_updates()
    assert {"eta", "zeta", "e"} <= set(updates)
    assert not {"beta_annual", "chi_b", "chi_n"} & set(updates)
    assert c.get_dict().keys() == updates.keys()
