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
        tax_func_path = os.path.join(
            CUR_PATH, "test_io_data", "TxFuncEst_policy.pkl"
        )
        c = Calibration(p)
        _, _ = c.read_tax_func_estimate(p, tax_func_path)


def test_read_tax_func_estimate():
    p = ogcore.Specifications()
    p.BW = 11
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
