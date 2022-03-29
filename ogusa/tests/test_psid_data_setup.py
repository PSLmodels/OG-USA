import os
import numpy as np
import math
import pytest
from ogusa import psid_data_setup
from ogusa.constants import PSID_NOMINAL_VARS, PSID_CONSTANT_VARS

CUR_PATH = os.path.abspath(os.path.dirname(__file__))


def test_prep_data():
    """
    Test that function works
    """
    _ = psid_data_setup.prep_data()


@pytest.mark.local
def test_num_obs_total():
    """
    Check that number of obs in final data equals what in psid
    after sample selection
    """
    panel_li = psid_data_setup.prep_data()
    num_obs_psid = 109155
    if panel_li.shape[0] != num_obs_psid:
        print("Number of observations in final data set is not right")
        print("Obs in PSID after selection = ", num_obs_psid)
        print("Obs in final panel = ", panel_li.shape[0])
        assert False


@pytest.mark.local
def test_indicator_vars():
    """
    Check that every observation has a group and decile and that
    fraction in each is correct. Note that can't check the latter with
    final unbalanced panel.
    """
    panel_li = psid_data_setup.prep_data()
    cats_pct = ["0-25", "26-50", "51-70", "71-80", "81-90", "91-99", "100"]
    cats_10 = ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10"]
    print("Checking counts of percentile groupings: ")
    for item in cats_10 + cats_pct:
        print(panel_li[item].count(), panel_li.shape[0])
        assert panel_li[item].count() == panel_li.shape[0]
    # NOTE: the following doesn't work with the unbalanced panel
    # print("Checking percentile groupings: ")
    # for d in cats_10:
    #     assert math.isclose(panel_li[d].mean(), 0.1, rel_tol=0.03)
    # for i, g in enumerate(cats_pct):
    #     percent_in_g = groups[i + 1] - groups[i]
    #     assert math.isclose(panel_li[g].mean(), percent_in_g, rel_tol=0.03)
