import multiprocessing
from distributed import Client, LocalCluster
import pytest
import os
from ogusa.calibrate import Calibration
from ogcore.parameters import Specifications
from taxcalc import Calculator

# from ogusa.constants import CPS_START_YEAR, PUF_START_YEAR, TC_LAST_YEAR
# from ogusa import get_micro_data
# from ogcore import utils
# from taxcalc import GrowFactors

NUM_WORKERS = min(multiprocessing.cpu_count(), 7)
# get path to puf if puf.csv in ogusa/ directory
CUR_PATH = os.path.abspath(os.path.dirname(__file__))


@pytest.fixture(scope="module")
def dask_client():
    cluster = LocalCluster(n_workers=NUM_WORKERS, threads_per_worker=2)
    client = Client(cluster)
    yield client
    # teardown
    client.close()
    cluster.close()


# Set up baseline parameterization
p1 = Specifications(
    baseline=True,
    num_workers=NUM_WORKERS,
    baseline_dir=CUR_PATH,
    output_base=CUR_PATH,
)
# Update parameters for baseline from default json file
baseline_url = (
    "github://PSLmodels:OG-USA@master/ogusa/ogusa_default_parameters.json"
)
baseline_json = Calculator.read_json_param_objects(baseline_url, None)
p1.update_specifications(baseline_json)


@pytest.mark.full_run
def test_tfunc_est(p1, dask_client):
    """
    Make sure that the Calibration class runs the tax function estimation (use
    baseline example)
    """
    c1 = Calibration(p1, estimate_tax_functions=True, client=dask_client)
    assert type(c1) == ogusa.calibrate.Calibration
    tfunc_obj_list = [
        "etr_params",
        "mtrx_params",
        "mtry_params",
        "taxcalc_version",
        "mean_income_data",
        "frac_tax_payroll",
    ]
    assert set(tfunc_obj_list).issubset(list(c1.tax_function_params.keys()))


def test_no_tfunc_default(p1, dask_client):
    """
    Make sure that the Calibration class can just choose the default tax
    function parameters from ogusa_default_parmameters.json
    """
    c1 = Calibration(p1, client=dask_client)
    assert type(c1) == ogusa.calibrate.Calibration
    assert type(p1.etr_params) == numpy.ndarray
    assert type(p1.mtrx_params) == numpy.ndarray
    assert type(p1.mtry_params) == numpy.ndarray


def test_no_tfunc_path(p1, dask_client):
    """
    Make sure that the Calibration class can choose the tax parameters saved in
    a pkl file from a previous tax function estimation
    """
    c1 = Calibration(p1, client=dask_client)
    assert type(c1) == ogusa.calibrate.Calibration
    tfunc_obj_list = [
        "etr_params",
        "mtrx_params",
        "mtry_params",
        "taxcalc_version",
        "mean_income_data",
        "frac_tax_payroll",
    ]
    assert set(tfunc_obj_list).issubset(list(c1.tax_function_params.keys()))
