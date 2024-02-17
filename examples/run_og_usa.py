import multiprocessing
from distributed import Client, LocalCluster
import os
import json
import time
from taxcalc import Calculator
from ogusa.calibrate import Calibration
from ogcore.parameters import Specifications
from ogcore import output_tables as ot
from ogcore import output_plots as op
from ogcore.execute import runner
from ogcore.utils import safe_read_pickle


def fetch_profiles(client):
    workers = client.scheduler_info()["workers"]
    profiles = client.run(
        lambda dask_worker: dask_worker.profile.dump_stats(
            "profile_worker_{address}.pstats".format(
                address=dask_worker.address[-5:]
            )
        )
    )
    return profiles


def main():

    cluster = LocalCluster(
        n_workers=7,
        threads_per_worker=13,
        worker_dashboard_address=":0",
        preload=[
            "/usr/local/google/home/talumbau/src/OG-USA/examples/worker_setup.py"
        ],
    )

    # Define parameters to use for multiprocessing
    num_workers = min(multiprocessing.cpu_count(), 7)
    client = Client(cluster)
    print("Number of workers = ", num_workers)

    # Directories to save data
    CUR_DIR = os.path.dirname(os.path.realpath(__file__))
    base_dir = os.path.join(CUR_DIR, "OG-USA-Example", "OUTPUT_BASELINE")
    reform_dir = os.path.join(CUR_DIR, "OG-USA-Example", "OUTPUT_REFORM")

    """
    ------------------------------------------------------------------------
    Run baseline policy
    ------------------------------------------------------------------------
    """
    # Set up baseline parameterization
    p = Specifications(
        baseline=True,
        num_workers=num_workers,
        baseline_dir=base_dir,
        output_base=base_dir,
    )
    # Update parameters for baseline from default json file
    p.update_specifications(
        json.load(
            open(
                os.path.join(
                    CUR_DIR, "..", "ogusa", "ogusa_default_parameters.json"
                )
            )
        )
    )
    p.tax_func_type = "GS"
    c = Calibration(p, estimate_tax_functions=True, client=client)
    # close and delete client bc cache is too large
    client.close()
    del client
    client = Client(n_workers=num_workers)
    d = c.get_dict()
    # # additional parameters to change
    updated_params = {
        "etr_params": d["etr_params"],
        "mtrx_params": d["mtrx_params"],
        "mtry_params": d["mtry_params"],
        "mean_income_data": d["mean_income_data"],
        "frac_tax_payroll": d["frac_tax_payroll"],
    }
    p.update_specifications(updated_params)
    # Run model
    start_time = time.time()
    runner(p, time_path=True, client=client)
    print("run time = ", time.time() - start_time)

    """
    ------------------------------------------------------------------------
    Run reform policy
    ------------------------------------------------------------------------
    """
    # Grab a reform JSON file already in Tax-Calculator
    # In this example the 'reform' is a change to 2017 law (the
    # baseline policy is tax law in 2018)
    reform_url = (
        "github://PSLmodels:examples@main/psl_examples/"
        + "taxcalc/2017_law.json"
    )
    ref = Calculator.read_json_param_objects(reform_url, None)
    iit_reform = ref["policy"]

    # create new Specifications object for reform simulation
    p2 = Specifications(
        baseline=False,
        num_workers=num_workers,
        baseline_dir=base_dir,
        output_base=reform_dir,
    )
    # Update parameters for baseline from default json file
    p2.update_specifications(
        json.load(
            open(
                os.path.join(
                    CUR_DIR, "..", "ogusa", "ogusa_default_parameters.json"
                )
            )
        )
    )
    p2.tax_func_type = "GS"
    # Use calibration class to estimate reform tax functions from
    # Tax-Calculator, specifying reform for Tax-Calculator in iit_reform
    c2 = Calibration(
        p2, iit_reform=iit_reform, estimate_tax_functions=True, client=client
    )
    # close and delete client bc cache is too large
    client.close()
    del client
    client = Client(n_workers=num_workers)
    # update tax function parameters in Specifications Object
    d = c2.get_dict()
    # # additional parameters to change
    updated_params = {
        "cit_rate": [[0.35]],
        "etr_params": d["etr_params"],
        "mtrx_params": d["mtrx_params"],
        "mtry_params": d["mtry_params"],
        "mean_income_data": d["mean_income_data"],
        "frac_tax_payroll": d["frac_tax_payroll"],
    }
    p2.update_specifications(updated_params)
    # Run model
    start_time = time.time()
    runner(p2, time_path=True, client=client)
    print("run time = ", time.time() - start_time)

    fetch_profiles(client)

    client.close()

    """
    ------------------------------------------------------------------------
    Save some results of simulations
    ------------------------------------------------------------------------
    """
    base_tpi = safe_read_pickle(os.path.join(base_dir, "TPI", "TPI_vars.pkl"))
    base_params = safe_read_pickle(os.path.join(base_dir, "model_params.pkl"))
    reform_tpi = safe_read_pickle(
        os.path.join(reform_dir, "TPI", "TPI_vars.pkl")
    )
    reform_params = safe_read_pickle(
        os.path.join(reform_dir, "model_params.pkl")
    )
    ans = ot.macro_table(
        base_tpi,
        base_params,
        reform_tpi=reform_tpi,
        reform_params=reform_params,
        var_list=["Y", "C", "K", "L", "r", "w"],
        output_type="pct_diff",
        num_years=10,
        start_year=base_params.start_year,
    )

    # create plots of output
    op.plot_all(
        base_dir, reform_dir, os.path.join(CUR_DIR, "OG-USA_example_plots")
    )

    print("Percentage changes in aggregates:", ans)
    # save percentage change output to csv file
    ans.to_csv("ogusa_example_output.csv")


if __name__ == "__main__":
    # execute only if run as a script
    main()
