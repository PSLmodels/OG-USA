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


def fetch_profiles(client, idx):
    workers = client.scheduler_info()["workers"]
    profiles = client.run(
        lambda dask_worker: dask_worker.profile.dump_stats(
            "profile_worker_{idx}_{address}.pstats".format(
                idx=idx, address=dask_worker.address[-5:]
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
    fetch_profiles(client, 0)
    client.close()
    del client
    cluster = LocalCluster(
        n_workers=7,
        threads_per_worker=13,
        worker_dashboard_address=":0",
        preload=[
            "/usr/local/google/home/talumbau/src/OG-USA/examples/worker_setup.py"
        ],
    )
    client = Client(cluster)
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
    fetch_profiles(client, 1)
    client.close()

    

if __name__ == "__main__":
    # execute only if run as a script
    main()
