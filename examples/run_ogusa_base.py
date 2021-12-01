import multiprocessing
from distributed import Client
import os
import json
import time
from taxcalc import Calculator
from ogusa.calibrate import Calibration
from ogcore.parameters import Specifications
from ogcore import output_tables as ot
from ogcore import output_plots as op
from ogcore.execute import runner
from ogcore.utils import safe_read_pickle, mkdirs


def main():
    # Define parameters to use for multiprocessing
    client = Client()
    num_workers = min(multiprocessing.cpu_count(), 7)
    print("Number of workers = ", num_workers)

    # Directories to save data
    CUR_DIR = os.path.dirname(os.path.realpath(__file__))
    base_dir = os.path.join(CUR_DIR, "OG-USA-DebtDef", "OUTPUT_BASELINE")
    # Make sure base directory images folder is made
    base_dir_images = os.path.join(base_dir, "images")
    mkdirs(base_dir_images)

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
    # Use calibration class to estimate baseline tax functions from
    # Tax-Calculator, specifing reform for Tax-Calculator in iit_reform
    # c2 = Calibration(
    #     p, estimate_tax_functions=True, client=client
    # )
    tax_func_path = os.path.join(base_dir, 'TxFuncEst_baseline.pkl')
    c2 = Calibration(
        p, tax_func_path=tax_func_path, client=client
    )
    # update tax function parameters in Specifications Object
    d = c2.get_dict()
    # additional parameters to change in baseline different from default values
    updated_params_base = {
        "tG1": 29,
        "initial_debt_ratio": 1.0,
        "alpha_T": [0.0725],
        "debt_ratio_ss": 2.2,
        "etr_params": d["etr_params"],
        "mtrx_params": d["mtrx_params"],
        "mtry_params": d["mtry_params"],
        "mean_income_data": d["mean_income_data"],
        "frac_tax_payroll": d["frac_tax_payroll"],
    }
    p.update_specifications(updated_params_base)

    # Run model
    start_time = time.time()
    runner(p, time_path=True, client=client)
    print("run time = ", time.time() - start_time)

    client.close()

    # create plots D/Y
    base_tpi = safe_read_pickle(os.path.join(base_dir, 'TPI', 'TPI_vars.pkl'))
    base_params = safe_read_pickle(os.path.join(base_dir, 'model_params.pkl'))
    D_Y_base_path = os.path.join(base_dir_images, 'DebtGDPratio.png')
    op.plot_gdp_ratio(base_tpi, base_params,
                      num_years_to_plot=int(base_params.tG1 + 10),
                      vertical_line_years=[base_params.start_year +
                                           base_params.tG1],
                      path=D_Y_base_path)


if __name__ == "__main__":
    # execute only if run as a script
    main()
