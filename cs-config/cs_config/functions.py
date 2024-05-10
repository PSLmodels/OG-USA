from ogusa.calibrate import Calibration
from ogcore.parameters import Specifications
from ogusa.constants import (
    REFORM_DIR,
    BASELINE_DIR,
    DEFAULT_START_YEAR,
    TC_LAST_YEAR,
)
from ogcore import output_plots as op
from ogcore import output_tables as ot
from ogcore import SS, TPI, utils
import os
import io
import pickle
import json
import inspect
import paramtools
from distributed import Client
from taxcalc import Policy
from collections import OrderedDict
from .helpers import retrieve_puf
from cs2tc import convert_policy_adjustment

AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
PUF_S3_FILE_LOCATION = os.environ.get(
    "PUF_S3_LOCATION", "s3://ospc-data-files/puf.20210720.csv.gz"
)
CUR_DIR = os.path.dirname(os.path.realpath(__file__))

# Get Tax-Calculator default parameters
TCPATH = inspect.getfile(Policy)
TCDIR = os.path.dirname(TCPATH)
with open(os.path.join(TCDIR, "policy_current_law.json"), "r") as f:
    pcl = json.loads(f.read())
RES = convert_policy_adjustment(pcl)


class TCParams(paramtools.Parameters):
    defaults = RES


class MetaParams(paramtools.Parameters):
    """
    Meta parameters class for COMP.  These parameters will be in a drop
    down menu on COMP.
    """

    array_first = True
    defaults = {
        "year": {
            "title": "Start year",
            "description": "Year for parameters.",
            "type": "int",
            "value": DEFAULT_START_YEAR,
            "validators": {
                "range": {"min": 2015, "max": Policy.LAST_BUDGET_YEAR}
            },
        },
        "data_source": {
            "title": "Data source",
            "description": "Data source for Tax-Calculator to use",
            "type": "str",
            "value": "CPS",
            "validators": {"choice": {"choices": ["PUF", "CPS"]}},
        },
        "time_path": {
            "title": "Solve for economy's transition path?",
            "description": (
                "Whether to solve for the transition path"
                + " in addition to the steady-state"
            ),
            "type": "bool",
            "value": True,
            "validators": {"range": {"min": False, "max": True}},
        },
    }


def get_version():
    return "0.1.2"


def get_inputs(meta_param_dict):
    meta_params = MetaParams()
    meta_params.adjust(meta_param_dict)
    # Set default OG-USA parameters
    ogusa_params = Specifications()
    ogusa_params.update_specifications(
        json.load(
            open(
                os.path.join(
                    CUR_DIR,
                    "..",
                    "..",
                    "ogusa",
                    "ogusa_default_parameters.json",
                )
            )
        )
    )
    ogusa_params.start_year = meta_params.year
    filtered_ogusa_params = OrderedDict()
    filter_list = [
        "chi_n_80",
        "chi_b",
        "eta",
        "zeta",
        "constant_demographics",
        "ltilde",
        "use_zeta",
        "constant_rates",
        "zero_taxes",
        "analytical_mtrs",
        "age_specific",
        "gamma",
        "epsilon",
        "start_year",
        "e",
        "chi_n",
        "omega_SS",
        "omega_S_preTP",
        "omega",
        "rho",
        "imm_rates",
        "g_n",
        "g_n_ss",
        "etr_params",
        "mtrx_params",
        "mtry_params",
        "frac_tax_payroll",
        "mean_income_data",
    ]
    for k, v in ogusa_params.dump().items():
        if (
            (k not in filter_list)
            and (v.get("section_1", False) != "Model Solution Parameters")
            and (v.get("section_2", False) != "Model Dimensions")
        ):
            filtered_ogusa_params[k] = v
            print("filtered ogusa = ", k)
    # Set default TC params
    iit_params = TCParams()
    iit_params.set_state(year=meta_params.year.tolist())
    filtered_iit_params = OrderedDict()
    for k, v in iit_params.dump().items():
        if k == "schema" or v.get("section_1", False):
            filtered_iit_params[k] = v

    default_params = {
        "OG-USA Parameters": filtered_ogusa_params,
        "Tax-Calculator Parameters": filtered_iit_params,
    }

    return {
        "meta_parameters": meta_params.dump(),
        "model_parameters": default_params,
    }


def validate_inputs(meta_param_dict, adjustment, errors_warnings):
    # ogusa doesn't look at meta_param_dict for validating inputs.
    params = Specifications()
    params.adjust(adjustment["OG-USA Parameters"], raise_errors=False)
    errors_warnings["OG-USA Parameters"]["errors"].update(params.errors)
    # Validate TC parameter inputs
    pol_params = {}
    # drop checkbox parameters.
    for param, data in list(adjustment["Tax-Calculator Parameters"].items()):
        if not param.endswith("checkbox"):
            pol_params[param] = data
    iit_params = TCParams()
    iit_params.adjust(pol_params, raise_errors=False)
    errors_warnings["Tax-Calculator Parameters"]["errors"].update(
        iit_params.errors
    )

    return {"errors_warnings": errors_warnings}


def run_model(meta_param_dict, adjustment):
    """
    Initializes classes from OG-USA that compute the model under
    different policies.  Then calls function get output objects.
    """
    print("Meta_param_dict = ", meta_param_dict)
    print("adjustment dict = ", adjustment)

    meta_params = MetaParams()
    meta_params.adjust(meta_param_dict)
    if meta_params.data_source == "PUF":
        data = retrieve_puf(
            PUF_S3_FILE_LOCATION, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
        )
        # set name of cached baseline file in case use below
        cached_pickle = "TxFuncEst_baseline_PUF.pkl"
    else:
        data = "cps"
        # set name of cached baseline file in case use below
        cached_pickle = "TxFuncEst_baseline_CPS.pkl"
    # Get TC params adjustments
    iit_mods = convert_policy_adjustment(
        adjustment["Tax-Calculator Parameters"]
    )
    # Create output directory structure
    base_dir = os.path.join(CUR_DIR, BASELINE_DIR)
    reform_dir = os.path.join(CUR_DIR, REFORM_DIR)
    dirs = [base_dir, reform_dir]
    for _dir in dirs:
        utils.mkdirs(_dir)

    # Dask parmeters
    num_workers = 2
    memory_limit = "10GiB"
    client = Client(
        n_workers=num_workers,
        threads_per_worker=1,
        memory_limit=memory_per_worker,
    )
    # TODO: Swap to these parameters when able to specify tax function
    # and model workers separately
    # num_workers_txf = 5
    # num_workers_mod = 6

    # whether to estimate tax functions from microdata
    run_micro = True
    time_path = meta_param_dict["time_path"][0]["value"]

    # filter out OG-USA params that will not change between baseline and
    # reform runs (these are the non-policy parameters)
    filtered_ogusa_params = {}
    constant_param_set = {
        "frisch",
        "beta_annual",
        "sigma",
        "g_y_annual",
        "gamma",
        "epsilon",
        "Z",
        "delta_annual",
        "small_open",
        "world_int_rate",
        "initial_debt_ratio",
        "initial_foreign_debt_ratio",
        "zeta_D",
        "zeta_K",
        "tG1",
        "tG2",
        "rho_G",
        "debt_ratio_ss",
        "budget_balance",
    }
    filtered_ogusa_params = OrderedDict()
    for k, v in adjustment["OG-USA Parameters"].items():
        if k in constant_param_set:
            filtered_ogusa_params[k] = v

    # Solve baseline model
    start_year = meta_param_dict["year"][0]["value"]
    base_spec = {
        **{
            "start_year": start_year,
            "tax_func_type": "GS",
            "age_specific": False,
        },
        **filtered_ogusa_params,
    }
    base_params = Specifications(
        output_base=base_dir,
        baseline_dir=base_dir,
        baseline=True,
        num_workers=num_workers,
    )
    base_params.update_specifications(
        json.load(
            open(
                os.path.join(
                    CUR_DIR,
                    "..",
                    "..",
                    "ogusa",
                    "ogusa_default_parameters.json",
                )
            )
        )
    )
    base_params.update_specifications(base_spec)
    BW = TC_LAST_YEAR - start_year + 1
    base_params.BW = BW
    # Will need to figure out how to handle default tax functions here
    # For now, estimating tax functions even for baseline
    c_base = Calibration(
        base_params,
        iit_reform={},
        estimate_tax_functions=True,
        data=data,
        client=client,
    )
    client.close()
    del client
    client = Client(
        n_workers=num_workers,
        threads_per_worker=1,
        memory_limit=memory_per_worker,
    )
    # update tax function parameters in Specifications Object
    d_base = c_base.get_dict()
    # additional parameters to change
    updated_txfunc_params = {
        "etr_params": d_base["etr_params"],
        "mtrx_params": d_base["mtrx_params"],
        "mtry_params": d_base["mtry_params"],
        "mean_income_data": d_base["mean_income_data"],
        "frac_tax_payroll": d_base["frac_tax_payroll"],
    }
    base_params.update_specifications(updated_txfunc_params)
    base_ss = SS.run_SS(base_params, client=client)
    utils.mkdirs(os.path.join(base_dir, "SS"))
    base_ss_dir = os.path.join(base_dir, "SS", "SS_vars.pkl")
    client.close()
    del client
    client = Client(
        n_workers=num_workers,
        threads_per_worker=1,
        memory_limit=memory_per_worker,
    )
    with open(base_ss_dir, "wb") as f:
        pickle.dump(base_ss, f)
    if time_path:
        base_tpi = TPI.run_TPI(base_params, client=client)
        tpi_dir = os.path.join(base_dir, "TPI", "TPI_vars.pkl")
        with open(tpi_dir, "wb") as f:
            pickle.dump(base_tpi, f)
    else:
        base_tpi = None

    # Solve reform model
    reform_spec = base_spec
    reform_spec.update(adjustment["OG-USA Parameters"])
    reform_params = Specifications(
        output_base=reform_dir,
        baseline_dir=base_dir,
        baseline=False,
        num_workers=num_workers,
    )
    reform_params.update_specifications(
        json.load(
            open(
                os.path.join(
                    CUR_DIR,
                    "..",
                    "..",
                    "ogusa",
                    "ogusa_default_parameters.json",
                )
            )
        )
    )
    reform_params.update_specifications(reform_spec)
    reform_params.BW = BW
    c_reform = Calibration(
        reform_params,
        iit_reform=iit_mods,
        estimate_tax_functions=True,
        data=data,
        client=client,
    )
    # update tax function parameters in Specifications Object
    d_reform = c_reform.get_dict()
    # additional parameters to change
    updated_txfunc_params = {
        "etr_params": d_reform["etr_params"],
        "mtrx_params": d_reform["mtrx_params"],
        "mtry_params": d_reform["mtry_params"],
        "mean_income_data": d_reform["mean_income_data"],
        "frac_tax_payroll": d_reform["frac_tax_payroll"],
    }
    reform_params.update_specifications(updated_txfunc_params)
    reform_ss = SS.run_SS(reform_params, client=client)
    utils.mkdirs(os.path.join(reform_dir, "SS"))
    reform_ss_dir = os.path.join(reform_dir, "SS", "SS_vars.pkl")
    with open(reform_ss_dir, "wb") as f:
        pickle.dump(reform_ss, f)
    if time_path:
        reform_tpi = TPI.run_TPI(reform_params, client=client)
    else:
        reform_tpi = None

    comp_dict = comp_output(
        base_params,
        base_ss,
        reform_params,
        reform_ss,
        time_path,
        base_tpi,
        reform_tpi,
    )

    # Shut down client and make sure all of its references are
    # cleaned up.
    client.close()
    del client

    return comp_dict


def comp_output(
    base_params,
    base_ss,
    reform_params,
    reform_ss,
    time_path,
    base_tpi=None,
    reform_tpi=None,
    var="cssmat",
):
    """
    Function to create output for the COMP platform
    """
    if time_path:
        macro_table_title = "Percentage Changes in Economic Aggregates Between"
        macro_table_title += " Baseline and Reform Policy"
        dynamic_rev_table_title = "Dynamic Revenue Estimate Decomposition"
        download_table_title = "Economic variables over the time path"
        plot1_title = "Pct Changes in Economic Aggregates Between"
        plot1_title += " Baseline and Reform Policy"
        plot2_title = "Pct Changes in Interest Rates and Wages"
        plot2_title += " Between Baseline and Reform Policy"
        plot3_title = "Differences in Fiscal Variables Relative to GDP"
        plot3_title += " Between Baseline and Reform Policy"
        out_table = ot.tp_output_dump_table(
            base_params,
            base_tpi,
            reform_params,
            reform_tpi,
            table_format="csv",
        )
        macro_table = ot.macro_table(
            base_tpi,
            base_params,
            reform_tpi,
            reform_params,
            var_list=["Y", "C", "I_total", "L", "D", "G", "r", "w"],
            output_type="pct_diff",
            num_years=10,
            include_SS=True,
            include_overall=True,
            start_year=base_params.start_year,
            table_format="html",
        )
        dynamic_rev_table = ot.dynamic_revenue_decomposition(
            base_params,
            base_tpi,
            base_ss,
            reform_params,
            reform_tpi,
            reform_ss,
            num_years=10,
            include_SS=True,
            include_overall=True,
            include_business_tax=True,
            full_break_out=False,
            start_year=base_params.start_year,
            table_format="html",
        )
        fig1 = op.plot_aggregates(
            base_tpi,
            base_params,
            reform_tpi,
            reform_params,
            var_list=["Y", "C", "K", "L"],
            plot_type="pct_diff",
            num_years_to_plot=50,
            start_year=base_params.start_year,
            vertical_line_years=[
                base_params.start_year + base_params.tG1,
                base_params.start_year + base_params.tG2,
            ],
            plot_title=None,
            path=None,
        )
        in_memory_file1 = io.BytesIO()
        fig1.savefig(in_memory_file1, format="png", bbox_inches="tight")
        in_memory_file1.seek(0)
        fig2 = op.plot_aggregates(
            base_tpi,
            base_params,
            reform_tpi,
            reform_params,
            var_list=["r_gov", "w"],
            plot_type="pct_diff",
            num_years_to_plot=50,
            start_year=base_params.start_year,
            vertical_line_years=[
                base_params.start_year + base_params.tG1,
                base_params.start_year + base_params.tG2,
            ],
            plot_title=None,
            path=None,
        )
        in_memory_file2 = io.BytesIO()
        fig2.savefig(in_memory_file2, format="png", bbox_inches="tight")
        in_memory_file2.seek(0)
        fig3 = op.plot_gdp_ratio(
            base_tpi,
            base_params,
            reform_tpi,
            reform_params,
            var_list=["D", "G", "total_tax_revenue"],
            plot_type="diff",
            num_years_to_plot=50,
            start_year=base_params.start_year,
            vertical_line_years=[
                base_params.start_year + base_params.tG1,
                base_params.start_year + base_params.tG2,
            ],
            plot_title=None,
            path=None,
        )
        in_memory_file3 = io.BytesIO()
        fig3.savefig(in_memory_file3, format="png", bbox_inches="tight")
        in_memory_file3.seek(0)

        comp_dict = {
            "renderable": [
                {
                    "media_type": "PNG",
                    "title": plot1_title,
                    "data": in_memory_file1.read(),
                },
                {
                    "media_type": "PNG",
                    "title": plot2_title,
                    "data": in_memory_file2.read(),
                },
                {
                    "media_type": "PNG",
                    "title": plot3_title,
                    "data": in_memory_file3.read(),
                },
                {
                    "media_type": "table",
                    "title": macro_table_title,
                    "data": macro_table,
                },
                {
                    "media_type": "table",
                    "title": dynamic_rev_table_title,
                    "data": dynamic_rev_table,
                },
            ],
            "downloadable": [
                {
                    "media_type": "CSV",
                    "title": download_table_title,
                    "data": out_table.to_csv(),
                }
            ],
        }
    else:
        macro_table_title = "Percentage Changes in Economic Aggregates Between"
        macro_table_title += " Baseline and Reform Policy"
        plot_title = "Percentage Changes in Consumption by Lifetime Income"
        plot_title += " Percentile Group"
        out_table = ot.macro_table_SS(
            base_ss,
            reform_ss,
            var_list=[
                "Yss",
                "Css",
                "Iss_total",
                "Gss",
                "total_tax_revenue",
                "Lss",
                "rss",
                "wss",
            ],
            table_format="csv",
        )
        macro_table_SS = ot.macro_table_SS(
            base_ss,
            reform_ss,
            var_list=[
                "Yss",
                "Css",
                "Iss_total",
                "Gss",
                "total_tax_revenue",
                "Lss",
                "rss",
                "wss",
            ],
            table_format="html",
        )
        fig = op.ability_bar_ss(
            base_ss, base_params, reform_ss, reform_params, var=var
        )
        in_memory_file = io.BytesIO()
        fig.savefig(in_memory_file, format="png", bbox_inches="tight")
        in_memory_file.seek(0)

        comp_dict = {
            "renderable": [
                {
                    "media_type": "PNG",
                    "title": plot_title,
                    "data": in_memory_file.read(),
                },
                {
                    "media_type": "table",
                    "title": macro_table_title,
                    "data": macro_table_SS,
                },
            ],
            "downloadable": [
                {
                    "media_type": "CSV",
                    "title": macro_table_title,
                    "data": out_table.to_csv(),
                }
            ],
        }

    return comp_dict
