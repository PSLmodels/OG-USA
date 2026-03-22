"""
Backend model functions for the OG-USA Panel app.

Adapted from cs-config/cs_config/functions.py with all Compute Studio
dependencies (cs2tc, cs_storage, cs-kit) removed:
  - convert_policy_adjustment is inlined below
  - comp_output returns matplotlib Figure objects and DataFrames
    instead of PNG bytes and HTML strings, so Panel can render them
    natively
"""
import importlib.resources
import inspect
import json
import os
import pickle
from collections import OrderedDict

import ogusa
import paramtools
from distributed import Client
from ogcore import SS, TPI, output_plots as op, output_tables as ot, utils
from ogcore.parameters import Specifications
from ogusa.calibrate import Calibration
from ogusa.constants import (
    BASELINE_DIR,
    DEFAULT_START_YEAR,
    REFORM_DIR,
    TC_LAST_YEAR,
)
from taxcalc import GrowFactors, Policy, Records

from .helpers import retrieve_puf, retrieve_tmd

# ---------------------------------------------------------------------------
# Inline replacement for cs2tc.convert_policy_adjustment
# ---------------------------------------------------------------------------

def _convert_policy_adjustment(policy_adjustment: dict) -> dict:
    """
    Replace ``param_checkbox`` keys with ``param-indexed`` and drop the
    checkbox entries.  This is the only behaviour from cs2tc that
    functions.py relied on.
    """
    params = {}
    for param, data in policy_adjustment.items():
        if param.endswith("_checkbox"):
            base = param[: -len("_checkbox")]
            params[f"{base}-indexed"] = data
        else:
            params[param] = data
    return params


# ---------------------------------------------------------------------------
# Load Tax-Calculator policy defaults once at import time
# ---------------------------------------------------------------------------

_TCPATH = inspect.getfile(Policy)
_TCDIR = os.path.dirname(_TCPATH)
with open(os.path.join(_TCDIR, "policy_current_law.json"), "r") as _f:
    _PCL = json.load(_f)

# Apply the checkbox conversion so TCParams matches the original behaviour
_TC_DEFAULTS = _convert_policy_adjustment(_PCL)


class TCParams(paramtools.Parameters):
    defaults = _TC_DEFAULTS


# ---------------------------------------------------------------------------
# Meta-parameters (start year, data source, time path)
# ---------------------------------------------------------------------------

class MetaParams(paramtools.Parameters):
    array_first = True
    defaults = {
        "year": {
            "title": "Start year",
            "description": "First year of the budget window.",
            "type": "int",
            "value": DEFAULT_START_YEAR,
            "validators": {
                "range": {"min": 2015, "max": Policy.LAST_BUDGET_YEAR}
            },
        },
        "data_source": {
            "title": "Data source",
            "description": "Micro-data used for Tax-Calculator estimation.",
            "type": "str",
            "value": "CPS",
            "validators": {"choice": {"choices": ["CPS", "PUF"]}},
        },
        "time_path": {
            "title": "Solve transition path?",
            "description": (
                "Whether to solve the full transition-path equilibrium "
                "in addition to the steady state.  Adds ~30 min of "
                "compute time."
            ),
            "type": "bool",
            "value": True,
            "validators": {"range": {"min": False, "max": True}},
        },
    }


# ---------------------------------------------------------------------------
# Parameters hidden from the user interface
# ---------------------------------------------------------------------------

# These are either calibrated internally, set via meta-parameters, or
# are large arrays that cannot be edited widget-by-widget.
_FILTER_LIST = {
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
}

# OG-USA parameters that are fixed across baseline and reform (structural
# parameters).  Changes to these apply to both runs equally.
_CONSTANT_PARAM_SET = {
    "frisch",
    "beta_annual",
    "sigma",
    "g_y_annual",
    "gamma",
    "epsilon",
    "Z",
    "delta_annual",
    "small_open",
    "world_int_rate_annual",
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


# ---------------------------------------------------------------------------
# Public API used by the Panel UI
# ---------------------------------------------------------------------------

def get_version():
    return ogusa.__version__


def get_inputs(meta_param_dict):
    """
    Return default parameter specifications for the UI.

    Parameters
    ----------
    meta_param_dict : dict
        Paramtools-style dict for MetaParams (year, data_source,
        time_path).

    Returns
    -------
    dict with keys:
        ``meta_parameters``  – MetaParams dump
        ``model_parameters`` – dict with keys
            ``"OG-USA Parameters"``      – filtered Specifications dump
            ``"Tax-Calculator Parameters"`` – filtered TCParams dump
    """
    meta_params = MetaParams()
    meta_params.adjust(meta_param_dict)

    # OG-USA parameters -------------------------------------------------------
    ogusa_params = Specifications()
    with importlib.resources.open_text(
        "ogusa", "ogusa_default_parameters.json"
    ) as f:
        ogusa_defaults = json.load(f)
    # update_specifications can fail for multi-sector parameters when J/S/M
    # are not set; wrap individual failures so the rest still load.
    try:
        ogusa_params.update_specifications(ogusa_defaults)
    except Exception:
        pass
    ogusa_params.start_year = meta_params.year

    filtered_ogusa = OrderedDict()
    for k, v in ogusa_params.dump().items():
        if (
            k not in _FILTER_LIST
            and v.get("section_1") != "Model Solution Parameters"
            and v.get("section_2") != "Model Dimensions"
        ):
            filtered_ogusa[k] = v

    # Tax-Calculator parameters -----------------------------------------------
    iit_params = TCParams()
    try:
        iit_params.set_state(year=meta_params.year.tolist())
    except Exception:
        pass
    filtered_iit = OrderedDict()
    for k, v in iit_params.dump().items():
        if k == "schema" or v.get("section_1"):
            filtered_iit[k] = v

    return {
        "meta_parameters": meta_params.dump(),
        "model_parameters": {
            "OG-USA Parameters": filtered_ogusa,
            "Tax-Calculator Parameters": filtered_iit,
        },
    }


def validate_inputs(meta_param_dict, adjustment, errors_warnings):
    """
    Validate user-supplied parameter adjustments.

    Returns
    -------
    dict  ``{"errors_warnings": errors_warnings}``
    """
    params = Specifications()
    params.adjust(
        adjustment.get("OG-USA Parameters", {}), raise_errors=False
    )
    errors_warnings["OG-USA Parameters"]["errors"].update(params.errors)

    pol_params = {
        k: v
        for k, v in adjustment.get(
            "Tax-Calculator Parameters", {}
        ).items()
        if not k.endswith("checkbox")
    }
    iit_params = TCParams()
    iit_params.adjust(pol_params, raise_errors=False)
    errors_warnings["Tax-Calculator Parameters"]["errors"].update(
        iit_params.errors
    )
    return {"errors_warnings": errors_warnings}


def run_model(
    meta_param_dict,
    adjustment,
    output_base,
    phase_callback=None,
):
    """
    Run baseline + reform models.

    Parameters
    ----------
    meta_param_dict : dict
        Paramtools-style dict for MetaParams.
    adjustment : dict
        User adjustments with keys ``"OG-USA Parameters"`` and
        ``"Tax-Calculator Parameters"``.
    output_base : str
        Directory in which to write baseline/reform output sub-folders.
    phase_callback : callable or None
        Optional ``phase_callback(message: str)`` called before each
        major computation phase so the UI can update a progress label.

    Returns
    -------
    dict  from ``comp_output()``
    """
    def _phase(msg):
        print(msg)
        if phase_callback is not None:
            phase_callback(msg)

    meta_params = MetaParams()
    meta_params.adjust(meta_param_dict)

    # ---- Data source -------------------------------------------------------
    if meta_params.data_source == "PUF":
        data = retrieve_puf()
        weights = Records.PUF_WEIGHTS_FILENAME
        records_start_year = Records.PUFCSV_YEAR
        if data is None:
            _phase("PUF unavailable – falling back to CPS")
            meta_params.adjust({"data_source": "CPS"})

    if meta_params.data_source == "CPS":
        data = "cps"
        weights = Records.PUF_WEIGHTS_FILENAME
        records_start_year = Records.CPSCSV_YEAR

    iit_mods = _convert_policy_adjustment(
        adjustment.get("Tax-Calculator Parameters", {})
    )

    # ---- Output directories ------------------------------------------------
    base_dir = os.path.join(output_base, BASELINE_DIR)
    reform_dir = os.path.join(output_base, REFORM_DIR)
    for d in [base_dir, reform_dir]:
        utils.mkdirs(d)

    num_workers = 2
    time_path = meta_param_dict["time_path"][0]["value"]
    start_year = meta_param_dict["year"][0]["value"]
    BW = TC_LAST_YEAR - start_year + 1

    # Parameters shared between baseline and reform
    filtered_ogusa = {
        k: v
        for k, v in adjustment.get("OG-USA Parameters", {}).items()
        if k in _CONSTANT_PARAM_SET
    }
    base_spec = {
        "start_year": start_year,
        "tax_func_type": "GS",
        "age_specific": False,
        **filtered_ogusa,
    }

    with importlib.resources.open_text(
        "ogusa", "ogusa_default_parameters.json"
    ) as f:
        ogusa_defaults = json.load(f)

    # ========================================================================
    # BASELINE
    # ========================================================================
    _phase("Estimating baseline tax functions…")
    client = Client(
        n_workers=num_workers,
        threads_per_worker=1,
        memory_limit="10GiB",
    )

    base_params = Specifications(
        output_base=base_dir,
        baseline_dir=base_dir,
        baseline=True,
        num_workers=num_workers,
    )
    base_params.update_specifications(ogusa_defaults)
    base_params.update_specifications(base_spec)
    base_params.BW = BW

    c_base = Calibration(
        base_params,
        iit_reform={},
        estimate_tax_functions=True,
        data=data,
        client=client,
    )
    client.close()
    del client

    d_base = c_base.get_dict()
    base_params.update_specifications(
        {
            "etr_params": d_base["etr_params"],
            "mtrx_params": d_base["mtrx_params"],
            "mtry_params": d_base["mtry_params"],
            "mean_income_data": d_base["mean_income_data"],
            "frac_tax_payroll": d_base["frac_tax_payroll"],
        }
    )

    _phase("Solving baseline steady state…")
    client = Client(
        n_workers=num_workers,
        threads_per_worker=1,
        memory_limit="10GiB",
    )
    base_ss = SS.run_SS(base_params, client=client)
    utils.mkdirs(os.path.join(base_dir, "SS"))
    with open(os.path.join(base_dir, "SS", "SS_vars.pkl"), "wb") as f:
        pickle.dump(base_ss, f)

    if time_path:
        _phase("Solving baseline transition path…")
        base_tpi = TPI.run_TPI(base_params, client=client)
        utils.mkdirs(os.path.join(base_dir, "TPI"))
        with open(
            os.path.join(base_dir, "TPI", "TPI_vars.pkl"), "wb"
        ) as f:
            pickle.dump(base_tpi, f)
    else:
        base_tpi = None

    client.close()
    del client

    # ========================================================================
    # REFORM
    # ========================================================================
    _phase("Estimating reform tax functions…")
    client = Client(
        n_workers=num_workers,
        threads_per_worker=1,
        memory_limit="10GiB",
    )

    reform_spec = {
        **base_spec,
        **adjustment.get("OG-USA Parameters", {}),
    }
    reform_params = Specifications(
        output_base=reform_dir,
        baseline_dir=base_dir,
        baseline=False,
        num_workers=num_workers,
    )
    reform_params.update_specifications(ogusa_defaults)
    reform_params.update_specifications(reform_spec)
    reform_params.BW = BW

    c_reform = Calibration(
        reform_params,
        iit_reform=iit_mods,
        estimate_tax_functions=True,
        data=data,
        gfactors=GrowFactors.FILE_NAME,
        weights=weights,
        records_start_year=records_start_year,
        client=client,
    )
    client.close()
    del client

    d_reform = c_reform.get_dict()
    reform_params.update_specifications(
        {
            "etr_params": d_reform["etr_params"],
            "mtrx_params": d_reform["mtrx_params"],
            "mtry_params": d_reform["mtry_params"],
            "mean_income_data": d_reform["mean_income_data"],
            "frac_tax_payroll": d_reform["frac_tax_payroll"],
        }
    )

    _phase("Solving reform steady state…")
    client = Client(
        n_workers=num_workers,
        threads_per_worker=1,
        memory_limit="10GiB",
    )
    reform_ss = SS.run_SS(reform_params, client=client)
    utils.mkdirs(os.path.join(reform_dir, "SS"))
    with open(
        os.path.join(reform_dir, "SS", "SS_vars.pkl"), "wb"
    ) as f:
        pickle.dump(reform_ss, f)

    if time_path:
        _phase("Solving reform transition path…")
        reform_tpi = TPI.run_TPI(reform_params, client=client)
        utils.mkdirs(os.path.join(reform_dir, "TPI"))
        with open(
            os.path.join(reform_dir, "TPI", "TPI_vars.pkl"), "wb"
        ) as f:
            pickle.dump(reform_tpi, f)
    else:
        reform_tpi = None

    client.close()
    del client

    _phase("Generating output…")
    return comp_output(
        base_params,
        base_ss,
        reform_params,
        reform_ss,
        time_path,
        base_tpi,
        reform_tpi,
    )


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
    Build the output dict consumed by the Panel results viewer.

    Unlike the original Compute Studio version this returns matplotlib
    Figure objects (not PNG bytes) and pandas DataFrames / HTML strings
    so Panel can render them natively.

    Returns
    -------
    dict with keys:
        ``figures``   – list of ``{"title": str, "fig": Figure}``
        ``tables``    – list of ``{"title": str, "html": str}``
        ``downloads`` – list of ``{"title": str, "data": str,
                                    "filename": str}``
    """
    vlines = [
        base_params.start_year + base_params.tG1,
        base_params.start_year + base_params.tG2,
    ]

    if time_path:
        # ---- Tables --------------------------------------------------------
        macro_table_html = ot.macro_table(
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
        rev_table_html = ot.dynamic_revenue_decomposition(
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
        out_csv = ot.tp_output_dump_table(
            base_params,
            base_tpi,
            reform_params,
            reform_tpi,
            table_format="csv",
        )

        # ---- Figures -------------------------------------------------------
        fig1 = op.plot_aggregates(
            base_tpi,
            base_params,
            reform_tpi,
            reform_params,
            var_list=["Y", "C", "K", "L"],
            plot_type="pct_diff",
            num_years_to_plot=50,
            start_year=base_params.start_year,
            vertical_line_years=vlines,
            plot_title="% Changes in Macro Aggregates",
            path=None,
        )
        fig2 = op.plot_aggregates(
            base_tpi,
            base_params,
            reform_tpi,
            reform_params,
            var_list=["r_gov", "w"],
            plot_type="pct_diff",
            num_years_to_plot=50,
            start_year=base_params.start_year,
            vertical_line_years=vlines,
            plot_title="% Changes in Interest Rate and Wage",
            path=None,
        )
        fig3 = op.plot_gdp_ratio(
            base_tpi,
            base_params,
            reform_tpi,
            reform_params,
            var_list=["D", "G", "total_tax_revenue"],
            plot_type="diff",
            num_years_to_plot=50,
            start_year=base_params.start_year,
            vertical_line_years=vlines,
            plot_title="Fiscal Variables Relative to GDP (pp diff)",
            path=None,
        )

        return {
            "figures": [
                {
                    "title": "% Changes in Macro Aggregates",
                    "fig": fig1,
                },
                {
                    "title": "% Changes in Interest Rate and Wage",
                    "fig": fig2,
                },
                {
                    "title": "Fiscal Variables Relative to GDP",
                    "fig": fig3,
                },
            ],
            "tables": [
                {
                    "title": "% Changes in Economic Aggregates",
                    "html": macro_table_html,
                },
                {
                    "title": "Dynamic Revenue Decomposition",
                    "html": rev_table_html,
                },
            ],
            "downloads": [
                {
                    "title": "Economic Variables – Full Time Series",
                    "data": out_csv.to_csv(),
                    "filename": "ogusa_output.csv",
                }
            ],
        }

    else:
        # Steady-state only
        macro_table_html = ot.macro_table_SS(
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
        out_csv = ot.macro_table_SS(
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
        fig = op.ability_bar_ss(
            base_ss, base_params, reform_ss, reform_params, var=var
        )

        return {
            "figures": [
                {
                    "title": "Consumption by Lifetime Income Group",
                    "fig": fig,
                }
            ],
            "tables": [
                {
                    "title": "Steady-State Macro Aggregates",
                    "html": macro_table_html,
                }
            ],
            "downloads": [
                {
                    "title": "Steady-State Results",
                    "data": out_csv.to_csv(),
                    "filename": "ogusa_ss_output.csv",
                }
            ],
        }
