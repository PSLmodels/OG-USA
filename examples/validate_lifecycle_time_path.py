"""
Validate calibrated lifecycle preference parameters along a time path.

Large ``chi_n`` at old ages (thousands at 75 to 79) is the part of the
calibrated parameter set most likely to trouble OG-Core's time-path
solver, so this script solves a baseline and one reform transition path
with the calibrated ``beta_annual``, ``chi_b``, and ``chi_n`` and records
whether each converged, the largest Euler errors, and the usual macro
comparison table.

The reform is a corporate income tax rate of 35 percent (the same reform
``run_ogusa.py`` uses) unless ``--reform-json`` supplies another
``update_specifications`` dictionary.

Example::

    uv run python examples/validate_lifecycle_time_path.py \
        --params examples/lifecycle_calibration/calibrated_params.json

Outputs land in ``examples/lifecycle_calibration/time_path_validation``
(``OUTPUT_BASELINE``, ``OUTPUT_REFORM``, ``macro_pct_diff.csv``, and
``validation_summary.json``).
"""

import argparse
import copy
import json
import logging
import multiprocessing
import os
import pickle
import time
import traceback
from importlib import resources

import cloudpickle
import numpy as np
from distributed import Client

import ogcore
from ogcore import SS, TPI
from ogcore import output_tables as ot
from ogcore.parameters import Specifications
from ogcore.utils import mkdirs, safe_read_pickle

CUR_DIR = os.path.dirname(os.path.realpath(__file__))
DEFAULT_OUTPUT_DIR = os.path.join(
    CUR_DIR, "lifecycle_calibration", "time_path_validation"
)
DEFAULT_REFORM = {"cit_rate": [[0.35]]}


def load_default_spec(**kwargs):
    """
    Loads the OG-USA default parameters into a Specifications object.

    Args:
        kwargs (dict): keyword arguments for ``Specifications`` (for
            example ``baseline``, ``output_base``, ``baseline_dir``)

    Returns:
        p (OG-Core Specifications object): parameters object
    """
    p = Specifications(**kwargs)
    with (
        resources.files("ogusa")
        .joinpath("ogusa_default_parameters.json")
        .open() as file
    ):
        p.update_specifications(json.load(file))
    return p


def tpi_summary(output_dir):
    """
    Reads convergence diagnostics from a saved ``TPI_vars.pkl``.

    Args:
        output_dir (str): model output directory containing ``TPI/``

    Returns:
        summary (dict): ``tpi_saved`` plus, when the pickle exists, the
            largest absolute Euler and resource-constraint errors and the
            first and mid-path values of ``r``, ``w``, ``Y``, ``K``, ``L``
    """
    path = os.path.join(output_dir, "TPI", "TPI_vars.pkl")
    if not os.path.exists(path):
        return {"tpi_saved": False}
    tpi = safe_read_pickle(path)
    summary = {"tpi_saved": True}
    for key in ("euler_savings", "euler_labor_leisure", "RC_error"):
        if key in tpi:
            summary[f"max_abs_{key}"] = float(np.abs(tpi[key]).max())
    for key in ("r", "w", "Y", "K", "L"):
        if key in tpi:
            series = np.asarray(tpi[key], dtype=float)
            summary[f"{key}_first"] = float(series[0])
            summary[f"{key}_last_T"] = float(series[len(series) // 2])
    return summary


def run_one(p, client, label, ss_pickle=None):
    """
    Solves a steady state serially, then the time path with Dask.

    Mirrors ``ogcore.execute.runner`` but solves the steady state without
    the Dask client, which on one machine is several times faster than
    the scattered solve, and accepts a saved steady state for the
    baseline.  Solver failures are recorded, not raised.  The record is
    also written to ``run_record.json`` in ``p.output_base``.

    Args:
        p (OG-Core Specifications object): parameters object whose
            ``output_base`` receives ``SS/``, ``TPI/``, and
            ``model_params.pkl``
        client (Dask Client object): client for the time path
        label (str): run label (``"baseline"`` or ``"reform"``)
        ss_pickle (str or None): saved steady state to load instead of
            solving

    Returns:
        record (dict): label, convergence flag, error message, timings,
            steady-state prices, and the ``tpi_summary`` diagnostics
    """
    ss_dir = os.path.join(p.output_base, "SS")
    tpi_dir = os.path.join(p.output_base, "TPI")
    mkdirs(ss_dir)
    mkdirs(tpi_dir)
    record = {"label": label, "converged": False, "error": None}
    start = time.time()
    try:
        if ss_pickle is not None and os.path.exists(ss_pickle):
            print(f"{label}: loading steady state from {ss_pickle}")
            ss_outputs = safe_read_pickle(ss_pickle)
        else:
            print(f"{label}: solving steady state serially", flush=True)
            ss_outputs = SS.run_SS(p, client=None)
        record["ss_seconds"] = time.time() - start
        with open(os.path.join(ss_dir, "SS_vars.pkl"), "wb") as f:
            pickle.dump(ss_outputs, f)
        with open(os.path.join(p.output_base, "model_params.pkl"), "wb") as f:
            cloudpickle.dump(p, f)
        for key in ("r_p", "w", "factor"):
            record[f"ss_{key}"] = float(np.squeeze(ss_outputs[key]))
        print(f"{label}: steady state done, starting TPI", flush=True)
        tpi_start = time.time()
        tpi_output = TPI.run_TPI(p, client=client)
        record["tpi_seconds"] = time.time() - tpi_start
        with open(os.path.join(tpi_dir, "TPI_vars.pkl"), "wb") as f:
            pickle.dump(tpi_output, f)
        record["converged"] = True
    except Exception as err:  # noqa: BLE001 - report, do not stop
        record["error"] = f"{type(err).__name__}: {err}"
        traceback.print_exc()
    record["seconds"] = time.time() - start
    record.update(tpi_summary(p.output_base))
    with open(
        os.path.join(p.output_base, "run_record.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(record, f, indent=1)
    print(f"{label}: {json.dumps(record, indent=1)}", flush=True)
    return record


def main():
    """
    Runs the baseline and reform time paths (or summarizes saved runs)
    and writes ``validation_summary.json`` and ``macro_pct_diff.csv``.

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--params",
        required=True,
        help="JSON with calibrated beta_annual, chi_b, and chi_n.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--baseline-ss",
        default=None,
        help="Pickle of the baseline steady state at the calibrated "
        "parameters (skips the cold solve when it exists).",
    )
    parser.add_argument(
        "--reform-json",
        default=None,
        help="JSON dictionary of reform parameter updates "
        "(default: cit_rate 0.35).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=min(multiprocessing.cpu_count(), 7),
    )
    parser.add_argument(
        "--skip-reform", action="store_true", help="Baseline only."
    )
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Skip solving; rebuild the summary and macro table from the "
        "run records and pickles already in the output directory.",
    )
    args = parser.parse_args()

    logging.getLogger("ogcore").setLevel(logging.INFO)
    ogcore.config.set_logging_level(verbose=True)
    base_dir = os.path.join(args.output_dir, "OUTPUT_BASELINE")
    reform_dir = os.path.join(args.output_dir, "OUTPUT_REFORM")
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(reform_dir, exist_ok=True)

    with open(args.params, "r", encoding="utf-8") as file:
        calibrated = json.load(file)
    pref_params = {k: calibrated[k] for k in ("beta_annual", "chi_b", "chi_n")}
    if args.reform_json is not None:
        with open(args.reform_json, "r", encoding="utf-8") as file:
            reform = json.load(file)
    else:
        reform = DEFAULT_REFORM

    client = None
    if args.summarize_only:
        records = []
        for directory in (base_dir, reform_dir):
            record_path = os.path.join(directory, "run_record.json")
            if os.path.exists(record_path):
                with open(record_path, "r", encoding="utf-8") as file:
                    records.append(json.load(file))
    else:
        client = Client(n_workers=args.num_workers, threads_per_worker=1)
        print("Number of workers = ", args.num_workers, flush=True)

        p = load_default_spec(
            baseline=True,
            num_workers=args.num_workers,
            baseline_dir=base_dir,
            output_base=base_dir,
        )
        p.update_specifications(pref_params)
        records = [run_one(p, client, "baseline", ss_pickle=args.baseline_ss)]

        if not args.skip_reform and records[0]["converged"]:
            p2 = copy.deepcopy(p)
            p2.baseline = False
            p2.output_base = reform_dir
            p2.update_specifications(reform)
            records.append(run_one(p2, client, "reform"))

    summary = {
        "params": os.path.abspath(args.params),
        "reform": reform,
        "runs": records,
    }
    if len(records) == 2 and all(r["converged"] for r in records):
        base_tpi = safe_read_pickle(
            os.path.join(base_dir, "TPI", "TPI_vars.pkl")
        )
        base_params = safe_read_pickle(
            os.path.join(base_dir, "model_params.pkl")
        )
        reform_tpi = safe_read_pickle(
            os.path.join(reform_dir, "TPI", "TPI_vars.pkl")
        )
        reform_params = safe_read_pickle(
            os.path.join(reform_dir, "model_params.pkl")
        )
        table = ot.macro_table(
            base_tpi,
            base_params,
            reform_tpi=reform_tpi,
            reform_params=reform_params,
            var_list=["Y", "C", "K", "L", "r", "w"],
            output_type="pct_diff",
            num_years=10,
            start_year=base_params.start_year,
        )
        table_path = os.path.join(args.output_dir, "macro_pct_diff.csv")
        table.to_csv(table_path)
        print(table)
        summary["macro_table"] = table_path
    with open(
        os.path.join(args.output_dir, "validation_summary.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=2)
    print("Wrote", os.path.join(args.output_dir, "validation_summary.json"))
    if client is not None:
        try:
            client.close(timeout=10)
        except Exception as err:  # noqa: BLE001 - teardown only
            print(f"Dask client close failed: {err}")


if __name__ == "__main__":
    main()
