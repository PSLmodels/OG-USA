"""
Calibrate OG-USA's lifecycle preference parameters and report the fit.

Runs the nested general-equilibrium calibration of ``beta_annual`` by
lifetime-income type, ``chi_b``, and the ``chi_n`` age profile through the
``Calibration`` class (``estimate_lifecycle_prefs=True``), then writes

* ``calibrated_params.json``: the parameters in ``update_specifications``
  format (read back on later runs instead of re-calibrating);
* ``ss_calibrated.pkl`` and ``ss_current_params.pkl``: steady states at
  the calibrated and the current default parameters;
* ``moment_comparison.csv``: every calibration moment with its data and
  model value and the log gap at the final general equilibrium;
* ``outer_loop_history.csv``: parameter and price changes, damping,
  inner-loop effort, and prices for each pass of the outer loop;
* ``preference_standard_errors.csv`` (with ``--standard-errors``): beta
  and chi_b by type with standard errors from the household-only
  Jacobian, plus the overidentification test when a bootstrap covariance
  of the data moments is computed;
* the figures and tables of ``plot_lifecycle_calibration.py`` (hours by
  age against CPS hours as a share of the time endowment, mean net worth
  by age in 2019 dollars against the SCF, chi_n by age, wealth moment
  tables) comparing the current defaults with the calibrated parameters.

The calibration takes 20 to 40 minutes serially (eight to twelve outer
passes).  Serial solves are the default because the steady state is
dominated by parameter-scattering overhead under Dask.  Time-path
validation of the calibrated parameters lives in
``validate_lifecycle_time_path.py``.

Example::

    uv run python examples/run_lifecycle_calibration.py
    uv run python examples/run_lifecycle_calibration.py \
        --params examples/lifecycle_calibration/calibrated_params.json \
        --standard-errors
"""

import argparse
import copy
import importlib.util
import logging
import os
import pickle
import sys
import time
import warnings
from dataclasses import replace

import numpy as np
import pandas as pd

import ogcore
from ogusa import calibrate_lifecycle as cl
from ogusa import estimate_lifecycle_params as elp
from ogusa.calibrate import Calibration

CUR_DIR = os.path.dirname(os.path.realpath(__file__))
DEFAULT_OUTPUT_DIR = os.path.join(CUR_DIR, "lifecycle_calibration")


def _load_plot_module():
    """
    Imports the sibling plotting script as a module.

    Returns:
        module (module): ``plot_lifecycle_calibration`` with its data,
            model, figure, and table helpers
    """
    spec = importlib.util.spec_from_file_location(
        "plot_lifecycle_calibration",
        os.path.join(CUR_DIR, "plot_lifecycle_calibration.py"),
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def history_frame(outcome):
    """
    Tabulates the outer-loop diagnostics, one row per pass.

    Args:
        outcome (LifecycleCalibrationOutcome): calibration outcome

    Returns:
        frame (Pandas DataFrame): parameter and price changes, damping,
            inner-loop effort, GE solve time, and prices by pass
    """
    rows = []
    for rec in outcome.history:
        row = {
            "iteration": rec.iteration,
            "param_change": rec.param_change,
            "price_change": rec.price_change,
            "damping": rec.damping,
            "chi_n_iterations": rec.chi_n_iterations,
            "pref_nfev": rec.pref_nfev,
            "pref_cost": rec.pref_cost,
            "ge_seconds": rec.ge_seconds,
        }
        row.update(rec.prices)
        rows.append(row)
    return pd.DataFrame(rows)


def moment_frame(data_moments, model_moments):
    """
    Tabulates data, model, and log gap for each moment.

    Args:
        data_moments (MomentSet): data moments
        model_moments (MomentSet): model moments in the same order

    Returns:
        frame (Pandas DataFrame): columns ``moment``, ``data``,
            ``model``, ``log_gap`` (NaN where the data value is not
            positive)
    """
    frame = data_moments.to_frame(model_moments)
    with np.errstate(divide="ignore", invalid="ignore"):
        frame["log_gap"] = np.where(
            frame["data"] > 0, np.log(frame["model"] / frame["data"]), np.nan
        )
    return frame


def standard_errors(p, ss, data_moments, config, options, args, output_dir):
    """
    Computes standard errors for beta and chi_b at the calibrated point.

    Runs one least-squares iteration of ``calibrate_beta_chi_b`` on a copy
    of ``p`` to obtain the Jacobian of the weighted log residuals over
    household-only solves, optionally bootstraps the data moments for a
    sandwich covariance and the overidentification test, and writes the
    table.

    Args:
        p (OG-Core Specifications object): calibrated parameters
        ss (dict): steady state at the calibrated parameters
        data_moments (MomentSet): data moments
        config (LifecycleCalibrationConfig): moment configuration
        options (PreferenceCalibrationOptions): beta / chi_b options
        args (argparse Namespace): command-line arguments (uses
            ``bootstrap_iterations``)
        output_dir (str): directory for the table

    Returns:
        path (str): path of ``preference_standard_errors.csv``
    """
    print("Computing standard errors at the calibrated point...")
    # One least-squares iteration on a copy of p: this evaluates the
    # residuals and the Jacobian at the calibrated point without moving it.
    pref = cl.calibrate_beta_chi_b(
        ss,
        copy.deepcopy(p),
        data_moments,
        config=config,
        options=replace(options, max_nfev=1),
    )
    moment_vcv = None
    if args.bootstrap_iterations > 0:
        boot_config = elp.LifecycleCalibrationConfig(
            **{
                **config.__dict__,
                "bootstrap_iterations": args.bootstrap_iterations,
            }
        )
        draws = elp.bootstrap_data_moments(p, boot_config, seed=0)
        full_vcv = np.cov(draws, rowvar=False)
        A = cl.preference_target_selection(data_moments, p, config, options)
        moment_vcv = A @ full_vcv @ A.T
    inference = cl.preference_inference(
        pref, p, options=options, moment_vcv=moment_vcv
    )
    table = inference.to_frame(p)
    path = os.path.join(output_dir, "preference_standard_errors.csv")
    table.to_csv(path, index=False)
    print(table.round(4).to_string(index=False))
    print(
        f"Method: {inference.method}; {inference.n_params} free parameters "
        f"({int(inference.at_bound.sum())} at a bound), "
        f"{inference.n_moments} targets; J = {inference.j_stat:.3f} on "
        f"{inference.j_df} df (p = {inference.j_pvalue:.3f})"
    )
    return path


def main():
    """
    Runs the calibration and writes tables and figures (see the module
    docstring for the outputs).

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--params",
        default=None,
        help="JSON with calibrated parameters to reuse; the calibration "
        "runs (and writes this file) when it is missing.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--current-ss",
        default=None,
        help="Pickle of the steady state at the current defaults.",
    )
    parser.add_argument(
        "--calibrated-ss",
        default=None,
        help="Pickle of the steady state at the calibrated parameters "
        "(only used with --params).",
    )
    parser.add_argument(
        "--income-concept",
        choices=["pre_transfer", "total"],
        default="pre_transfer",
        help="SCF income concept for the wealth-to-income target.",
    )
    parser.add_argument(
        "--bequest-flow-weight",
        type=float,
        default=1.0,
        help="Weight on the bequest-flow target (0 drops it).",
    )
    parser.add_argument(
        "--chi-b-mode", choices=["by_type", "common_scale"], default="by_type"
    )
    parser.add_argument("--max-outer", type=int, default=15)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Dask workers for household solves; 0 solves serially.",
    )
    parser.add_argument(
        "--standard-errors",
        action="store_true",
        help="Compute standard errors for beta and chi_b.",
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=200,
        help="Bootstrap draws for the moment covariance behind the "
        "sandwich standard errors and the overidentification test; "
        "0 uses the homoskedastic least-squares variance instead.",
    )
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("ogcore").setLevel(logging.WARNING)
    ogcore.config.set_logging_level(verbose=False)
    os.makedirs(args.output_dir, exist_ok=True)
    plots = _load_plot_module()

    client = None
    if args.num_workers > 0:
        from distributed import Client

        client = Client(n_workers=args.num_workers, threads_per_worker=1)

    p = plots.load_default_spec()
    p.output_base = args.output_dir
    config = elp.LifecycleCalibrationConfig(
        scf_income_concept=args.income_concept
    )
    options = cl.PreferenceCalibrationOptions(
        chi_b_mode=args.chi_b_mode,
        bequest_flow_weight=args.bequest_flow_weight,
    )
    params_path = args.params or os.path.join(
        args.output_dir, "calibrated_params.json"
    )

    start = time.time()
    calib = Calibration(
        p,
        estimate_lifecycle_prefs=True,
        lifecycle_params_path=params_path,
        lifecycle_config=config,
        lifecycle_options=options,
        lifecycle_kwargs={"max_outer": args.max_outer},
        client=client,
    )
    print(f"Calibration step took {time.time() - start:.0f} s")
    p_new = plots.load_default_spec()
    p_new.update_specifications(
        {k: calib.get_dict()[k] for k in ("beta_annual", "chi_b", "chi_n")}
    )

    outcome = calib.lifecycle_outcome
    ss_new_path = os.path.join(args.output_dir, "ss_calibrated.pkl")
    if outcome is not None:
        ss_new = outcome.ss_output
        with open(ss_new_path, "wb") as file:
            pickle.dump(ss_new, file)
        history_frame(outcome).to_csv(
            os.path.join(args.output_dir, "outer_loop_history.csv"),
            index=False,
        )
        print(
            f"Outer loop: {outcome.iterations} passes, "
            f"converged={outcome.converged}"
        )
        print(history_frame(outcome).round(4).to_string(index=False))
        data_moments = outcome.data_moments
        model_moments = outcome.model_moments
    else:
        ss_new = plots.load_or_solve_ss(
            p_new, args.calibrated_ss, ss_new_path, "calibrated"
        )
        data_moments = elp.compute_data_moments(p_new, config)
        model_moments = elp.compute_model_moments(ss_new, p_new, config)

    moments = moment_frame(data_moments, model_moments)
    moments_path = os.path.join(args.output_dir, "moment_comparison.csv")
    moments.to_csv(moments_path, index=False)
    print("\nMoment comparison at the final general equilibrium:")
    print(
        moments[~moments["moment"].str.startswith("labor_supply")]
        .round(4)
        .to_string(index=False)
    )
    labor = moments[moments["moment"].str.startswith("labor_supply")]
    print(
        f"Hours: max abs log gap {labor['log_gap'].abs().max():.4f} over "
        f"{len(labor)} ages"
    )

    if args.standard_errors:
        standard_errors(
            p_new, ss_new, data_moments, config, options, args, args.output_dir
        )

    # Comparison with the current default parameters
    p_current = plots.load_default_spec()
    ss_current = plots.load_or_solve_ss(
        p_current,
        args.current_ss,
        os.path.join(args.output_dir, "ss_current_params.pkl"),
        "current",
    )
    labels = ("Current parameters", "Calibrated parameters")
    min_age, max_age = config.min_age, config.max_age
    hours_data = plots.cps_hours_profile(min_age, max_age)
    plots.plot_labor_profiles(
        ss_current,
        p_current,
        ss_new,
        p_new,
        hours_data,
        labels,
        args.output_dir,
    )
    plots.plot_chi_n_profiles(p_current, p_new, labels, args.output_dir)
    wealth_data = plots.scf_wealth_profile(
        max(min_age, p_current.starting_age + 1),
        max_age,
        config.scf_yrs_list,
    )
    plots.plot_wealth_profiles(
        ss_current,
        p_current,
        ss_new,
        p_new,
        wealth_data,
        labels,
        args.output_dir,
    )
    extended = plots.extended_wealth_fit(
        ss_current, p_current, ss_new, p_new, labels, config
    )
    extended.to_csv(
        os.path.join(args.output_dir, "wealth_fit_extended.csv"), index=False
    )
    print("\nWealth fit, current versus calibrated parameters:")
    print(extended.round(4).to_string(index=False))

    if client is not None:
        client.close()
    print("\nOutputs written to", args.output_dir)


if __name__ == "__main__":
    main()
