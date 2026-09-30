"""
Compare lifecycle fit between two parameterizations of OG-USA.

Produces figures and tables that put the current OG-USA default parameters
and a newly calibrated parameter set side by side against the data:

* ``labor_supply_profiles.png``: steady-state labor supply by age for both
  parameter sets with the CPS profile (survey-weighted mean weekly hours by
  age as a share of the 112-hour time endowment), drawn with
  ``ogcore.output_plots.ss_profiles``.
* ``chi_n_comparison.png`` and ``chi_n_comparison_log.png``: chi_n by age
  from ``ogcore.parameter_plots.plot_chi_n``, in levels and on a log scale.
* ``wealth_profiles.png``: mean net worth by age in 2019 dollars for both
  parameter sets (model savings converted with each steady state's income
  scaling factor and aligned so that wealth at age ``a`` is the saving
  chosen at ``a - 1``) against the SCF survey-weighted mean by age.
* ``wealth_moments_ogcore.csv``: ``ogcore.output_tables.wealth_moments_table``
  for both parameter sets merged with the SCF column (seven percentile
  bins, Gini, variance of log wealth).
* ``wealth_fit_extended.csv``: the calibration moments from
  ``ogusa.estimate_lifecycle_params`` (wealth shares by lifetime-income
  type bin, mean wealth over mean income, bequest flow over living wealth,
  the aggregate 75-79 over 60-64 ratio, and old-age tilt by wealth bin)
  for the data and both parameter sets, with log gaps.
* ``labor_supply_and_chi_n.csv``: hours and chi_n by age.

Steady states are solved serially unless saved pickles are supplied, and
any newly solved steady state is saved next to the figures for reuse.

Example::

    uv run python examples/plot_lifecycle_calibration.py \
        --params examples/lifecycle_calibration/calibrated_params_total_income.json \
        --current-ss examples/lifecycle_calibration/ss_current_params.pkl \
        --new-ss examples/lifecycle_calibration/ss_calibrated_total_income.pkl
"""

import argparse
import json
import logging
import os
import pickle
import time
import warnings
from importlib import resources

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import ogcore  # noqa: E402
from ogcore import SS  # noqa: E402
from ogcore import output_plots as op  # noqa: E402
from ogcore import output_tables as ot  # noqa: E402
from ogcore import parameter_plots as pp  # noqa: E402
from ogcore.parameters import Specifications  # noqa: E402

from ogusa import compute_moments as cm  # noqa: E402
from ogusa import estimate_lifecycle_params as elp  # noqa: E402
from ogusa import wealth  # noqa: E402

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_DIR = os.path.join(CUR_DIR, "lifecycle_calibration")
OGCORE_WEALTH_BINS = np.array([0.25, 0.25, 0.2, 0.1, 0.1, 0.09, 0.01])


def load_default_spec():
    """OG-USA default parameters as a baseline Specifications object."""
    p = Specifications(baseline=True, num_workers=1)
    with (
        resources.files("ogusa")
        .joinpath("ogusa_default_parameters.json")
        .open() as file
    ):
        p.update_specifications(json.load(file))
    return p


def load_or_solve_ss(p, pickle_path, save_path, label):
    """Load a steady state from ``pickle_path`` or solve and save it."""
    if pickle_path is not None and os.path.exists(pickle_path):
        with open(pickle_path, "rb") as file:
            return pickle.load(file)
    print(f"Solving steady state for {label} parameters (serial)...")
    start = time.time()
    ss_output = SS.run_SS(p, client=None)
    print(f"  done in {time.time() - start:.0f}s")
    with open(save_path, "wb") as file:
        pickle.dump(ss_output, file)
    return ss_output


# ---------------------------------------------------------------------------
# Data profiles
# ---------------------------------------------------------------------------


def cps_hours_profile(min_age, max_age):
    """Survey-weighted mean weekly hours by age as a share of the endowment.

    ``compute_moments.get_age_profile_moments`` returns the profile indexed
    by model age 20 through 99 with NaN outside ``[min_age, max_age]``,
    already scaled by the 112-hour weekly time endowment.
    """
    profile = cm.get_age_profile_moments(
        "hours", min_age=min_age, max_age=max_age, hours_source="cps"
    )
    return profile.to_numpy(dtype=float)


def scf_wealth_profile(min_age, max_age, scf_years):
    """SCF survey-weighted mean net worth by age in 2019 dollars."""
    profile = cm.get_age_profile_moments(
        "wealth",
        min_age=min_age,
        max_age=max_age,
        wealth_source="scf",
        scf_yrs_list=list(scf_years),
    )
    return profile.to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# Model profiles
# ---------------------------------------------------------------------------


def aggregate_labor(ss_output, p):
    """Population-weighted labor supply by age, matching the moments code."""
    weights = elp._type_weights_by_age(p)
    return (np.asarray(ss_output["n"]) * weights).sum(axis=1)


def model_wealth_profile_dollars(ss_output, p):
    """Mean wealth held at each age in dollars, aligned with the SCF.

    ``b_sp1[s]`` is saving chosen at age index ``s`` and held at ``s + 1``,
    so wealth at age ``a`` is ``b_sp1[a - starting_age - 1]``; the starting
    age holds nothing.  Model units convert to dollars with the steady
    state's income scaling factor.
    """
    b_sp1 = np.asarray(ss_output["b_sp1"], dtype=float)
    weights = elp._type_weights_by_age(p)
    profile = np.full(p.S, np.nan)
    profile[1:] = (b_sp1[:-1, :] * weights[1:, :]).sum(axis=1)
    profile[0] = 0.0
    return profile * float(ss_output["factor"])


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def plot_labor_profiles(
    ss_current, p_current, ss_new, p_new, data_profile, labels, output_dir
):
    """Labor supply by age for both parameterizations plus the CPS data.

    ``ss_profiles`` labels the two model series "Baseline" and "Reform" and
    rescales the data series so its first point equals the baseline model's
    first point.  Both are undone here: the lines are relabeled, and the
    data line is reset to the survey levels so the level gap is visible.
    """
    fig = op.ss_profiles(
        ss_current,
        p_current,
        reform_ss=ss_new,
        reform_params=p_new,
        by_j=False,
        var="n",
        plot_data=data_profile,
        plot_title="Labor supply by age",
    )
    ax = fig.gca()
    relabel = {"Baseline": labels[0], "Reform": labels[1], "Data": "CPS data"}
    for line in ax.get_lines():
        name = line.get_label()
        if name == "Data":
            line.set_ydata(data_profile)
        if name in relabel:
            line.set_label(relabel[name])
    ax.relim()
    ax.autoscale_view()
    ax.set_ylabel("Labor supply, share of time endowment")
    ax.legend(loc="upper right")
    fig.tight_layout()
    path = os.path.join(output_dir, "labor_supply_profiles.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def plot_chi_n_profiles(p_current, p_new, labels, output_dir):
    """chi_n by age for both parameterizations, in levels and log scale."""
    year = int(p_current.start_year)
    paths = []
    for suffix, log_scale in (("", False), ("_log", True)):
        fig = pp.plot_chi_n(
            [p_current, p_new], labels=list(labels), years_to_plot=[year]
        )
        ax = fig.gca()
        if log_scale:
            ax.set_yscale("log")
        ax.legend(loc="upper left")
        fig.tight_layout()
        path = os.path.join(output_dir, f"chi_n_comparison{suffix}.png")
        fig.savefig(path, dpi=300)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_wealth_profiles(
    ss_current, p_current, ss_new, p_new, data_profile, labels, output_dir
):
    """Mean net worth by age in thousands of 2019 dollars, data and models."""
    ages = np.arange(
        p_current.starting_age, p_current.starting_age + p_current.S
    )
    current = model_wealth_profile_dollars(ss_current, p_current)
    new = model_wealth_profile_dollars(ss_new, p_new)
    fig, ax = plt.subplots()
    ax.plot(ages, current / 1e3, label=labels[0])
    ax.plot(ages, new / 1e3, linestyle="--", label=labels[1])
    ax.plot(
        ages,
        data_profile / 1e3,
        linestyle=":",
        linewidth=2.0,
        label="SCF data",
    )
    ax.set_xlabel("Age")
    ax.set_ylabel("Mean net worth, thousands of 2019 dollars")
    ax.set_title("Wealth by age")
    ax.legend(loc="upper left")
    fig.tight_layout()
    path = os.path.join(output_dir, "wealth_profiles.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path, current, new


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def ogcore_wealth_moments(ss_current, p_current, ss_new, p_new, labels, scf):
    """OG-Core's wealth moments table for both parameter sets plus SCF.

    ``wealth_moments_table`` handles one model at a time and uses seven
    fixed percentile bins with nearest-cell cutoffs; the two calls are
    merged on the moment name.
    """
    data = wealth.compute_wealth_moments(scf.copy(), OGCORE_WEALTH_BINS)
    current = ot.wealth_moments_table(
        ss_current, p_current, data_moments=list(data)
    )
    new = ot.wealth_moments_table(ss_new, p_new, data_moments=list(data))
    table = current.rename(columns={"Model": labels[0], "Data": "SCF data"})
    table[labels[1]] = new["Model"].to_numpy()
    return table[["Moment", "SCF data", labels[0], labels[1]]]


def extended_wealth_fit(ss_current, p_current, ss_new, p_new, labels, config):
    """Calibration moments for data and both models, with log gaps."""
    moment_config = elp.LifecycleCalibrationConfig(
        include_labor_profile=False,
        include_wealth_distribution=True,
        include_wealth_income_ratio=True,
        include_bequest_flow_ratio=True,
        include_old_age_wealth_ratio=True,
        include_old_age_ratio_by_type=True,
        scf_income_concept=config.scf_income_concept,
        scf_yrs_list=config.scf_yrs_list,
    )
    data = elp.compute_data_moments(p_current, moment_config)
    current = elp.compute_model_moments(ss_current, p_current, moment_config)
    new = elp.compute_model_moments(ss_new, p_new, moment_config)
    table = pd.DataFrame(
        {
            "moment": data.names,
            "SCF data": data.values,
            labels[0]: current.values,
            labels[1]: new.values,
        }
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        positive = table["SCF data"] > 0
        for label in labels:
            table[f"log_gap_{label}"] = np.where(
                positive, np.log(table[label] / table["SCF data"]), np.nan
            )
    return table


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--params",
        required=True,
        help="JSON file with beta_annual, chi_b, and chi_n for the new "
        "parameterization (update_specifications format).",
    )
    parser.add_argument(
        "--current-ss",
        default=None,
        help="Pickle of the steady state at the current default parameters "
        "(solved and saved if missing).",
    )
    parser.add_argument(
        "--new-ss",
        default=None,
        help="Pickle of the steady state at the new parameters (solved and "
        "saved if missing).",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for figures, tables, and newly solved steady states.",
    )
    parser.add_argument(
        "--labels",
        nargs=2,
        default=("Current parameters", "Calibrated parameters"),
        metavar=("CURRENT", "NEW"),
        help="Legend and column labels for the two parameterizations.",
    )
    parser.add_argument("--min-age", type=int, default=20)
    parser.add_argument("--max-age", type=int, default=79)
    parser.add_argument(
        "--scf-years",
        nargs="+",
        type=int,
        default=[2019],
        help="SCF survey years pooled for the wealth data.",
    )
    parser.add_argument(
        "--income-concept",
        choices=["pre_transfer", "total"],
        default="total",
        help="SCF income concept for the wealth-to-income ratio.",
    )
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    logging.getLogger("ogcore").setLevel(logging.WARNING)
    ogcore.config.VERBOSE = False
    os.makedirs(args.output_dir, exist_ok=True)
    pd.set_option("display.width", 160)

    p_current = load_default_spec()
    p_new = load_default_spec()
    with open(args.params, "r", encoding="utf-8") as file:
        new_params = json.load(file)
    p_new.update_specifications(
        {k: new_params[k] for k in ("beta_annual", "chi_b", "chi_n")}
    )

    ss_current = load_or_solve_ss(
        p_current,
        args.current_ss,
        os.path.join(args.output_dir, "ss_current_params.pkl"),
        "current",
    )
    ss_new = load_or_solve_ss(
        p_new,
        args.new_ss,
        os.path.join(args.output_dir, "ss_new_params.pkl"),
        "new",
    )
    labels = tuple(args.labels)
    config = elp.LifecycleCalibrationConfig(
        scf_income_concept=args.income_concept,
        scf_yrs_list=tuple(args.scf_years),
    )

    # Labor supply and chi_n
    hours_data = cps_hours_profile(args.min_age, args.max_age)
    labor_path = plot_labor_profiles(
        ss_current,
        p_current,
        ss_new,
        p_new,
        hours_data,
        labels,
        args.output_dir,
    )
    chi_paths = plot_chi_n_profiles(p_current, p_new, labels, args.output_dir)

    # Wealth by age
    wealth_data = scf_wealth_profile(
        max(args.min_age, p_current.starting_age + 1),
        args.max_age,
        args.scf_years,
    )
    wealth_path, wealth_current, wealth_new = plot_wealth_profiles(
        ss_current,
        p_current,
        ss_new,
        p_new,
        wealth_data,
        labels,
        args.output_dir,
    )

    # Tables
    scf = wealth.get_wealth_data(
        scf_yrs_list=list(args.scf_years), include_age=True
    )
    ogcore_table = ogcore_wealth_moments(
        ss_current, p_current, ss_new, p_new, labels, scf
    )
    extended = extended_wealth_fit(
        ss_current, p_current, ss_new, p_new, labels, config
    )
    ages = np.arange(
        p_current.starting_age, p_current.starting_age + p_current.S
    )
    profiles = pd.DataFrame(
        {
            "age": ages,
            "cps_hours": hours_data,
            f"hours_{labels[0]}": aggregate_labor(ss_current, p_current),
            f"hours_{labels[1]}": aggregate_labor(ss_new, p_new),
            f"chi_n_{labels[0]}": p_current.chi_n[-1, :],
            f"chi_n_{labels[1]}": p_new.chi_n[-1, :],
            "scf_wealth": wealth_data,
            f"wealth_{labels[0]}": wealth_current,
            f"wealth_{labels[1]}": wealth_new,
        }
    )

    ogcore_path = os.path.join(args.output_dir, "wealth_moments_ogcore.csv")
    extended_path = os.path.join(args.output_dir, "wealth_fit_extended.csv")
    profiles_path = os.path.join(args.output_dir, "labor_supply_and_chi_n.csv")
    ogcore_table.to_csv(ogcore_path, index=False)
    extended.to_csv(extended_path, index=False)
    profiles.to_csv(profiles_path, index=False)

    print("\nOG-Core wealth moments table (SCF vs. both parameter sets):")
    print(ogcore_table.round(4).to_string(index=False))
    print("\nExtended wealth fit (calibration moments):")
    print(extended.round(4).to_string(index=False))
    print("\nProfiles by age (every fifth age):")
    print(profiles.iloc[::5].round(3).to_string(index=False))
    print("\nWrote:")
    for path in [
        labor_path,
        *chi_paths,
        wealth_path,
        ogcore_path,
        extended_path,
        profiles_path,
    ]:
        print("  ", path)


if __name__ == "__main__":
    main()
