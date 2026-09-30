"""
Moments and helpers for calibrating beta, chi_b, and chi_n in OG-USA.

This module holds the data and model moment construction used to calibrate
the household preference parameters that govern savings (``beta_annual``),
bequests (``chi_b``), and labor supply (``chi_n``).  The default moment set
is:

* mean hours by single year of age from the CPS ASEC (targets for ``chi_n``),
* wealth shares held by each lifetime-income type, with the SCF percentile
  bins taken from ``p.lambdas`` (targets for ``beta_annual`` by type),
* the ratio of mean net worth at ages 75-79 to ages 60-64 from the SCF
  (target for ``chi_b``).

Aggregate bequests over GDP, the income Gini, the gross saving rate, a
normalized wealth-by-age profile, and SCF inheritance moments are available
as optional or diagnostic moments.

The module also retains a joint DFO-LS SMM driver
(:func:`estimate_lifecycle_params`) that solves the full general-equilibrium
steady state on every evaluation.  It is intended for inference (standard
errors and overidentification tests) starting from an already calibrated
point, not as the primary calibration path.  See
``LIFECYCLE_CALIBRATION_PLAN.md`` in the repository root.
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np
import pandas as pd
import ogcore
from ogcore import SS
from ogcore.utils import Inequality

from ogusa import compute_moments, wealth

try:  # optional dependency used only by the SMM inference driver
    import dfols
except ImportError:  # pragma: no cover - exercised only without dfo-ls
    dfols = None

ogcore.config.VERBOSE = False
logger = logging.getLogger(__name__)

WeightingMethod = Literal["identity", "diagonal", "optimal"]
TailMethod = Literal["scaled_default", "flat"]
WealthProfileMoment = Literal["anchor_window", "level", "mean_normalized"]
MomentDistanceMethod = Literal["absolute", "relative"]
SAVINGS_RATE_DATA_LABEL = r"Gross savings rate $(S/Y)$"
HOURS_IN_TIME_ENDOWMENT = (24 - 8) * 7  # weekly hours net of sleep


@dataclass(frozen=True)
class LifecycleCalibrationConfig:
    """
    Configuration for the lifecycle preference moments and calibration.

    Default targets are mean hours by single year of age from 20 through 79,
    one SCF wealth share per lifetime-income type (bins from ``p.lambdas``),
    and the old-age wealth ratio (mean net worth at 75-79 over 60-64).  The
    normalized wealth-by-age profile, income Gini, gross saving rate,
    wealth Gini, variance of log wealth, aggregate bequests over GDP, and
    inheritance moments are off by default and can be switched on for
    diagnostics or robustness checks.

    Wealth held at age ``a`` corresponds to model savings ``b_sp1`` chosen
    at age ``a - 1``, so wealth-based ages must be at least one year above
    the model's starting age.
    """

    # Labor moments
    min_age: int = 20
    max_age: int = 79
    include_labor_profile: bool = True
    labor_smoothing_window: int = 3
    # Wealth-by-age profile (diagnostic by default)
    include_wealth_profile: bool = False
    wealth_profile_min_age: int = 21
    wealth_profile_max_age: int = 79
    wealth_profile_moment: WealthProfileMoment = "mean_normalized"
    wealth_anchor_min_age: int = 20
    wealth_anchor_max_age: int = 24
    # Wealth distribution moments
    include_wealth_distribution: bool = True
    include_wealth_gini: bool = False
    include_wealth_var_log: bool = False
    # Wealth level: mean net worth over mean pre-transfer income (beta scale)
    include_wealth_income_ratio: bool = True
    scf_income_concept: Literal["pre_transfer", "total"] = "pre_transfer"
    # Bequest flow: wealth of decedents over wealth of the living, using the
    # model's mortality rates on both sides (chi_b target)
    include_bequest_flow_ratio: bool = True
    # Old-age wealth ratio, aggregate (diagnostic; barely responds to chi_b)
    include_old_age_wealth_ratio: bool = False
    old_age_ratio_numerator_ages: tuple[int, int] = (75, 79)
    old_age_ratio_denominator_ages: tuple[int, int] = (60, 64)
    # Old-age wealth ratio by wealth-percentile bin within age bands
    # (type-specific chi_b targets; off by default)
    include_old_age_ratio_by_type: bool = True
    tilt_numerator_ages: tuple[int, int] = (80, 89)
    tilt_denominator_ages: tuple[int, int] = (60, 64)
    tilt_bin_weights: tuple[float, ...] | None = None
    # Optional aggregate and inequality moments
    include_bequest_to_output: bool = False
    bequest_to_output_data: float | None = None
    include_income_gini: bool = False
    include_savings_rate: bool = False
    include_inheritance_moments: bool = False
    macro_year: int = 2025
    # Data sources
    cps_years: tuple[int, ...] = (2023, 2022)
    scf_yrs_list: tuple[int, ...] = (2019,)
    cps_directory: str | None = None
    scf_directory: str | None = None
    scf_web: bool = False
    bootstrap_iterations: int = 1000
    # chi_n handling for the legacy direct-estimation helpers
    estimate_chi_n_min_age: int = 20
    estimate_chi_n_max_age: int = 79
    chi_n_tail_method: TailMethod = "scaled_default"
    chi_n_n_spline_knots: int = 10
    chi_n_spline_degree: int = 3
    # SMM inference driver settings
    weighting_method: WeightingMethod = "identity"
    weighting_ridge: float = 1e-8
    n_starts: int = 3
    start_radius: float = 0.2
    dfols_rhoend: float = 1e-6
    dfols_maxfun: int | None = None
    bound_epsilon: float = 1e-4
    beta_annual_bounds: tuple[float, float] = (0.8, 0.999)
    chi_b_bounds: tuple[float, float] = (0.1, 200.0)
    chi_n_bounds: tuple[float, float] | None = None
    failure_residual: float = 10.0
    moment_distance_method: MomentDistanceMethod = "relative"
    moment_distance_floor: float = 1e-8
    use_ss_solver_restart: bool = True
    log_optimizer_progress: bool = True

    @property
    def moment_ages(self) -> np.ndarray:
        """Return age labels used in labor-profile moments."""
        return np.arange(self.min_age, self.max_age + 1)

    @property
    def wealth_profile_ages(self) -> np.ndarray:
        """Return age labels used in wealth-profile moments."""
        return np.arange(
            self.wealth_profile_min_age,
            self.wealth_profile_max_age + 1,
        )

    @property
    def wealth_anchor_ages(self) -> np.ndarray:
        """Return age labels used to normalize wealth-profile moments."""
        return np.arange(
            self.wealth_anchor_min_age, self.wealth_anchor_max_age + 1
        )

    @property
    def old_age_numerator_ages(self) -> np.ndarray:
        """Return age labels in the old-age wealth ratio numerator."""
        lo, hi = self.old_age_ratio_numerator_ages
        return np.arange(lo, hi + 1)

    @property
    def old_age_denominator_ages(self) -> np.ndarray:
        """Return age labels in the old-age wealth ratio denominator."""
        lo, hi = self.old_age_ratio_denominator_ages
        return np.arange(lo, hi + 1)

    @property
    def tilt_numerator_age_labels(self) -> np.ndarray:
        """Return age labels in the by-type tilt numerator band."""
        lo, hi = self.tilt_numerator_ages
        return np.arange(lo, hi + 1)

    @property
    def tilt_denominator_age_labels(self) -> np.ndarray:
        """Return age labels in the by-type tilt denominator band."""
        lo, hi = self.tilt_denominator_ages
        return np.arange(lo, hi + 1)

    def tilt_bins(self, p) -> np.ndarray:
        """Population bin weights for the by-type tilt moments.

        Defaults to ``p.lambdas`` with the bottom half of the population
        merged into one bin (its mean wealth is near zero, so ratios of it
        are noise) and every type inside the top one percent merged into
        another, since SCF cells at old ages are too thin to split the top
        one percent.
        """
        if self.tilt_bin_weights is not None:
            return _as_vector(self.tilt_bin_weights)
        return merged_type_bins(_lambdas(p))

    @property
    def estimated_chi_n_ages(self) -> np.ndarray:
        """Return the age labels for directly estimated chi_n values."""
        return np.arange(
            self.estimate_chi_n_min_age,
            self.estimate_chi_n_max_age + 1,
        )

    def validate(self, p) -> None:
        """Validate age and dimension settings against an OG-Core spec."""
        starting_age = int(getattr(p, "starting_age", 20))
        ending_age = int(getattr(p, "ending_age", 100))
        min_model_age = starting_age
        max_model_age = ending_age - 1
        for label, lo, hi in (
            ("labor", self.min_age, self.max_age),
            (
                "wealth profile",
                self.wealth_profile_min_age,
                self.wealth_profile_max_age,
            ),
            (
                "wealth anchor",
                self.wealth_anchor_min_age,
                self.wealth_anchor_max_age,
            ),
            (
                "chi_n",
                self.estimate_chi_n_min_age,
                self.estimate_chi_n_max_age,
            ),
            ("old-age numerator", *self.old_age_ratio_numerator_ages),
            ("old-age denominator", *self.old_age_ratio_denominator_ages),
            ("tilt numerator", *self.tilt_numerator_ages),
            ("tilt denominator", *self.tilt_denominator_ages),
        ):
            if hi < lo:
                raise ValueError(f"{label} max age must be at least min age.")
            if lo < min_model_age or hi > max_model_age:
                raise ValueError(
                    f"{label} ages [{lo}, {hi}] fall outside model ages "
                    f"[{min_model_age}, {max_model_age}]."
                )
        # Wealth held at age a is b_sp1 chosen at a - 1, so a > starting_age.
        wealth_age_sets = []
        if self.include_wealth_profile:
            wealth_age_sets.append(self.wealth_profile_ages)
            if self.wealth_profile_moment == "anchor_window":
                wealth_age_sets.append(self.wealth_anchor_ages)
        if self.include_old_age_wealth_ratio:
            wealth_age_sets.append(self.old_age_numerator_ages)
            wealth_age_sets.append(self.old_age_denominator_ages)
        if self.include_old_age_ratio_by_type:
            wealth_age_sets.append(self.tilt_numerator_age_labels)
            wealth_age_sets.append(self.tilt_denominator_age_labels)
        if wealth_age_sets:
            wealth_ages = np.concatenate(wealth_age_sets)
            if wealth_ages.min() <= min_model_age:
                raise ValueError(
                    "Wealth-based ages must exceed the model starting age "
                    "because wealth at age a is savings chosen at a - 1."
                )
        if self.labor_smoothing_window < 1:
            raise ValueError("labor_smoothing_window must be at least 1.")
        if self.include_bequest_to_output and (
            self.bequest_to_output_data is None
        ):
            raise ValueError(
                "bequest_to_output_data is required when "
                "include_bequest_to_output is True."
            )
        if self.failure_residual <= 0:
            raise ValueError("failure_residual must be positive.")


@dataclass(frozen=True)
class MomentSet:
    """Named vector of moments."""

    names: tuple[str, ...]
    values: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float)
        object.__setattr__(self, "values", values)
        if len(self.names) != values.size:
            raise ValueError("Moment names and values have different lengths.")

    def to_frame(self, other: "MomentSet" | None = None) -> pd.DataFrame:
        """Return the moments as a DataFrame, optionally beside another set."""
        frame = pd.DataFrame({"moment": self.names, "value": self.values})
        if other is not None:
            if other.names != self.names:
                raise ValueError("Moment sets are not aligned.")
            frame = frame.rename(columns={"value": "data"})
            frame["model"] = other.values
        return frame


@dataclass
class LifecycleCalibrationResult:
    """Container for joint lifecycle preference calibration results."""

    beta_annual: np.ndarray
    chi_b: np.ndarray
    chi_n: np.ndarray
    objective_value: float
    optimizer_result: object
    data_moments: MomentSet
    model_moments: MomentSet
    weighting_matrix: np.ndarray
    best_start_index: int = 0
    all_start_results: list = field(default_factory=list)

    @property
    def parameter_dict(self) -> dict[str, list[float]]:
        """Return calibrated values in OG-Core update_specifications format."""
        return {
            "beta_annual": self.beta_annual.tolist(),
            "chi_b": self.chi_b.tolist(),
            "chi_n": self.chi_n.tolist(),
        }


@dataclass
class SSSolutionCache:
    """Mutable cache for warm-starting repeated SS solves."""

    previous_output: dict | None = None
    use_ss_solver: bool = True

    def reset(self) -> None:
        """Clear the cached SS output."""
        self.previous_output = None


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _as_vector(values) -> np.ndarray:
    """Return values as a one-dimensional float array."""
    return np.asarray(values, dtype=float).reshape(-1)


def _lambdas(p) -> np.ndarray:
    """Return lifetime-income weights as a one-dimensional array."""
    return _as_vector(p.lambdas)


def _ss_chi_n(p) -> np.ndarray:
    """Return the steady-state chi_n age profile from a spec object."""
    chi_n = np.asarray(p.chi_n, dtype=float)
    if chi_n.ndim == 1:
        return chi_n.copy()
    return chi_n[-1, :].copy()


def _age_to_index(age: int, p) -> int:
    """Map an age label to the model age index of people at that age."""
    return int(age) - int(getattr(p, "starting_age", 20))


def _age_indices(ages: np.ndarray, p) -> np.ndarray:
    """Map age labels to model indices for flow variables (n, c, y)."""
    return np.array([_age_to_index(age, p) for age in ages], dtype=int)


def _wealth_age_indices(ages: np.ndarray, p) -> np.ndarray:
    """Map age labels to ``b_sp1`` indices.

    ``b_sp1[s, j]`` is savings chosen at model age index ``s`` and held at
    the start of age index ``s + 1``.  Wealth observed in the data for people
    of age ``a`` therefore corresponds to ``b_sp1[a - starting_age - 1]``.
    """
    idx = _age_indices(ages, p) - 1
    if np.any(idx < 0):
        raise ValueError(
            "Wealth ages must exceed the model starting age by at least one."
        )
    return idx


def _joint_pop_weights(p) -> np.ndarray:
    """Return the (S, J) steady-state population distribution.

    Newer OG-Core versions carry ``omega_SS`` as an (S, J) joint distribution
    when demographics differ by lifetime-income type.  Older versions carry
    a length-S vector, in which case the joint distribution is the outer
    product with ``lambdas``.
    """
    omega = np.asarray(p.omega_SS, dtype=float)
    lambdas = _lambdas(p)
    if omega.ndim == 2:
        return omega
    return omega.reshape(-1, 1) * lambdas.reshape(1, -1)


def _type_weights_by_age(p) -> np.ndarray:
    """Return (S, J) weights of each type within each age, summing to one."""
    joint = _joint_pop_weights(p)
    totals = joint.sum(axis=1, keepdims=True)
    totals = np.where(totals > 0, totals, 1.0)
    return joint / totals


def _smooth_profile(values: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average with shrinking windows at the edges."""
    values = np.asarray(values, dtype=float)
    if window <= 1 or values.size == 0:
        return values.copy()
    return (
        pd.Series(values)
        .rolling(window=int(window), center=True, min_periods=1)
        .mean()
        .to_numpy(dtype=float)
    )


def _build_chi_n_spline_basis(
    ages: np.ndarray,
    n_basis: int,
    degree: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a B-spline design matrix for the chi_n age profile.

    The spline is defined in log space: evaluating ``B @ gamma`` gives
    ``log(chi_n)`` at each age, so ``chi_n = exp(B @ gamma)`` is always
    positive regardless of the coefficient values.

    Args:
        ages: Age values at which to evaluate the basis (length N).
        n_basis: Number of B-spline basis functions (= number of free
            coefficients).  Must satisfy n_basis >= degree + 1.
        degree: Polynomial degree of the spline (default 3 = cubic).

    Returns:
        B: Design matrix of shape (N, n_basis).
        knots: Full knot vector used to construct the basis.
    """
    from scipy.interpolate import BSpline

    ages_f = np.asarray(ages, dtype=float)
    age_min, age_max = ages_f[0], ages_f[-1]
    n_internal = n_basis - degree - 1
    if n_internal < 0:
        raise ValueError(
            f"n_basis={n_basis} is too small for degree={degree}. "
            f"Need n_basis >= degree + 1 = {degree + 1}."
        )
    internal = (
        np.linspace(age_min, age_max, n_internal + 2)[1:-1]
        if n_internal > 0
        else np.array([], dtype=float)
    )
    knots = np.concatenate(
        [
            np.repeat(age_min, degree + 1),
            internal,
            np.repeat(age_max, degree + 1),
        ]
    )
    n_cols = len(knots) - degree - 1
    B = np.zeros((len(ages_f), n_cols))
    for i in range(n_cols):
        c = np.zeros(n_cols)
        c[i] = 1.0
        B[:, i] = BSpline(knots, c, degree)(ages_f)
    return B, knots


def _weighted_mean(values, weights) -> float:
    """Return a weighted mean after dropping nonfinite observations."""
    data = pd.DataFrame({"value": values, "weight": weights})
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    data = data[data["weight"] > 0]
    if data.empty:
        return np.nan
    return float((data["value"] * data["weight"]).sum() / data["weight"].sum())


def _weighted_mean_by_age(
    data: pd.DataFrame,
    value_col: str,
    weight_col: str | None,
    ages: np.ndarray,
    age_col: str = "age",
) -> np.ndarray:
    """Compute weighted means by single-year age."""
    columns = [age_col, value_col]
    if weight_col is not None:
        columns.append(weight_col)
    age_data = data[columns].copy()
    age_data[age_col] = pd.to_numeric(age_data[age_col], errors="coerce")
    age_data[value_col] = pd.to_numeric(age_data[value_col], errors="coerce")
    age_data = age_data.replace([np.inf, -np.inf], np.nan).dropna()
    age_data = age_data[age_data[age_col].isin(ages)].copy()
    age_data[age_col] = age_data[age_col].astype(int)

    if weight_col is None:
        profile = age_data.groupby(age_col)[value_col].mean()
    else:
        age_data[weight_col] = pd.to_numeric(
            age_data[weight_col], errors="coerce"
        )
        age_data = age_data[age_data[weight_col] > 0].copy()
        age_data["weighted_value"] = age_data[value_col] * age_data[weight_col]
        by_age = age_data.groupby(age_col)[
            ["weighted_value", weight_col]
        ].sum()
        profile = by_age["weighted_value"] / by_age[weight_col]

    return profile.reindex(ages).to_numpy(dtype=float)


def _weighted_mean_in_age_range(
    data: pd.DataFrame,
    value_col: str,
    weight_col: str,
    ages: np.ndarray,
    age_col: str = "age",
) -> float:
    """Weighted mean of a column over all observations with age in ages."""
    subset = data[[age_col, value_col, weight_col]].copy()
    subset[age_col] = pd.to_numeric(subset[age_col], errors="coerce")
    subset = subset[subset[age_col].isin(ages)]
    return _weighted_mean(subset[value_col], subset[weight_col])


def _require_finite(values: np.ndarray, label: str) -> np.ndarray:
    """Validate that all values are finite."""
    values = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{label} contains missing or nonfinite values.")
    return values


def _normalize_wealth_profile(
    profile: np.ndarray,
    anchor_profile: np.ndarray,
    method: WealthProfileMoment,
) -> np.ndarray:
    """Scale a wealth age profile according to the requested convention."""
    profile = np.asarray(profile, dtype=float)
    if method == "level":
        return profile
    if method == "mean_normalized":
        mean = np.nanmean(profile)
        if not np.isfinite(mean) or np.isclose(mean, 0.0):
            raise ValueError("Cannot normalize wealth profile with zero mean.")
        return profile / mean
    if method == "anchor_window":
        anchor_mean = np.nanmean(np.asarray(anchor_profile, dtype=float))
        if not np.isfinite(anchor_mean) or np.isclose(anchor_mean, 0.0):
            raise ValueError(
                "Cannot normalize wealth profile with zero anchor mean."
            )
        return profile / anchor_mean
    raise ValueError(f"Unsupported wealth profile moment: {method}")


# ---------------------------------------------------------------------------
# Wealth distribution moments (bins from lambdas)
# ---------------------------------------------------------------------------


def merged_type_groups(
    lambdas,
    bottom_share: float = 0.5,
    top_share: float = 0.01,
) -> list[list[int]]:
    """Group type indices: bottom ``bottom_share`` merged, top merged.

    Types whose cumulative population share lies within ``bottom_share`` of
    the bottom form one group, types inside the top ``top_share`` form one
    group, and every other type is its own group.  For the default OG-USA
    lambdas this gives ``[[0, 1], [2], [3], [4], [5], [6, 7, 8, 9]]``.
    """
    lambdas = _as_vector(lambdas)
    cum = np.cumsum(lambdas)
    start = np.concatenate([[0.0], cum[:-1]])
    groups: list[list[int]] = []
    bottom = [j for j in range(lambdas.size) if cum[j] <= bottom_share + 1e-12]
    top = [
        j for j in range(lambdas.size) if start[j] >= 1.0 - top_share - 1e-12
    ]
    if bottom:
        groups.append(bottom)
    for j in range(lambdas.size):
        if j not in bottom and j not in top:
            groups.append([j])
    if top:
        groups.append(top)
    return groups


def merged_type_bins(lambdas, **kwargs) -> np.ndarray:
    """Population bin weights for :func:`merged_type_groups`."""
    lambdas = _as_vector(lambdas)
    return np.array(
        [
            lambdas[group].sum()
            for group in merged_type_groups(lambdas, **kwargs)
        ]
    )


def _percent_label(share: float) -> str:
    """Format a cumulative population share as a percentile label."""
    pct = 100.0 * share
    text = f"{pct:.4f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def wealth_share_bin_names(lambdas) -> tuple[str, ...]:
    """Return one wealth-share moment name per lifetime-income bin.

    Bins are cumulative population shares of ``lambdas``; for the default
    ten OG-USA types the names run from ``wealth_share_0_25`` through
    ``wealth_share_99p99_100``.
    """
    cum = np.concatenate([[0.0], np.cumsum(_as_vector(lambdas))])
    cum[-1] = 1.0
    return tuple(
        f"wealth_share_{_percent_label(lo)}_{_percent_label(hi)}"
        for lo, hi in zip(cum[:-1], cum[1:])
    )


def _wealth_distribution_moment_names(
    lambdas,
    include_gini: bool = False,
    include_var_log: bool = False,
) -> tuple[str, ...]:
    """Return names for the wealth distribution moments in order."""
    names = list(wealth_share_bin_names(lambdas))
    if include_gini:
        names.append("wealth_gini")
    if include_var_log:
        names.append("wealth_var_log")
    return tuple(names)


def _living_wealth_distribution(ss_output: dict, p):
    """Return wealth held by living households and matching weights.

    ``b_sp1[s]`` is held at age index ``s + 1``, so the last row is wealth
    carried out of the model at death rather than held by anyone alive.  The
    distribution therefore pairs ``b_sp1[:-1]`` with population weights for
    age indices ``1`` through ``S - 1``.
    """
    b_sp1 = np.asarray(ss_output["b_sp1"], dtype=float)
    joint = _joint_pop_weights(p)
    weights = joint[1:, :]
    # Renormalize so cumulative weights reach one; otherwise percentile
    # targets above the retained population share all map to the top cell.
    return b_sp1[:-1, :], weights / weights.sum()


def percentile_bin_shares(
    dist: np.ndarray,
    weights: np.ndarray,
    bin_weights: np.ndarray,
) -> np.ndarray:
    """Share of a distribution held by consecutive population bins.

    Cells are sorted by ``dist``; the cumulative population weight is
    compared with the cumulative ``bin_weights``.  A cell that straddles a
    cutoff is split in proportion, so the result does not depend on how
    coarse the cells are.  ``weights`` are normalized to sum to one.

    Args:
        dist: values (any shape), e.g. wealth by age and type.
        weights: population weights with the same shape as ``dist``.
        bin_weights: population shares of the bins, summing to one.

    Returns:
        Array with one share per bin, summing to one.
    """
    values = np.asarray(dist, dtype=float).ravel()
    w = np.asarray(weights, dtype=float).ravel()
    w = w / w.sum()
    order = np.argsort(values, kind="stable")
    values = values[order]
    w = w[order]
    cum_w = np.concatenate([[0.0], np.cumsum(w)])
    cum_wealth = np.concatenate([[0.0], np.cumsum(values * w)])
    total_wealth = cum_wealth[-1]
    cutoffs = np.concatenate([[0.0], np.cumsum(_as_vector(bin_weights))])
    cutoffs[-1] = 1.0
    # Wealth held below each cutoff, interpolating within the straddling
    # cell (constant value within a cell, so linear in population weight).
    below = np.interp(cutoffs, cum_w, cum_wealth)
    if np.isclose(total_wealth, 0.0):
        raise ValueError("Total wealth is zero; shares are undefined.")
    return np.diff(below) / total_wealth


def model_wealth_shares(ss_output: dict, p) -> np.ndarray:
    """Compute the share of wealth held by each lambdas bin in the model.

    Percentile cutoffs are cumulative ``p.lambdas``.  Because all households
    (not only those in type ``j``) are sorted by wealth, the bin for type
    ``j`` is a population percentile bin, exactly as in the SCF data moment
    from :func:`ogusa.wealth.compute_wealth_moments`.
    """
    dist, weights = _living_wealth_distribution(ss_output, p)
    return percentile_bin_shares(dist, weights, _lambdas(p))


def _model_wealth_distribution_moments(
    ss_output: dict,
    p,
    config: LifecycleCalibrationConfig,
) -> np.ndarray:
    """Compute model wealth shares plus optional Gini and var(log)."""
    values = list(model_wealth_shares(ss_output, p))
    if config.include_wealth_gini or config.include_wealth_var_log:
        dist, weights = _living_wealth_distribution(ss_output, p)
        ineq = Inequality(dist, weights, _lambdas(p), dist.shape[0], p.J)
        if config.include_wealth_gini:
            values.append(ineq.gini())
        if config.include_wealth_var_log:
            values.append(ineq.var_of_logs())
    return np.asarray(values, dtype=float)


def _data_wealth_distribution_moments(
    scf: pd.DataFrame,
    p,
    config: LifecycleCalibrationConfig,
) -> np.ndarray:
    """Compute SCF wealth shares (bins from lambdas) plus optional extras."""
    raw = wealth.compute_wealth_moments(scf.copy(), _lambdas(p))
    shares = list(raw[: p.J])
    if config.include_wealth_gini:
        shares.append(raw[-2])
    if config.include_wealth_var_log:
        shares.append(raw[-1])
    return np.asarray(shares, dtype=float)


# ---------------------------------------------------------------------------
# Data loading and data moments
# ---------------------------------------------------------------------------


def load_cps_hours_data(
    cps_years: tuple[int, ...] = (2023, 2022),
    cps_directory: str | None = None,
) -> pd.DataFrame:
    """Load packaged CPS ASEC hours data used for labor moments."""
    if cps_directory is None:
        cps_directory = compute_moments.CPS_DATA_DIR
    data = []
    for year in cps_years:
        path = f"{cps_directory}/cps_asec_hours_{year}.csv"
        data.append(pd.read_csv(path))
    return pd.concat(data, ignore_index=True)


def load_scf_wealth_data(config: LifecycleCalibrationConfig) -> pd.DataFrame:
    """Load SCF wealth data with ages (and income) for wealth moments."""
    return wealth.get_wealth_data(
        scf_yrs_list=list(config.scf_yrs_list),
        web=config.scf_web,
        directory=config.scf_directory,
        include_age=True,
        include_income=config.include_wealth_income_ratio,
    )


def _mortality_by_age(p) -> np.ndarray:
    """Steady-state mortality by model age, averaged over types."""
    rho = np.asarray(p.rho, dtype=float)
    if rho.ndim == 3:
        rho = rho[-1]
    if rho.ndim == 2:
        joint = _joint_pop_weights(p)
        totals = joint.sum(axis=1)
        totals = np.where(totals > 0, totals, 1.0)
        return (rho * joint).sum(axis=1) / totals
    return rho.reshape(-1)


def _scf_model_age_frame(
    scf: pd.DataFrame, p, value_cols: tuple[str, ...]
) -> pd.DataFrame:
    """SCF rows at model ages with numeric columns and positive weights."""
    starting_age = int(getattr(p, "starting_age", 20))
    max_age = starting_age + p.S - 1
    cols = ["age", "wgt", *value_cols]
    data = scf[cols].copy()
    for col in cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    data = data[(data["wgt"] > 0) & (data["age"] >= starting_age)].copy()
    # The SCF top-codes age; treat anyone above the model's last age as
    # being at that age.
    data["age"] = data["age"].clip(upper=max_age).astype(int)
    return data


def scf_income_series(
    scf: pd.DataFrame, concept: Literal["pre_transfer", "total"]
) -> pd.Series:
    """Household income from the SCF under the requested concept.

    ``pre_transfer`` subtracts Social Security, pension, and other transfer
    income from total income, matching the model's before-tax income of
    capital plus labor earnings.  ``total`` uses SCF total income.
    """
    if "income" not in scf:
        raise ValueError(
            "SCF data lack income columns; regenerate the extracts with "
            "data/download_moment_data.py or load with include_income."
        )
    income = pd.to_numeric(scf["income"], errors="coerce")
    if concept == "total":
        return income
    if concept == "pre_transfer":
        transfers = pd.to_numeric(
            scf["ssretinc"], errors="coerce"
        ) + pd.to_numeric(scf["transfothinc"], errors="coerce")
        return income - transfers
    raise ValueError(f"Unsupported SCF income concept: {concept}")


def wealth_income_ratio_from_scf(
    scf: pd.DataFrame, p, config: LifecycleCalibrationConfig
) -> float:
    """SCF mean net worth over mean income at model ages."""
    data = scf.copy()
    data["income_used"] = scf_income_series(data, config.scf_income_concept)
    data = _scf_model_age_frame(data, p, ("networth_infadj", "income_used"))
    mean_wealth = _weighted_mean(data["networth_infadj"], data["wgt"])
    mean_income = _weighted_mean(data["income_used"], data["wgt"])
    if not np.isfinite(mean_income) or mean_income <= 0:
        raise ValueError("SCF mean income is not positive.")
    return float(mean_wealth / mean_income)


def model_wealth_income_ratio(ss_output: dict, p) -> float:
    """Mean wealth held by the living over mean before-tax income.

    Both are population-weighted over ages and types.  The income scaling
    factor cancels, so this compares directly with the SCF ratio.
    """
    b_sp1 = np.asarray(ss_output["b_sp1"], dtype=float)
    income = np.asarray(ss_output["before_tax_income"], dtype=float)
    joint = _joint_pop_weights(p)
    mean_wealth = (b_sp1[:-1, :] * joint[1:, :]).sum() / joint.sum()
    mean_income = (income * joint).sum() / joint.sum()
    if np.isclose(mean_income, 0.0):
        raise ValueError("Model mean income is zero.")
    return float(mean_wealth / mean_income)


def bequest_flow_ratio_from_scf(scf: pd.DataFrame, p) -> float:
    """Mortality-weighted SCF wealth over total SCF wealth.

    Applies the model's own mortality rates by age to SCF net worth by age,
    so the data moment is wealth of those who die within the year relative
    to wealth of the living, the same concept as the model's aggregate
    bequests without the return factor.
    """
    data = _scf_model_age_frame(scf, p, ("networth_infadj",))
    starting_age = int(getattr(p, "starting_age", 20))
    rho = _mortality_by_age(p)
    data["rho"] = rho[data["age"].to_numpy() - starting_age]
    weighted_wealth = data["wgt"] * data["networth_infadj"]
    total = weighted_wealth.sum()
    if np.isclose(total, 0.0):
        raise ValueError("SCF total wealth is zero.")
    return float((data["rho"] * weighted_wealth).sum() / total)


def model_bequest_flow_ratio(ss_output: dict, p) -> float:
    """Wealth left by decedents over wealth held, OG-Core bequest timing.

    OG-Core's aggregate bequests are ``(1 + r) * sum(rho * omega * b_sp1)``
    with ``rho[s]`` the probability of dying at the end of age index ``s``
    and ``b_sp1[s]`` the savings chosen at that age.  This moment drops the
    return factor and divides by ``sum(omega * b_sp1)``.
    """
    b_sp1 = np.asarray(ss_output["b_sp1"], dtype=float)
    joint = _joint_pop_weights(p)
    rho = _mortality_by_age(p).reshape(-1, 1)
    total = (joint * b_sp1).sum()
    if np.isclose(total, 0.0):
        raise ValueError("Model total wealth is zero.")
    return float((rho * joint * b_sp1).sum() / total)


def tilt_moment_names(
    config: LifecycleCalibrationConfig, p
) -> tuple[str, ...]:
    """Names for the by-bin old-age wealth ratio moments."""
    n_lo, n_hi = config.tilt_numerator_ages
    d_lo, d_hi = config.tilt_denominator_ages
    bins = wealth_share_bin_names(config.tilt_bins(p))
    return tuple(
        f"tilt_{name.replace('wealth_share_', '')}_{n_lo}_{n_hi}_over_"
        f"{d_lo}_{d_hi}"
        for name in bins
    )


def _bin_mean_wealth(
    values: np.ndarray, weights: np.ndarray, bin_weights: np.ndarray
) -> np.ndarray:
    """Mean wealth in each population bin after sorting by wealth."""
    values = np.asarray(values, dtype=float).ravel()
    weights = np.asarray(weights, dtype=float).ravel()
    shares = percentile_bin_shares(values, weights, bin_weights)
    mean_all = (values * weights).sum() / weights.sum()
    return shares * mean_all / _as_vector(bin_weights)


def old_age_ratio_by_type_from_scf(
    scf: pd.DataFrame, p, config: LifecycleCalibrationConfig
) -> np.ndarray:
    """Within-band wealth-percentile bin means at old ages over pre-retirement.

    Households in each age band are sorted by net worth and cut into the
    population bins from ``config.tilt_bins``; the moment is the ratio of
    bin mean wealth in the numerator band to that in the denominator band.
    Sorting by wealth within an age band proxies for lifetime-income type.
    """
    data = _scf_model_age_frame(scf, p, ("networth_infadj",))
    bins = config.tilt_bins(p)

    def _means(ages):
        band = data[data["age"].isin(ages)]
        if band.empty:
            raise ValueError("SCF age band for tilt moments is empty.")
        return _bin_mean_wealth(
            band["networth_infadj"].to_numpy(), band["wgt"].to_numpy(), bins
        )

    numerator = _means(config.tilt_numerator_age_labels)
    denominator = _means(config.tilt_denominator_age_labels)
    return numerator / denominator


def model_old_age_ratio_by_type(
    ss_output: dict, p, config: LifecycleCalibrationConfig
) -> np.ndarray:
    """Model counterpart of :func:`old_age_ratio_by_type_from_scf`."""
    b_sp1 = np.asarray(ss_output["b_sp1"], dtype=float)
    joint = _joint_pop_weights(p)
    bins = config.tilt_bins(p)

    def _means(ages):
        hold_idx = _age_indices(ages, p)
        save_idx = _wealth_age_indices(ages, p)
        return _bin_mean_wealth(b_sp1[save_idx, :], joint[hold_idx, :], bins)

    numerator = _means(config.tilt_numerator_age_labels)
    denominator = _means(config.tilt_denominator_age_labels)
    return numerator / denominator


def labor_profile_from_cps(
    cps: pd.DataFrame,
    config: LifecycleCalibrationConfig,
) -> np.ndarray:
    """Compute mean hours by age from CPS data as a share of the endowment.

    Hours include non-workers (zero hours), so the profile is aggregate labor
    input per person, matching the model's ``n``.  A centered moving average
    of width ``config.labor_smoothing_window`` is applied so sampling noise
    in single-year cells is not carried into ``chi_n``.
    """
    cps = cps.copy()
    if "hours" not in cps:
        if "hours_per_week" in cps:
            cps["hours"] = cps["hours_per_week"]
        else:
            raise ValueError("CPS data must include hours or hours_per_week.")
    cps["hours"] = pd.to_numeric(cps["hours"], errors="coerce").fillna(0.0)
    cps.loc[cps["hours"] < 0, "hours"] = 0.0
    weight_col = None
    for possible_weight in ("weight", "wtsupp", "s006", "wgt"):
        if possible_weight in cps:
            weight_col = possible_weight
            break
    hours = _weighted_mean_by_age(
        cps,
        "hours",
        weight_col,
        config.moment_ages,
    )
    labor = hours / HOURS_IN_TIME_ENDOWMENT
    labor = _smooth_profile(labor, config.labor_smoothing_window)
    return _require_finite(labor, "labor profile")


def wealth_profile_from_scf(
    scf: pd.DataFrame,
    config: LifecycleCalibrationConfig,
) -> np.ndarray:
    """Compute the (normalized) net-wealth age profile from SCF data."""
    profile = _weighted_mean_by_age(
        scf,
        "networth_infadj",
        "wgt",
        config.wealth_profile_ages,
    )
    anchor_profile = None
    if config.wealth_profile_moment == "anchor_window":
        anchor_profile = _weighted_mean_by_age(
            scf,
            "networth_infadj",
            "wgt",
            config.wealth_anchor_ages,
        )
    profile = _normalize_wealth_profile(
        profile,
        anchor_profile,
        config.wealth_profile_moment,
    )
    return _require_finite(profile, "wealth profile")


def old_age_ratio_moment_name(config: LifecycleCalibrationConfig) -> str:
    """Return the moment name for the old-age wealth ratio."""
    n_lo, n_hi = config.old_age_ratio_numerator_ages
    d_lo, d_hi = config.old_age_ratio_denominator_ages
    return f"old_age_wealth_ratio_{n_lo}_{n_hi}_over_{d_lo}_{d_hi}"


def old_age_wealth_ratio_from_scf(
    scf: pd.DataFrame,
    config: LifecycleCalibrationConfig,
) -> float:
    """Mean SCF net worth in the numerator ages over the denominator ages.

    Both means are survey-weighted over all households in the age range, so
    the ratio is population weighted across ages as well as within them.
    """
    numerator = _weighted_mean_in_age_range(
        scf, "networth_infadj", "wgt", config.old_age_numerator_ages
    )
    denominator = _weighted_mean_in_age_range(
        scf, "networth_infadj", "wgt", config.old_age_denominator_ages
    )
    if not np.isfinite(denominator) or np.isclose(denominator, 0.0):
        raise ValueError("Old-age wealth ratio denominator is zero.")
    return float(numerator / denominator)


def model_old_age_wealth_ratio(
    ss_output: dict,
    p,
    config: LifecycleCalibrationConfig,
) -> float:
    """Population-weighted mean wealth at old ages over pre-retirement ages."""
    b_sp1 = np.asarray(ss_output["b_sp1"], dtype=float)
    joint = _joint_pop_weights(p)

    def _mean_wealth(ages):
        hold_idx = _age_indices(ages, p)
        save_idx = _wealth_age_indices(ages, p)
        weights = joint[hold_idx, :]
        return (b_sp1[save_idx, :] * weights).sum() / weights.sum()

    denominator = _mean_wealth(config.old_age_denominator_ages)
    if np.isclose(denominator, 0.0):
        raise ValueError("Model old-age wealth ratio denominator is zero.")
    return float(_mean_wealth(config.old_age_numerator_ages) / denominator)


def model_bequest_to_output(ss_output: dict, p) -> float:
    """Aggregate bequests over GDP in the model steady state."""
    output = float(ss_output["Y"])
    if np.isclose(output, 0.0):
        raise ValueError("Cannot compute bequests over output with Y = 0.")
    return float(np.sum(np.asarray(ss_output["BQ"], dtype=float)) / output)


def income_gini_data_moment(
    income_year: int | None = None,
) -> float:
    """Compute the before-tax income Gini data moment."""
    moments = compute_moments._taxcalc_cps_income_ginis(income_year)
    return float(moments["Gini coefficient, income"])


def savings_rate_data_moment(macro_year: int = 2025) -> float:
    """Compute the aggregate savings-rate data moment."""
    moments = compute_moments.get_macro_moments(year=macro_year)
    return float(moments[SAVINGS_RATE_DATA_LABEL])


def model_savings_rate_moment(ss_output: dict, p) -> float:
    """Compute the model gross aggregate savings rate."""
    output = float(ss_output["Y"])
    if np.isclose(output, 0.0):
        raise ValueError("Cannot compute savings rate with zero output.")
    growth = (1 + p.g_n_ss) * np.exp(p.g_y)
    gross_saving_flow = (growth - 1.0) * float(ss_output["B"]) + (
        p.delta * float(ss_output["K_d"])
    )
    return gross_saving_flow / output


def compute_inheritance_moments_from_scf(
    scf: pd.DataFrame,
    amount_col: str,
    received_col: str | None = None,
    weight_col: str = "wgt",
    networth_col: str = "networth_infadj",
) -> MomentSet:
    """
    Compute optional inherited-transfer moments from full SCF extracts.

    The trimmed SCF files packaged in OG-USA do not include the inheritance
    variables needed here.  Pass a full SCF extract with the relevant Section X
    variables and specify the amount and receipt indicator columns.
    """
    if amount_col not in scf:
        raise ValueError(f"SCF data are missing {amount_col}.")
    if weight_col not in scf:
        raise ValueError(f"SCF data are missing {weight_col}.")

    data = scf.copy()
    data[amount_col] = pd.to_numeric(data[amount_col], errors="coerce")
    data[weight_col] = pd.to_numeric(data[weight_col], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[amount_col, weight_col]
    )
    data = data[data[weight_col] > 0].copy()
    if received_col is None:
        received = data[amount_col] > 0
    else:
        if received_col not in data:
            raise ValueError(f"SCF data are missing {received_col}.")
        received = pd.to_numeric(data[received_col], errors="coerce") > 0

    names = ["inheritance_received_rate"]
    values = [_weighted_mean(received.astype(float), data[weight_col])]

    recipients = data[received].copy()
    if not recipients.empty:
        names.append("inheritance_amount_conditional_mean")
        values.append(
            _weighted_mean(recipients[amount_col], recipients[weight_col])
        )

    if networth_col in data:
        data[networth_col] = pd.to_numeric(data[networth_col], errors="coerce")
        positive_networth = data[networth_col] > 0
        ratio_data = data[positive_networth].copy()
        if not ratio_data.empty:
            names.append("inheritance_to_networth_mean")
            values.append(
                _weighted_mean(
                    ratio_data[amount_col] / ratio_data[networth_col],
                    ratio_data[weight_col],
                )
            )

    return MomentSet(tuple(names), np.asarray(values, dtype=float))


def compute_data_moments(
    p,
    config: LifecycleCalibrationConfig | None = None,
    cps: pd.DataFrame | None = None,
    scf: pd.DataFrame | None = None,
    income_year: int | None = None,
    inheritance_moments: MomentSet | None = None,
    savings_rate: float | None = None,
) -> MomentSet:
    """Compute the stacked data moment vector.

    Moment order: labor profile, wealth profile, income Gini, savings rate,
    wealth distribution, old-age wealth ratio, bequests over output,
    inheritance moments.  :func:`compute_model_moments` uses the same order.
    """
    if config is None:
        config = LifecycleCalibrationConfig()
    config.validate(p)
    names: list[str] = []
    values: list[float] = []

    needs_scf = (
        config.include_wealth_profile
        or config.include_wealth_distribution
        or config.include_old_age_wealth_ratio
        or config.include_wealth_income_ratio
        or config.include_bequest_flow_ratio
        or config.include_old_age_ratio_by_type
    )
    if needs_scf and scf is None:
        scf = load_scf_wealth_data(config)

    if config.include_labor_profile:
        if cps is None:
            cps = load_cps_hours_data(config.cps_years, config.cps_directory)
        labor = labor_profile_from_cps(cps, config)
        names.extend(f"labor_supply_age_{age}" for age in config.moment_ages)
        values.extend(labor)

    if config.include_wealth_profile:
        wealth_profile = wealth_profile_from_scf(scf, config)
        names.extend(
            f"net_wealth_age_{age}" for age in config.wealth_profile_ages
        )
        values.extend(wealth_profile)

    if config.include_income_gini:
        names.append("income_gini")
        values.append(income_gini_data_moment(income_year=income_year))

    if config.include_savings_rate:
        if savings_rate is None:
            savings_rate = savings_rate_data_moment(config.macro_year)
        names.append("savings_rate")
        values.append(float(savings_rate))

    if config.include_wealth_distribution:
        names.extend(
            _wealth_distribution_moment_names(
                _lambdas(p),
                config.include_wealth_gini,
                config.include_wealth_var_log,
            )
        )
        values.extend(_data_wealth_distribution_moments(scf, p, config))

    if config.include_wealth_income_ratio:
        names.append("wealth_income_ratio")
        values.append(wealth_income_ratio_from_scf(scf, p, config))

    if config.include_bequest_flow_ratio:
        names.append("bequest_flow_ratio")
        values.append(bequest_flow_ratio_from_scf(scf, p))

    if config.include_old_age_wealth_ratio:
        names.append(old_age_ratio_moment_name(config))
        values.append(old_age_wealth_ratio_from_scf(scf, config))

    if config.include_old_age_ratio_by_type:
        names.extend(tilt_moment_names(config, p))
        values.extend(old_age_ratio_by_type_from_scf(scf, p, config))

    if config.include_bequest_to_output:
        names.append("bequest_to_output")
        values.append(float(config.bequest_to_output_data))

    if config.include_inheritance_moments:
        if inheritance_moments is None:
            raise ValueError(
                "inheritance_moments must be supplied when "
                "include_inheritance_moments is True."
            )
        names.extend(inheritance_moments.names)
        values.extend(inheritance_moments.values)

    return MomentSet(tuple(names), np.asarray(values, dtype=float))


def compute_model_moments(
    ss_output: dict,
    p,
    config: LifecycleCalibrationConfig | None = None,
    inheritance_moments: MomentSet | None = None,
) -> MomentSet:
    """Compute model moments in the same order as compute_data_moments."""
    if config is None:
        config = LifecycleCalibrationConfig()
    config.validate(p)
    type_weights = _type_weights_by_age(p)
    names: list[str] = []
    values: list[float] = []

    if config.include_labor_profile:
        age_idx = _age_indices(config.moment_ages, p)
        n = np.asarray(ss_output["n"], dtype=float)
        labor = (n[age_idx, :] * type_weights[age_idx, :]).sum(axis=1)
        names.extend(f"labor_supply_age_{age}" for age in config.moment_ages)
        values.extend(labor)

    if config.include_wealth_profile:
        b_sp1 = np.asarray(ss_output["b_sp1"], dtype=float)
        factor = float(ss_output.get("factor", 1.0))

        def _profile(ages):
            hold_idx = _age_indices(ages, p)
            save_idx = _wealth_age_indices(ages, p)
            return (
                b_sp1[save_idx, :] * factor * type_weights[hold_idx, :]
            ).sum(axis=1)

        anchor = None
        if config.wealth_profile_moment == "anchor_window":
            anchor = _profile(config.wealth_anchor_ages)
        wealth_profile = _normalize_wealth_profile(
            _profile(config.wealth_profile_ages),
            anchor,
            config.wealth_profile_moment,
        )
        names.extend(
            f"net_wealth_age_{age}" for age in config.wealth_profile_ages
        )
        values.extend(wealth_profile)

    if config.include_income_gini:
        income = np.asarray(ss_output["before_tax_income"], dtype=float)
        income_ineq = Inequality(
            income, _joint_pop_weights(p), _lambdas(p), p.S, p.J
        )
        names.append("income_gini")
        values.append(income_ineq.gini())

    if config.include_savings_rate:
        names.append("savings_rate")
        values.append(model_savings_rate_moment(ss_output, p))

    if config.include_wealth_distribution:
        names.extend(
            _wealth_distribution_moment_names(
                _lambdas(p),
                config.include_wealth_gini,
                config.include_wealth_var_log,
            )
        )
        values.extend(_model_wealth_distribution_moments(ss_output, p, config))

    if config.include_wealth_income_ratio:
        names.append("wealth_income_ratio")
        values.append(model_wealth_income_ratio(ss_output, p))

    if config.include_bequest_flow_ratio:
        names.append("bequest_flow_ratio")
        values.append(model_bequest_flow_ratio(ss_output, p))

    if config.include_old_age_wealth_ratio:
        names.append(old_age_ratio_moment_name(config))
        values.append(model_old_age_wealth_ratio(ss_output, p, config))

    if config.include_old_age_ratio_by_type:
        names.extend(tilt_moment_names(config, p))
        values.extend(model_old_age_ratio_by_type(ss_output, p, config))

    if config.include_bequest_to_output:
        names.append("bequest_to_output")
        values.append(model_bequest_to_output(ss_output, p))

    if config.include_inheritance_moments:
        if inheritance_moments is None:
            raise ValueError(
                "Model inheritance moments must be supplied when "
                "include_inheritance_moments is True."
            )
        names.extend(inheritance_moments.names)
        values.extend(inheritance_moments.values)

    return MomentSet(tuple(names), np.asarray(values, dtype=float))


# ---------------------------------------------------------------------------
# Parameter packing for the SMM inference driver
# ---------------------------------------------------------------------------


def build_chi_n_profile(
    estimated_chi_n: np.ndarray,
    base_chi_n: np.ndarray,
    p,
    config: LifecycleCalibrationConfig | None = None,
) -> np.ndarray:
    """Build a full-S chi_n profile from directly estimated values."""
    if config is None:
        config = LifecycleCalibrationConfig()
    config.validate(p)
    full_chi_n = np.asarray(base_chi_n, dtype=float).reshape(-1).copy()
    if full_chi_n.size != p.S:
        raise ValueError("base_chi_n length must equal p.S.")

    est_values = np.asarray(estimated_chi_n, dtype=float).reshape(-1)
    est_ages = config.estimated_chi_n_ages
    if est_values.size != est_ages.size:
        raise ValueError("estimated_chi_n length does not match config ages.")

    est_idx = _age_indices(est_ages, p)
    full_chi_n[est_idx] = est_values
    tail_start = int(est_idx[-1] + 1)
    if tail_start >= p.S:
        return full_chi_n

    if config.chi_n_tail_method == "flat":
        full_chi_n[tail_start:] = est_values[-1]
    elif config.chi_n_tail_method == "scaled_default":
        base_anchor = base_chi_n[tail_start - 1]
        if np.isclose(base_anchor, 0.0):
            scale = 1.0
        else:
            scale = est_values[-1] / base_anchor
        full_chi_n[tail_start:] = base_chi_n[tail_start:] * scale
    else:
        raise ValueError(
            f"Unsupported chi_n tail method: {config.chi_n_tail_method}"
        )

    return full_chi_n


def pack_lifecycle_params(
    beta_annual: np.ndarray,
    chi_b: np.ndarray,
    chi_n: np.ndarray,
    p,
    config: LifecycleCalibrationConfig | None = None,
    transform: bool = True,
) -> np.ndarray:
    """Pack natural lifecycle parameters into an optimizer vector.

    chi_n (length p.S) is projected onto the B-spline basis in log space.
    The returned vector layout is [beta_trans, chi_b_trans, gamma] where
    gamma holds the chi_n spline coefficients (unconstrained when transform
    is True, since the log-space spline already enforces positivity via exp).
    """
    if config is None:
        config = LifecycleCalibrationConfig()
    config.validate(p)
    beta_annual = _as_vector(beta_annual)
    chi_b = _as_vector(chi_b)
    chi_n = _as_vector(chi_n)

    starting_age = int(getattr(p, "starting_age", 20))
    all_ages = np.arange(starting_age, starting_age + p.S)
    B, _ = _build_chi_n_spline_basis(
        all_ages, config.chi_n_n_spline_knots, config.chi_n_spline_degree
    )

    if not transform:
        # Project chi_n directly (no log); coefficients are in natural space.
        gamma, _, _, _ = np.linalg.lstsq(B, chi_n, rcond=None)
        return np.concatenate([beta_annual, chi_b, gamma])

    if np.any((beta_annual <= 0) | (beta_annual >= 1)):
        raise ValueError(
            "beta_annual values must be strictly between 0 and 1."
        )
    if np.any(chi_b <= 0):
        raise ValueError("chi_b values must be positive.")
    if np.any(chi_n <= 0):
        raise ValueError("chi_n values must be positive.")

    beta_trans = np.log(beta_annual / (1 - beta_annual))
    chi_b_trans = np.log(chi_b)
    # Fit spline to log(chi_n) over all model ages; gamma is unconstrained.
    gamma, _, _, _ = np.linalg.lstsq(B, np.log(chi_n), rcond=None)
    return np.concatenate([beta_trans, chi_b_trans, gamma])


def unpack_lifecycle_params(
    theta: np.ndarray,
    p,
    config: LifecycleCalibrationConfig | None = None,
    base_chi_n: np.ndarray | None = None,
    transform: bool = True,
) -> dict[str, np.ndarray]:
    """Unpack an optimizer vector into natural lifecycle parameters.

    The chi_n block of theta holds B-spline coefficients (gamma).  When
    transform=True the spline operates in log space and chi_n = exp(B @ gamma).
    The base_chi_n argument is accepted for backward compatibility but is no
    longer used; the spline covers all model ages directly.
    """
    if config is None:
        config = LifecycleCalibrationConfig()
    config.validate(p)

    theta = _as_vector(theta)
    n_beta = p.J
    n_chi_b = p.J
    n_gamma = config.chi_n_n_spline_knots
    expected = n_beta + n_chi_b + n_gamma
    if theta.size != expected:
        raise ValueError(f"Expected {expected} parameters, got {theta.size}.")

    beta_raw = theta[:n_beta]
    chi_b_raw = theta[n_beta : n_beta + n_chi_b]
    gamma = theta[n_beta + n_chi_b :]

    starting_age = int(getattr(p, "starting_age", 20))
    all_ages = np.arange(starting_age, starting_age + p.S)
    B, _ = _build_chi_n_spline_basis(
        all_ages, config.chi_n_n_spline_knots, config.chi_n_spline_degree
    )

    if transform:
        beta_annual = 1 / (1 + np.exp(-beta_raw))
        chi_b = np.exp(chi_b_raw)
        chi_n = np.exp(B @ gamma)
    else:
        beta_annual = beta_raw
        chi_b = chi_b_raw
        chi_n = B @ gamma

    return {
        "beta_annual": beta_annual,
        "chi_b": chi_b,
        "chi_n": chi_n,
    }


def initial_lifecycle_theta(
    p,
    config: LifecycleCalibrationConfig | None = None,
    transform: bool = True,
) -> np.ndarray:
    """Build an optimizer vector from the current spec values."""
    if config is None:
        config = LifecycleCalibrationConfig()
    return pack_lifecycle_params(
        p.beta_annual,
        p.chi_b,
        _ss_chi_n(p),
        p,
        config,
        transform=transform,
    )


def apply_lifecycle_params(
    p,
    beta_annual: np.ndarray,
    chi_b: np.ndarray,
    chi_n: np.ndarray,
) -> None:
    """Apply natural lifecycle parameters to a spec object in-place."""
    p.update_specifications(
        {
            "beta_annual": _as_vector(beta_annual).tolist(),
            "chi_b": _as_vector(chi_b).tolist(),
            "chi_n": _as_vector(chi_n).tolist(),
        }
    )


# ---------------------------------------------------------------------------
# Steady-state solves
# ---------------------------------------------------------------------------

_SS_WARM_START_ERRORS = (
    AssertionError,
    FloatingPointError,
    KeyError,
    RuntimeError,
    TypeError,
    ValueError,
)


def _ss_solver_kwargs(previous: dict, p, client) -> dict:
    """Build keyword arguments for ``SS.SS_solver`` from a prior solution.

    The solver's positional signature has changed across OG-Core releases
    (a ``G`` argument was added), so arguments are matched by name against
    the installed signature.  Missing required arguments raise ``TypeError``
    rather than being misassigned positionally.
    """
    signature = inspect.signature(SS.SS_solver).parameters
    ig_baseline = (
        previous.get("I_g") if getattr(p, "baseline_spending", False) else None
    )
    candidates = {
        "bmat": previous["b_sp1"],
        "nmat": previous["n"],
        "r_p": float(previous["r_p"]),
        "r": float(previous["r"]),
        "w": float(previous["w"]),
        "p_m": previous["p_m"],
        "Y": float(previous["Y"]),
        "BQ": previous["BQ"],
        "TR": float(previous["TR"]),
        "Ig_baseline": ig_baseline,
        "factor": float(previous["factor"]),
        "p": p,
        "client": client,
    }
    if "G" in signature and "G" in previous:
        candidates["G"] = float(previous["G"])
    accepts_var_keywords = any(
        param.kind is inspect.Parameter.VAR_KEYWORD
        for param in signature.values()
    )
    if accepts_var_keywords:
        kwargs = dict(candidates)
    else:
        kwargs = {k: v for k, v in candidates.items() if k in signature}
    named_kinds = (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    )
    missing = [
        name
        for name, param in signature.items()
        if param.kind in named_kinds
        and param.default is inspect.Parameter.empty
        and name not in kwargs
    ]
    if missing:
        raise TypeError(
            "Cannot warm start SS.SS_solver; missing required arguments: "
            + ", ".join(missing)
        )
    return kwargs


def solve_ss_with_cache(
    p,
    client=None,
    ss_cache: SSSolutionCache | None = None,
) -> dict:
    """
    Solve SS, optionally warm-starting from the previous SS output.

    The direct SS_solver path keeps p.baseline unchanged, so baseline solves
    still update the model scaling factor.  If the warm start fails, a
    warning is logged and the solve falls back to SS.run_SS, refreshing the
    cache with that solution.
    """
    use_cache = (
        ss_cache is not None
        and ss_cache.use_ss_solver
        and ss_cache.previous_output is not None
    )
    ss_output = None
    if use_cache:
        try:
            kwargs = _ss_solver_kwargs(ss_cache.previous_output, p, client)
            ss_output = SS.SS_solver(**kwargs)
        except _SS_WARM_START_ERRORS as err:
            logger.warning(
                "SS warm start failed (%s: %s); falling back to a cold "
                "SS.run_SS solve.",
                type(err).__name__,
                err,
            )
            ss_output = None
    if ss_output is None:
        ss_output = SS.run_SS(p, client=client)

    if ss_cache is not None:
        ss_cache.previous_output = ss_output
    return ss_output


# ---------------------------------------------------------------------------
# Weighting, distance, and the SMM inference driver
# ---------------------------------------------------------------------------


def weighting_matrix(
    moment_count: int,
    method: WeightingMethod = "identity",
    bootstrap_moments: np.ndarray | None = None,
    ridge: float = 1e-8,
) -> np.ndarray:
    """Construct a weighting matrix for the SMM objective."""
    if method == "identity":
        return np.eye(moment_count)
    if bootstrap_moments is None:
        raise ValueError("bootstrap_moments are required for this method.")

    boot = np.asarray(bootstrap_moments, dtype=float)
    if boot.ndim != 2 or boot.shape[1] != moment_count:
        raise ValueError("bootstrap_moments must be n x moment_count.")
    vcv = np.cov(boot.T)
    vcv = np.atleast_2d(vcv)
    if method == "diagonal":
        diag = np.diag(vcv).copy()
        diag[diag < ridge] = ridge
        return np.diag(1 / diag)
    if method == "optimal":
        vcv = vcv + ridge * np.eye(moment_count)
        return np.linalg.pinv(vcv)
    raise ValueError(f"Unsupported weighting method: {method}")


def bootstrap_data_moments(
    p,
    config: LifecycleCalibrationConfig | None = None,
    cps: pd.DataFrame | None = None,
    scf: pd.DataFrame | None = None,
    seed: int | None = None,
    savings_rate: float | None = None,
) -> np.ndarray:
    """Bootstrap the data moments available from CPS and SCF microdata.

    Moments that do not come from the CPS or SCF microdata (income Gini,
    savings rate, bequests over output, inheritance moments) are held at
    their point values.  Rows are resampled independently; SCF implicates
    of the same household are therefore treated as independent draws, which
    understates the sampling variance somewhat.
    """
    if config is None:
        config = LifecycleCalibrationConfig()
    if cps is None and config.include_labor_profile:
        cps = load_cps_hours_data(config.cps_years, config.cps_directory)
    needs_scf = (
        config.include_wealth_profile
        or config.include_wealth_distribution
        or config.include_old_age_wealth_ratio
    )
    if scf is None and needs_scf:
        scf = load_scf_wealth_data(config)

    point_moments = compute_data_moments(
        p,
        config,
        cps=cps,
        scf=scf,
        savings_rate=savings_rate,
    )
    resampled_config = replace(
        config,
        include_income_gini=False,
        include_savings_rate=False,
        include_bequest_to_output=False,
        include_inheritance_moments=False,
    )
    rng = np.random.default_rng(seed)
    boot = np.zeros((config.bootstrap_iterations, point_moments.values.size))
    for i in range(config.bootstrap_iterations):
        cps_boot = None
        scf_boot = None
        if cps is not None:
            cps_boot = cps.iloc[
                rng.integers(0, len(cps), size=len(cps))
            ].reset_index(drop=True)
        if scf is not None:
            scf_boot = scf.iloc[
                rng.integers(0, len(scf), size=len(scf))
            ].reset_index(drop=True)
        resampled_moments = compute_data_moments(
            p,
            resampled_config,
            cps=cps_boot,
            scf=scf_boot,
        )
        resampled_values = dict(
            zip(resampled_moments.names, resampled_moments.values)
        )
        boot[i, :] = [
            resampled_values.get(name, value)
            for name, value in zip(point_moments.names, point_moments.values)
        ]
    return boot


def moment_residuals(
    model_moments: MomentSet,
    data_moments: MomentSet,
    method: MomentDistanceMethod = "relative",
    floor: float = 1e-8,
) -> np.ndarray:
    """Return model-minus-data residuals, optionally relative to the data."""
    if model_moments.names != data_moments.names:
        raise ValueError("Model and data moments are not aligned.")
    m = model_moments.values
    d = data_moments.values
    if method == "relative":
        safe_denom = np.where(np.abs(d) > floor, np.abs(d), floor)
        return (m - d) / safe_denom
    if method == "absolute":
        return m - d
    raise ValueError(f"Unsupported moment distance method: {method}")


def smm_distance(
    model_moments: MomentSet,
    data_moments: MomentSet,
    W: np.ndarray,
    method: MomentDistanceMethod = "relative",
    floor: float = 1e-8,
) -> float:
    """Compute the quadratic SMM distance.

    Args:
        model_moments: Simulated model moments.
        data_moments: Empirical data moments.
        W: Positive semi-definite weighting matrix.
        method: ``"relative"`` divides each residual by ``|data_moment|``
            (floored at ``floor``) so all moments are on a comparable
            percentage-deviation scale.  ``"absolute"`` uses raw levels.
        floor: Minimum absolute value used as the denominator when
            method="relative", preventing division by zero for near-zero
            moments such as bottom wealth shares.
    """
    diff = moment_residuals(model_moments, data_moments, method, floor)
    if not np.all(np.isfinite(diff)):
        return np.inf
    return float(diff.T @ W @ diff)


def _apply_weight_sqrt(residuals: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Return L.T @ residuals where W = L @ L.T, so ||result||^2 = r^T W r.

    For identity W this is a no-op.  For diagonal W it multiplies element-wise
    by sqrt of the diagonal.  For a full positive definite W it uses the
    Cholesky factor, falling back to the diagonal if W is not PD.
    """
    if np.allclose(W, np.eye(len(W))):
        return residuals
    try:
        L = np.linalg.cholesky(W)
        return L.T @ residuals
    except np.linalg.LinAlgError:
        return np.sqrt(np.maximum(np.diag(W), 0.0)) * residuals


def _evaluate_model_moments(
    theta: np.ndarray,
    p,
    config: LifecycleCalibrationConfig,
    base_chi_n: np.ndarray | None,
    client,
    transform: bool,
    ss_cache: SSSolutionCache | None,
) -> MomentSet:
    """Apply theta to p, solve the SS, and return model moments."""
    params = unpack_lifecycle_params(
        theta, p, config, base_chi_n=base_chi_n, transform=transform
    )
    apply_lifecycle_params(
        p, params["beta_annual"], params["chi_b"], params["chi_n"]
    )
    ss_output = solve_ss_with_cache(p, client=client, ss_cache=ss_cache)
    return compute_model_moments(ss_output, p, config)


def smm_objective(
    theta: np.ndarray,
    data_moments: MomentSet,
    W: np.ndarray,
    p,
    config: LifecycleCalibrationConfig | None = None,
    base_chi_n: np.ndarray | None = None,
    client=None,
    transform: bool = True,
    ss_cache: SSSolutionCache | None = None,
) -> float:
    """Evaluate the joint lifecycle SMM objective.

    On solver failure returns the objective implied by a residual vector of
    ``config.failure_residual`` in every moment, so failed regions look bad
    to the optimizer without producing astronomically large values.
    """
    if config is None:
        config = LifecycleCalibrationConfig()
    if base_chi_n is None:
        base_chi_n = _ss_chi_n(p)
    n_moments = data_moments.values.size
    penalty = float(n_moments) * config.failure_residual**2
    try:
        model_moments = _evaluate_model_moments(
            theta, p, config, base_chi_n, client, transform, ss_cache
        )
        distance = smm_distance(
            model_moments,
            data_moments,
            W,
            method=config.moment_distance_method,
            floor=config.moment_distance_floor,
        )
    except _SS_WARM_START_ERRORS as err:
        logger.warning("SMM objective evaluation failed: %s", err)
        return penalty
    if not np.isfinite(distance):
        return penalty
    return float(min(distance, penalty))


def smm_residual(
    theta: np.ndarray,
    data_moments: MomentSet,
    W: np.ndarray,
    p,
    config: LifecycleCalibrationConfig,
    base_chi_n: np.ndarray | None = None,
    client=None,
    transform: bool = True,
    ss_cache: SSSolutionCache | None = None,
) -> np.ndarray:
    """Return the weighted moment residual vector for DFO-LS.

    DFO-LS minimises ``||f(x)||^2``.  Pre-multiplying by ``W^{1/2}`` ensures
    that ``||result||^2 == r^T W r``, matching the SMM objective.  On solver
    failure returns a vector of ``config.failure_residual`` so the optimizer
    avoids that region without the interpolation model being poisoned by
    enormous values.
    """
    n_moments = data_moments.values.size
    penalty = np.full(n_moments, config.failure_residual)
    try:
        model_moments = _evaluate_model_moments(
            theta, p, config, base_chi_n, client, transform, ss_cache
        )
        residuals = moment_residuals(
            model_moments,
            data_moments,
            config.moment_distance_method,
            config.moment_distance_floor,
        )
        weighted = _apply_weight_sqrt(residuals, W)
    except _SS_WARM_START_ERRORS as err:
        logger.warning("SMM residual evaluation failed: %s", err)
        return penalty
    if not np.all(np.isfinite(weighted)):
        return penalty
    return np.clip(weighted, -config.failure_residual, config.failure_residual)


def _validator_range(p, param_name: str) -> tuple[float, float]:
    """Return the ParamTools range validator bounds for a parameter."""
    data = getattr(p, "_data", {})
    validators = data.get(param_name, {}).get("validators", {})
    value_range = validators.get("range", {})
    lo = float(value_range.get("min", -np.inf))
    hi = float(value_range.get("max", np.inf))
    return lo, hi


def _intersect_bounds(
    validator_bounds: tuple[float, float],
    config_bounds: tuple[float, float] | None,
) -> tuple[float, float]:
    """Intersect validator bounds with optional tighter config bounds."""
    lo, hi = validator_bounds
    if config_bounds is not None:
        lo = max(lo, float(config_bounds[0]))
        hi = min(hi, float(config_bounds[1]))
    if not hi > lo:
        raise ValueError(f"Empty parameter bounds: [{lo}, {hi}].")
    return lo, hi


def _extract_dfols_bounds(
    p,
    config: LifecycleCalibrationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Build DFO-LS bound arrays in the transformed parameter space.

    Natural bounds are the intersection of the ParamTools range validators
    on ``p`` with the (optional, tighter) bounds in ``config``.  They are
    then mapped into the optimizer's space:

    * ``beta_annual`` uses a logit transform, with ``bound_epsilon`` keeping
      the natural bounds strictly inside (0, 1).
    * ``chi_b`` uses a log transform with the same epsilon floor.
    * ``chi_n`` spline coefficients ``gamma`` are bounded by
      ``log(chi_n_min)`` and ``log(chi_n_max)``.  By the convex-hull property
      of B-splines this keeps ``chi_n = exp(B @ gamma)`` inside the natural
      bounds at every age.

    Returns:
        lower: 1-D array of length 2J + n_gamma.
        upper: 1-D array of the same length.
    """
    eps = config.bound_epsilon

    b_lo, b_hi = _intersect_bounds(
        _validator_range(p, "beta_annual"), config.beta_annual_bounds
    )
    b_lo = max(b_lo, eps)
    b_hi = min(b_hi, 1.0 - eps)
    beta_lo = np.full(p.J, np.log(b_lo / (1.0 - b_lo)))
    beta_hi = np.full(p.J, np.log(b_hi / (1.0 - b_hi)))

    cb_lo, cb_hi = _intersect_bounds(
        _validator_range(p, "chi_b"), config.chi_b_bounds
    )
    cb_lo = max(cb_lo, eps)
    chi_b_lo = np.full(p.J, np.log(cb_lo))
    chi_b_hi = np.full(p.J, np.log(cb_hi))

    cn_lo, cn_hi = _intersect_bounds(
        _validator_range(p, "chi_n"), config.chi_n_bounds
    )
    cn_lo = max(cn_lo, eps)
    gamma_lo = np.full(config.chi_n_n_spline_knots, np.log(cn_lo))
    gamma_hi = np.full(config.chi_n_n_spline_knots, np.log(cn_hi))

    lower = np.concatenate([beta_lo, chi_b_lo, gamma_lo])
    upper = np.concatenate([beta_hi, chi_b_hi, gamma_hi])
    return lower, upper


def _generate_starts(
    theta0: np.ndarray,
    n_starts: int,
    radius: float,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
    seed: int | None = None,
) -> list[np.ndarray]:
    """Return n_starts starting vectors centred on theta0.

    The first element is always theta0 itself (the warm start from prior
    calibration values).  Additional points are drawn by adding additive
    noise scaled by ``radius * max(|theta0_i|, 1.0)`` in each dimension,
    so near-zero transformed parameters still receive meaningful perturbations.
    All generated points are clipped to ``[lower, upper]`` when provided.
    """

    def _clip(x):
        if lower is not None:
            x = np.maximum(x, lower)
        if upper is not None:
            x = np.minimum(x, upper)
        return x

    starts = [_clip(theta0.copy())]
    if n_starts <= 1:
        return starts
    rng = np.random.default_rng(seed)
    scale = np.maximum(np.abs(theta0), 1.0)
    for _ in range(n_starts - 1):
        noise = rng.uniform(-radius, radius, size=theta0.shape) * scale
        starts.append(_clip(theta0 + noise))
    return starts


def estimate_lifecycle_params(
    p,
    config: LifecycleCalibrationConfig | None = None,
    theta0: np.ndarray | None = None,
    data_moments: MomentSet | None = None,
    W: np.ndarray | None = None,
    bootstrap_moments: np.ndarray | None = None,
    client=None,
    transform: bool = True,
    savings_rate: float | None = None,
) -> LifecycleCalibrationResult:
    """Estimate beta_annual, chi_b, and chi_n jointly by SMM using DFO-LS.

    Every residual evaluation solves the full general-equilibrium steady
    state, so this is expensive.  It is intended for inference from an
    already calibrated starting point rather than as the primary calibration
    routine.
    """
    if dfols is None:
        raise ImportError(
            "dfo-ls is required for estimate_lifecycle_params; install it "
            "with `uv add dfo-ls`."
        )
    if config is None:
        config = LifecycleCalibrationConfig()
    config.validate(p)
    base_chi_n = _ss_chi_n(p)
    if theta0 is None:
        theta0 = initial_lifecycle_theta(p, config, transform=transform)
    if data_moments is None:
        data_moments = compute_data_moments(
            p,
            config,
            savings_rate=savings_rate,
        )
    if W is None:
        W = weighting_matrix(
            data_moments.values.size,
            method=config.weighting_method,
            bootstrap_moments=bootstrap_moments,
            ridge=config.weighting_ridge,
        )

    lower, upper = _extract_dfols_bounds(p, config)
    starts = _generate_starts(
        theta0,
        config.n_starts,
        config.start_radius,
        lower=lower,
        upper=upper,
    )
    best_result = None
    best_obj = np.inf
    best_start_index = 0
    all_start_results = []

    for i, theta_start in enumerate(starts):
        logger.info(
            "Lifecycle SMM: DFO-LS start %d/%d", i + 1, config.n_starts
        )
        ss_cache = SSSolutionCache(use_ss_solver=config.use_ss_solver_restart)

        def residual_fn(theta, _cache=ss_cache):
            return smm_residual(
                theta,
                data_moments,
                W,
                p,
                config,
                base_chi_n=base_chi_n,
                client=client,
                transform=transform,
                ss_cache=_cache,
            )

        dfols_result = dfols.solve(
            residual_fn,
            theta_start,
            bounds=(lower, upper),
            rhoend=config.dfols_rhoend,
            maxfun=config.dfols_maxfun,
            do_logging=config.log_optimizer_progress,
            print_progress=False,
        )
        all_start_results.append(dfols_result)
        obj = float(dfols_result.obj)
        logger.info(
            "Lifecycle SMM: start %d/%d finished, objective=%.6e, "
            "evals=%d, msg=%s",
            i + 1,
            config.n_starts,
            obj,
            dfols_result.nf,
            dfols_result.msg,
        )
        if obj < best_obj:
            best_obj = obj
            best_result = dfols_result
            best_start_index = i

    params = unpack_lifecycle_params(
        best_result.x,
        p,
        config,
        base_chi_n=base_chi_n,
        transform=transform,
    )
    apply_lifecycle_params(
        p,
        params["beta_annual"],
        params["chi_b"],
        params["chi_n"],
    )
    ss_cache = SSSolutionCache(use_ss_solver=config.use_ss_solver_restart)
    ss_output = solve_ss_with_cache(p, client=client, ss_cache=ss_cache)
    model_moments = compute_model_moments(ss_output, p, config)

    return LifecycleCalibrationResult(
        beta_annual=params["beta_annual"],
        chi_b=params["chi_b"],
        chi_n=params["chi_n"],
        objective_value=best_obj,
        optimizer_result=best_result,
        data_moments=data_moments,
        model_moments=model_moments,
        weighting_matrix=W,
        best_start_index=best_start_index,
        all_start_results=all_start_results,
    )


def compute_parameter_vcv(
    theta_hat: np.ndarray,
    W: np.ndarray,
    p,
    config: LifecycleCalibrationConfig | None = None,
    base_chi_n: np.ndarray | None = None,
    h: float = 1e-4,
    client=None,
    transform: bool = True,
) -> np.ndarray:
    """Compute a numerical GMM parameter VCV matrix."""
    if config is None:
        config = LifecycleCalibrationConfig()
    if base_chi_n is None:
        base_chi_n = _ss_chi_n(p)
    ss_cache = SSSolutionCache(use_ss_solver=config.use_ss_solver_restart)
    theta_hat = _as_vector(theta_hat)
    base_moments = _evaluate_model_moments(
        theta_hat, p, config, base_chi_n, client, transform, ss_cache
    )
    deriv = np.zeros((base_moments.values.size, theta_hat.size))

    for i in range(theta_hat.size):
        step = h * max(1.0, abs(theta_hat[i]))
        high = theta_hat.copy()
        low = theta_hat.copy()
        high[i] += step
        low[i] -= step
        high_moments = _evaluate_model_moments(
            high, p, config, base_chi_n, client, transform, ss_cache
        )
        low_moments = _evaluate_model_moments(
            low, p, config, base_chi_n, client, transform, ss_cache
        )
        deriv[:, i] = (high_moments.values - low_moments.values) / (2 * step)

    return np.linalg.pinv(deriv.T @ W @ deriv)
