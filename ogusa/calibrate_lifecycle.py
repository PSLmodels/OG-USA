"""
Nested calibration of household preference parameters in OG-USA.

This module implements the calibration path described in
``LIFECYCLE_CALIBRATION_PLAN.md``.  Phase 2 (this file's first block)
provides a household-only steady-state solve: given prices, taxes, bequests,
transfers, and the income scaling factor from a general-equilibrium
steady-state solution, it re-solves only the household Euler equations for
every lifetime-income type.  That is the inner workhorse for the ``chi_n``
inversion and the ``beta`` / ``chi_b`` fixed point in later phases, which
would otherwise need a full general-equilibrium solve on every evaluation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from ogcore import SS, aggregates as aggr, household

logger = logging.getLogger(__name__)

_HOUSEHOLD_INPUT_KEYS = ("p_i", "p_tilde", "bq", "rm", "tr", "ubi")


@dataclass(frozen=True)
class HouseholdEnvironment:
    """
    Everything the household block takes as given in the steady state.

    Attributes:
        r_p (float): return on the household portfolio
        w (float): wage rate
        p_tilde (float): composite consumption good price
        p_i (Numpy array): consumption good prices, length I
        bq (Numpy array): bequests received by age and type, shape (S, J)
        rm (Numpy array): remittances received by age and type, shape (S, J)
        tr (Numpy array): government transfers received by age and type,
            shape (S, J)
        ubi (Numpy array): universal basic income by age and type, shape
            (S, J)
        factor (float): scaling factor from model units to dollars
    """

    r_p: float
    w: float
    p_tilde: float
    p_i: np.ndarray
    bq: np.ndarray
    rm: np.ndarray
    tr: np.ndarray
    ubi: np.ndarray
    factor: float

    @classmethod
    def from_ss_output(cls, ss_output: dict, p) -> "HouseholdEnvironment":
        """
        Builds the household environment from an OG-Core steady-state output.

        The output dictionary written by ``SS.run_SS`` already carries the
        household-level arrays (``bq``, ``rm``, ``tr``, ``ubi``) and prices
        (``p_i``, ``p_tilde``).  When any of those are missing, for example
        from an older pickle, they are rebuilt from the aggregates ``BQ``,
        ``RM``, ``TR``, ``Y``, and ``p_m`` with the same OG-Core helpers the
        steady-state inner loop uses.

        Args:
            ss_output (dict): OG-Core steady-state output
            p (OG-Core Specifications object): parameters object

        Returns:
            env (HouseholdEnvironment): prices, transfers, bequests, and
                scaling factor held fixed in household-only solves
        """
        factor = float(ss_output["factor"])
        if all(key in ss_output for key in _HOUSEHOLD_INPUT_KEYS):
            p_i = np.atleast_1d(np.asarray(ss_output["p_i"], dtype=float))
            p_tilde = float(np.squeeze(ss_output["p_tilde"]))
            bq = np.asarray(ss_output["bq"], dtype=float)
            rm = np.asarray(ss_output["rm"], dtype=float)
            tr = np.asarray(ss_output["tr"], dtype=float)
            ubi = np.asarray(ss_output["ubi"], dtype=float)
        else:
            p_m = np.atleast_1d(np.asarray(ss_output["p_m"], dtype=float))
            p_i = _consumption_good_prices(p_m, p)
            p_tilde = float(
                np.squeeze(aggr.get_ptilde(p_i, p.tau_c[-1, :], p.alpha_c))
            )
            BQ = np.asarray(ss_output["BQ"], dtype=float)
            if "RM" in ss_output:
                RM = np.asarray(ss_output["RM"], dtype=float)
            else:
                RM = np.asarray(
                    aggr.get_RM(float(ss_output["Y"]), p, "SS"), dtype=float
                )
            bq = np.asarray(household.get_bq(BQ, None, p, "SS"), dtype=float)
            rm = np.asarray(household.get_rm(RM, None, p, "SS"), dtype=float)
            tr = np.asarray(
                household.get_tr(float(ss_output["TR"]), None, p, "SS"),
                dtype=float,
            )
            ubi = np.asarray(p.ubi_nom_array[-1, :, :], dtype=float) / factor
        return cls(
            r_p=float(ss_output["r_p"]),
            w=float(ss_output["w"]),
            p_tilde=p_tilde,
            p_i=p_i,
            bq=bq,
            rm=rm,
            tr=tr,
            ubi=ubi,
            factor=factor,
        )


@dataclass
class HouseholdSolution:
    """
    Household savings and labor supply from a partial-equilibrium solve.

    Attributes:
        b_sp1 (Numpy array): savings chosen at each age, shape (S, J)
        n (Numpy array): labor supply at each age, shape (S, J)
        euler_errors (Numpy array): savings and labor Euler equation
            errors, shape (2S, J)
        success (Numpy array): whether each type's root finder reported
            success, boolean of length J
    """

    b_sp1: np.ndarray
    n: np.ndarray
    euler_errors: np.ndarray
    success: np.ndarray

    @property
    def max_abs_euler_error(self) -> float:
        """
        Largest absolute Euler equation error across ages and types.

        Returns:
            max_error (float): largest absolute Euler equation error
        """
        return float(np.max(np.abs(self.euler_errors)))

    @property
    def all_converged(self) -> bool:
        """
        Whether every type's root finder reported success.

        Returns:
            converged (bool): True when all J types converged
        """
        return bool(np.all(self.success))


def _consumption_good_prices(p_m: np.ndarray, p) -> np.ndarray:
    """
    Computes consumption good prices from production good prices.

    Newer OG-Core releases expose ``aggregates.get_io_prices``; older ones
    compute the prices inline as ``io_matrix @ p_m``.

    Args:
        p_m (Numpy array): production good prices, length M
        p (OG-Core Specifications object): parameters object

    Returns:
        p_i (Numpy array): consumption good prices, length I
    """
    get_io_prices = getattr(aggr, "get_io_prices", None)
    if get_io_prices is not None:
        p_i, _, _ = get_io_prices(p_m, p, "SS")
        return np.atleast_1d(np.asarray(p_i, dtype=float))
    return np.atleast_1d(np.dot(p.io_matrix, p_m).astype(float))


def _scatter_params(p, client):
    """
    Scatters the parameters object to Dask workers once.

    Uses ``SS.scatter_params`` when the installed OG-Core provides it.
    Otherwise it mirrors the steady-state inner loop: the ParamTools schema
    objects are unpicklable, so they are detached before scattering and
    restored afterwards.

    Args:
        p (OG-Core Specifications object): parameters object
        client (Dask Client object): client

    Returns:
        scattered_p (Dask Future object): the parameters on the workers
    """
    scatter = getattr(SS, "scatter_params", None)
    if scatter is not None:
        return scatter(p, client)
    backup = {}
    for attr in ("_defaults_schema", "_validator_schema", "sel"):
        if hasattr(p, attr):
            backup[attr] = getattr(p, attr)
            try:
                delattr(p, attr)
            except Exception:  # pragma: no cover - defensive, as in OG-Core
                pass
    try:
        return client.scatter(p, broadcast=True)
    finally:
        for attr, value in backup.items():
            try:
                setattr(p, attr, value)
            except Exception:  # pragma: no cover - defensive, as in OG-Core
                pass


def _solve_one_type(env, p_or_future, guesses, j):
    """
    Calls ``SS.solve_for_j`` for one lifetime-income type.

    Args:
        env (HouseholdEnvironment): prices, transfers, and scaling factor
        p_or_future (OG-Core Specifications object or Dask Future
            object): parameters object, possibly already scattered to
            the workers
        guesses (Numpy array): stacked savings and labor guesses for type
            ``j``, length 2S
        j (int): lifetime-income type index

    Returns:
        result (SciPy OptimizeResult object): root-finder output with
            ``x`` (savings then labor) and ``fun`` (Euler errors)
    """
    return SS.solve_for_j(
        guesses,
        env.r_p,
        env.w,
        env.p_tilde,
        env.p_i,
        env.bq[:, j],
        env.rm[:, j],
        env.tr[:, j],
        env.ubi[:, j],
        env.factor,
        j,
        p_or_future,
    )


def solve_households(
    env: HouseholdEnvironment,
    p,
    b_guess: np.ndarray,
    n_guess: np.ndarray,
    client=None,
    scattered_p=None,
) -> HouseholdSolution:
    """
    Solves every type's lifecycle problem at a fixed environment.

    Args:
        env (HouseholdEnvironment): prices, transfers, and scaling factor
            held fixed
        p (OG-Core Specifications object): parameters object with the
            current preference parameters
        b_guess (Numpy array): initial savings guesses, shape (S, J)
        n_guess (Numpy array): initial labor guesses, shape (S, J)
        client (Dask Client object): client; types are solved in parallel
            when given, with a serial fallback if the parallel run fails
        scattered_p (Dask Future object): ``p`` already scattered to the
            workers, to avoid re-scattering on repeated calls

    Returns:
        solution (HouseholdSolution): savings, labor, Euler errors
            (2S x J), and per-type convergence flags
    """
    b_guess = np.asarray(b_guess, dtype=float)
    n_guess = np.asarray(n_guess, dtype=float)
    if b_guess.shape != (p.S, p.J) or n_guess.shape != (p.S, p.J):
        raise ValueError("b_guess and n_guess must have shape (S, J).")
    guesses = [np.append(b_guess[:, j], n_guess[:, j]) for j in range(p.J)]

    results = None
    if client is not None:
        try:
            p_future = (
                scattered_p
                if scattered_p is not None
                else _scatter_params(p, client)
            )
            futures = [
                client.submit(_solve_one_type, env, p_future, guesses[j], j)
                for j in range(p.J)
            ]
            results = client.gather(futures)
        except Exception as err:  # noqa: BLE001 - mirror OG-Core fallback
            logger.warning(
                "Parallel household solve failed (%s: %s); solving types "
                "serially.",
                type(err).__name__,
                err,
            )
            results = None
    if results is None:
        results = [_solve_one_type(env, p, guesses[j], j) for j in range(p.J)]

    b_sp1 = np.zeros((p.S, p.J))
    n = np.zeros((p.S, p.J))
    euler_errors = np.zeros((2 * p.S, p.J))
    success = np.zeros(p.J, dtype=bool)
    for j, result in enumerate(results):
        b_sp1[:, j] = result.x[: p.S]
        n[:, j] = result.x[p.S :]
        euler_errors[:, j] = result.fun
        success[j] = bool(getattr(result, "success", True))
    if not np.all(success):
        logger.warning(
            "Household root finder did not report success for types %s.",
            np.flatnonzero(~success).tolist(),
        )
    return HouseholdSolution(
        b_sp1=b_sp1, n=n, euler_errors=euler_errors, success=success
    )


def partial_equilibrium_ss(
    ss_output: dict,
    p,
    client=None,
    b_guess: np.ndarray | None = None,
    n_guess: np.ndarray | None = None,
    scattered_p=None,
) -> tuple[dict, HouseholdSolution]:
    """
    Re-solves the household block at the prices in ``ss_output``.

    Aggregates such as ``BQ``, ``Y``, and ``factor`` are left at their
    general-equilibrium values, so the result can be passed straight to
    ``ogusa.estimate_lifecycle_params.compute_model_moments``.

    Args:
        ss_output (dict): OG-Core steady-state output supplying prices,
            transfers, bequests, and the scaling factor
        p (OG-Core Specifications object): parameters object with the
            current preference parameters
        client (Dask Client object): client for parallel solves
        b_guess (Numpy array): initial savings guesses, shape (S, J);
            defaults to ``ss_output["b_sp1"]``
        n_guess (Numpy array): initial labor guesses, shape (S, J);
            defaults to ``ss_output["n"]``
        scattered_p (Dask Future object): ``p`` already scattered to the
            workers

    Returns:
        updated (dict): copy of ``ss_output`` with ``b_sp1``, ``b_s``,
            ``n``, and ``before_tax_income`` replaced by the new household
            solution
        solution (HouseholdSolution): Euler errors and convergence flags
    """
    env = HouseholdEnvironment.from_ss_output(ss_output, p)
    if b_guess is None:
        b_guess = np.asarray(ss_output["b_sp1"], dtype=float)
    if n_guess is None:
        n_guess = np.asarray(ss_output["n"], dtype=float)
    solution = solve_households(
        env, p, b_guess, n_guess, client=client, scattered_p=scattered_p
    )
    updated = dict(ss_output)
    updated["b_sp1"] = solution.b_sp1
    b_s = np.vstack([np.zeros((1, p.J)), solution.b_sp1[:-1, :]])
    updated["b_s"] = b_s
    updated["n"] = solution.n
    updated["before_tax_income"] = np.asarray(
        household.get_y(env.r_p, env.w, b_s, solution.n, p, "SS"),
        dtype=float,
    )
    return updated, solution


# ---------------------------------------------------------------------------
# Phase 3: concentrate out chi_n by inverting the labor first-order condition
# ---------------------------------------------------------------------------


@dataclass
class ChiNInversionResult:
    """
    Outcome of the chi_n inversion at fixed prices.

    Attributes:
        chi_n (Numpy array): steady-state chi_n by age, length S
        ages (Numpy array): ages at which hours were targeted
        labor_model (Numpy array): population-weighted model hours at
            ``ages``
        labor_target (Numpy array): target hours at ``ages``
        iterations (int): number of inversion steps taken
        converged (bool): whether the log gap fell below the tolerance
        max_abs_log_gap (float): largest absolute log gap between model and
            target hours at the end
        history (list): largest absolute log gap after each step
        capped_ages (Numpy array): ages where chi_n hit a bound
        ss_output (dict): household solution at the final chi_n
        solution (HouseholdSolution): Euler errors and convergence flags
    """

    chi_n: np.ndarray
    ages: np.ndarray
    labor_model: np.ndarray
    labor_target: np.ndarray
    iterations: int
    converged: bool
    max_abs_log_gap: float
    history: list
    capped_ages: np.ndarray
    ss_output: dict
    solution: HouseholdSolution


def aggregate_labor_by_age(n: np.ndarray, p, ages: np.ndarray) -> np.ndarray:
    """
    Computes population-weighted mean labor supply at each requested age.

    Args:
        n (Numpy array): labor supply by age and type, shape (S, J)
        p (OG-Core Specifications object): parameters object
        ages (Numpy array): ages to aggregate over

    Returns:
        labor (Numpy array): mean labor supply at each age in ``ages``
    """
    from ogusa import estimate_lifecycle_params as elp

    n = np.asarray(n, dtype=float)
    weights = elp._type_weights_by_age(p)
    idx = elp._age_indices(np.asarray(ages), p)
    return (n[idx, :] * weights[idx, :]).sum(axis=1)


def chi_n_update(
    chi_n_values: np.ndarray,
    labor_model: np.ndarray,
    labor_target: np.ndarray,
    p,
    damping: float = 1.0,
) -> np.ndarray:
    """
    Takes one inversion step of the labor first-order condition.

    The steady-state labor FOC is ``chi_n[s] * MDU(n) = LHS[s]`` where the
    right-hand side depends on wages, taxes, and consumption.  Holding that
    side fixed, the ``chi_n`` that delivers the target hours is
    ``chi_n * MDU(n_model) / MDU(n_target)``.  ``damping`` raises the ratio
    to a power: 1 is the full step, below 1 damps, above 1 over-relaxes to
    offset the consumption response that makes hours move less than the
    fixed-LHS step predicts.

    Args:
        chi_n_values (Numpy array): current chi_n at the target ages
        labor_model (Numpy array): model hours at the target ages
        labor_target (Numpy array): target hours at the target ages
        p (OG-Core Specifications object): parameters object (supplies
            ``ltilde`` and ``upsilon``)
        damping (float): exponent on the update ratio

    Returns:
        chi_n_new (Numpy array): updated chi_n at the target ages
    """
    chi_n_values = np.asarray(chi_n_values, dtype=float)
    ratio = household.marg_ut_labor(
        np.asarray(labor_model, dtype=float), 1.0, p
    ) / household.marg_ut_labor(np.asarray(labor_target, dtype=float), 1.0, p)
    return chi_n_values * np.asarray(ratio, dtype=float) ** damping


def apply_chi_n(p, chi_n: np.ndarray) -> None:
    """
    Sets the steady-state chi_n age profile on the parameters object.

    Args:
        p (OG-Core Specifications object): parameters object
        chi_n (Numpy array): chi_n by age, length S

    Returns:
        None
    """
    p.update_specifications(
        {"chi_n": np.asarray(chi_n, dtype=float).reshape(-1).tolist()}
    )


def _chi_n_bounds(p, chi_n_min: float | None, chi_n_max: float | None):
    """
    Returns bounds for chi_n from the validators unless overridden.

    Args:
        p (OG-Core Specifications object): parameters object
        chi_n_min (float or None): lower bound override
        chi_n_max (float or None): upper bound override

    Returns:
        lower (float): lower bound on chi_n
        upper (float): upper bound on chi_n
    """
    from ogusa import estimate_lifecycle_params as elp

    lo, hi = elp._validator_range(p, "chi_n")
    lo = max(lo, 1e-8) if chi_n_min is None else float(chi_n_min)
    hi = hi if chi_n_max is None else float(chi_n_max)
    return lo, hi


def invert_chi_n(
    ss_output: dict,
    p,
    labor_target: np.ndarray,
    config=None,
    max_iter: int = 30,
    tol: float = 1e-3,
    damping: float = 1.0,
    chi_n_min: float | None = None,
    chi_n_max: float | None = None,
    client=None,
) -> ChiNInversionResult:
    """
    Chooses chi_n by age so model hours match ``labor_target`` at fixed prices.

    Iterates: solve the household block at the prices in ``ss_output``,
    form population-weighted hours at each target age, update ``chi_n`` at
    those ages with ``chi_n_update``, and repeat until the largest absolute
    log gap between model and target hours is below ``tol``.  Ages above the
    last target age are filled with
    ``ogusa.estimate_lifecycle_params.build_chi_n_profile`` using
    ``config.chi_n_tail_method`` (default: the initial profile's tail
    rescaled to join the last estimated value).  Values are clipped to the
    ParamTools range for ``chi_n`` unless narrower bounds are given, and the
    ages where the cap binds are reported.

    On return ``p`` carries the final ``chi_n`` and ``result.ss_output`` is
    the household solution at that profile, ready for
    ``ogusa.estimate_lifecycle_params.compute_model_moments``.

    Args:
        ss_output (dict): OG-Core steady-state output supplying prices
        p (OG-Core Specifications object): parameters object; updated in place
        labor_target (Numpy array): target hours at ``config.moment_ages``
        config (LifecycleCalibrationConfig): moment configuration; default
            configuration when None
        max_iter (int): maximum number of inversion steps
        tol (float): convergence tolerance on the absolute log gap
        damping (float): exponent on the update ratio in ``chi_n_update``
        chi_n_min (float or None): lower bound override for chi_n
        chi_n_max (float or None): upper bound override for chi_n
        client (Dask Client object): client for household solves

    Returns:
        result (ChiNInversionResult): calibrated profile, fit, and the
            household solution at the final chi_n
    """
    from dataclasses import replace

    from ogusa import estimate_lifecycle_params as elp

    if config is None:
        config = elp.LifecycleCalibrationConfig()
    config = replace(
        config,
        estimate_chi_n_min_age=config.min_age,
        estimate_chi_n_max_age=config.max_age,
    )
    config.validate(p)
    ages = config.moment_ages
    labor_target = np.asarray(labor_target, dtype=float).reshape(-1)
    if labor_target.size != ages.size:
        raise ValueError("labor_target must have one value per target age.")
    if np.any(labor_target <= 0) or np.any(labor_target >= p.ltilde):
        raise ValueError("labor_target must lie strictly inside (0, ltilde).")
    lo, hi = _chi_n_bounds(p, chi_n_min, chi_n_max)

    base_chi_n = elp._ss_chi_n(p)
    est_idx = elp._age_indices(ages, p)
    chi_n = base_chi_n.copy()
    b_guess = np.asarray(ss_output["b_sp1"], dtype=float)
    n_guess = np.asarray(ss_output["n"], dtype=float)
    history: list[float] = []
    converged = False
    capped = np.zeros(ages.size, dtype=bool)
    updated = ss_output
    solution = None
    labor_model = np.full(ages.size, np.nan)
    iterations = 0

    for iterations in range(1, max_iter + 1):
        apply_chi_n(p, chi_n)
        updated, solution = partial_equilibrium_ss(
            ss_output, p, client=client, b_guess=b_guess, n_guess=n_guess
        )
        labor_model = aggregate_labor_by_age(updated["n"], p, ages)
        gap = np.log(labor_model / labor_target)
        max_gap = float(np.max(np.abs(gap)))
        history.append(max_gap)
        logger.info(
            "chi_n inversion iteration %d: max |log(n_model/n_target)| = %.3e",
            iterations,
            max_gap,
        )
        if max_gap < tol:
            converged = True
            break
        if iterations == max_iter:
            break
        new_values = chi_n_update(
            chi_n[est_idx], labor_model, labor_target, p, damping=damping
        )
        capped = new_values >= hi
        new_values = np.clip(new_values, lo, hi)
        chi_n = elp.build_chi_n_profile(new_values, base_chi_n, p, config)
        chi_n = np.clip(chi_n, lo, hi)
        b_guess, n_guess = solution.b_sp1, solution.n

    if not converged:
        logger.warning(
            "chi_n inversion did not converge in %d iterations "
            "(max |log gap| = %.3e).",
            iterations,
            history[-1],
        )
    if np.any(capped):
        logger.warning(
            "chi_n hit its upper bound %.0f at ages %s.",
            hi,
            ages[capped].tolist(),
        )
    return ChiNInversionResult(
        chi_n=chi_n,
        ages=ages,
        labor_model=labor_model,
        labor_target=labor_target,
        iterations=iterations,
        converged=converged,
        max_abs_log_gap=history[-1],
        history=history,
        capped_ages=ages[capped],
        ss_output=updated,
        solution=solution,
    )


# ---------------------------------------------------------------------------
# Phase 4: beta by type and chi_b from wealth shares, level, and bequest flow
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PreferenceCalibrationOptions:
    """
    Options for the beta / chi_b calibration at fixed prices.

    Attributes:
        chi_b_mode (str): ``"common_scale"`` moves every type's ``chi_b``
            by one common factor, identified by the aggregate bequest-flow
            ratio. ``"by_type"`` gives each type group its own ``chi_b``
            factor, identified by the old-age wealth tilt of the matching
            wealth percentile bin, with the bequest-flow ratio as an
            additional residual.
        bottom_share (float): types whose cumulative population share lies
            in the bottom ``bottom_share`` form the bottom group
        top_share (float): types inside the top ``top_share`` share one
            ``chi_b`` factor in ``by_type`` mode (each still has its own
            ``beta``)
        exclude_bottom (bool): when True (default), the bottom group's
            wealth-share and tilt bins are dropped from the targets and the
            bottom types share their ``beta`` (and ``chi_b``) factor with
            the next type up.  A deterministic model with no within-type
            heterogeneity cannot deliver the SCF bottom-half share of about
            one percent, because young households of every type fill the
            bottom percentiles, so targeting it drives the bottom betas to
            zero.  When False the bottom group gets its own factor and one
            merged share target.
        beta_annual_max (float or None): ceiling on beta tighter than the
            ParamTools validator (0.9999)
        chi_b_max (float or None): ceiling on chi_b tighter than the
            ParamTools validator (10,000)
        bequest_flow_weight (float): weight on the bequest-flow residual
        failure_residual (float): value of every residual when the
            household solve fails to converge, in log units
        max_nfev (int): cap on least-squares function evaluations counted
            by SciPy; Jacobian columns are not counted, so total household
            solves are about ``max_nfev * (1 + n_params)``
        diff_step (float): absolute finite-difference step for the
            Jacobian, in the transformed parameter space (shifts to
            ``logit(beta)`` and ``log(chi_b)``).  Every column is
            differenced from the same household guess so solver noise does
            not enter the derivatives.  SciPy's own ``diff_step`` is
            relative to the parameter value and would fall back to about
            1.5e-8 here because the parameterization starts at zero.
        ftol (float): least-squares cost tolerance
        xtol (float): least-squares parameter tolerance
    """

    chi_b_mode: str = "by_type"
    bottom_share: float = 0.5
    top_share: float = 0.01
    exclude_bottom: bool = True
    beta_annual_max: float | None = None
    chi_b_max: float | None = None
    bequest_flow_weight: float = 1.0
    failure_residual: float = 3.0
    max_nfev: int = 150
    diff_step: float = 1e-3
    ftol: float = 1e-8
    xtol: float = 1e-8


@dataclass
class PreferenceCalibrationResult:
    """
    Outcome of the beta / chi_b calibration at fixed prices.

    Attributes:
        beta_annual (Numpy array): calibrated beta by type, length J
        chi_b (Numpy array): calibrated chi_b by type, length J
        theta (Numpy array): free parameters at the solution (transformed
            shifts)
        residuals (Numpy array): weighted log residuals at the solution
        residual_names (tuple): target names, one per residual
        data_values (Numpy array): data value of each target
        model_values (Numpy array): model value of each target
        cost (float): least-squares cost, half the sum of squared residuals
        nfev (int): number of household solves used
        success (bool): SciPy's convergence flag
        message (str): SciPy's termination message
        ss_output (dict): household solution at the calibrated parameters
        solution (HouseholdSolution or None): Euler errors and flags
        jacobian (Numpy array or None): Jacobian of the residuals at the
            solution, shape (targets, free parameters)
        weights (Numpy array or None): residual weights, one per target
    """

    beta_annual: np.ndarray
    chi_b: np.ndarray
    theta: np.ndarray
    residuals: np.ndarray
    residual_names: tuple
    data_values: np.ndarray
    model_values: np.ndarray
    cost: float
    nfev: int
    success: bool
    message: str
    ss_output: dict
    solution: HouseholdSolution | None
    jacobian: np.ndarray | None = None
    weights: np.ndarray | None = None

    def to_frame(self):
        """
        Tabulates data, model, and log residual for each target.

        Returns:
            frame (Pandas DataFrame): columns ``target``, ``data``, ``model``,
                ``log_residual``
        """
        import pandas as pd

        return pd.DataFrame(
            {
                "target": self.residual_names,
                "data": self.data_values,
                "model": self.model_values,
                "log_residual": self.residuals,
            }
        )


def _group_bounds(base, groups, lo, hi, transform):
    """
    Computes per-group bounds on an additive shift in transformed space.

    Args:
        base (Numpy array): base parameter values by type
        groups (list): lists of type indices sharing one shift
        lo (float): lower bound on the parameter in levels
        hi (float): upper bound on the parameter in levels
        transform (function): map from levels to the transformed space

    Returns:
        lower (Numpy array): lower bound on each group's shift
        upper (Numpy array): upper bound on each group's shift
    """
    lower = np.empty(len(groups))
    upper = np.empty(len(groups))
    for g, members in enumerate(groups):
        t_base = transform(base[members])
        lower[g] = np.max(transform(lo) - t_base)
        upper[g] = np.min(transform(hi) - t_base)
    return lower, upper


def _logit(x):
    """
    Logit transform.

    Args:
        x (Numpy array or float): values in (0, 1)

    Returns:
        z (Numpy array or float): ``log(x / (1 - x))``
    """
    x = np.asarray(x, dtype=float)
    return np.log(x / (1.0 - x))


def _logistic(z):
    """
    Logistic transform, the inverse of ``_logit``.

    Args:
        z (Numpy array or float): real values

    Returns:
        x (Numpy array or float): ``1 / (1 + exp(-z))``
    """
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


class _PreferenceParameterization:
    """
    Maps a free parameter vector to beta_annual and chi_b by type.

    The free parameters are additive shifts to ``logit(beta_annual)`` for
    each beta group and to ``log(chi_b)`` for each chi_b group, relative to
    the values on the parameters object when the instance was created.

    Attributes:
        base_beta (Numpy array): starting beta by type, clipped to bounds
        base_chi_b (Numpy array): starting chi_b by type, clipped to bounds
        beta_groups (list): lists of type indices sharing one beta shift
        chi_b_groups (list): lists of type indices sharing one chi_b shift
        n_beta (int): number of beta groups
        n_chi_b (int): number of chi_b groups
        beta_bounds (tuple): (lower, upper) bounds on beta in levels
        chi_b_bounds (tuple): (lower, upper) bounds on chi_b in levels
        lower (Numpy array): lower bounds on the free parameters
        upper (Numpy array): upper bounds on the free parameters
    """

    def __init__(self, p, options: PreferenceCalibrationOptions):
        """
        Builds the parameterization from the current parameters and options.

        Args:
            p (OG-Core Specifications object): parameters object with
                the starting beta and chi_b
            options (PreferenceCalibrationOptions): grouping and bound options

        Returns:
            None
        """
        from ogusa import estimate_lifecycle_params as elp

        self.base_beta = np.asarray(p.beta_annual, dtype=float).copy()
        self.base_chi_b = np.asarray(p.chi_b, dtype=float).copy()
        lambdas = elp._lambdas(p)
        groups = _type_groups(lambdas, options)
        bottom = groups[0]
        # beta: the bottom group shares one factor, every other type is free.
        self.beta_groups = [bottom] + [
            [j] for j in range(p.J) if j not in bottom
        ]
        if options.chi_b_mode == "common_scale":
            self.chi_b_groups = [list(range(p.J))]
        elif options.chi_b_mode == "by_type":
            self.chi_b_groups = groups
        else:
            raise ValueError(f"Unsupported chi_b_mode: {options.chi_b_mode}")
        self.n_beta = len(self.beta_groups)
        self.n_chi_b = len(self.chi_b_groups)
        eps = 1e-4
        b_lo, b_hi = elp._validator_range(p, "beta_annual")
        b_lo, b_hi = max(b_lo, eps), min(b_hi, 1.0 - eps)
        if options.beta_annual_max is not None:
            b_hi = min(b_hi, float(options.beta_annual_max))
        c_lo, c_hi = elp._validator_range(p, "chi_b")
        c_lo = max(c_lo, eps)
        if options.chi_b_max is not None:
            c_hi = min(c_hi, float(options.chi_b_max))
        # Base values may already sit on a bound; keep them inside it.
        self.base_beta = np.clip(self.base_beta, b_lo, b_hi)
        self.base_chi_b = np.clip(self.base_chi_b, c_lo, c_hi)
        self.beta_bounds = (b_lo, b_hi)
        self.chi_b_bounds = (c_lo, c_hi)
        beta_lo, beta_hi = _group_bounds(
            self.base_beta, self.beta_groups, b_lo, b_hi, _logit
        )
        chi_lo, chi_hi = _group_bounds(
            self.base_chi_b, self.chi_b_groups, c_lo, c_hi, np.log
        )
        self.lower = np.concatenate([beta_lo, chi_lo])
        self.upper = np.concatenate([beta_hi, chi_hi])

    @property
    def size(self) -> int:
        """
        Number of free parameters.

        Returns:
            size (int): beta groups plus chi_b groups
        """
        return self.n_beta + self.n_chi_b

    def unpack(self, theta):
        """
        Maps free parameters to beta_annual and chi_b by type.

        Round-tripping through logit/log at a bound can overshoot it by
        floating-point error, which ParamTools rejects, so the results are
        clipped to the bounds.

        Args:
            theta (Numpy array): free parameters, length ``size``

        Returns:
            beta (Numpy array): beta_annual by type, length J
            chi_b (Numpy array): chi_b by type, length J
        """
        theta = np.asarray(theta, dtype=float)
        beta = self.base_beta.copy()
        for g, members in enumerate(self.beta_groups):
            beta[members] = _logistic(
                _logit(self.base_beta[members]) + theta[g]
            )
        chi_b = self.base_chi_b.copy()
        for g, members in enumerate(self.chi_b_groups):
            chi_b[members] = self.base_chi_b[members] * np.exp(
                theta[self.n_beta + g]
            )
        # Round-tripping through logit/log at a bound can overshoot it by
        # floating-point error, which ParamTools rejects; clip to be safe.
        beta = np.clip(beta, *self.beta_bounds)
        chi_b = np.clip(chi_b, *self.chi_b_bounds)
        return beta, chi_b


def _type_groups(lambdas, options: PreferenceCalibrationOptions):
    """
    Forms the type groups for the parameters.

    The bottom types are merged (plus the next type when the bottom bin is
    excluded from the targets), the top types are merged, and every other
    type stands alone.

    Args:
        lambdas (Numpy array): population share of each type
        options (PreferenceCalibrationOptions): grouping options

    Returns:
        groups (list): lists of type indices, in order from bottom to top
    """
    from ogusa import estimate_lifecycle_params as elp

    groups = elp.merged_type_groups(
        lambdas, options.bottom_share, options.top_share
    )
    if options.exclude_bottom and len(groups) > 1:
        bottom = groups[0] + groups[1]
        groups = [bottom] + groups[2:]
    return groups


def _merge_bins(values, groups):
    """
    Sums per-type values over groups.

    Args:
        values (Numpy array): one value per type
        groups (list): lists of type indices

    Returns:
        merged (Numpy array): one sum per group
    """
    values = np.asarray(values, dtype=float)
    return np.array([values[g].sum() for g in groups])


def preference_targets(
    data_moments, p, config, options: PreferenceCalibrationOptions
):
    """
    Selects and merges the data moments the beta / chi_b calibration uses.

    Targets are: wealth shares by type bin (the bottom group's bins merged
    into one target, or dropped when ``options.exclude_bottom``), the
    wealth-to-income ratio, the by-bin old-age tilts in ``by_type`` mode
    (bottom tilt bin dropped when ``exclude_bottom``), and the bequest-flow
    ratio.

    Args:
        data_moments (MomentSet): full data moment vector
        p (OG-Core Specifications object): parameters object
        config (LifecycleCalibrationConfig): moment configuration
        options (PreferenceCalibrationOptions): grouping options

    Returns:
        names (tuple): target names
        values (Numpy array): target values
        selection (dict): ``share_bins`` (lists of type indices per share
            target) and ``tilt_idx`` (indices of the tilt bins used), needed
            to compute matching model values
    """
    from ogusa import estimate_lifecycle_params as elp

    lookup = dict(zip(data_moments.names, data_moments.values))
    lambdas = elp._lambdas(p)
    share_names = elp.wealth_share_bin_names(lambdas)
    shares = np.array([lookup[name] for name in share_names])
    raw_groups = elp.merged_type_groups(
        lambdas, options.bottom_share, options.top_share
    )
    bottom = raw_groups[0]
    if options.exclude_bottom:
        share_bins = [[j] for j in range(p.J) if j not in bottom]
    else:
        share_bins = [bottom] + [[j] for j in range(p.J) if j not in bottom]
    names = []
    values = []
    cum = np.cumsum(lambdas)
    for members in share_bins:
        if len(members) == 1:
            names.append(share_names[members[0]])
        else:
            lo = _as_pct(cum[members[0]] - lambdas[members[0]])
            hi = _as_pct(cum[members[-1]])
            names.append(f"wealth_share_{lo}_{hi}")
        values.append(shares[members].sum())
    names.append("wealth_income_ratio")
    values.append(lookup["wealth_income_ratio"])
    tilt_idx = []
    if options.chi_b_mode == "by_type":
        tilt_names = elp.tilt_moment_names(config, p)
        start = 1 if options.exclude_bottom else 0
        for k in range(start, len(tilt_names)):
            names.append(tilt_names[k])
            values.append(lookup[tilt_names[k]])
            tilt_idx.append(k)
    names.append("bequest_flow_ratio")
    values.append(lookup["bequest_flow_ratio"])
    selection = {"share_bins": share_bins, "tilt_idx": tilt_idx}
    return tuple(names), np.asarray(values, dtype=float), selection


def _as_pct(share):
    """
    Formats a population share as a percentile label.

    Args:
        share (float): cumulative population share

    Returns:
        label (str): percentile label such as ``"99p5"``
    """
    from ogusa import estimate_lifecycle_params as elp

    return elp._percent_label(share)


def _preference_model_values(ss_output, p, config, options, selection):
    """
    Computes the model values matching ``preference_targets``.

    Args:
        ss_output (dict): household solution
        p (OG-Core Specifications object): parameters object
        config (LifecycleCalibrationConfig): moment configuration
        options (PreferenceCalibrationOptions): grouping options
        selection (dict): the selection returned by ``preference_targets``

    Returns:
        values (Numpy array): model value of each target
    """
    from ogusa import estimate_lifecycle_params as elp

    shares = elp.model_wealth_shares(ss_output, p)
    values = list(_merge_bins(shares, selection["share_bins"]))
    values.append(elp.model_wealth_income_ratio(ss_output, p))
    if options.chi_b_mode == "by_type":
        tilt = elp.model_old_age_ratio_by_type(ss_output, p, config)
        values.extend(tilt[selection["tilt_idx"]])
    values.append(elp.model_bequest_flow_ratio(ss_output, p))
    return np.asarray(values, dtype=float)


def _preference_weights(n_targets, options):
    """
    Builds the residual weights for the beta / chi_b targets.

    Args:
        n_targets (int): number of targets
        options (PreferenceCalibrationOptions): supplies the bequest-flow
            weight, applied to the last target

    Returns:
        weights (Numpy array): one weight per target
    """
    weights = np.ones(n_targets)
    weights[-1] = options.bequest_flow_weight
    return weights


def calibrate_beta_chi_b(
    ss_output: dict,
    p,
    data_moments,
    config=None,
    options: PreferenceCalibrationOptions | None = None,
    client=None,
) -> PreferenceCalibrationResult:
    """
    Calibrates beta by type and chi_b at fixed prices.

    Solves a bounded nonlinear least-squares problem over additive shifts
    to ``logit(beta_annual)`` by type (bottom types tied) and to
    ``log(chi_b)`` by group, with log residuals between model and data for
    the merged wealth shares, the wealth-to-income ratio, and the bequest
    flow ratio (plus by-bin old-age tilts in ``by_type`` mode).  Every
    residual evaluation is a household-only solve at the prices in
    ``ss_output``; failed solves return ``options.failure_residual``.  The
    Jacobian is formed by forward differences with the absolute step
    ``options.diff_step``, every column from the same household guess.

    On return ``p`` carries the calibrated parameters and
    ``result.ss_output`` the matching household solution.

    Args:
        ss_output (dict): household solution at the current chi_n whose
            ``b_sp1`` and ``n`` serve as starting guesses (for example the
            ``ss_output`` returned by ``invert_chi_n``)
        p (OG-Core Specifications object): parameters object; updated in place
        data_moments (MomentSet): data moments including the wealth shares,
            ``wealth_income_ratio``, ``bequest_flow_ratio``, and tilts
        config (LifecycleCalibrationConfig): moment configuration; default
            configuration when None
        options (PreferenceCalibrationOptions): calibration options;
            defaults when None
        client (Dask Client object): client for household solves

    Returns:
        result (PreferenceCalibrationResult): calibrated parameters, fit,
            Jacobian, and the household solution
    """
    from scipy import optimize

    from ogusa import estimate_lifecycle_params as elp

    if config is None:
        config = elp.LifecycleCalibrationConfig()
    if options is None:
        options = PreferenceCalibrationOptions()
    if (
        options.chi_b_mode == "by_type"
        and not config.include_old_age_ratio_by_type
    ):
        raise ValueError(
            "chi_b_mode='by_type' needs config.include_old_age_ratio_by_type."
        )
    for needed in ("wealth_income_ratio", "bequest_flow_ratio"):
        if needed not in data_moments.names:
            raise ValueError(f"data_moments must include {needed}.")

    names, data_values, selection = preference_targets(
        data_moments, p, config, options
    )
    weights = _preference_weights(len(names), options)
    param = _PreferenceParameterization(p, options)
    state = {
        "b_guess": np.asarray(ss_output["b_sp1"], dtype=float),
        "n_guess": np.asarray(ss_output["n"], dtype=float),
        "last": None,
        "nfev": 0,
    }

    initial_guesses = (state["b_guess"], state["n_guess"])

    def _solve(theta):
        """
        Solves the household block at ``theta``, retrying from the initial
        guesses when the warm guess fails.

        Args:
            theta (Numpy array): free parameters

        Returns:
            updated (dict): household solution as ``ss_output``
            solution (HouseholdSolution): Euler errors and convergence flags
        """
        beta, chi_b = param.unpack(theta)
        p.update_specifications(
            {"beta_annual": beta.tolist(), "chi_b": chi_b.tolist()}
        )
        guess_sets = [(state["b_guess"], state["n_guess"])]
        if state["b_guess"] is not initial_guesses[0]:
            guess_sets.append(initial_guesses)
        for b_guess, n_guess in guess_sets:
            updated, solution = partial_equilibrium_ss(
                ss_output, p, client=client, b_guess=b_guess, n_guess=n_guess
            )
            state["nfev"] += 1
            if solution.all_converged:
                return updated, solution
        return updated, solution

    updated0, solution0 = _solve(theta0 := np.zeros(param.size))
    if not solution0.all_converged:
        raise RuntimeError(
            "The household solve did not converge at the starting parameters "
            "with the guesses in ss_output. Pass an ss_output whose b_sp1 "
            "and n were solved at the current chi_n (for example the "
            "ss_output returned by invert_chi_n)."
        )

    def residuals(theta):
        """
        Weighted log residuals at ``theta``, with a penalty on failed solves.

        Args:
            theta (Numpy array): free parameters

        Returns:
            res (Numpy array): one clipped residual per target
        """
        theta = np.asarray(theta, dtype=float)
        cached = state.get("cache")
        if cached is not None and np.array_equal(cached[0], theta):
            return cached[1].copy()
        updated, solution = _solve(theta)
        if not solution.all_converged:
            logger.warning(
                "Household solve failed at theta=%s; penalizing.",
                np.round(theta, 4).tolist(),
            )
            res = np.full(len(names), options.failure_residual)
        else:
            state["b_guess"], state["n_guess"] = solution.b_sp1, solution.n
            model_values = _preference_model_values(
                updated, p, config, options, selection
            )
            state["last"] = (updated, solution, model_values)
            with np.errstate(divide="ignore", invalid="ignore"):
                res = weights * np.log(model_values / data_values)
            if not np.all(np.isfinite(res)):
                res = np.full(len(names), options.failure_residual)
            res = np.clip(
                res, -options.failure_residual, options.failure_residual
            )
        state["cache"] = (theta.copy(), res.copy())
        return res

    def jacobian(theta):
        """
        Forward-difference Jacobian with an absolute step from one guess.

        Args:
            theta (Numpy array): free parameters

        Returns:
            J (Numpy array): derivatives of the residuals, shape (targets,
                free parameters)
        """
        theta = np.asarray(theta, dtype=float)
        f0 = residuals(theta)
        b0, n0, last0 = state["b_guess"], state["n_guess"], state["last"]
        J = np.empty((len(names), theta.size))
        for k in range(theta.size):
            h = float(options.diff_step)
            if theta[k] + h > param.upper[k]:
                h = -h
            shifted = theta.copy()
            shifted[k] += h
            state["b_guess"], state["n_guess"] = b0, n0
            J[:, k] = (residuals(shifted) - f0) / h
        state["b_guess"], state["n_guess"], state["last"] = b0, n0, last0
        state["cache"] = (theta.copy(), f0.copy())
        return J

    result = optimize.least_squares(
        residuals,
        theta0,
        jac=jacobian,
        bounds=(param.lower, param.upper),
        method="trf",
        max_nfev=options.max_nfev,
        ftol=options.ftol,
        xtol=options.xtol,
    )
    # Re-evaluate at the solution so p and the cached state match result.x.
    final_res = residuals(result.x)
    beta, chi_b = param.unpack(result.x)
    if state["last"] is None:
        raise RuntimeError("No converged household solve during calibration.")
    updated, solution, model_values = state["last"]
    logger.info(
        "beta/chi_b calibration: cost=%.3e, nfev=%d, success=%s, %s",
        result.cost,
        state["nfev"],
        result.success,
        result.message,
    )
    return PreferenceCalibrationResult(
        beta_annual=beta,
        chi_b=chi_b,
        theta=result.x,
        residuals=final_res,
        residual_names=names,
        data_values=data_values,
        model_values=model_values,
        cost=float(result.cost),
        nfev=state["nfev"],
        success=bool(result.success),
        message=str(result.message),
        ss_output=updated,
        solution=solution,
        jacobian=np.asarray(result.jac, dtype=float),
        weights=weights,
    )


# ---------------------------------------------------------------------------
# Phase 5: outer general-equilibrium loop
# ---------------------------------------------------------------------------


def _ss_solver_has_G() -> bool:
    """
    Whether the installed OG-Core steady-state solver carries G.

    Returns:
        has_G (bool): True when ``SS.SS_solver`` takes a ``G`` argument
    """
    import inspect

    return "G" in inspect.signature(SS.SS_solver).parameters


def _ss_guesses_from_solution(previous: dict, p) -> list:
    """
    Builds the outer-loop guess vector for ``SS.SS_fsolve`` from a prior
    solution.

    Layout follows ``SS.run_SS`` for a baseline solve: ``[r_p, r, w]``,
    then ``p_m``, ``Y``, the bequest items, ``G`` on OG-Core versions whose
    solver carries it, ``TR``, and ``factor``.

    Args:
        previous (dict): earlier OG-Core steady-state output
        p (OG-Core Specifications object): parameters object

    Returns:
        guesses (list): outer-loop guess vector
    """
    BQ = np.atleast_1d(np.asarray(previous["BQ"], dtype=float))
    bq_items = [float(BQ.sum())] if p.use_zeta else BQ.tolist()
    guesses = (
        [float(previous["r_p"]), float(previous["r"]), float(previous["w"])]
        + np.atleast_1d(np.asarray(previous["p_m"], dtype=float)).tolist()
        + [float(previous["Y"])]
        + bq_items
    )
    if _ss_solver_has_G():
        guesses.append(float(previous["G"]))
    guesses.append(float(previous["TR"]))
    guesses.append(float(previous["factor"]))
    return guesses


def _unpack_ss_solution(x: np.ndarray, p) -> dict:
    """
    Splits the root-finder solution into named outer-loop variables.

    Args:
        x (Numpy array): solution vector in the layout of
            ``_ss_guesses_from_solution``
        p (OG-Core Specifications object): parameters object

    Returns:
        vals (dict): ``r_p``, ``r``, ``w``, ``p_m``, ``Y``, ``BQ``, ``TR``,
            ``factor``, and ``G`` when the solver carries it
    """
    x = np.asarray(x, dtype=float)
    has_G = _ss_solver_has_G()
    out = {
        "r_p": float(x[0]),
        "r": float(x[1]),
        "w": float(x[2]),
        "p_m": x[3 : 3 + p.M],
        "Y": float(x[3 + p.M]),
    }
    tail = 3 if has_G else 2
    out["BQ"] = x[3 + p.M + 1 : -tail]
    if has_G:
        out["G"] = float(x[-3])
    out["TR"] = float(x[-2])
    out["factor"] = float(x[-1])
    if not p.budget_balance and not p.baseline_spending:
        out["Y"] = out["TR"] / p.alpha_T[-1]
    return out


def solve_ge_steady_state(
    p, previous: dict | None = None, client=None
) -> dict:
    """
    Solves the baseline general-equilibrium steady state.

    With ``previous`` (an earlier OG-Core steady-state output), the outer
    root finder starts from that solution's prices, aggregates, and
    household arrays instead of OG-Core's cold guesses, which matters when
    the calibrated parameters move the equilibrium far from the defaults.
    If the warm start fails to converge the solve falls back to
    ``SS.run_SS``.  Serial solves (``client=None``) are much faster than
    Dask on a single machine because the steady state is dominated by
    parameter-scattering overhead.

    Args:
        p (OG-Core Specifications object): parameters object with
            ``baseline=True``
        previous (dict): earlier steady-state output for the warm start;
            None for a cold solve
        client (Dask Client object): client

    Returns:
        ss_output (dict): OG-Core steady-state output
    """
    from scipy import optimize

    if not p.baseline:
        raise ValueError("solve_ge_steady_state supports baseline solves.")
    if previous is None or p.baseline_spending:
        return SS.run_SS(p, client=client)

    guesses = _ss_guesses_from_solution(previous, p)
    b_guess = np.asarray(previous["b_sp1"], dtype=float)
    n_guess = np.asarray(previous["n"], dtype=float)
    args = [b_guess, n_guess, None, None, None, p, client]
    if _ss_solver_has_G():
        args.append(None)  # scattered_p
    try:
        sol = optimize.root(
            SS.SS_fsolve,
            guesses,
            args=tuple(args),
            method=p.SS_root_method,
            tol=p.mindist_SS,
        )
    except _WARM_START_ERRORS as err:
        logger.warning(
            "Warm-started GE solve raised %s: %s; using SS.run_SS.",
            type(err).__name__,
            err,
        )
        return SS.run_SS(p, client=client)
    if not sol.success:
        logger.warning(
            "Warm-started GE solve did not converge (%s); using SS.run_SS.",
            sol.message,
        )
        return SS.run_SS(p, client=client)

    vals = _unpack_ss_solution(sol.x, p)
    kwargs = {
        "bmat": b_guess,
        "nmat": n_guess,
        "r_p": vals["r_p"],
        "r": vals["r"],
        "w": vals["w"],
        "p_m": vals["p_m"],
        "Y": vals["Y"],
        "BQ": vals["BQ"],
        "TR": vals["TR"],
        "Ig_baseline": None,
        "factor": vals["factor"],
        "p": p,
        "client": client,
        "fsolve_flag": True,
    }
    if "G" in vals:
        kwargs["G"] = vals["G"]
    return SS.SS_solver(**kwargs)


_WARM_START_ERRORS = (
    AssertionError,
    FloatingPointError,
    KeyError,
    RuntimeError,
    TypeError,
    ValueError,
)


@dataclass
class OuterIterationRecord:
    """
    Diagnostics for one pass of the outer calibration loop.

    Attributes:
        iteration (int): pass number, starting at 1
        param_change (float): largest absolute change in the transformed
            parameters over the pass
        price_change (float): largest relative change in the outer-loop
            prices after the GE re-solve
        damping (float): damping factor applied to the parameter update
        chi_n_iterations (int): steps taken by the last chi_n inversion
        pref_nfev (int): household solves used by the beta / chi_b step
        pref_cost (float): least-squares cost of the beta / chi_b step
        ge_seconds (float): wall time of the GE re-solve
        prices (dict): ``r_p``, ``r``, ``w``, ``factor``, ``TR`` after the
            re-solve
        residuals (dict): relative moment residuals by moment name
    """

    iteration: int
    param_change: float
    price_change: float
    damping: float
    chi_n_iterations: int
    pref_nfev: int
    pref_cost: float
    ge_seconds: float
    prices: dict
    residuals: dict


@dataclass
class LifecycleCalibrationOutcome:
    """
    Result of the full nested preference calibration.

    Attributes:
        beta_annual (Numpy array): calibrated beta by type, length J
        chi_b (Numpy array): calibrated chi_b by type, length J
        chi_n (Numpy array): calibrated steady-state chi_n by age, length S
        ss_output (dict): final general-equilibrium steady state
        iterations (int): number of outer passes taken
        converged (bool): whether parameters and prices met the tolerances
        history (list): one OuterIterationRecord per pass
        data_moments (MomentSet): data moments
        model_moments (MomentSet): model moments at the final steady state
        chi_n_result (ChiNInversionResult or None): last chi_n inversion
        pref_result (PreferenceCalibrationResult or None): last beta /
            chi_b calibration
    """

    beta_annual: np.ndarray
    chi_b: np.ndarray
    chi_n: np.ndarray
    ss_output: dict
    iterations: int
    converged: bool
    history: list
    data_moments: object
    model_moments: object
    chi_n_result: ChiNInversionResult | None
    pref_result: PreferenceCalibrationResult | None

    @property
    def parameter_dict(self) -> dict:
        """
        Calibrated values in ``update_specifications`` format.

        Returns:
            params (dict): ``beta_annual``, ``chi_b``, ``chi_n`` as lists
        """
        return {
            "beta_annual": np.asarray(self.beta_annual).tolist(),
            "chi_b": np.asarray(self.chi_b).tolist(),
            "chi_n": np.asarray(self.chi_n).tolist(),
        }

    def to_frame(self):
        """
        Tabulates data versus model moments at the final general equilibrium.

        Returns:
            frame (Pandas DataFrame): columns ``moment``, ``data``, ``model``
        """
        return self.data_moments.to_frame(self.model_moments)


def _theta_from_p(p) -> np.ndarray:
    """
    Stacks the transformed preference parameters for change tracking.

    Args:
        p (OG-Core Specifications object): parameters object

    Returns:
        theta (Numpy array): ``logit(beta_annual)``, ``log(chi_b)``, and
            ``log(chi_n)`` stacked, length 2J + S
    """
    from ogusa import estimate_lifecycle_params as elp

    beta = np.asarray(p.beta_annual, dtype=float)
    return np.concatenate(
        [
            _logit(beta),
            np.log(np.asarray(p.chi_b, dtype=float)),
            np.log(elp._ss_chi_n(p)),
        ]
    )


def _apply_theta(theta: np.ndarray, p) -> None:
    """
    Applies a stacked transformed parameter vector to the parameters
    object; the inverse of ``_theta_from_p``.

    Args:
        theta (Numpy array): stacked transformed parameters, length 2J + S
        p (OG-Core Specifications object): parameters object; updated in place

    Returns:
        None
    """
    J = p.J
    beta = _logistic(theta[:J])
    chi_b = np.exp(theta[J : 2 * J])
    chi_n = np.exp(theta[2 * J :])
    p.update_specifications(
        {
            "beta_annual": beta.tolist(),
            "chi_b": chi_b.tolist(),
            "chi_n": chi_n.tolist(),
        }
    )


def _price_change(new: dict, old: dict) -> tuple[float, dict]:
    """
    Computes the largest relative change across the outer-loop prices.

    Args:
        new (dict): steady-state output after the re-solve
        old (dict): steady-state output before the re-solve

    Returns:
        max_change (float): largest relative change
        changes (dict): relative change for ``r_p``, ``r``, ``w``,
            ``factor``, ``TR``, and total ``BQ``
    """
    keys = ("r_p", "r", "w", "factor", "TR")
    changes = {}
    for key in keys:
        a, b = float(np.squeeze(old[key])), float(np.squeeze(new[key]))
        changes[key] = abs(b - a) / max(abs(a), 1e-12)
    bq_old = np.asarray(old["BQ"], dtype=float).sum()
    bq_new = np.asarray(new["BQ"], dtype=float).sum()
    changes["BQ"] = abs(bq_new - bq_old) / max(abs(bq_old), 1e-12)
    return max(changes.values()), changes


def calibrate_lifecycle_preferences(
    p,
    config=None,
    options: PreferenceCalibrationOptions | None = None,
    data_moments=None,
    initial_ss: dict | None = None,
    max_outer: int = 15,
    param_tol: float = 1e-2,
    price_tol: float = 1e-3,
    outer_damping: float = 1.0,
    adaptive_damping: bool = True,
    reinvert_chi_n: bool = True,
    client=None,
    ge_client=None,
) -> LifecycleCalibrationOutcome:
    """
    Calibrates chi_n, beta by type, and chi_b to joint general-equilibrium
    convergence.

    Each outer pass: solve (or reuse) the general-equilibrium steady state,
    invert the labor FOC for ``chi_n`` at those prices, calibrate ``beta``
    and ``chi_b`` at those prices, optionally re-invert ``chi_n`` so hours
    stay on target after the preference change, blend the new parameters
    with the old ones by ``outer_damping`` in transformed space, and
    re-solve the general equilibrium warm-started from the previous
    solution.  Stops when the largest transformed-parameter change and the
    largest relative price change both fall below their tolerances.  The
    parameter tolerance is looser than the price tolerance by default
    because beta and chi_b trade off along a ridge that leaves the moments
    unchanged, so transformed parameters can drift at the percent level
    after prices and moments have settled.

    With ``adaptive_damping`` the damping factor halves whenever the
    parameter change fails to shrink by at least ten percent from one pass
    to the next, which guards against the oscillation that strong
    general-equilibrium feedback (saving down, interest rate up, hours up)
    can produce.

    On return ``p`` carries the calibrated parameters.

    Args:
        p (OG-Core Specifications object): parameters object with
            ``baseline=True``; updated in place
        config (LifecycleCalibrationConfig): moment configuration; default
            configuration when None
        options (PreferenceCalibrationOptions): beta / chi_b options;
            defaults when None
        data_moments (MomentSet): data moments; computed from the
            configuration when None
        initial_ss (dict): steady-state output to start from; solved cold
            when None
        max_outer (int): maximum number of outer passes
        param_tol (float): tolerance on the largest transformed-parameter
            change
        price_tol (float): tolerance on the largest relative price change
        outer_damping (float): initial damping on the parameter update
        adaptive_damping (bool): whether to halve the damping when the
            parameter change stalls
        reinvert_chi_n (bool): whether to re-invert chi_n after the beta /
            chi_b step
        client (Dask Client object): client for household solves
        ge_client (Dask Client object): client for the GE solves; serial
            when None

    Returns:
        outcome (LifecycleCalibrationOutcome): calibrated parameters, final
            steady state, per-pass diagnostics, and data-versus-model
            moments
    """
    import time

    from ogusa import estimate_lifecycle_params as elp

    if config is None:
        config = elp.LifecycleCalibrationConfig()
    if options is None:
        options = PreferenceCalibrationOptions()
    config.validate(p)
    if data_moments is None:
        data_moments = elp.compute_data_moments(p, config)
    labor_target = np.array(
        [
            dict(zip(data_moments.names, data_moments.values))[
                f"labor_supply_age_{age}"
            ]
            for age in config.moment_ages
        ]
    )

    t = time.time()
    ss = (
        initial_ss
        if initial_ss is not None
        else solve_ge_steady_state(p, client=ge_client)
    )
    logger.info("Outer loop: initial GE solve took %.0f s", time.time() - t)

    history: list[OuterIterationRecord] = []
    damping = float(outer_damping)
    converged = False
    chi_n_result = None
    pref_result = None
    previous_change = np.inf
    for iteration in range(1, max_outer + 1):
        theta_old = _theta_from_p(p)
        chi_n_result = invert_chi_n(
            ss, p, labor_target, config=config, client=client
        )
        pref_result = calibrate_beta_chi_b(
            chi_n_result.ss_output,
            p,
            data_moments,
            config=config,
            options=options,
            client=client,
        )
        if reinvert_chi_n:
            chi_n_result = invert_chi_n(
                pref_result.ss_output,
                p,
                labor_target,
                config=config,
                client=client,
            )
        theta_new = _theta_from_p(p)
        param_change = float(np.max(np.abs(theta_new - theta_old)))
        if adaptive_damping and param_change > 0.9 * previous_change:
            # Not shrinking fast enough (or growing): damp harder.
            damping = max(damping / 2.0, 0.05)
            logger.info("Outer loop: damping reduced to %.3f", damping)
        previous_change = param_change
        if damping < 1.0:
            _apply_theta(theta_old + damping * (theta_new - theta_old), p)

        t = time.time()
        ss_new = solve_ge_steady_state(p, previous=ss, client=ge_client)
        ge_seconds = time.time() - t
        price_change, changes = _price_change(ss_new, ss)
        model_moments = elp.compute_model_moments(ss_new, p, config)
        residuals = dict(
            zip(
                data_moments.names,
                elp.moment_residuals(model_moments, data_moments, "relative"),
            )
        )
        history.append(
            OuterIterationRecord(
                iteration=iteration,
                param_change=param_change,
                price_change=price_change,
                damping=damping,
                chi_n_iterations=chi_n_result.iterations,
                pref_nfev=pref_result.nfev,
                pref_cost=pref_result.cost,
                ge_seconds=ge_seconds,
                prices={
                    k: float(np.squeeze(ss_new[k]))
                    for k in ("r_p", "r", "w", "factor", "TR")
                },
                residuals=residuals,
            )
        )
        logger.info(
            "Outer loop %d: max param change %.3e, max price change %.3e "
            "(%s), GE %.0f s, pref cost %.3e",
            iteration,
            param_change,
            price_change,
            max(changes, key=changes.get),
            ge_seconds,
            pref_result.cost,
        )
        ss = ss_new
        if param_change < param_tol and price_change < price_tol:
            converged = True
            break

    if not converged:
        logger.warning(
            "Outer loop did not converge in %d passes (param change %.3e, "
            "price change %.3e).",
            max_outer,
            history[-1].param_change,
            history[-1].price_change,
        )
    model_moments = elp.compute_model_moments(ss, p, config)
    return LifecycleCalibrationOutcome(
        beta_annual=np.asarray(p.beta_annual, dtype=float).copy(),
        chi_b=np.asarray(p.chi_b, dtype=float).copy(),
        chi_n=elp._ss_chi_n(p),
        ss_output=ss,
        iterations=len(history),
        converged=converged,
        history=history,
        data_moments=data_moments,
        model_moments=model_moments,
        chi_n_result=chi_n_result,
        pref_result=pref_result,
    )


# ---------------------------------------------------------------------------
# Phase 6: inference on beta and chi_b at the calibrated point
# ---------------------------------------------------------------------------


@dataclass
class PreferenceInference:
    """
    Standard errors and overidentification test for beta and chi_b.

    Attributes:
        theta_se (Numpy array): standard errors of the free (transformed)
            parameters, in the order of ``_PreferenceParameterization``:
            beta groups (logit shifts) then chi_b groups (log shifts); NaN
            for parameters fixed at a bound
        beta_se (Numpy array): delta-method standard errors of beta by
            type, length J; NaN at a bound
        chi_b_se (Numpy array): delta-method standard errors of chi_b by
            type, length J; NaN at a bound
        vcv_theta (Numpy array): covariance matrix of the free parameters;
            NaN rows and columns for parameters at a bound
        j_stat (float): Hansen-type test statistic of the overidentifying
            restrictions, or NaN when no moment covariance was supplied
        j_df (int): degrees of freedom of the test, targets minus free
            parameters
        j_pvalue (float): p-value of the test, or NaN
        method (str): ``"nls"`` (homoskedastic residual variance) or
            ``"sandwich"`` (supplied moment covariance)
        n_moments (int): number of targets
        n_params (int): number of free parameters not at a bound
        at_bound (Numpy array): whether each free parameter sits at a
            bound, boolean
    """

    theta_se: np.ndarray
    beta_se: np.ndarray
    chi_b_se: np.ndarray
    vcv_theta: np.ndarray
    j_stat: float
    j_df: int
    j_pvalue: float
    method: str
    n_moments: int
    n_params: int
    at_bound: np.ndarray

    def to_frame(self, p):
        """
        Tabulates point estimates and standard errors by type.

        Args:
            p (OG-Core Specifications object): parameters object carrying
                the calibrated values

        Returns:
            frame (Pandas DataFrame): columns ``type``, ``beta_annual``,
                ``beta_se``, ``beta_at_bound``, ``chi_b``, ``chi_b_se``,
                ``chi_b_at_bound``; a standard error is NaN for a type whose
                parameter sits on a bound
        """
        import pandas as pd

        return pd.DataFrame(
            {
                "type": np.arange(1, p.J + 1),
                "beta_annual": np.asarray(p.beta_annual, dtype=float),
                "beta_se": self.beta_se,
                "beta_at_bound": np.isnan(self.beta_se),
                "chi_b": np.asarray(p.chi_b, dtype=float),
                "chi_b_se": self.chi_b_se,
                "chi_b_at_bound": np.isnan(self.chi_b_se),
            }
        )


def preference_target_selection(
    data_moments, p, config, options: PreferenceCalibrationOptions
) -> np.ndarray:
    """
    Builds the matrix mapping the full moment vector to the beta / chi_b
    targets.

    Rows follow ``preference_targets``; merged wealth-share bins sum the
    underlying per-type shares.  ``A @ data_moments.values`` reproduces the
    target values, and ``A @ V @ A.T`` carries a bootstrap covariance of the
    full moment set over to the targets.

    Args:
        data_moments (MomentSet): full data moment vector
        p (OG-Core Specifications object): parameters object
        config (LifecycleCalibrationConfig): moment configuration
        options (PreferenceCalibrationOptions): grouping options

    Returns:
        A (Numpy array): selection matrix, shape (targets, moments)
    """
    from ogusa import estimate_lifecycle_params as elp

    names, values, selection = preference_targets(
        data_moments, p, config, options
    )
    index = {name: k for k, name in enumerate(data_moments.names)}
    share_names = elp.wealth_share_bin_names(elp._lambdas(p))
    A = np.zeros((len(names), len(data_moments.names)))
    row = 0
    for members in selection["share_bins"]:
        for j in members:
            A[row, index[share_names[j]]] = 1.0
        row += 1
    A[row, index["wealth_income_ratio"]] = 1.0
    row += 1
    if options.chi_b_mode == "by_type":
        tilt_names = elp.tilt_moment_names(config, p)
        for k in selection["tilt_idx"]:
            A[row, index[tilt_names[k]]] = 1.0
            row += 1
    A[row, index["bequest_flow_ratio"]] = 1.0
    assert row + 1 == len(names)
    if not np.allclose(A @ data_moments.values, values):
        raise RuntimeError("Selection matrix does not reproduce targets.")
    return A


def preference_inference(
    result: PreferenceCalibrationResult,
    p,
    options: PreferenceCalibrationOptions | None = None,
    moment_vcv: np.ndarray | None = None,
    bound_tol: float = 1e-6,
) -> PreferenceInference:
    """
    Computes standard errors for beta and chi_b from the least-squares
    Jacobian.

    The calibration minimizes the sum of squared weighted log residuals
    ``r(theta) = w * (log m(theta) - log d)`` over household-only solves,
    so it is a GMM estimator with an identity weighting matrix on the
    weighted log moments.  With ``G`` the Jacobian of ``r`` at the solution
    the parameter covariance is

    * ``s^2 (G'G)^-1`` with ``s^2 = r'r / (m - k)`` when no moment
      covariance is given (classical nonlinear least squares; treats every
      target as equally noisy), or
    * ``(G'G)^-1 G' V_r G (G'G)^-1`` with ``V_r = D V_d D`` when
      ``moment_vcv`` (the covariance ``V_d`` of the data targets in levels,
      for example from ``estimate_lifecycle_params.bootstrap_data_moments``
      carried through ``preference_target_selection``) is given, where
      ``D = diag(w / d)`` converts to weighted log units.  In that case the
      overidentification statistic ``r' (M V_r M')^+ r`` with
      ``M = I - G (G'G)^-1 G'`` is chi-squared with ``m - k`` degrees of
      freedom under the null that the model matches every target.

    Parameters live in the transformed space of the calibration (shifts to
    ``logit(beta)`` per beta group and ``log(chi_b)`` per chi_b group);
    standard errors for ``beta_annual`` and ``chi_b`` by type follow from
    the delta method.

    A parameter within ``bound_tol`` of one of its bounds (in the
    transformed space) is treated as fixed there: its column is dropped
    from the Jacobian, it does not count toward ``k``, and its standard
    error is NaN.  The types at the ``beta_annual`` ceiling are the usual
    case; their logit derivative is essentially zero and a delta-method
    standard error would be meaningless.

    Args:
        result (PreferenceCalibrationResult): calibration result carrying
            the Jacobian, residuals, weights, and data values
        p (OG-Core Specifications object): parameters object carrying
            the calibrated parameters that ``result`` was computed at
        options (PreferenceCalibrationOptions): grouping options used in
            the calibration; defaults when None
        moment_vcv (Numpy array or None): covariance of the data targets
            in levels, shape (targets, targets); None for the classical
            form
        bound_tol (float): distance from a bound below which a parameter is
            treated as fixed

    Returns:
        inference (PreferenceInference): standard errors, covariance, and
            the overidentification test
    """
    from scipy import stats

    if options is None:
        options = PreferenceCalibrationOptions()
    if result.jacobian is None:
        raise ValueError(
            "result has no Jacobian; re-run calibrate_beta_chi_b."
        )
    param = _PreferenceParameterization(p, options)
    theta = np.asarray(result.theta, dtype=float)
    at_bound = (theta - param.lower <= bound_tol) | (
        param.upper - theta <= bound_tol
    )
    G_full = np.asarray(result.jacobian, dtype=float)
    r = np.asarray(result.residuals, dtype=float)
    m = G_full.shape[0]
    free = np.flatnonzero(~at_bound)
    G = G_full[:, free]
    k = free.size
    if m <= k:
        raise ValueError("Need more targets than free parameters.")
    GtG_inv = np.linalg.pinv(G.T @ G)
    if moment_vcv is None:
        s2 = float(r @ r) / (m - k)
        vcv = s2 * GtG_inv
        j_stat = j_pvalue = np.nan
        method = "nls"
    else:
        V_d = np.asarray(moment_vcv, dtype=float)
        if V_d.shape != (m, m):
            raise ValueError(
                f"moment_vcv must be {(m, m)} to match the targets."
            )
        weights = (
            np.ones(m)
            if result.weights is None
            else np.asarray(result.weights)
        )
        D = np.diag(weights / np.asarray(result.data_values, dtype=float))
        V_r = D @ V_d @ D
        bread = GtG_inv @ G.T
        vcv = bread @ V_r @ bread.T
        M = np.eye(m) - G @ bread
        j_stat = float(r @ np.linalg.pinv(M @ V_r @ M.T) @ r)
        j_pvalue = float(stats.chi2.sf(j_stat, m - k))
        method = "sandwich"
    theta_se = np.full(theta.size, np.nan)
    theta_se[free] = np.sqrt(np.clip(np.diag(vcv), 0.0, None))
    vcv_full = np.full((theta.size, theta.size), np.nan)
    vcv_full[np.ix_(free, free)] = vcv
    vcv = vcv_full

    beta = np.asarray(p.beta_annual, dtype=float)
    chi_b = np.asarray(p.chi_b, dtype=float)
    beta_se = np.zeros(p.J)
    chi_b_se = np.zeros(p.J)
    for g, members in enumerate(param.beta_groups):
        # d beta / d theta = beta (1 - beta) for a logit shift.
        beta_se[members] = beta[members] * (1 - beta[members]) * theta_se[g]
    for g, members in enumerate(param.chi_b_groups):
        chi_b_se[members] = chi_b[members] * theta_se[param.n_beta + g]
    return PreferenceInference(
        theta_se=theta_se,
        beta_se=beta_se,
        chi_b_se=chi_b_se,
        vcv_theta=vcv,
        j_stat=j_stat,
        j_df=m - k,
        j_pvalue=j_pvalue,
        method=method,
        n_moments=m,
        n_params=k,
        at_bound=at_bound,
    )
