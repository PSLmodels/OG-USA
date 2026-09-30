# Plan: calibrating beta, chi_b, and chi_n in OG-USA

Status as of 2026-09-21. Branch: `smm`.

## Background

The `smm` branch added `ogusa/estimate_lifecycle_params.py`, a joint
steady-state simulated method of moments (SMM) estimator for `beta_annual`
(one per lifetime-income type), `chi_b` (one per type), and a 10-knot
log B-spline for the `chi_n` age profile, against roughly 130 moments.
Each objective evaluation solved the full general-equilibrium steady state.

Review findings that motivate this plan:

- About three quarters of the objective came from hours at ages 70 to 79.
  The CPS decline there is almost entirely participation, which the model
  lacks. With a Frisch elasticity of 0.4 the elliptical disutility has
  upsilon of about 2.86, so moving hours from 0.30 to 0.035 needs `chi_n`
  to rise by a factor of roughly 55. A 10-knot spline cannot produce that
  cliff without distorting ages 55 to 70.
- `beta_j` and `chi_b_j` both raise wealth for type `j` and are not
  separately identified by aggregate age profiles. The hard-coded seven
  wealth-share bins put the four top-1% types behind a single moment.
- Relative deviations explode on near-zero data moments: the bottom-25%
  wealth share is negative in the SCF, and the wealth profile was anchored
  to mean net worth at ages 20 to 24, a thin and noisy cell.
- Income Gini and variance of log wealth do not depend on the preference
  parameters to first order and cannot be matched by a deterministic model
  with ten types.
- The committed DFO-LS driver had never run (bounds bug, `dfo-ls` not
  installed). Saved results came from an earlier L-BFGS-B version that
  stopped after three iterations.

## On J

Data and model moments always agree with each other; nothing uses J = 7
when the model has J = 10. But the wealth-share bins were hard-coded to
seven cuts on both sides, so the moment set did not scale with J. Using
`p.lambdas` as the bins gives one share moment per type. SCF 2019 shares
with the ten default lambdas (row-bootstrap standard errors understate the
true ones, especially in the top bin, which rests on about 87 households;
the SCF also excludes the Forbes 400):

| Bin (percentile) | Lambda | SCF share | Row-bootstrap SE |
|---|---|---|---|
| 0 to 25 | 0.25 | -0.005 | 0.0002 |
| 25 to 50 | 0.25 | 0.020 | 0.0005 |
| 50 to 70 | 0.20 | 0.055 | 0.0009 |
| 70 to 80 | 0.10 | 0.056 | 0.0009 |
| 80 to 90 | 0.10 | 0.110 | 0.0015 |
| 90 to 99 | 0.09 | 0.392 | 0.0047 |
| 99 to 99.5 | 0.005 | 0.092 | 0.0016 |
| 99.5 to 99.9 | 0.004 | 0.140 | 0.0026 |
| 99.9 to 99.99 | 0.0009 | 0.089 | 0.0020 |
| 99.99 to 100 | 0.0001 | 0.051 | 0.0042 |

The bottom bin is negative and cannot be matched, so `beta` for type 1 is
tied to type 2 and the bottom 50% is treated as one target.

## Phases

### Phase 0. Make the branch runnable and decide what to keep

Status: done 2026-09-16 (all items below).

- Install `dfo-ls` into the `.venv` (`uv sync`) and fix the failing test.
- Keep the data-side functions, `MomentSet`, and `compute_model_moments`.
- Fix the off-by-one in the wealth-profile age index (`b_sp1[s]` is wealth
  held at age `s + 1`).
- Replace the silent warm-start fallback with a logged warning and call
  `SS.SS_solver` by keyword so signature changes across OG-Core versions
  fail loudly instead of silently.
- Replace the 1e15 failure penalty with a bounded value so it does not
  poison the DFO-LS interpolation model.
- Retire the 30-parameter DFO-LS driver as the main path; it becomes the
  optional inference layer in Phase 6.
- Remove `ogusa/calibrate_chi_n.py`, which targets an OG-Core API that no
  longer exists.

### Phase 1. Redesign the moment set

Status: done 2026-09-16. Defaults in `LifecycleCalibrationConfig` now give
71 moments for J = 10 (60 hours ages, 10 shares, 1 old-age ratio). The SCF
old-age wealth ratio (75-79 over 60-64, household weighted) is 0.80. The
ability profiles are hourly-wage based (CWHS earnings with imputed hours,
see `docs/book/content/calibration/earnings.md`), so hours targets do not
double count. While fixing the bins, `wealth.compute_wealth_moments` was
found to drop the wealthiest observation from the top bin; fixed.

- Wealth shares use `p.lambdas` as bins on both sides; generalize the model
  side from seven hard-coded cuts to cumulative lambdas.
- Hours targets stay at single years of age 20 to 79 (the `chi_n` inversion
  needs them) with a light smoother on the CPS profile.
- Add one `chi_b` target: the ratio of mean SCF net worth at ages 75 to 79
  to that at 60 to 64. Report aggregate bequests over GDP as a check.
- Drop income Gini, variance of log wealth, and the anchored wealth profile
  as targets. Keep a wealth-by-age profile normalized by its mean over ages
  21 to 79 for diagnostics only.
- Confirm the ability profiles `e` are hourly-wage based; if they embed
  hours, the labor targets double count.

### Phase 2. Household-only solver wrapper

Status: done 2026-09-16 in `ogusa/calibrate_lifecycle.py`
(`HouseholdEnvironment`, `solve_households`, `partial_equilibrium_ss`).
Validated against a cold general-equilibrium solve with the default
parameters: the household-only re-solve reproduces `b_sp1` and `n` to
about 1e-12 relative error. Timings on the development machine:

| Solve | Time |
|---|---|
| General-equilibrium steady state, cold, 5 Dask workers | 1028 s |
| Household-only re-solve, serial | 0.5 s |
| Household-only re-solve, 5 Dask workers | 14 s |

The household block itself is cheap; the Dask path is dominated by
scattering the parameters object. Use `client=None` for the inner loops in
Phases 3 and 4. The general-equilibrium solve's cost is therefore mostly
outer-loop iterations and per-iteration Dask overhead, which is worth
revisiting in Phase 5 (a serial general-equilibrium solve may be faster).

A function that takes a steady-state output dictionary and solves only the
household block at fixed prices, taxes, bequests, transfers, and scaling
factor, using `SS.solve_for_j`, `aggregates.get_io_prices` (or
`io_matrix @ p_m` on older OG-Core), `aggregates.get_ptilde`, and the
`household.get_bq` / `get_tr` / `get_rm` helpers. Parallelize across types
with the existing Dask pattern, with a serial fallback.

### Phase 3. Concentrate out chi_n

Status: done 2026-09-16 (`invert_chi_n`, `chi_n_update`,
`aggregate_labor_by_age` in `ogusa/calibrate_lifecycle.py`). At the
default-parameter equilibrium prices the inversion converges in 7 passes
and about 4 seconds to a maximum absolute log gap of 6e-4 between model and
CPS hours at every age 20 to 79. The resulting `chi_n` is about 254 at age
20, 42 to 58 over ages 30 to 50, 110 at 60, 267 at 65, 1,047 at 70, 3,471
at 75, and 4,542 at 79, held flat from 80 on. The 10,000 cap does not bind.
Whether values in the thousands at old ages are acceptable, or whether the
target at 70 plus should switch to hours conditional on working, remains
the maintainer's call.

The general-equilibrium feedback is large and must be handled by the
Phase 5 outer loop: re-solving the steady state with the inverted `chi_n`
lowers aggregate labor and output by about 36 percent, raises the income
scaling factor by 57 percent, and pushes hours back above target by 8 to
20 percent (log gap) because lower consumption raises the marginal utility
of consumption. That same re-solve, run serially from cold guesses, took
20 seconds, against 1,028 seconds for the Phase 2 solve with five Dask
workers. Dask overhead dominates the steady-state solve on this machine;
Phase 5 should run the general-equilibrium solve serially, which makes
even a dozen outer iterations cheap.

Iterate: solve households at fixed prices, form lambda-weighted mean hours
by age, update each age's `chi_n` by the ratio of marginal disutility at
model hours to that at data hours, repeat. Hold `chi_n` flat in logs for
ages 80 to 99 at the age-79 value. Report where `chi_n` lands relative to
the 10,000 validator cap. If ages 75 to 79 need values near the cap, that
is the signal to switch the target at those ages to hours conditional on
working; that decision is the maintainer's.

### Phase 4. beta by type and chi_b at fixed prices

Status: implemented 2026-09-17 (`calibrate_beta_chi_b`,
`PreferenceCalibrationOptions` in `ogusa/calibrate_lifecycle.py`), with
results that call for maintainer decisions listed below.

Design as built (it replaced the damped secant with a bounded nonlinear
least-squares solve, since each residual evaluation is a half-second
household solve):

- Free parameters are additive shifts to `logit(beta_annual)` by type and
  to `log(chi_b)` by type group. `chi_b_mode="by_type"` (default) gives one
  `chi_b` factor to each of the groups `[1-3], [4], [5], [6], [7-10]`;
  `"common_scale"` moves every type's `chi_b` by one factor.
- Targets, all as log model-over-data residuals: wealth shares for the
  bins of types 3 through 10 (eight), SCF mean wealth over mean income,
  the by-bin old-age tilt (mean wealth at 80-89 over 60-64 within
  wealth-percentile bins 50-70, 70-80, 80-90, 90-99, 99-100) in by-type
  mode, and the bequest-flow ratio (mortality-weighted wealth over total
  wealth, model mortality on both sides).
- The bottom-half share bin is excluded and types 1 and 2 share type 3's
  factors (`exclude_bottom=True`). A deterministic model with no
  within-type heterogeneity cannot deliver the SCF bottom-half share of
  about 1.5 percent because young households of every type fill the bottom
  percentiles; targeting it drove the bottom betas to zero.
- Failed household solves retry from the initial guesses, then return a
  bounded penalty. The initial solve must converge or the call raises.

Sensitivity facts that shaped the design (household-only solves at the
default-parameter prices):

| Change | Total wealth | Bequest flow | Bequest flow / wealth | Wealth 95-99 / 60-64 |
|---|---|---|---|---|
| (1 - beta) down 10%, all types | +4.3% | +3.3% | -1.0% | -2.5% |
| chi_b doubled | +39% | +41% | +1.6% | +5.6% |

Both parameters scale the whole wealth profile; only the tilt toward the
very old separates them, and it is small. The aggregate 75-79 over 60-64
ratio from Phase 1 moved 0.2 percent when chi_b doubled and was dropped as
a target. Type 10's wealth rises only 9 percent when its beta goes from
0.995 to 0.9999, so top-type wealth is nearly insensitive to beta near one.

Results at the default-parameter equilibrium prices, after the Phase 3
chi_n inversion, with the bottom bin excluded:

| Mode, income concept | Solves | Time | Betas | chi_b | Notes |
|---|---|---|---|---|---|
| by type, SCF total income | 258 | 136 s | 0.96, 0.96, 0.97, 0.94, 0.91, then 0.9999 for types 6-9, 0.994 | 3.3 (types 1-3), 15, 11, 23, 50 (top) | shares within 9% except 99-99.5 bin (-46%); wealth/income 5.0 vs 7.0; tilt bins 50-90 within 8%, 90-99 bin +51%; bequest flow +56% |
| common scale, SCF total income | 3,726 | 32 min | 0.61, 0.61, 0.64, 0.85, 0.72, 0.99, then 0.9998 | 73 | hit evaluation cap; crawled along the beta/chi_b ridge |
| common scale, SCF pre-transfer income | 3,643 | 32 min | 0.41, 0.41, 0.45, 0.80, 0.58, 0.97, then 0.9997 | 218 | matched wealth/income 8.3 entirely through chi_b |

The common-scale runs show the identification problem directly: with
one chi_b factor and a level target, the solver trades low betas for a
huge bequest motive. The by-type mode converges quickly to plausible
values because the tilt bins pin each group's chi_b.

Decisions for the maintainer:

1. **Income concept for the level target.** The model's before-tax income
   (`r_p * B + w * L`) is 0.78 of output; SCF pre-transfer income is about
   0.62 of GDP and SCF total income about 0.74. The pre-transfer target of
   8.4 is therefore inflated by an accounting mismatch. The code defaults to
   `scf_income_concept="pre_transfer"` as requested; the runs above suggest
   `"total"` (target 7.0) is the more comparable choice, and household net
   worth over GDP from the Financial Accounts (5.5 versus model 4.9) is a
   third option.
2. **Bequest-flow target.** Model 0.030 to 0.040 against data 0.017 in
   every run. The model's old do not decumulate (no medical expense risk,
   no annuities, consumption roughly flat with beta times gross return near
   one), so this gap is structural. Keeping it as a target pulls chi_b down
   and distorts the betas; treating it as a diagnostic is the alternative.
3. **The 99 to 99.5 percentile bin** (type 7) is 42 to 46 percent short in
   every run even with its beta at the upper bound. Type 7's ability
   profile is 8.5 times the mean at age 45 while the SCF bin holds 18 times
   mean wealth per person; the shortfall points at the ability profile
   between types 7 and 8, not at preferences.
4. **Hours drift.** Changing beta and chi_b at fixed prices moves hours by
   6 to 14 percent (log), so Phase 5 must alternate the chi_n inversion
   and this calibration.

General-equilibrium re-solve with the by-type parameters and the Phase 3
`chi_n` (serial, 157 seconds after two failed initial guesses): the
interest rate rose from 4.3 to 5.6 percent, output fell 40 percent, the
income scaling factor rose 71 percent, hours ended 10 to 27 percent (log)
above target, wealth over income fell to 4.6 against 7.0, and the top-1%
bins stayed within 9 percent while the 80-90 bin dropped 25 percent below
target. The tilt bins moved the most (the 50-70 bin from -7 to -49
percent), so the type-specific chi_b factors are sensitive to prices.
Phase 5 must therefore iterate to convergence rather than apply one pass,
and the higher interest rate means the fixed-price step overstates how
much saving the calibrated parameters deliver.

### Phase 5. Outer general-equilibrium loop

Status: done 2026-09-18 (`calibrate_lifecycle_preferences`,
`solve_ge_steady_state` in `ogusa/calibrate_lifecycle.py`). The warm start
is implemented on the OG-USA side by calling OG-Core's `SS.SS_fsolve` root
finder from the previous solution and assembling output with
`SS.SS_solver(fsolve_flag=True)`, handling solver layouts with and without
`G`; no OG-Core change was needed. Parameter updates are blended in
transformed space with adaptive damping (halved whenever the parameter
change fails to shrink by ten percent).

Run with the default options (SCF pre-transfer income, bequest flow as a
target, by-type chi_b, bottom bin excluded), starting from the
default-parameter steady state:

| Pass | Max param change | Max price change | Pref solves | GE solve |
|---|---|---|---|---|
| 1 | 5.89 | 0.64 (BQ) | 1,062 | 151 s (warm start failed, cold fallback) |
| 2 | 1.19 | 0.15 (factor) | 222 | 15 s |
| 3 | 0.22 | 0.045 | 327 | 21 s |
| 4 | 0.057 | 0.021 | 141 | 18 s |
| 5 | 0.004 | 0.0005 | 30 | 18 s |
| 6 to 8 | 0.048, 0.017, 3e-11 | below 0.003 | 42 to 74 | 10 to 15 s |

Total 22 minutes, converged at pass 8 (damping fell to 0.5 at pass 6).
Final general equilibrium: interest rate 4.8 percent (portfolio return
3.55 percent), wage 1.394, scaling factor 302k. Hours match CPS at every
age to a 0.001 log gap. Wealth shares: the 80-90, 90-99, and top three
bins are within 6 percent; 50-70 is +13 percent, 70-80 is -14 percent, and
99-99.5 is -51 percent (the type 7 ability issue). Wealth over income is
5.4 against the pre-transfer target of 8.4; the bequest-flow ratio is
0.030 against 0.017. Betas: 0.953 (types 1-3), 0.969, 0.923, then 0.9999
for types 6-9 and 0.996 for type 10. chi_b: 9.3 (types 1-3), 25.9, 20.7,
30.4, 77.2 (top). chi_n: 520 at 20, 71 to 76 over 30-45, 156 at 60, 390
at 65, 1,556 at 70, 5,296 at 75, 7,124 at 79 (the 10,000 cap is close).

Pass 1's warm start failed because the first parameter jump is large;
later passes warm-start in 10 to 20 seconds against 150 seconds cold.

Second run with SCF total income as the level target and the bequest-flow
weight set to zero (diagnostic only): 12 passes, 34 minutes, prices settled
to relative changes below 0.0005 from pass 9 but the transformed parameter
change hovered at 0.01 to 0.015, which is the beta versus chi_b ridge
wandering with no effect on the moments (fixed-price cost flat at 0.2365
from pass 4). The default parameter tolerance was loosened to 0.01
afterwards. Final fit: hours to 0.001 log gap; wealth shares within 6
percent for the 80-90, 90-99, and top three bins, +11 percent for 50-70,
-15 percent for 70-80, -47 percent for 99-99.5; tilt bins 50-70, 70-80,
and 80-90 within 1 percent, 90-99 +29 percent, top +17 percent; wealth over
income 5.1 against 7.0; bequest flow 0.030 against 0.017. Betas 0.948
(types 1-3), 0.930, 0.950, 0.9999 for types 6-8, 0.998, 0.995; chi_b 9.2
(types 1-3), 34.9, 13.3, 29.2, 66.0 (top); chi_n 7,412 at age 79.

What the two runs say together:

- The loop works: eight to twelve passes, 20 to 35 minutes, hours exact,
  and the type-specific tilt moments for the middle of the distribution
  are matched almost exactly once general-equilibrium prices are
  consistent.
- The wealth level cannot be reached under either income concept even
  with types 6 to 8 at the 0.9999 beta ceiling, because the middle types'
  betas are held down by their share targets. Wealth concentration in the
  SCF exceeds what the model's ability profiles can generate through
  patience alone at a 3.7 percent portfolio return; matching it would need
  return heterogeneity, a larger ability gap between types 7 and 8, or a
  bequest motive that is stronger at the top than the by-type tilt
  supports.
- The rich do not decumulate in the model (tilt for the 90-99 bin 1.0
  against 0.76 in the SCF), which is the same structural gap the aggregate
  bequest flow shows.
- chi_n at ages 75 to 79 is 5,300 to 7,400 and rises as the portfolio
  return falls; the 10,000 validator cap is within reach of any further
  drop in returns.

### Phase 6. Integrate and validate

Status: done 2026-09-21.

- Done: `Calibration(p, estimate_lifecycle_prefs=True)` runs
  `calibrate_lifecycle_preferences` on a deep copy of `p` that already
  carries the class's other outputs (tax functions, `e`, `eta`, `zeta`,
  demographics, macro parameters), so the preferences are consistent with
  the rest of `get_dict()`, which now returns `beta_annual`, `chi_b`, and
  `chi_n`. `lifecycle_params_path` mirrors the tax-function cache: a JSON
  whose `S` and `J` match is read instead of re-calibrated, otherwise the
  calibration runs and writes it (with a `_meta` block recording
  convergence). `lifecycle_config`, `lifecycle_options`,
  `lifecycle_initial_ss`, and `lifecycle_kwargs` pass through.
  `estimate_chi_n` is a deprecated alias (with a warning) because `chi_n`
  is only meaningful jointly with `beta` and `chi_b`. The legacy
  `estimate_beta` path, which passed the class instance as the initial
  guess and read an attribute that was never set, was repaired in passing.
- Done: `check_smm.py` (untracked) removed. `examples/run_lifecycle_calibration.py`
  runs the calibration through the class and writes `calibrated_params.json`,
  `moment_comparison.csv` (data, model, log gap), `outer_loop_history.csv`,
  optional `preference_standard_errors.csv`, and the hours (share of the
  112-hour endowment against CPS), wealth (2019 dollars against SCF), and
  `chi_n` figures by reusing `plot_lifecycle_calibration.py`.
  `examples/validate_lifecycle_time_path.py` solves the baseline and a
  reform (corporate rate 35 percent) transition path at calibrated
  parameters, solving steady states serially and the time path under Dask,
  and records convergence, Euler errors, and the macro comparison table.
- Done: `preference_inference` in `calibrate_lifecycle.py` gives standard
  errors for `beta` and `chi_b` from the household-only Jacobian that
  `calibrate_beta_chi_b` now stores (SciPy's `least_squares` Jacobian at
  the solution, so no extra solves are needed at the calibrated point).
  Two forms: classical nonlinear least squares, or a sandwich with a
  bootstrap covariance of the data moments mapped to the merged targets by
  `preference_target_selection`; the latter also yields the Hansen-type
  overidentification statistic on `m - k` degrees of freedom. DFO-LS was
  not needed for this: the calibration already is a least-squares problem
  with an analytic-enough Jacobian from finite differences.
- Time-path validation at the pre-transfer-income, bequest-flow
  calibration (`calibrated_params_pretransfer_bq.json`), six Dask workers
  for the transition path:

  | Run | Wall time | Max Euler error (savings, labor) | Outcome |
  |---|---|---|---|
  | Baseline | 8.1 min | 2.3e-12, 3.1e-12 | converged |
  | Reform (corporate rate 35 percent) | 8.4 min (plus 15 s serial SS) | 2.3e-12, 3.2e-12 | converged |

  The large old-age `chi_n` (5,300 to 7,400 at ages 75 to 79) gave the
  time-path solver no trouble: household Euler errors are at machine
  precision on both paths and the outer loops converged normally. The
  reform's macro effects are the expected sign (GDP -0.7 percent, capital
  -1.9 percent, and labor +0.03 percent over 2026-2035; steady-state GDP
  -0.8 percent, capital -2.5 percent, wage -1.0 percent). Output in
  `examples/lifecycle_calibration/time_path_validation/` (gitignored).
  One wrinkle: Dask's `client.close()` timed out at the end of the first
  run, which the script now tolerates; run records are written per run so
  `--summarize-only` can rebuild the table.
- Found while adding the standard errors: SciPy's `diff_step` in
  `least_squares` is relative to the parameter value, and the calibration
  parameterizes shifts that start at zero, so the Jacobian in Phase 4 and
  5 was built with steps of about 1.5e-8 against household solves that
  are reproducible to about 1e-6. The returned Jacobian had entries of
  1e4 to 1e12 and many exact zeros. `calibrate_beta_chi_b` now supplies
  its own forward-difference Jacobian with an absolute step of 1e-3 from
  a common household guess; the columns match manual derivatives
  (magnitudes 0.01 to 0.7). The calibration reached its fit despite the
  noisy Jacobian (`trf` is robust to it), but the ridge wandering between
  `beta` and `chi_b` noted in Phase 5 may partly reflect it.
- Full calibration re-run through the `Calibration` class with the fixed
  Jacobian (`examples/run_lifecycle_calibration.py`, default options,
  serial): 7 outer passes, 30 minutes, converged with damping never
  reduced (Phase 5 needed 8 passes and a damping cut). Parameter changes
  by pass: 5.7, 1.9, 0.20, 0.089, 0.014, 0.0049, 0.0024; price changes
  fell to 0.0009. Household solves in the beta/chi_b step per pass: 1,740,
  212, 127, 353, 239, 141, 113. Results are within rounding of the Phase 5
  run: betas 0.953 (types 1-3), 0.959, 0.933, 0.9999 (types 6-9), 0.997;
  chi_b 9.1 (types 1-3), 24.7, 20.0, 33.1, 75.9 (top); chi_n 526 at 20,
  71 to 75 over 30-45, 157 at 60, 392 at 65, 1,566 at 70, 5,331 at 75,
  7,182 at 79. Hours match to a 0.0006 log gap; wealth-share and tilt fits
  are unchanged from Phase 5 (the same structural gaps: 99-99.5 share -49
  percent, wealth over income 5.4 against 8.4, bequest flow 0.030 against
  0.017, top tilts +19 to +31 percent). Outputs in
  `examples/lifecycle_calibration/` (`calibrated_params.json`,
  `moment_comparison.csv`, `outer_loop_history.csv`,
  `preference_standard_errors.csv`, figures).
- Standard errors (sandwich form, 200 SCF/CPS row-bootstrap draws,
  4 of 13 parameters fixed at the beta ceiling): beta 0.004 for types 1-3,
  0.023 for types 4-5, 0.005 for type 10; chi_b 1.2 (types 1-3), 6.3, 5.9,
  9.4, 10.3 (top). The overidentification statistic is 2,234 on 6 degrees
  of freedom: with row-bootstrap sampling variances this small, the
  structural misfit in the wealth level, bequest flow, and top-end tilts is
  overwhelmingly rejected, which is the statistical face of the Phase 5
  conclusion that the model needs return heterogeneity or a stronger
  top-end bequest motive to reach the SCF wealth concentration.

Remaining for Phase 6: none of the listed items. Open questions carry over
to the modelling side (see the Phase 5 discussion).

## Tests

- Unit tests for the `chi_n` and `beta` update rules on synthetic data.
- Lambdas-based data shares sum to one and match the model-side cuts.
- The household-only wrapper reproduces general-equilibrium savings and
  hours when handed equilibrium prices.
- The full calibration gets a `local` marker.
- `Calibration` wiring: the lifecycle routine sees a copy of `p` carrying
  `e` and `eta`, `get_dict` returns the three parameters, a saved JSON is
  reused and a dimension mismatch triggers re-calibration
  (`tests/test_calibrate.py`).
- `preference_target_selection` reproduces the targets from the full
  moment vector; `preference_inference` returns finite standard errors in
  both forms with tied types sharing one standard error, and treats a
  parameter on its bound as fixed (`tests/test_calibrate_lifecycle.py`).
- The Jacobian stored by `calibrate_beta_chi_b` equals a manual forward
  difference with the absolute step on the synthetic household block;
  `_theta_from_p` / `_apply_theta` round-trip; `_price_change` and
  `_chi_n_bounds` behave as documented (`tests/test_calibrate_lifecycle.py`).
- `read_lifecycle_parameters` branches (missing file, missing keys,
  dimension mismatch, consistent file) and `_parameter_updates`
  (`tests/test_calibrate.py`).
- Example-script helpers: `history_frame`, `moment_frame`,
  `standard_errors` (with the inner steps faked), `tpi_summary`,
  `run_one` (success, saved steady state, and TPI failure), and
  `load_default_spec` (`tests/test_lifecycle_examples.py`).

## Order and dependencies

Phases 0 and 1 are independent and can start immediately. Phase 2 does not
depend on Phase 1. Phases 3 and 4 depend on 2. Phase 5 depends on the
OG-Core change. Phase 6 depends on everything before it.

## Environment

Use `uv run python ...` for everything in this repo. The conda environments
on the development machine are stale (OG-Core 0.16.1) and cannot load the
current default parameters.
