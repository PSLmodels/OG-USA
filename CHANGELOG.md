# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [0.6.0] - 2026-09-23 09:00:00

### Added

- Documentation chapter `docs/book/content/calibration/matching_lwi.md` describing the calibration of `chi_n`, `beta_annual`, and `chi_b`: data targets, the inversion and least-squares steps, the general-equilibrium loop, standard errors, time-path validation, and the fit against CPS hours and SCF wealth moments. `PREFERENCE_CALIBRATION_GUIDE.md` summarizes the approach for other country calibrations.
- `Calibration(estimate_lifecycle_prefs=True)` runs the nested lifecycle preference calibration on a copy of the parameters that already carries the class's other outputs (tax functions, `e`, `eta`, `zeta`, demographics, macro parameters) and returns `beta_annual`, `chi_b`, and `chi_n` from `get_dict()`. `lifecycle_params_path` reads a saved JSON when its dimensions match the model and writes the result otherwise; `lifecycle_config`, `lifecycle_options`, `lifecycle_initial_ss`, and `lifecycle_kwargs` pass through to `calibrate_lifecycle_preferences`. `estimate_chi_n` is a deprecated alias.
- `ogusa/calibrate_lifecycle.py`: `preference_inference` computes standard errors for `beta_annual` and `chi_b` by type from the least-squares Jacobian of the household-only calibration step (classical nonlinear least squares, or a sandwich with a bootstrap covariance of the data moments carried to the targets by `preference_target_selection`) and, with a moment covariance, the overidentification test. `PreferenceCalibrationResult` now stores the Jacobian and target weights.
- `examples/run_lifecycle_calibration.py` runs the calibration through the `Calibration` class and writes the parameter JSON, moment comparison, outer-loop history, optional standard errors, and the hours, wealth, and `chi_n` comparison figures. `examples/validate_lifecycle_time_path.py` solves a baseline and a reform time path at calibrated parameters and records convergence and Euler errors.
- `ogusa/calibrate_lifecycle.py`: household-only steady-state solve (`HouseholdEnvironment`, `solve_households`, `partial_equilibrium_ss`) that re-solves every lifetime-income type's Euler equations at fixed prices, transfers, bequests, and scaling factor from an OG-Core steady-state output. It reproduces the general-equilibrium household solution at equilibrium prices in well under a second serially and is the inner loop for the preference-parameter calibration.
- `ogusa/calibrate_lifecycle.py`: `calibrate_lifecycle_preferences` runs the full nested calibration: general-equilibrium steady state, `chi_n` inversion, `beta` and `chi_b` calibration, re-inversion, and a warm-started general-equilibrium re-solve (`solve_ge_steady_state`), repeated with adaptive damping until parameters and prices settle. Returns a `LifecycleCalibrationOutcome` with per-pass diagnostics and a data-versus-model moment table.
- `ogusa/calibrate_lifecycle.py`: `calibrate_beta_chi_b` calibrates `beta_annual` by type and `chi_b` by type group at fixed prices with bounded nonlinear least squares over household-only solves, targeting SCF wealth shares by type bin, SCF mean wealth over mean income, by-bin old-age wealth tilts, and the mortality-weighted bequest-flow ratio. `PreferenceCalibrationOptions` selects the `chi_b` grouping and whether the structurally unmatchable bottom-half bin is excluded.
- New moments in `estimate_lifecycle_params.py`: `wealth_income_ratio` (SCF mean net worth over mean income, pre-transfer or total concept), `bequest_flow_ratio` (wealth of decedents over wealth of the living using the model's mortality on both sides), and by-bin old-age tilts (`tilt_*`); `merged_type_groups` and `percentile_bin_shares` helpers.
- SCF extracts in `ogusa/data/SCF` now carry total pre-tax income and its components (`data/download_moment_data.py`); `wealth.get_wealth_data` accepts `include_income`.
- `ogusa/calibrate_lifecycle.py`: `invert_chi_n` chooses the `chi_n` age profile so population-weighted model hours match CPS hours at each age 20 to 79 at fixed prices, by iterating on the labor first-order condition (`chi_n_update`). Ages beyond the last target are filled by the configured tail method, values are clipped to the ParamTools range, and ages where the cap binds are reported.

### Changed

- `ogusa/estimate_lifecycle_params.py` now builds the default preference-calibration moment set as hours by single year of age (CPS, lightly smoothed), one SCF wealth share per lifetime-income type with percentile bins taken from `p.lambdas`, and the ratio of mean SCF net worth at ages 75-79 to ages 60-64. The normalized wealth-by-age profile, income Gini, gross saving rate, wealth Gini, variance of log wealth, and aggregate bequests over GDP are optional or diagnostic moments. See `LIFECYCLE_CALIBRATION_PLAN.md`.
- Age aggregation of model moments uses the (S, J) steady-state population distribution from OG-Core rather than `lambdas` alone.
- `SS.SS_solver` warm starts are called by keyword against the installed OG-Core signature, and a failed warm start logs a warning before falling back to a cold solve.
- Moved `tool.uv.dev-dependencies` into `[dependency-groups] dev` in `pyproject.toml`.

### Fixed

- `calibrate_beta_chi_b` now supplies its own finite-difference Jacobian with an absolute step (`PreferenceCalibrationOptions.diff_step`, now absolute in the transformed space) taken from a common household guess. SciPy's `diff_step` is relative to the parameter value, and because the parameterization starts at zero it silently fell back to a step of about 1.5e-8, far below the reproducibility of the household solve, so the least-squares Jacobian was mostly solver noise.
- `Calibration.get_dict` referenced attributes that were never set for `estimate_beta` and `estimate_chi_n`; the legacy `estimate_beta` path now passes the current `beta_annual` as the initial guess and returns the estimate.
- `wealth.compute_wealth_moments` no longer drops the wealthiest observation from the top percentile bin, so shares sum to one.
- Wealth-by-age model moments map age `a` to `b_sp1[a - starting_age - 1]`, the savings actually held at age `a`.
- DFO-LS bounds in `estimate_lifecycle_params` are built in the transformed (logit/log) parameter space from the ParamTools validators intersected with configurable bounds; the previous code raised on scalar concatenation.
- Solver failures inside the SMM residual and objective return a bounded penalty instead of `1e15`.

### Removed

- `ogusa/calibrate_chi_n.py`, which targeted an OG-Core API that no longer exists.

## [0.5.0] - 2026-07-25 12:00:00

### Fixed

- Updates parameters and `income.py` to work with neew OG-Core demographics parameter object arrays.

## [0.4.0] - 2026-06-15 12:00:00

### Changed

- Migrated the project from conda to uv. Install with `uv sync --extra dev`; `pyproject.toml` is the single source of truth for dependencies and `uv.lock` pins exact versions.
- CI uses `astral-sh/setup-uv`, and ruff replaces black for formatting and linting (`check_format.yml` -> `check_ruff.yml`).
- Updated the README, `AGENTS.md`, and the Makefile to the uv workflow.

### Removed

- `setup.py`, `environment.yml`, `pytest.ini`, and `MANIFEST.in` (their settings moved into `pyproject.toml`).


## [0.3.3] - 2026-06-11 15:30:00

### Added

- Module `compute_moments.py` to calculate moments from US data. Data from the CPS and SCF used in `compute_moments.py` are checked into the repo in the `data/` directory. ([PR #1138](https://github.com/PSLmodels/OG-USA/pull/1138))

## [0.3.2] - 2026-04-03 18:30:00

### Added

- Streamlines the building of the exogenous parameters table in the docs.
- Replaces `pandas-datareader` with `fredapi` in `macro_params.py`
- Adds additional tests coverate for `calibrate.py` and `macro_params.py` modules.
- New module, `update_baseline.py` added
- `Makefile` has new or updated commands to update baseline calibration, build docs
- Updates documents and fixes `README.md` and intro

## [0.3.1] - 2025-09-24 18:00:00

### Bug Fixes

- Updates the `default_parameters.json` file specify HSV tax functions, which is what the tax function parameters were already specificed for.


## [0.3.0] - 2025-09-06 11:00:00

### Added

- Updates the `default_parameters.json` file to represent post-2025 reconciliation bill values of USA parameters and economic conditions.

## [0.2.4] - 2025-08-15 21:00:00

### Added

- Updates for Python 3.13

## [0.2.3] - 2025-06-12 12:00:00

### Added

- Updates `utils.read_cbo_forecast` to use 2025 forecasts
- Environment uses more recent `paramtools` to avoid `marshmallow` dependency issues

## [0.2.2] - 2025-04-25 12:00:00

### Added

- Updates `get_micro_data.py` to use the new taxcalc TMD constructor method. To do this, arguments had to be passed all the way through from the Calibration class object in `calibrate.py`.
- Renames and updates `run_ogusa.py`, `run_ogusa_tmd.py`, and `run_current_policy_baseline.py` run scripts.
- Updates `.gitignore` and `README.md`.
- Updates `test_get_micro_data.py` test `test_tmd_path()` to be consistent with new code in `calibrate.py` and `get_micro_data.py`
- Updates the `test_run_example.py` test to correspond to the new run script names.
- Adds a `make format` command to `Makefile`.

## [0.2.1] - 2024-10-05 12:00:00

### Added

- Adds `eta_RM` to default calibration to work with new OG-Core parameterization.

## [0.2.0] - 2024-08-27 12:00:00

### Added

- Updated default calibration to represent 2024 values of USA parameters and economic conditions.

## [0.1.12] - 2024-08-26 12:00:00

### Added

- Streamlined the `run_og_usa.py` script to make the example more clear, run faster, and save output in a common directory.

## [0.1.11] - 2024-07-26 12:00:00

### Added

- Adds a module to update Tax-Calculator growth factors using OG-USA simulations.


## [0.1.10] - 2024-06-10 12:00:00

### Added

- Removes the `rpy2` dependency from the `environment.yml` and `setup.py` files, and modifies use of PSID data to avoid needing this package in OG-USA.


## [0.1.9] - 2024-06-07 12:00:00

### Added

- Updates the `get_micro_data.py` and `calibration.py` modules to allow for the user to use the CPS, PUF, and TMD files with Tax-Calculator or to provide their own custom datafile, with associated grow factors and weights.


## [0.1.8] - 2024-05-20 12:00:00

### Added

- Updates the `ogusa` package to include the zipped `psid_lifetime_income.csv.gz` file, which is now called in some calibration modules (`bequest_transmission.py`,  `deterministic_profiles.py`, and `transfer_distirbution.py`), but with an option for the user to provide their own custom datafile.  These changes allow for Jupyter notebook users to execute the `Calibration` class object and for those who install the `ogusa` package from PyPI to have the required datafile for the major calibration modules.


## [0.1.7] - 2024-05-14 16:30:00

### Added

- Updates the dependency `rpy2>=3.5.12` in `environment.yml` and `setup.py`.


## [0.1.6] - 2024-05-08 10:30:00

### Added

- PR [#99](https://github.com/PSLmodels/OG-USA/pull/99), updating the continuous integration tests
- PR [#101](https://github.com/PSLmodels/OG-USA/pull/101), which sets plotting to "off" by default for the  `Calibrate` class
- PR [#102](https://github.com/PSLmodels/OG-USA/pull/102), PR [#103](https://github.com/PSLmodels/OG-USA/pull/103), PR [#104](https://github.com/PSLmodels/OG-USA/pull/104), which change dask client parameters for better memory performance
- PR [#106](https://github.com/PSLmodels/OG-USA/pull/106), which allows for alternative policy baselines and updates calls to the `ogcore.txfunc` module.
- Updated `build_and_test.yml` to run on Python 3.10 and 3.11 (dropped Python 3.9)


## [0.1.5] - 2024-04-12 10:00:00

### Added

- Adds a list of file change event triggers to `build_and_test.yml` so that those tests only run when one of those files is changed.
- Updates the codecov GH Action to version 4 and adds a secret token.
- Adds a list of file change event triggers to `deploy_docs.yml` and `docs_check.yml`, and limits `docs_check.yml` to only run on pull requests.
- Fixes a small typo in `tax_functions.md` in order to test if the event triggers worked properly (yes, they worked)
- Updated some dependencies in `environment.yml`.
- Updated three data files in the `/tests/test_io_data/` file that used output from the taxcalc package. This package was recently updated. I also changed the `test_get_data()` test in the `test_get_micro_data.py` file because the new taxcalc data included four years instead of two years. In order to conserve repo memory footprint, we deleted the last two years of the output.

## [0.1.4] - 2024-04-03 15:00:00

### Added

- PRs, #91, #93, and #94 update the configuration of Compute Studio hosted OG-USA web apps
- PR #89 adds more CI tests, updates the Jupyter Book documentation, and make fixes for the latest `pandas-datareader`
- PR #87 updates the `run_og_usa.py` script for better use of `dask` multiprocessing

## [0.1.3] - 2024-02-12 15:00:00

### Added

- Restricts Python version in `environment.yml` and `setup.py` to be <3.12
- Updates the Jupyter Book copyright to 2024 in `_config.yml`
- Updates the pandas_datareader quarterly calls in `macro_params.py` to be "QE" instead of just "Q"
- Adds Jupyter Book and Black tags to `README.md` and `intro.md`
- Adds back Windows tests to `build_and_test.yml`
- PR #84 fixed some formatting
- PR #85 updated the way the dask client is set in `run_og_usa.py` script
- PR #86 moved `demographics.py` out of OG-USA and into OG-Core

## [0.1.2] - 2023-10-26 15:00:00

### Added

- Simple update of version in `setup.py` and `cs-config/cs_config/functions.py` to make sure that the `publish_to_pypi.yml` GitHub Action works
- Removes Windows OS tests from `build_and_test.yml`, which are not working right now for some reason.

## [0.1.1] - 2023-10-25 17:00:00

### Added

- Updates `README.md`
- Changes `check_black.yml` to `check_format.yml`
- Updates other GH Action files: `build_and_test.yml`, `docs_check.yml`, and `deploy_docs.yml`
- Updates `publish_to_pypi.yml`
- Adds changes from PRs [#73](https://github.com/PSLmodels/OG-USA/pull/73) and [#67](https://github.com/PSLmodels/OG-USA/pull/67)

## [0.1.0] - 2023-07-19 12:00:00

### Added

- Restarts the release numbering to follow semantic versioning and the OG-USA version numbering as separate from the OG-Core version numbering.
- Adds restriction `python<3.11` to `environment.yml` and `setup.py`.
- Changes the format of `setup.py`.
- Updates `build_and_test.yml` to test Python 3.9 and 3.10.
- Updates some GH Action script versions in `check_black.yml`.
- Updates the Python version to 3.10 in  `docs_check.yml` and `deploy_docs.yml`.
- Updated the `LICENSE` file to one that GitHub recognizes.
- Updates the `run_og_usa.py` run script.
- Updates some tests and associated data.
- Pins the version of `rpy2` package in `environment.yml` and `setup.py`


## Previous versions

### Summary

- Version [0.7.0] on August 30, 2021 was the first time that the OG-USA repository was detached from all of the core model logic, which was named OG-Core. Before this version, OG-USA was part of what is now the [`OG-Core`](https://github.com/PSLmodels/OG-Core) repository. In the next version of OG-USA, we adjusted the version numbering to begin with 0.1.0. This initial version of 0.7.0, was sequential from what OG-USA used to be when the OG-Core project was called OG-USA.
- Any earlier versions of OG-USA can be found in the [`OG-Core`](https://github.com/PSLmodels/OG-Core) repository [release history](https://github.com/PSLmodels/OG-Core/releases) from [v.0.6.4](https://github.com/PSLmodels/OG-Core/releases/tag/v0.6.4) (Jul. 20, 2021) or earlier.


[0.5.0]: https://github.com/PSLmodels/OG-USA/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/PSLmodels/OG-USA/compare/v0.3.3...v0.4.0
[0.3.3]: https://github.com/PSLmodels/OG-USA/compare/v0.3.2...v0.3.3
[0.3.2]: https://github.com/PSLmodels/OG-USA/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/PSLmodels/OG-USA/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/PSLmodels/OG-USA/compare/v0.2.4...v0.3.0
[0.2.4]: https://github.com/PSLmodels/OG-USA/compare/v0.2.3...v0.2.4
[0.2.3]: https://github.com/PSLmodels/OG-USA/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/PSLmodels/OG-USA/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/PSLmodels/OG-USA/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/PSLmodels/OG-USA/compare/v0.1.12...v0.2.0
[0.1.12]: https://github.com/PSLmodels/OG-USA/compare/v0.1.11...v0.1.12
[0.1.11]: https://github.com/PSLmodels/OG-USA/compare/v0.1.10...v0.1.11
[0.1.10]: https://github.com/PSLmodels/OG-USA/compare/v0.1.9...v0.1.10
[0.1.9]: https://github.com/PSLmodels/OG-USA/compare/v0.1.8...v0.1.9
[0.1.8]: https://github.com/PSLmodels/OG-USA/compare/v0.1.7...v0.1.8
[0.1.7]: https://github.com/PSLmodels/OG-USA/compare/v0.1.6...v0.1.7
[0.1.6]: https://github.com/PSLmodels/OG-USA/compare/v0.1.5...v0.1.6
[0.1.5]: https://github.com/PSLmodels/OG-USA/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/PSLmodels/OG-USA/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/PSLmodels/OG-USA/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/PSLmodels/OG-USA/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/PSLmodels/OG-USA/compare/v0.1.0...v0.1.1
