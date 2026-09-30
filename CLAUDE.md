# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository. `AGENTS.md` holds the same conventions in shorter form; keep the two consistent.

## Project Overview

OG-USA is an overlapping-generations (OG) model for dynamic general equilibrium analysis of fiscal policy in the United States. It is built on the [OG-Core](https://github.com/PSLmodels/OG-Core) framework. The model calibrates US-specific parameters including:

- Tax functions estimated from microdata (Tax-Calculator integration)
- Demographics and population dynamics
- Wealth and income distributions from PSID and SCF data
- Macroeconomic parameters from CBO forecasts
- Transfer and bequest distributions
- Household preference parameters (`beta_annual`, `chi_b`, `chi_n`); see `LIFECYCLE_CALIBRATION_PLAN.md`

## Development Setup

### Environment Setup

The project uses [`uv`](https://docs.astral.sh/uv/) as its package manager. Conda is no longer used; the conda environments that may exist on a development machine are stale and cannot load the current default parameters.

```bash
uv sync --extra dev          # creates .venv at the repo root
uv sync --extra dev --extra docs   # also for Jupyter Book work
```

Run everything through `uv run` so the project virtualenv is used without activation:

```bash
uv run python examples/run_ogusa.py
uv run pytest tests/test_calibrate.py
```

**Python versions**: 3.12 and 3.13 (`requires-python` in `pyproject.toml`).

`pyproject.toml` is the single source of truth for dependencies and `uv.lock` pins exact versions. Add dependencies with `uv add <pkg>`, not pip.

### Running Tests
```bash
# Default suite (matches CI; skips long local-only, PUF, and TMD tests)
uv run python -m pytest -m "not local" -q
make test

# Targeted, fast
uv run python -m pytest tests/test_macro_params.py tests/test_calibrate.py -q

# Coverage
make coverage
```

**Test markers**: `local`, `needs_puf`, `needs_tmd`, `needs_fred` (see `[tool.pytest.ini_options]` in `pyproject.toml`). Tests that hit FRED need the `FRED_API_KEY` environment variable.

### Code Formatting and Linting

Ruff replaces black. Line length is 79.

```bash
make format   # uv run ruff format . ; uv run ruff check . --fix ; uv run linecheck . --fix
make lint     # CI check, no changes: ruff format --check . ; ruff check .
```

Sequence: edit, format, test, stage, commit. Re-run tests after formatting.

### Building Documentation
The project uses Jupyter Book for documentation in the `docs/book/` directory. See `.github/workflows/deploy_docs.yml` and `docs_check.yml` for the build process.

## Running the Model

### Basic Example
```bash
uv run python examples/run_ogusa.py
```

**Runtime**: Full model runs (baseline + reform time path equilibrium) take roughly 35 minutes to 2 hours depending on machine.

**Output Structure**:
- `./examples/Example/OUTPUT_BASELINE/`: Baseline policy results
  - `model_params.pkl`: ParamTools parameters object
  - `SS/SS_vars.pkl`: Steady-state solution dictionary
  - `TPI/TPI_vars.pkl`: Time path iteration solution dictionary
- `./examples/Example/OUTPUT_REFORM/`: Reform policy results (same structure)
- `./examples/Example/example_plots_tables/`: Visualizations and summary tables
- `./examples/Example/example_output.csv`: Summary of macro variable changes

### Alternative Run Scripts
- `run_ogusa_tmd.py`: Uses TMD microdata instead of CPS
- `run_current_policy_baseline.py`: Current policy baseline
- Other example scripts in `examples/` for specific policy simulations

## Architecture

### Core Module: `ogusa/`

**calibrate.py**: Main `Calibration` class that orchestrates parameter estimation
- Instantiate with a `Specifications` object from OG-Core
- Estimates tax functions, demographics, income/wealth profiles, transfer matrices
- Key methods: `get_tax_function_parameters()`, `get_dict()`
- Returns calibrated parameters via `get_dict()` for updating `Specifications`

**Tax Function Estimation**:
- `get_micro_data.py`: Interface with Tax-Calculator for microsimulation
- Tax functions can use CPS (default), PUF, or TMD data
- Functions estimated by age and year over a "budget window" (BW)
- Parameters saved/loaded from pickle files (`TxFuncEst_baseline.pkl`, `TxFuncEst_policy.pkl`)

**Microdata-Based Calibration**:
- `income.py`: Earnings ability profiles (`e`) from CWHS hourly wages (hours imputed from CPS)
- `wealth.py`: SCF net worth data and wealth distribution moments
- `bequest_transmission.py`: Bequest matrix (`zeta`) by age/ability
- `transfer_distribution.py`: Government transfer matrix (`eta`) by age/ability
- `estimate_beta_j.py`: Legacy SMM for time preference by ability type
- `estimate_lifecycle_params.py`: Moment construction (CPS hours by age, SCF wealth shares with bins from `p.lambdas`, old-age wealth ratio) and an optional DFO-LS SMM inference driver for `beta_annual`, `chi_b`, `chi_n`
- `calibrate_lifecycle.py`: Nested calibration of `beta_annual`, `chi_b`, `chi_n` (household-only solves, `chi_n` inversion, `beta`/`chi_b` least squares, warm-started GE outer loop, standard errors via `preference_inference`); example flow in `examples/run_lifecycle_calibration.py`, time-path check in `examples/validate_lifecycle_time_path.py`
- `compute_moments.py`: Data moments from FRED, CPS, SCF, PSID, and Tax-Calculator
- Default PSID data: `psid_lifetime_income.csv.gz`; trimmed CPS and SCF extracts in `ogusa/data/`

**Macro Calibration**:
- `macro_params.py`: Parameters from national accounts and CBO forecasts
- `utils.py`: Contains `read_cbo_forecast()` to parse CBO Excel files
- CBO data provides: GDP, interest rates, wages, labor, government spending/revenue

**Parameter Specification**:
- Uses `ogcore.Specifications` (ParamTools-based) for all model parameters
- Default US parameters: `ogusa_default_parameters.json`
- Update with: `p.update_specifications(dict)`

### Key Dependencies

- **OG-Core** (`ogcore`): Core OG model framework (see `pyproject.toml` for the pinned minimum)
  - Provides: `Specifications`, `runner()`, solution algorithms, output utilities
  - `omega_SS` is an (S, J) joint distribution in current versions; aggregate over types with it rather than with `lambdas` alone
  - `SS.SS_solver`'s positional signature has changed across releases (a `G` argument was added); call it by keyword
- **Tax-Calculator** (`taxcalc`): Microsimulation for tax functions
- **Dask/Distributed**: Parallel computing for estimation and model solution
- **ParamTools**: Parameter handling and validation
- **DFO-LS** (`dfols`): Derivative-free least squares, used by the optional SMM inference driver

### Typical Workflow

1. Create `Specifications` object with `baseline=True`
2. Load default parameters from `ogusa_default_parameters.json`
3. Instantiate `Calibration` class with options:
   - `estimate_tax_functions`: Estimate from microdata or load cached
   - `estimate_beta`: Estimate time preference (legacy path)
   - `estimate_lifecycle_prefs`: Calibrate `beta_annual` by type, `chi_b`, and the `chi_n` age profile with the nested general-equilibrium routine (`calibrate_lifecycle.calibrate_lifecycle_preferences`); `lifecycle_params_path` caches the result as JSON. `estimate_chi_n` is a deprecated alias.
   - `estimate_pop`: Estimate demographics from UN data
4. Get calibrated parameters with `c.get_dict()`
5. Update `Specifications`: `p.update_specifications(c.get_dict())`
6. Run model: `ogcore.runner(p, time_path=True, client=dask_client)`
7. Repeat for reform policy with `baseline=False`
8. Generate output tables/plots using `ogcore.output_tables` and `ogcore.output_plots`

## Git Workflow

- **Main branch**: `master`
- **Current development**: See branch from `git status`
- CI runs on: `build_and_test.yml`, `check_ruff.yml`, `deploy_docs.yml`, `docs_check.yml`
- Platforms tested: Ubuntu, macOS, Windows (Python 3.12, 3.13)
- Read-only actions (status, log, file reads) need no approval. Mutating actions (edits, commits, pushes, rebases, branch deletion) need a plan and explicit approval first.
- Add a `CHANGELOG.md` entry with user-visible changes.

## Important Notes

- **Tax function caching**: Estimation is expensive; cached pickle files are reused if parameters match
- **Dask client**: Models use multiprocessing via Dask; initialize with `Client(n_workers=N)`
- **Budget window (BW)**: Number of years for which tax functions are estimated (from `start_year`)
- **Time path vs. SS**: `time_path=False` only solves steady state (faster for testing)
- **Model years**: `T` total periods, `S` age groups, `J` ability types
- **Age indexing**: `b_sp1[s]` is savings chosen at age index `s` and held at age index `s + 1`; wealth observed at age `a` maps to `b_sp1[a - starting_age - 1]`
- **OG-Core documentation**: https://pslmodels.github.io/OG-Core (for solution methods, theory)
- **OG-USA documentation**: https://pslmodels.github.io/OG-USA (for calibration details)

## Common Gotchas

- Ensure `baseline_dir` is set when running reform (to load baseline solution)
- Tax function parameters must match model dimensions (S, BW, start_year)
- PSID data file must be available for wealth/transfer calibrations
- CBO Excel URL formats change; `read_cbo_forecast()` may need updates for new years
- Large output files are not tracked in git (see `.gitignore`)
- `uv run` from the repo root uses `.venv`; the bare `python` on a developer machine may be an unrelated interpreter with an old OG-Core
