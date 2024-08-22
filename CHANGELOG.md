# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.12] - 2024-08-21 12:00:00

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
