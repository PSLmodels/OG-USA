(Chap_Intro)=
# OG-USA

| | |
| --- | --- |
| Org | [![PSL cataloged](https://img.shields.io/badge/PSL-cataloged-a0a0a0.svg)](https://www.PSLmodels.org) [![OS License: CC0-1.0](https://img.shields.io/badge/OS%20License-CC0%201.0-yellow)](https://github.com/PSLmodels/OG-USA/blob/master/LICENSE) [![Jupyter Book Badge](https://jupyterbook.org/badge.svg)](https://pslmodels.github.io/OG-Core/) |
| Package | [![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3116/) [![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3129/) [![PyPI Latest Release](https://img.shields.io/pypi/v/ogusa.svg)](https://pypi.org/project/ogusa/) [![PyPI Downloads](https://img.shields.io/pypi/dm/ogusa.svg?label=PyPI%20downloads)](https://pypi.org/project/ogusa/) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) |
| Testing | ![example event parameter](https://github.com/PSLmodels/OG-USA/actions/workflows/build_and_test.yml/badge.svg?branch=master) ![example event parameter](https://github.com/PSLmodels/OG-USA/actions/workflows/deploy_docs.yml/badge.svg?branch=master) ![example event parameter](https://github.com/PSLmodels/OG-USA/actions/workflows/check_format.yml/badge.svg?branch=master) [![Codecov](https://codecov.io/gh/PSLmodels/OG-USA/branch/master/graph/badge.svg)](https://codecov.io/gh/PSLmodels/OG-USA) |

`OG-USA` is a package that provides code and data to calibrate an overlapping-generations (OG) model to the economy of the United States (USA). `OG-USA` uses as a dependency the [`OG-Core`](https://pslmodels.github.io/OG-Core/) package, which contains the core theory and logic of a general OG model. The `OG-USA` calibration package and the `OG-Core` theory and logic make the model that allows for dynamic general equilibrium analysis of federal fiscal policy in the United States. The model output focuses on changes in macroeconomic aggregates (GDP, investment, consumption), wages, interest rates, and the stream of tax revenues over time. This documentation of the `OG-USA` package contains the following major sections, which are regularly updated.

* Contributing to `OG-USA`
* `OG-USA` API
* Calibration
* References
* Citations of `OG-USA`


(Sec_CoreMaintainers)=
## Core Maintainers

[Jason DeBacker](https://www.jasondebacker.com/) (GitHub handle [@jdebacker](https://github.com/jdebacker)) and [Richard W. Evans](https://sites.google.com/site/rickecon/) (GitHub handle [@rickecon](https://github.com/rickecon)) are the core maintainers of `OG-USA`. If you have questions about or contributions to the model or repository, please submit a GitHub "Issue" described in the {ref}`Sec_GitHubIssue` subsection or "Pull Request" as described in the {ref}`Sec_GitHubPR` subsection of the {ref}`Sec_Workflow` section of the `OG-USA` {ref}`Chap_ContribGuide`.


(Sec_Disclaimer)=
## Disclaimer

The model is continuously under development. Users will be notified through [closed PR threads](https://github.com/PSLmodels/OG-USA/pulls?q=is%3Apr+is%3Aclosed) and through the [release notes](https://github.com/PSLmodels/OG-USA/releases) what changes have been implemented. The package will have released versions, which will be checked against existing code prior to release. Stay tuned for an upcoming release!


(Sec_CitingOGUSA)=
## Citing OG-USA

`OG-USA` (Version #.#.#)[Source code], https://github.com/PSLmodels/OG-USA
