# OG-USA

| | |
| --- | --- |
| Org | [![PSL cataloged](https://img.shields.io/badge/PSL-cataloged-a0a0a0.svg)](https://www.PSLmodels.org) [![OS License: CC0-1.0](https://img.shields.io/badge/OS%20License-CC0%201.0-yellow)](https://github.com/PSLmodels/OG-USA/blob/master/LICENSE) [![Jupyter Book Badge](https://jupyterbook.org/badge.svg)](https://pslmodels.github.io/OG-Core/) |
| Package | [![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-31013/) [![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3116/) [![PyPI Latest Release](https://img.shields.io/pypi/v/ogusa.svg)](https://pypi.org/project/ogusa/) [![PyPI Downloads](https://img.shields.io/pypi/dm/ogusa.svg?label=PyPI%20downloads)](https://pypi.org/project/fiscalsim-us/) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) |
| Testing | ![example event parameter](https://github.com/PSLmodels/OG-USA/actions/workflows/build_and_test.yml/badge.svg?branch=master) ![example event parameter](https://github.com/PSLmodels/OG-USA/actions/workflows/deploy_docs.yml/badge.svg?branch=master) ![example event parameter](https://github.com/PSLmodels/OG-USA/actions/workflows/check_format.yml/badge.svg?branch=master) [![Codecov](https://codecov.io/gh/PSLmodels/OG-USA/branch/master/graph/badge.svg)](https://codecov.io/gh/PSLmodels/OG-USA) |

OG-USA is an overlapping-generations (OG) model that allows for dynamic general equilibrium analysis of fiscal policy for the United States. OG-USA is built on the [OG-Core](https://github.com/PSLmodels/OG-Core) framework. The model output includes changes in macroeconomic aggregates (GDP, investment, consumption), wages, interest rates, and the stream of tax revenues over time. Regularly updated documentation of the model theory--its output, and solution method--and the Python API is available at [https://pslmodels.github.io/OG-Core](https://pslmodels.github.io/OG-Core) and documentation of the specific United States calibration of the model is available at [https://pslmodels.github.io/OG-USA](https://pslmodels.github.io/OG-USA).


## Disclaimer

The model is constantly under development, and model components could change significantly. The package will have released versions, which will be checked against existing code prior to release. Stay tuned for an upcoming release!



## Using/contributing to OG-USA

* Install the [Anaconda distribution](https://www.anaconda.com/distribution/) of Python
* Clone this repository to a directory on your computer
* From the terminal (or Conda command prompt), navigate to the directory to which you cloned this repository and run `conda env create -f environment.yml`. The process of creating the `ogusa-dev` conda environment can take more than 20 minutes. The pip install of the `OG-Core` dependency from GitHub takes most of the time.
* Then, `conda activate ogusa-dev`
* Then install by `pip install -e .`
* Navigate to `./examples`
* Run the model with an example reform from terminal/command prompt by typing `python run_og_usa.py`
* You can adjust the `./examples/run_og_usa.py` by modifying model parameters specified in the dictionary passed to the `p.update_specifications()` calls.
* Model outputs will be saved in the following files:
  * `./examples/OG-USA_example_plots`
    * This folder will contain a number of plots generated from OG-Core to help you visualize the output from your run
  * `./examples/ogusa_example_output.csv`
    * This is a summary of the percentage changes in macro variables over the first ten years and in the steady-state.
  * `./examples/OG-USA-Example/OUTPUT_BASELINE/model_params.pkl`
    * Model parameters used in the baseline run
    * See [`ogcore.execute.py`](https://github.com/PSLmodels/OG-Core/blob/master/ogcore/execute.py) for items in the dictionary object in this pickle file
  * `./examples/OG-USA-Example/OUTPUT_BASELINE/SS/SS_vars.pkl`
    * Outputs from the model steady state solution under the baseline policy
    * See [`ogcore.SS.py`](https://github.com/PSLmodels/OG-Core/blob/master/ogcore/SS.py) for what is in the dictionary object in this pickle file
  * `./examples/OG-USA-Example/OUTPUT_BASELINE/TPI/TPI_vars.pkl`
    * Outputs from the model timepath solution under the baseline policy
    * See [`ogcore.TPI.py`](https://github.com/PSLmodels/OG-Core/blob/master/ogcore/TPI.py) for what is in the dictionary object in this pickle file
  * An analogous set of files in the `./examples/OUTPUT_REFORM` directory, which represent objects from the simulation of the reform policy

Note that, depending on your machine, a full model run (solving for the full time path equilibrium for the baseline and reform policies) can take more than two hours of compute time.

If you run into errors running the example script, please open a new issue in the OG-USA repo with a description of the issue and any relevant tracebacks you receive.

Once the package is installed, one can adjust parameters in the OG-Core `Specifications` object using the `Calibration` class as follows:

```
from ogcore.parameters import Specifications
from ogusa.calibrate import Calibration
p = Specifications()
c = Calibration(p)
updated_params = c.get_dict()
p.update_specifications({'initial_debt_ratio': updated_params['initial_debt_ratio']})
```


## Core Maintainers

The core maintainers of the OG-Core repository are:

* [Jason DeBacker](https://www.jasondebacker.com/) (GitHub handle: [jdebacker](https://github.com/jdebacker)), Associate Professor, Department of Economics, Darla Moore School of Business, University of South Carolina; President, PSL Foundation; Vice President of Research and Co-founder, Open Research Group, Inc.
* [Richard W. Evans](https://sites.google.com/site/rickecon/) (GitHub handle: [rickecon](https://github.com/rickecon)), Senior Research Fellow and Director of Open Policy, Center for Growth and Opportunity at Utah State University; President, Open Research Group, Inc.; Director, Open Source Economics Laboratory.

## Citing OG-USA

OG-USA (Version #.#.#)[Source code], https://github.com/PSLmodels/OG-USA.
