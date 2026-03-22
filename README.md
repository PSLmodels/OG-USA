# OG-USA

| | |
| --- | --- |
| Org | [![PSL cataloged](https://img.shields.io/badge/PSL-cataloged-a0a0a0.svg)](https://www.PSLmodels.org) [![OS License: CC0-1.0](https://img.shields.io/badge/OS%20License-CC0%201.0-yellow)](https://github.com/PSLmodels/OG-USA/blob/master/LICENSE) [![Jupyter Book Badge](https://jupyterbook.org/badge.svg)](https://pslmodels.github.io/OG-Core/) |
| Package | [![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3129/) [![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/release/python-3137/) [![PyPI Latest Release](https://img.shields.io/pypi/v/ogusa.svg)](https://pypi.org/project/ogusa/) [![PyPI Downloads](https://img.shields.io/pypi/dm/ogusa.svg?label=PyPI%20downloads)](https://pypi.org/project/ogusa/) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) |
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
* Run the model with an example reform from terminal/command prompt by typing `python run_ogusa.py`
* You can adjust the `./examples/run_ogusa.py` by modifying model parameters specified in the dictionary passed to the `p.update_specifications()` calls.
* Model outputs will be saved in the following files:
    * `./examples/Example/`: This folder will contain all of the output from the `run_ogusa.py` run script.
        * `./examples/Example/example_plots_tables`: This folder will contain a number of plots and tables generated from the `run_ogusa.py` run script to help you visualize the output.
        * `./examples/Example/example_output.csv`: This is a summary of the percentage changes in macro variables over the first ten years and in the steady-state.
    * `./examples/Example/OUTPUT_BASELINE/`: This folder contains all of the inputs to and outputs from the baseline equilibrium computation from `run_ogusa.py`
        * `./examples/Example/OUTPUT_BASELINE/model_params.pkl`: Pickle binary file of ParamTools object of model parameters used in the baseline run
        * `./examples/Example/OUTPUT_BASELINE/SS/SS_vars.pkl`: Pickle binary file of Python dictionary of outputs from the model steady state solution under the baseline policy. See [`ogcore.SS.py`](https://github.com/PSLmodels/OG-Core/blob/master/ogcore/SS.py) for what is in the dictionary object in this pickle file
        * `./examples/Example/OUTPUT_BASELINE/TPI/TPI_vars.pkl`: Pickle binary file of Python dictionary of outputs from the model timepath solution under the baseline policy. See [`ogcore.TPI.py`](https://github.com/PSLmodels/OG-Core/blob/master/ogcore/TPI.py) for what is in the dictionary object in this pickle file
    * An analogous set of files in the `./examples/OUTPUT_REFORM` directory, which represent objects from the simulation of the reform policy.

Note that, depending on your machine, a full model run (solving for the full time path equilibrium for the baseline and reform policies) can take more than two hours of compute time.

If you run into errors running the example script, please open a new issue in the OG-USA repo with a description of the issue and any relevant tracebacks you receive.

## Interactive Web Application

OG-USA includes a browser-based GUI that lets you adjust model parameters, launch a run, and view results without writing any Python.

### Launch locally

**Option 1 — double-click (macOS)**

After completing the setup steps above, make the launcher executable once:
```bash
chmod +x launch_ogusa.command
```
Then double-click `launch_ogusa.command` in Finder. A Terminal window will open, activate the `ogusa-dev` environment, start the Panel server, and open the app in your default browser at `http://localhost:5006/app`.

**Option 2 — command line**

```bash
conda activate ogusa-dev
panel serve ogusa/app/app.py --show --address localhost --port 5006 --prefix /app --allow-websocket-origin localhost:5006
```

### Hosted deployment

To serve the app on a remote machine (e.g., an EC2 instance or a university server), run:
```bash
panel serve ogusa/app/app.py --address 0.0.0.0 --port 5006 --allow-websocket-origin <your-domain-or-ip>:5006
```
Place an nginx reverse proxy in front of it for HTTPS in production.

### Email notifications (optional)

For runs that take 30–90 minutes, the app can email you when a run finishes. Set the following environment variables before launching:

```bash
export OGUSA_SMTP_HOST=smtp.gmail.com   # your SMTP server
export OGUSA_SMTP_PORT=587
export OGUSA_SMTP_USER=you@example.com
export OGUSA_SMTP_PASS=your-app-password
```

Then enter your address in the **Notify email** field in the app sidebar before clicking **Run Model**.

### Output location

Each run saves baseline and reform output to a timestamped subdirectory of `~/.ogusa_app/runs/`, e.g.:
```
~/.ogusa_app/runs/20250321_143022/OUTPUT_BASELINE/
~/.ogusa_app/runs/20250321_143022/OUTPUT_REFORM/
```

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

* [Jason DeBacker](https://www.jasondebacker.com/) (GitHub handle: [jdebacker](https://github.com/jdebacker)), Professor, Department of Economics, Darla Moore School of Business, University of South Carolina; President, PSL Foundation; Vice President of Research and Co-founder, Open Research Group, Inc.
* [Richard W. Evans](https://sites.google.com/site/rickecon/) (GitHub handle: [rickecon](https://github.com/rickecon)), Senior Economist, Abundance Institute; President, Open Research Group, Inc.; Director, Open Source Economics Laboratory.

## Citing OG-USA

OG-USA (Version #.#.#)[Source code], https://github.com/PSLmodels/OG-USA.
