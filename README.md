# OG-USA-Calibration
This repository contains source code and data used to calibrate the OG-USA model.

## Using/contributing to OG-USA-Calibration

* Install the [Anaconda distribution](https://www.anaconda.com/distribution/) of Python
* Clone this repository to a directory on your computer
* From the terminal (or Conda command prompt), navigate to the directory to which you cloned this repository and run `conda env create -f environment.yml`
* Then, `conda activate ogusa-dev`
* Then install by `pip install -e .`
* You will also need to install OG-USA package, which you can build from source following the instructions in the [OG-USA repo README](https://github.com/PSLmodels/OG-USA/)

Once the package is installed, you an example of this to update the parameters used in OG-USA using Python is the following:

```
from ogcore.parameters import Specifications
from ogusa.calibrate import Calibration
p = Specifications()
c = Calibration(p)
updated_params = c.get_dict()
p.update_specifications({'initial_debt_ratio': updated_params['initial_debt_ratio']})
```
# Disclaimer:
The organization of this repository will be changing rapidly, but the `OG-USA/examples/run_og_usa.py` script will be kept up to date to run with the master branch of this repo.

