# OG-USA-Calibration
This repository contains source code and data used to calibrate the OG-USA model.

The organization of this repository will be changing rapidly, but we offer the following guide for working with the existing scripts:

* The calibration of macro parameters needs no source data - one can simply run the script `/Macro/macro_params_FRED.py`, which downloads all necessary data. The resulting parameter values will be printed to the screen from a Python dictionary object.
* The calibration of earnings processes, the bequest transmission process, and the distribution of government transfers all rely on a common dataset built from the PSID.
  * This dataset is built through the following steps:
    * (Optional) Run `/Data/PSID/psid_download.R` to download data from the PSID. This saves a file, `/Data/PSID/psid1968to2017.RData`.  It can take a while to download these data so the result, `psid1968to2017.RData` is checked into the repo.  But you may wish to add addtitional variables or years and so may want to run the download script again.
    * Create a Python Pandas DataFrame by running `/Data/PSID/psid_data_setup.py`, which saves a Pickle file with the DataFrame to `/Data/PSID/psid_lifetime_income.pkl`.  This DataFrame contains the PSID data with inflation adjustments to nominal values and adds some additional constructed variables, including the lifetime income group for each household.
* To estimate lifetime earnings processes, run `EarningsProcesses/psid_deterministic_profiles.py`
* To estimate the bequest transmission processes, run `BequestTransmission/bequest_tabs.py`
* To estimate the distribution of government transfers, run `TransferDistribution/transfer_tabs.py`
