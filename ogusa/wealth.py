import os
import numpy as np
import pandas as pd
from ogcore import utils

CUR_PATH = os.path.split(os.path.abspath(__file__))[0]
SCF_DATA_DIR = os.path.abspath(os.path.join(CUR_PATH, "data", "SCF"))
# Pre-tax household income and components in the SCF summary extract.
SCF_INCOME_COLUMNS = [
    "income",
    "wageinc",
    "bussefarminc",
    "intdivinc",
    "kginc",
    "ssretinc",
    "transfothinc",
]


def get_wealth_data(
    scf_yrs_list=[2019, 2016, 2013, 2010, 2007],
    web=False,
    directory=None,
    include_age=False,
    include_income=False,
):
    """
    Reads wealth data from the 2007, 2010, 2013, 2016, and 2019 Survey of
    Consumer Finances (SCF) files.

    Args:
        scf_yrs_list (list): list of SCF years to import. Currently the
            largest set of years that will work is
            [2019, 2016, 2013, 2010, 2007]
        web (Boolean): =True if function retrieves data from internet.
            Defaults to False and uses local trimmed CSVs in ogusa/data/SCF.
        directory (string or None): local directory location if data are
            stored on local drive, not use internet (web=False)
        include_age (Boolean): =True if function keeps the respondent age
            from the SCF summary extract.
        include_income (Boolean): =True to keep total pre-tax household
            income and its components (``income``, ``wageinc``,
            ``bussefarminc``, ``intdivinc``, ``kginc``, ``ssretinc``,
            ``transfothinc``) plus ``income_infadj`` in 2019 dollars.


    Returns:
        df_scf (Pandas DataFrame): pooled cross-sectional data from SCFs

    """
    # Hard code cpi list for given years. Index values are annual average index
    # values from monthly FRED Consumer Price Index for All Urban Consumers:
    # All Items Less Food and Energy in U.S. City Average (CPILFESL,
    # https://fred.stlouisfed.org/series/CPILFESL). Base year is 1982-1984=100.
    # Values are [263.209, 247.585, 233.810, 221.336, 210.725].
    # We then reset the base year to 2019 by dividing each annual average by
    # the 2019 annual average and multiply by 100. Base year is 2019=100
    cpi_dict = {
        "cpi2019": 100.000,
        "cpi2016": 94.06403464,
        "cpi2013": 88.83067929,
        "cpi2010": 84.09125952,
        "cpi2007": 80.05995867,
    }
    if web:
        # Throw an error if the machine is not connected to the internet
        if utils.not_connected():
            err_msg = (
                "SCF DATA ERROR: The local machine is not "
                + "connected to the internet and web=True was "
                + "selected."
            )
            raise RuntimeError(err_msg)

        file_urls = []
        for yr in scf_yrs_list:
            zipfilename = (
                "https://www.federalreserve.gov/econres/"
                + "files/scfp"
                + str(yr)
                + "s.zip"
            )
            file_urls.append(zipfilename)

        file_paths = utils.fetch_files_from_web(file_urls)

    elif not web:
        file_paths = []
        if directory is None:
            directory = SCF_DATA_DIR
        full_directory = os.path.expanduser(directory)
        filename_list = []
        for yr in scf_yrs_list:
            csv_filename = "scf_wealth_" + str(yr) + ".csv"
            dta_filename = "rscfp" + str(yr) + ".dta"
            if os.path.isfile(os.path.join(full_directory, csv_filename)):
                filename = csv_filename
            else:
                filename = dta_filename
            filename_list.append(filename)

        for name in filename_list:
            file_paths.append(os.path.join(full_directory, name))
        # Check to make sure the necessary files are present in the
        # local directory
        err_msg = (
            "hrs_by_age() ERROR: The file %s was not found in "
            + "the directory %s"
        )
        for path in file_paths:
            if not os.path.isfile(path):
                raise ValueError(err_msg % (path, full_directory))

    # read in raw SCF data to calculate moments
    scf_dict = {}
    columns = ["networth", "wgt"]
    if include_age:
        columns.append("age")
    if include_income:
        columns.extend(SCF_INCOME_COLUMNS)
    for filename, year in zip(file_paths, scf_yrs_list):
        cpi = cpi_dict["cpi" + str(year)]
        if filename.endswith(".csv"):
            csv_columns = ["networth", "networth_infadj", "wgt"]
            if include_age:
                csv_columns.append("age")
            if include_income:
                csv_columns.extend(SCF_INCOME_COLUMNS + ["income_infadj"])
            df_yr = pd.read_csv(filename, usecols=csv_columns)
        else:
            df_yr = pd.read_stata(filename, columns=columns)
            # Add inflation adjusted net worth
            df_yr["networth_infadj"] = df_yr["networth"] * (100.0 / cpi)
            if include_income:
                df_yr["income_infadj"] = df_yr["income"] * (100.0 / cpi)
        scf_dict[str(year)] = df_yr

    df_scf = scf_dict[str(scf_yrs_list[0])]
    num_yrs = len(scf_yrs_list)
    if num_yrs >= 2:
        for year in scf_yrs_list[1:]:
            df_scf = pd.concat(
                [df_scf, scf_dict[str(year)]], ignore_index=True
            )

    return df_scf


def compute_wealth_moments(scf, bin_weights):
    """
    This function computes moments (wealth shares, Gini coefficient,
    var[ln(wealth)]) from the distribution of wealth using SCF data.

    Args:
        scf (Pandas DataFrame): pooled cross-sectional data from SCFs
        bin_weights (Numpy Array) = ability weights
        J (int) = number of ability groups

    Returns:
        wealth_moments (Numpy Array): array of wealth moments

    """
    # calculate percentile shares (percentiles based on lambdas input)
    scf.sort_values(by="networth_infadj", ascending=True, inplace=True)
    scf["weight_networth"] = scf["wgt"] * scf["networth_infadj"]
    total_weight_wealth = scf.weight_networth.sum()
    cumsum = scf.wgt.cumsum()
    J = bin_weights.shape[0]
    wealth = np.zeros((J,))
    cum_weights = bin_weights.cumsum()
    for i in range(J):
        # Get number of individuals at top of percentile bin
        cutoff = scf.wgt.sum() * cum_weights[i]
        if i == J - 1:
            # The top bin runs to the end of the distribution.  A strict
            # inequality would drop the wealthiest observation(s) whose
            # cumulative weight equals the total, so shares would not sum
            # to one.
            wealth[i] = 1.0
        else:
            wealth[i] = (
                scf.weight_networth[cumsum < cutoff].sum()
            ) / total_weight_wealth

    wealth_share = np.zeros(J)
    wealth_share[0] = wealth[0]
    wealth_share[1:] = wealth[1:] - wealth[0:-1]

    # compute gini coeff
    scf.sort_values(by="networth_infadj", ascending=True, inplace=True)
    p = (scf.wgt.cumsum() / scf.wgt.sum()).values
    nu = ((scf.wgt * scf.networth_infadj).cumsum()).values
    nu = nu / nu[-1]
    gini_coeff = (nu[1:] * p[:-1]).sum() - (nu[:-1] * p[1:]).sum()

    # compute variance in logs
    df = scf.drop(scf[scf["networth_infadj"] <= 0.0].index)
    df["ln_networth"] = np.log(df["networth_infadj"])
    df.sort_values(by="ln_networth", ascending=True, inplace=True)
    weight_mean = ((df.ln_networth * df.wgt).sum()) / (df.wgt.sum())
    var_ln_wealth = (
        (df.wgt * ((df.ln_networth - weight_mean) ** 2)).sum()
    ) * (1.0 / (df.wgt.sum() - 1))

    wealth_moments = np.append([wealth_share], [gini_coeff, var_ln_wealth])

    return wealth_moments
