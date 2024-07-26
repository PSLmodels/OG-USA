import numpy as np
import pandas as pd
import os
from ogcore.utils import safe_read_pickle

# Relate OG output variables to grow factors
GROWFACTOR_MAPPING = {
    "ABOOK": "corp_profits",
    "ASCHCI": "Y",
    "ASCHCL": "Y",
    "ASCHF": "Y",
    "ATXPY": "personal_income",
    "ACGNS": "hh_capital_income",
    "ADIVS": "hh_capital_income",
    "AIPD": "interest_income",
    "AINTS": "interest_income",
    "AWAGE": "wages_paid",
    "ASOCSEC": "w",
}


def growth_rate_diff(base, reform):
    """
    This function calculates the difference in growth rates between the
    base and reform policy scenarios.

    Args:
        base (DataFrame): DataFrame containing the base policy scenario
        reform (DataFrame): DataFrame containing the reform policy scenario

    Returns:
        growth_diff (DataFrame): The difference in growth rates between
            the base and reform policy scenarios
    """
    # Calculate the growth rates for the base and reform policy scenarios
    base_growth = base.pct_change()
    reform_growth = reform.pct_change()

    # Calculate the difference in growth rates
    growth_diff = reform_growth - base_growth
    # Calculate the level shift in the first year
    level_shift = (reform.iloc[0] - base.iloc[0]) / base.iloc[0]
    # shift index back one so year is the year of the growth rate
    growth_diff.index = growth_diff.index - 1
    # Drop index 2024
    growth_diff = growth_diff.drop(2024)
    # And add the level shift in the 1st year
    growth_diff.iloc[0] += level_shift

    return growth_diff


def compute_input_vars(tpi_dict, start_year, T):
    """
    This function takes a dictionary of output from OG-Core and computes
    the variables necessary for a mapping to the series used to
    compute grow factors.

    Args:
        tpi_dict (dict): Dictionary of output from OG-Core
        start_year (int): The first year of the simulation
        T (int): The number of years in the simulation

    Returns:
        df (DataFrame): DataFrame of input variables necessary for
            computing grow factors

    """

    # make turn the dict into a DataFrame
    tpi_filtered = {
        k: v
        for k, v in tpi_dict.items()
        if k in ["Y", "r", "w", "B", "r_p", "L", "K_d"]
    }
    df = pd.DataFrame(
        data=tpi_filtered, index=np.arange(start_year, start_year + T)
    )

    # Create some new variables:
    df["corp_profits"] = df["Y"] - df["w"] * df["L"]
    df["personal_income"] = df["w"] * df["L"] + df["r_p"] * df["B"]
    df["hh_capital_income"] = df["r"] * df["K_d"]
    df["interest_income"] = df["r_p"] * df["B"]
    df["wages_paid"] = df["w"] * df["L"]

    return df


def update_growfactors(
    tpi_base, tpi_reform, initial_growfactors, start_year=2025, T=320
):
    """
    This function takes the output from OG-Core for the base and reform
    policy scenarios and computes the difference in growth rates between
    the two scenarios for the variables necessary for computing grow
    factors. The function then updates the grow factors based on the
    difference in growth rates.

    Args:
        tpi_base (dict): Dictionary of output from OG-Core for the base
            policy scenario
        tpi_reform (dict): Dictionary of output from OG-Core for the
            reform policy scenario
        initial_growfactors (DataFrame): Initial grow factors
        start_year (int): The first year of the simulation
        T (int): The number of years in the simulation

    Returns:
        updated_growfactors (DataFrame): Updated grow factors
    """

    # Compute the input variables for the base and reform policy scenarios
    base = compute_input_vars(tpi_base, start_year, T)
    reform = compute_input_vars(tpi_reform, start_year, T)

    # Calculate the difference in growth rates between the base and
    # reform policy scenarios
    growth_diff = growth_rate_diff(base, reform)

    # Keep only growth_diffs for years in the growfactors
    max_year = growth_diff.index.max()
    growth_diff = growth_diff.loc[start_year:max_year]

    # Update the grow factors based on the difference in growth rates
    updated_growfactors = initial_growfactors.copy()
    for key, value in GROWFACTOR_MAPPING.items():
        updated_growfactors.loc[start_year:, key] = initial_growfactors.loc[
            start_year:, key
        ] * (1 + growth_diff[value])

    return updated_growfactors


def update(base_sim_path, reform_sim_path, growfactor_path, output_path):
    """
    Takes paths for model output and grow factors and updates the
    grow factors

    Args:
        base_sim_path (str): Path to the base policy scenario model output
        reform_sim_path (str): Path to the reform policy scenario
            model output
        growfactor_path (str): Path to the initial grow factors
        output_path (str): Path to save the updated grow factors

    Returns:
        None
    """
    tpi_base = safe_read_pickle(
        os.path.join(base_sim_path, "TPI", "TPI_vars.pkl")
    )
    tpi_reform = safe_read_pickle(
        os.path.join(reform_sim_path, "TPI", "TPI_vars.pkl")
    )
    base_params = safe_read_pickle(
        os.path.join(base_sim_path, "model_params.pkl")
    )
    start_year = base_params.start_year
    T = base_params.T

    initial_growfactors = pd.read_csv(growfactor_path, index_col=0)

    updated_growfactors = update_growfactors(
        tpi_base, tpi_reform, initial_growfactors, start_year, T
    )

    updated_growfactors.to_csv(output_path)
