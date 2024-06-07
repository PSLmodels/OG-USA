"""
------------------------------------------------------------------------
This program extracts tax rate and income data from the microsimulation
model (Tax-Calculator).
------------------------------------------------------------------------
"""

from taxcalc import Records, Calculator, Policy
from pandas import DataFrame
from dask import delayed, compute
import dask.multiprocessing
import numpy as np
import os
import pickle
import pkg_resources
from ogcore import utils
from ogusa.constants import DEFAULT_START_YEAR, TC_LAST_YEAR

CUR_PATH = os.path.split(os.path.abspath(__file__))[0]


def get_calculator(
    calculator_start_year,
    iit_baseline=None,
    iit_reform=None,
    data=None,
    gfactors=None,
    weights=None,
    records_start_year=Records.PUFCSV_YEAR,
):
    """
    This function creates the tax calculator object with the policy
    specified in reform and the data specified with the data kwarg.

    Args:
        calculator_start_year (int): first year of budget window
        reform (dictionary): IIT policy reform parameters, None if
            baseline
        data (DataFrame or str): DataFrame or path to datafile for
            Records object
        gfactors (Tax-Calculator GrowthFactors object): growth factors
            to use to extrapolate data over budget window
        weights (DataFrame): weights for Records object
        records_start_year (int): the start year for the data and
            weights dfs (default is set to the PUF start year as defined
            in the Tax-Calculator project)

    Returns:
        calc1 (Tax-Calculator Calculator object): Calculator object with
            current_year equal to calculator_start_year

    """
    # create a calculator
    policy1 = Policy()
    if data is not None and "cps" in data:
        print("Using CPS")
        records1 = Records.cps_constructor()
        # impute short and long term capital gains if using CPS data
        # in 2012 SOI data 6.587% of CG as short-term gains
        records1.p22250 = 0.06587 * records1.e01100
        records1.p23250 = (1 - 0.06587) * records1.e01100
        # set total capital gains to zero
        records1.e01100 = np.zeros(records1.e01100.shape[0])
    elif data is None or "puf" in data:  # pragma: no cover
        print("Using PUF")
        records1 = Records()
    elif data is not None and "tmd" in data:  # pragma: no cover
        print("Using TMD")
        records1 = Records.tmd_constructor("tmd.csv.gz")
    elif data is not None:  # pragma: no cover
        print("Data is ", data)
        print("Weights are ", weights)
        print("Records start year is ", records_start_year)
        records1 = Records(
            data=data,
            gfactors=gfactors,
            weights=weights,
            start_year=records_start_year,
        )  # pragma: no cover
    else:  # pragma: no cover
        raise ValueError("Please provide data or use CPS, PUF, or TMD.")

    if iit_baseline:  # if something other than current law policy baseline
        update_policy(policy1, iit_baseline)
    if iit_reform:  # if there is a reform
        update_policy(policy1, iit_reform)

    # the default set up increments year to 2013
    calc1 = Calculator(records=records1, policy=policy1)
    print("Calculator initial year = ", calc1.current_year)

    # Check that start_year is appropriate
    if calculator_start_year > TC_LAST_YEAR:
        raise RuntimeError("Start year is beyond data extrapolation.")

    return calc1


def get_data(
    baseline=False,
    start_year=DEFAULT_START_YEAR,
    iit_baseline=None,
    iit_reform={},
    data=None,
    gfactors=None,
    weights=None,
    records_start_year=Records.CPSCSV_YEAR,
    path=CUR_PATH,
    client=None,
    num_workers=1,
):
    """
    This function creates dataframes of micro data with marginal tax
    rates and information to compute effective tax rates from the
    Tax-Calculator output.  The resulting dictionary of dataframes is
    returned and saved to disk in a pickle file.

    Args:
        baseline (boolean): True if baseline tax policy
        calculator_start_year (int): first year of budget window
        iit_baseline (dictionary): IIT policy parameters for baseline
        iit_reform (dictionary): IIT policy reform parameters, None if
            baseline
        data (str or Pandas DataFrame): path or DataFrame with
            data for Tax-Calculator model
        gfactors (str or Pandas DataFrame ): path or DataFrame with
            growth factors for Tax-Calculator model
        weights (str or Pandas DataFrame): path or DataFrame with
            weights for Tax-Calculator model
        records_start_year (int): year micro data begins
        path (str): path to save microdata files to
        client (Dask Client object): client for Dask multiprocessing
        num_workers (int): number of workers to use for Dask
            multiprocessing

    Returns:
        micro_data_dict (dict): dict of Pandas Dataframe, one for each
            year from start_year to the maximum year Tax-Calculator can
            analyze
        taxcalc_version (str): version of Tax-Calculator used

    """
    # Compute MTRs and taxes or each year, but not beyond TC_LAST_YEAR
    lazy_values = []
    for year in range(start_year, TC_LAST_YEAR + 1):
        lazy_values.append(
            delayed(taxcalc_advance)(
                start_year,
                iit_baseline,
                iit_reform,
                data,
                gfactors,
                weights,
                records_start_year,
                year,
            )
        )
    if client:  # pragma: no cover
        futures = client.compute(lazy_values, num_workers=num_workers)
        results = client.gather(futures)
    else:
        results = results = compute(
            *lazy_values,
            scheduler=dask.multiprocessing.get,
            num_workers=num_workers,
        )

    # dictionary of data frames to return
    micro_data_dict = {}
    for i, result in enumerate(results):
        year = start_year + i
        micro_data_dict[str(year)] = DataFrame(result)

    if baseline:
        pkl_path = os.path.join(path, "micro_data_baseline.pkl")
    else:
        pkl_path = os.path.join(path, "micro_data_policy.pkl")
    utils.mkdirs(path)
    with open(pkl_path, "wb") as f:
        pickle.dump(micro_data_dict, f)

    # Do some garbage collection
    del results

    # Pull Tax-Calc version for reference
    taxcalc_version = pkg_resources.get_distribution("taxcalc").version

    return micro_data_dict, taxcalc_version


def taxcalc_advance(
    start_year,
    iit_baseline,
    iit_reform,
    data,
    gfactors,
    weights,
    records_start_year,
    year,
):
    """
    This function advances the year used in Tax-Calculator, compute
    taxes and rates, and save the results to a dictionary.

    Args:
        start_year (int): first year of budget window
        iit_baseline (dict): IIT policy parameters for baseline
        iit_reform (dict): IIT policy reform parameters for reform
        data (str or Pandas DataFrame): path or DataFrame with
            data for Tax-Calculator model
        gfactors (str or Pandas DataFrame ): path or DataFrame with
            growth factors for Tax-Calculator model
        weights (str or Pandas DataFrame): path or DataFrame with
            weights for Tax-Calculator model
        records_start_year (int): year micro data begins
        year (int): year to advance to in Tax-Calculator

    Returns:
        tax_dict (dict): a dictionary of microdata with marginal tax
            rates and other information computed in TC
    """
    calc1 = get_calculator(
        calculator_start_year=start_year,
        iit_baseline=iit_baseline,
        iit_reform=iit_reform,
        data=data,
        gfactors=gfactors,
        weights=weights,
        records_start_year=records_start_year,
    )
    calc1.advance_to_year(year)
    calc1.calc_all()
    print("Year: ", str(calc1.current_year))

    # define market income - taking expanded_income and excluding gov't
    # transfer benefits found in the Tax-Calculator expanded income
    market_income = calc1.array("expanded_income") - calc1.array(
        "benefit_value_total"
    )

    # Compute mtr on capital income
    mtr_combined_capinc = cap_inc_mtr(calc1)

    # Compute weighted avg mtr for labor income
    # Note the index [2] in the mtr results means that we are pulling
    # the combined mtr from the IIT + FICA taxes
    mtr_combined_labinc = (
        calc1.mtr("e00200p")[2] * np.abs(calc1.array("e00200"))
        + calc1.mtr("e00900p")[2] * np.abs(calc1.array("sey"))
    ) / (np.abs(calc1.array("sey")) + np.abs(calc1.array("e00200")))

    # Put MTRs, income, tax liability, and other variables in dict
    length = len(calc1.array("s006"))
    tax_dict = {
        "mtr_labinc": mtr_combined_labinc,
        "mtr_capinc": mtr_combined_capinc,
        "age": calc1.array("age_head"),
        "total_labinc": calc1.array("sey") + calc1.array("e00200"),
        "total_capinc": (
            market_income - calc1.array("sey") + calc1.array("e00200")
        ),
        "market_income": market_income,
        "total_tax_liab": calc1.array("combined"),
        "payroll_tax_liab": calc1.array("payrolltax"),
        "etr": (
            (calc1.array("combined") - calc1.array("ubi")) / market_income
        ),
        "year": calc1.current_year * np.ones(length),
        "weight": calc1.array("s006"),
    }

    # garbage collection
    del calc1

    return tax_dict


def cap_inc_mtr(calc1):  # pragma: no cover
    """
    This function computes the marginal tax rate on capital income,
    which is calculated as a weighted average of the marginal tax rates
    on different sources of capital income.

    Args:
        calc1 (Tax-Calculator Calculator object): TC calculator

    Returns:
        mtr_combined_capinc (Numpy array): array with marginal tax rates
            for each observation in the TC Records object

    """
    # Note: PUF does not have variable for non-taxable IRA distributions
    # Exclude Sch E income (e02000) from this list since we'll compute
    # MTRs for this income in two parts - one for overall Sch C and one
    # for S Corp and Partnerhsip income (e26270) (note that TaxCalc
    # doesn't allow for an MTR on rents and royalties alone)
    # e00300 = interest income
    # e00400 = nontaxable interest income
    # e00600 = ordinary dividend income
    # e00650 = qualified dividend income
    # e01400 = taxable IRA distributions
    # e01700 = pension and annuity income
    # p22250 = short term cap gain/loss
    # p23250 = long term cap gain/loss
    # e26270 = partnership and s corp income/loss
    # e02000 = Sch E income (includes e26270)
    capital_income_sources = (
        "e00300",
        "e00400",
        "e00600",
        "e00650",
        "e01400",
        "e01700",
        "p22250",
        "p23250",
        "e26270",
    )
    rent_royalty_inc = np.abs(calc1.array("e02000") - calc1.array("e26270"))
    # assign overall Sch E mtr to rent and royalities since TC can't do
    # this component separately
    rent_royalty_mtr = calc1.mtr("e02000")[2]
    # calculating MTRs separately - can skip items with zero tax
    all_mtrs = {
        income_source: calc1.mtr(income_source)
        for income_source in capital_income_sources
    }
    # Get each column of income sources, to include non-taxable income
    record_columns = [calc1.array(x) for x in capital_income_sources]
    # Compute weighted average of all those MTRs
    # first find total capital income
    total_cap_inc = sum(map(abs, record_columns)) + rent_royalty_inc
    # Note that all_mtrs gives fica (0), iit (1), and combined (2) mtrs
    # We'll use the combined - hence all_mtrs[source][2]
    capital_mtr = [
        abs(col) * all_mtrs[source][2]
        for col, source in zip(record_columns, capital_income_sources)
    ]
    mtr_combined_capinc = np.zeros_like(total_cap_inc)
    mtr_combined_capinc[total_cap_inc != 0] = (
        sum(capital_mtr + rent_royalty_mtr * rent_royalty_inc)[
            total_cap_inc != 0
        ]
        / total_cap_inc[total_cap_inc != 0]
    )
    mtr_combined_capinc[total_cap_inc == 0] = all_mtrs["e00300"][2][
        total_cap_inc == 0
    ]
    return mtr_combined_capinc


def update_policy(policy_obj, reform, **kwargs):
    """
    Convenience method that updates the Policy object with the reform
    dict using the appropriate method, given the reform format.

    Args:
        policy_obj (Policy object): Tax-Calculator Policy object
        reform (dict or str): JSON string or dictionary of reform
            parameters
        kwargs (dict): keyword arguments to pass to the adjust method

    Returns:
        None (updates policy object)

    """
    if is_paramtools_format(reform):
        policy_obj.adjust(reform, **kwargs)
    else:
        policy_obj.implement_reform(reform, **kwargs)


def is_paramtools_format(reform):
    """
    Check first item in reform to determine if it is using the ParamTools
    adjustment or the Tax-Calculator reform format.
    If first item is a dict, then it is likely be a Tax-Calculator reform:
    {
        param: {2020: 1000}
    }
    Otherwise, it is likely to be a ParamTools format.

    Args:
        reform (dict or str): JSON string or dictionary of reform
            parameters

    Returns:
        format (bool): True if reform is likely to be in PT format.
    """
    for _, data in reform.items():
        if isinstance(data, dict):
            return False  # taxcalc reform
        else:
            # Not doing a specific check to see if the value is a list
            # since it could be a list or just a scalar value.
            return True
