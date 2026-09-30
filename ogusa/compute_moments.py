"""
This module uses data from FRED and other sources to compute data
moments that are used in the calibration of the OG-USA model.
"""

# imports
from fredapi import Fred
import io
import os
import pandas as pd
import numpy as np
import datetime
import warnings
import zipfile
from urllib.request import Request, urlopen

NBER_CPS_ASEC_URLS = {
    2022: "https://data.nber.org/cps_supp_1/raw/2022/march/asecpub22csv.zip",
    2023: "https://data.nber.org/cps_supp_1/raw/2023/march/asecpub23csv.zip",
}
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))
CPS_DATA_DIR = os.path.join(DATA_DIR, "CPS")


def _mean_ratio(numerator, denominator):
    """
    Return the average of numerator / denominator over common observations.
    """
    ratio_data = pd.concat(
        [numerator.rename("numerator"), denominator.rename("denominator")],
        axis=1,
    ).dropna()
    return (ratio_data["numerator"] / ratio_data["denominator"]).mean()


def _ratio_moment(numerator, denominator, last_value_only=False):
    """
    Return either the last ratio or average ratio over common observations.
    """
    ratio_data = pd.concat(
        [numerator.rename("numerator"), denominator.rename("denominator")],
        axis=1,
    ).dropna()
    ratio = ratio_data["numerator"] / ratio_data["denominator"]
    if last_value_only:
        return ratio.iloc[-1]
    return ratio.mean()


def _mean_real_rate(nominal_rate, price_index):
    """
    Return the average nominal rate less inflation over common observations.
    """
    nominal_rate_a = nominal_rate.resample("YE").mean() / 100
    inflation = price_index.pct_change()
    rate_data = pd.concat(
        [
            nominal_rate_a.rename("nominal_rate"),
            inflation.rename("inflation"),
        ],
        axis=1,
    ).dropna()
    return (rate_data["nominal_rate"] - rate_data["inflation"]).mean()


def _weighted_gini(values, weights):
    """
    Compute the weighted Gini coefficient.
    """
    data = pd.DataFrame({"value": values, "weight": weights})
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    data = data[data["weight"] > 0].copy()
    if data.empty:
        raise ValueError("No observations with positive weight.")

    data.sort_values(by="value", ascending=True, inplace=True)
    weighted_value = data["value"] * data["weight"]
    total_weighted_value = weighted_value.sum()
    if np.isclose(total_weighted_value, 0.0):
        raise ValueError("Weighted sum of values is zero.")

    p = (data["weight"].cumsum() / data["weight"].sum()).values
    nu = (weighted_value.cumsum() / total_weighted_value).values
    return float((nu[1:] * p[:-1]).sum() - (nu[:-1] * p[1:]).sum())


def _weighted_mean_by_age(
    data, value_col, weight_col, min_age, max_age, age_col="age"
):
    """
    Return weighted mean values by single year of age.
    """
    if min_age > max_age:
        raise ValueError("min_age must be less than or equal to max_age.")

    columns = [age_col, value_col]
    if weight_col is not None:
        columns.append(weight_col)
    age_data = data[columns].copy()
    age_data[age_col] = pd.to_numeric(age_data[age_col], errors="coerce")
    age_data[value_col] = pd.to_numeric(age_data[value_col], errors="coerce")
    age_data = age_data.replace([np.inf, -np.inf], np.nan).dropna()
    age_data = age_data[
        (age_data[age_col] >= min_age) & (age_data[age_col] <= max_age)
    ].copy()
    age_data[age_col] = age_data[age_col].astype(int)

    ages = pd.Index(range(min_age, max_age + 1), name="age")
    if age_data.empty:
        return pd.Series(index=ages, dtype=float)

    if weight_col is None:
        return age_data.groupby(age_col)[value_col].mean().reindex(ages)

    age_data[weight_col] = pd.to_numeric(age_data[weight_col], errors="coerce")
    age_data = age_data[age_data[weight_col] > 0].copy()
    if age_data.empty:
        return pd.Series(index=ages, dtype=float)

    age_data["weighted_value"] = age_data[value_col] * age_data[weight_col]
    by_age = age_data.groupby(age_col)[["weighted_value", weight_col]].sum()
    profile = by_age["weighted_value"] / by_age[weight_col]

    return profile.reindex(ages)


def _model_age_profile(profile, min_age, max_age):
    """
    Return an 80-period model age profile with missing ages as NaN.
    """
    model_ages = pd.Index(range(20, 100), name="age")
    model_profile = pd.Series(np.nan, index=model_ages, dtype=float)
    valid_ages = [
        age
        for age in profile.index
        if age in model_profile.index and min_age <= age <= max_age
    ]
    model_profile.loc[valid_ages] = profile.loc[valid_ages].astype(float)

    return model_profile


def _demographic_moments_from_pop_path(E, S, omega, g_n):
    """
    Compute demographic moments from a population path and growth path.
    """
    ages = np.arange(E, E + S)
    omega = np.asarray(omega)
    g_n = np.asarray(g_n)
    omega0 = omega[0, :]

    demographic_moments = {}
    demographic_moments[r"Fraction 65+"] = float(omega0[ages >= 65].sum())
    demographic_moments[r"Pop growth rate"] = float(g_n[0])

    return demographic_moments


def _taxcalc_cps_earnings_by_age(min_age, max_age, income_year=None):
    """
    Compute individual earnings means by age from Tax-Calculator CPS data.
    """
    from taxcalc import Calculator, Policy, Records

    calc = Calculator(records=Records.cps_constructor(), policy=Policy())
    if income_year is not None:
        calc.advance_to_year(income_year)
    calc.calc_all()

    weights = calc.array("s006")
    cps = pd.concat(
        [
            pd.DataFrame(
                {
                    "age": calc.array("age_head"),
                    "earnings": calc.array("earned_p"),
                    "weight": weights,
                }
            ),
            pd.DataFrame(
                {
                    "age": calc.array("age_spouse"),
                    "earnings": calc.array("earned_s"),
                    "weight": weights,
                }
            ),
        ],
        ignore_index=True,
    )
    cps = cps[cps["age"] > 0]

    return _weighted_mean_by_age(cps, "earnings", "weight", min_age, max_age)


def _cps_hours_by_age(cps, min_age, max_age):
    """
    Compute hours means by age from a CPS dataframe.
    """
    if cps is None:
        raise ValueError("No CPS hours data were provided.")
    if "age" not in cps or "hours" not in cps:
        raise ValueError("CPS hours data must include age and hours columns.")

    weight_col = None
    for possible_weight_col in ("wtsupp", "s006", "weight", "wgt"):
        if possible_weight_col in cps:
            weight_col = possible_weight_col
            break

    return _weighted_mean_by_age(cps, "hours", weight_col, min_age, max_age)


def _read_nber_cps_asec_person_file(
    year,
    url=None,
    directory=None,
    web=False,
):
    """
    Read CPS ASEC person-level hours fields from local CSV or NBER zip.
    """
    if directory is None:
        directory = CPS_DATA_DIR
    if url is None:
        local_path = os.path.join(directory, f"cps_asec_hours_{year}.csv")
        if os.path.exists(local_path):
            return pd.read_csv(local_path)
        if not web:
            raise FileNotFoundError(
                f"Could not find {local_path}. Run "
                "data/download_moment_data.py to create local CPS data "
                "under ogusa/data/CPS."
            )
        url = NBER_CPS_ASEC_URLS[year]

    url_or_path = os.fspath(url)
    if os.path.exists(url_or_path) and url_or_path.endswith(".csv"):
        return pd.read_csv(url_or_path)
    if os.path.exists(url_or_path):
        with open(url_or_path, "rb") as zip_file_on_disk:
            zip_bytes = zip_file_on_disk.read()
    else:
        request = Request(url_or_path, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(request, timeout=60) as response:
            zip_bytes = response.read()

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zip_file:
        person_files = [
            name
            for name in zip_file.namelist()
            if name.lower().startswith("pppub")
            and name.lower().endswith(".csv")
        ]
        if len(person_files) != 1:
            raise ValueError(
                f"Expected one CPS person file in {url}, found "
                f"{len(person_files)}."
            )
        with zip_file.open(person_files[0]) as person_file:
            cps = pd.read_csv(
                person_file,
                usecols=["A_AGE", "HRSWK", "A_FNLWGT"],
            )

    cps.rename(
        columns={
            "A_AGE": "age",
            "HRSWK": "hours_per_week",
            "A_FNLWGT": "weight",
        },
        inplace=True,
    )
    cps["year"] = year

    return cps


def _nber_cps_hours_by_age(
    min_age,
    max_age,
    cps_years=(2023, 2022),
    cps_urls=None,
    cps_directory=None,
    cps_web=False,
):
    """
    Compute weekly hours means by age from NBER CPS ASEC files.
    """
    cps_data = []
    for year in cps_years:
        url = None
        if cps_urls is not None:
            url = cps_urls.get(year)
        cps_data.append(
            _read_nber_cps_asec_person_file(
                year,
                url=url,
                directory=cps_directory,
                web=cps_web,
            )
        )
    cps = pd.concat(cps_data, ignore_index=True)

    for col in ["hours_per_week"]:
        cps[col] = pd.to_numeric(cps[col], errors="coerce").fillna(0)
        cps.loc[cps[col] < 0, col] = 0
    cps["hours"] = cps["hours_per_week"]

    return _weighted_mean_by_age(cps, "hours", "weight", min_age, max_age)


def _default_psid_path():
    """
    Return the first packaged PSID lifetime-income file found.
    """
    cur_path = os.path.split(os.path.abspath(__file__))[0]
    candidate_paths = [
        os.path.join(cur_path, "psid_lifetime_income.csv.gz"),
        os.path.join(
            cur_path,
            "data",
            "PSID",
            "psid_lifetime_income.csv",
        ),
        os.path.join(
            cur_path,
            "data",
            "PSID",
            "psid_lifetime_income_archived.csv",
        ),
        os.path.join(
            cur_path,
            "..",
            "data",
            "PSID",
            "psid_lifetime_income_archived.csv",
        ),
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        "Could not find a packaged PSID lifetime-income file."
    )


def _read_psid_lifetime_income(columns, psid_path=None):
    """
    Read selected columns from the packaged PSID lifetime-income data.
    """
    if psid_path is None:
        psid_path = _default_psid_path()

    psid = pd.read_csv(psid_path, usecols=lambda col: col in columns)
    missing_columns = sorted(set(columns) - set(psid.columns))
    if missing_columns:
        raise ValueError(
            "PSID data are missing required columns: "
            + ", ".join(missing_columns)
        )

    return psid


def _psid_person_profile(
    var, min_age, max_age, psid_path=None, weight_col=None
):
    """
    Compute individual hours or earnings means by age from PSID data.
    """
    columns = [
        "age",
        "spouse_age",
        "head_annual_hours",
        "spouse_annual_hours",
        "head_labor_inc",
        "spouse_labor_inc",
        "head_noncorp_bus_labor_income",
        "spouse_noncorp_bus_labor_income",
    ]
    if weight_col is not None:
        columns.append(weight_col)
    psid = _read_psid_lifetime_income(columns, psid_path=psid_path)

    if var == "hours":
        head_value = psid["head_annual_hours"]
        spouse_value = psid["spouse_annual_hours"]
        value_col = "hours"
    elif var == "earnings":
        head_value = (
            psid["head_labor_inc"] + psid["head_noncorp_bus_labor_income"]
        )
        spouse_value = (
            psid["spouse_labor_inc"] + psid["spouse_noncorp_bus_labor_income"]
        )
        value_col = "earnings"
    else:
        raise ValueError(f"Unsupported PSID person profile variable: {var}")

    head_data = {
        "age": psid["age"],
        value_col: head_value,
    }
    spouse_data = {
        "age": psid["spouse_age"],
        value_col: spouse_value,
    }
    if weight_col is not None:
        head_data[weight_col] = psid[weight_col]
        spouse_data[weight_col] = psid[weight_col]

    people = pd.concat(
        [pd.DataFrame(head_data), pd.DataFrame(spouse_data)],
        ignore_index=True,
    )
    people = people[people["age"] > 0]

    return _weighted_mean_by_age(
        people, value_col, weight_col, min_age, max_age
    )


def _psid_consumption_by_age(
    min_age, max_age, psid_path=None, consumption_vars=None, weight_col=None
):
    """
    Compute household consumption-expenditure means by head age from PSID.
    """
    if consumption_vars is None:
        consumption_vars = ["food_out_expend", "food_in_expend"]
    columns = ["age"] + list(consumption_vars)
    if weight_col is not None:
        columns.append(weight_col)
    psid = _read_psid_lifetime_income(columns, psid_path=psid_path)

    psid["consumption"] = psid[list(consumption_vars)].sum(axis=1)

    return _weighted_mean_by_age(
        psid, "consumption", weight_col, min_age, max_age
    )


def _taxcalc_cps_income_ginis(income_year=None):
    """
    Compute income Ginis from Tax-Calculator CPS records.
    """
    from taxcalc import Calculator, Policy, Records

    calc = Calculator(records=Records.cps_constructor(), policy=Policy())
    if income_year is not None:
        calc.advance_to_year(income_year)
    calc.calc_all()

    weights = calc.array("s006")
    before_tax_income = calc.array("expanded_income") - calc.array(
        "benefit_value_total"
    )
    after_tax_income = calc.array("aftertax_income")

    return {
        "Gini coefficient, income": _weighted_gini(before_tax_income, weights),
        "Gini coefficient, after-tax income": _weighted_gini(
            after_tax_income, weights
        ),
    }


def _convert_nominal_to_base_year(nominal, deflator, base_year):
    """
    Convert nominal dollars to dollars in the deflator's base-year prices.
    """
    base_date = pd.Timestamp(base_year, 12, 31)
    base_deflator = deflator.loc[base_date]
    return nominal * base_deflator / deflator


def get_macro_moments(year=2025):
    """
    Compute moments that use macro data.

    Computes the following moments:

        r"Investment rate $(I/K)$",
        r"Capital-Output ratio $(K/Y)$",
        r"Consumption-Output ratio $(C/Y)$",
        r"Gross savings rate $(S/Y)$",
        r"Interest rate $(r)$",
        r"Capital share of output",
        r"Labor share of output",
    """

    # set beginning and end dates for data
    # format is year (1940),month (1),day (1)
    start = datetime.datetime(1947, 1, 1)
    end = min(datetime.date.today(), datetime.date(year, 12, 31))
    # Deflator conversion uses 2021 prices even if the ratio sample ends
    # earlier.
    observation_end = max(end, datetime.date(2021, 12, 31))

    variable_dict = {
        "GDP Per Capita": "A939RX0Q048SBEA",
        "Labor share": "LABSHPUSA156NRUG",
        "Debt held by foreigners": "FDHBFIN",
        "Debt held by public": "FYGFDPUN",
        "BAA Corp Bond Rates": "DBAA",
        "10 year treasury rate": "DGS10",
        "Total gov transfer payments": "B087RC1Q027SBEA",
        "Social Security payments": "W823RC1",
        "Gov expenditures": "FGEXPND",
        "Gov investment": "A782RC1Q027SBEA",
        "Gov interest payments": "A091RC1Q027SBEA",
        "Real GDP": "GDPC1",
        "Nominal GDP": "GDP",
        "Fixed private investment": "FPI",
        "Personal consumption expenditures": "PCE",
        "Gross private savings": "GPSAVE",
        "Real capital stock": "RKNANPUSA666NRUG",  # 2021 dollars, in millions
        "GDP deflator": "A191RD3A086NBEA",  # 2017 = 100
        "Fixed private investment deflator": "A007RD3A086NBEA",  # 2017 = 100
    }

    # pull series of interest using fredapi
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        raise ValueError(
            "FRED_API_KEY environment variable is not set. "
            "A free API key can be obtained at "
            "https://fred.stlouisfed.org/docs/api/api_key.html"
        )
    fred = Fred(api_key=api_key)
    series_list = []
    for name, series_id in variable_dict.items():
        s = fred.get_series(
            series_id, observation_start=start, observation_end=observation_end
        )
        s.name = name
        series_list.append(s)
    fred_data = pd.concat(series_list, axis=1)
    fred_data_common = fred_data.loc[: pd.Timestamp(end)].copy()

    # make sure all dollar value data are in billions
    fred_data_common["Debt held by public"] = (
        fred_data_common["Debt held by public"] / 1000
    )

    # Separate quarterly, monthly, and annual data series
    fred_data_q = (
        fred_data_common[
            [
                "Debt held by public",
                "Nominal GDP",
                "Real GDP",
                "Total gov transfer payments",
                "Social Security payments",
                "Gov expenditures",
                "Gov investment",
                "Gov interest payments",
                "GDP Per Capita",
                "Fixed private investment",
                "Personal consumption expenditures",
                "Gross private savings",
            ]
        ]
        .resample("QE")
        .mean()
    )
    fred_data_a = (
        fred_data_common[
            [
                "Labor share",
                "Real capital stock",
                "GDP deflator",
                "Fixed private investment deflator",
            ]
        ]
        .resample("YE")
        .mean()
    )
    fred_data_d = fred_data_common[
        ["BAA Corp Bond Rates", "10 year treasury rate"]
    ]
    fred_data_a_all = (
        fred_data[["GDP deflator", "Fixed private investment deflator"]]
        .resample("YE")
        .mean()
    )

    # Convert quarterly flow series to annual frequency for stock-flow ratios.
    fred_data_qa = fred_data_q.resample("YE").mean()
    capital_stock_billions = fred_data_a["Real capital stock"] / 1000
    fixed_private_investment_2021 = _convert_nominal_to_base_year(
        fred_data_qa["Fixed private investment"],
        fred_data_a_all["Fixed private investment deflator"],
        2021,
    )
    real_gdp_2021 = fred_data_qa["Real GDP"] * (
        fred_data_a_all["GDP deflator"].loc[pd.Timestamp(2021, 12, 31)] / 100
    )

    macro_moments = {}
    macro_moments[r"Investment rate $(I/K)$"] = _mean_ratio(
        fixed_private_investment_2021, capital_stock_billions
    )
    macro_moments[r"Capital-Output ratio $(K/Y)$"] = _mean_ratio(
        capital_stock_billions, real_gdp_2021
    )
    macro_moments[r"Consumption-Output ratio $(C/Y)$"] = _mean_ratio(
        fred_data_q["Personal consumption expenditures"],
        fred_data_q["Nominal GDP"],
    )
    macro_moments[r"Gross savings rate $(S/Y)$"] = _mean_ratio(
        fred_data_q["Gross private savings"], fred_data_q["Nominal GDP"]
    )
    macro_moments[r"Interest rate $(r)$"] = _mean_real_rate(
        fred_data_d["BAA Corp Bond Rates"], fred_data_a["GDP deflator"]
    )
    macro_moments[r"Capital share of output"] = (
        1 - fred_data_a["Labor share"].mean()
    )
    macro_moments[r"Labor share of output"] = fred_data_a["Labor share"].mean()

    return macro_moments


def get_fiscal_moments(year=2025, last_value_only=True):
    """
    Compute moments that use macro data.

    Computes the following moments:

        r"Revenue to GDP ratio $(T/Y)$"
        r"Gov't consumption to GDP ratio $(G/Y)$"
        r"Pension outlays to GDP ratio $(Pension/Y)$"
        r"Infrastructure spending to GDP ratio $(I_g/Y)$"
        r"Debt to GDP ratio $(D/Y)$"

    Args:
        year (int): Inclusive end year for FRED data.
        last_value_only (bool): If True, use the last common ratio
            observation. If False, use the mean ratio over all common
            observations.
    """

    # set beginning and end dates for data
    # format is year (1940),month (1),day (1)
    start = datetime.datetime(1947, 1, 1)
    end = min(datetime.date.today(), datetime.date(year, 12, 31))

    variable_dict = {
        "GDP Per Capita": "A939RX0Q048SBEA",
        "Labor share": "LABSHPUSA156NRUG",
        "Debt held by foreigners": "FDHBFIN",
        "Debt held by public": "FYGFDPUN",
        "BAA Corp Bond Rates": "DBAA",
        "10 year treasury rate": "DGS10",
        "Total gov transfer payments": "B087RC1Q027SBEA",
        "Social Security payments": "W823RC1",
        "Gov expenditures": "FGEXPND",
        "Gov investment": "A782RC1Q027SBEA",
        "Gov interest payments": "A091RC1Q027SBEA",
        "Real GDP": "GDPC1",
        "Nominal GDP": "GDP",
        "Fixed private investment": "FPI",
        "Personal consumption expenditures": "PCE",
        "Gross private savings": "GPSAVE",
        "Federal tax receipts": "W006RC1Q027SBEA",
    }

    # pull series of interest using fredapi
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        raise ValueError(
            "FRED_API_KEY environment variable is not set. "
            "A free API key can be obtained at "
            "https://fred.stlouisfed.org/docs/api/api_key.html"
        )
    fred = Fred(api_key=api_key)
    series_list = []
    for name, series_id in variable_dict.items():
        s = fred.get_series(
            series_id, observation_start=start, observation_end=end
        )
        s.name = name
        series_list.append(s)
    fred_data = pd.concat(series_list, axis=1)

    # make sure all dollar value data are in billions
    fred_data["Debt held by public"] = fred_data["Debt held by public"] / 1000

    # Separate quarterly, monthly, and annual data series
    fred_data_q = (
        fred_data[
            [
                "Debt held by public",
                "Nominal GDP",
                "Total gov transfer payments",
                "Social Security payments",
                "Gov expenditures",
                "Gov investment",
                "Gov interest payments",
                "GDP Per Capita",
                "Federal tax receipts",
            ]
        ]
        .resample("QE")
        .mean()
    )

    # initialize a dictionary of parameters
    fiscal_moments = {}

    fiscal_moments[r"Revenue to GDP ratio $(T/Y)$"] = _ratio_moment(
        fred_data_q["Federal tax receipts"],
        fred_data_q["Nominal GDP"],
        last_value_only,
    )
    fiscal_moments[r"Debt to GDP ratio $(D/Y)$"] = _ratio_moment(
        fred_data_q["Debt held by public"],
        fred_data_q["Nominal GDP"],
        last_value_only,
    )
    gov_consumption = (
        fred_data_q["Gov expenditures"]
        - fred_data_q["Total gov transfer payments"]
        - fred_data_q["Gov interest payments"]
        - fred_data_q["Gov investment"]
    )
    fiscal_moments[r"Gov't consumption to GDP ratio $(G/Y)$"] = _ratio_moment(
        gov_consumption, fred_data_q["Nominal GDP"], last_value_only
    )
    fiscal_moments[r"Pension outlays to GDP ratio $(Pension/Y)$"] = (
        _ratio_moment(
            fred_data_q["Social Security payments"],
            fred_data_q["Nominal GDP"],
            last_value_only,
        )
    )

    # find alpha_I
    fiscal_moments[r"Infrastructure spending to GDP ratio $(I_g/Y)$"] = (
        _ratio_moment(
            fred_data_q["Gov investment"],
            fred_data_q["Nominal GDP"],
            last_value_only,
        )
    )

    return fiscal_moments


def get_demographic_moments(p, demographic_data_path=None):
    """
    Compute moments that use demographic data.

    Computes the following moments:

        r"Fraction 65+"
        r"Pop growth rate"

    Args:
        p (OG-Core Specifications object): model parameters.
        demographic_data_path (str): path to save downloaded demographic data.
    """
    from ogcore import demographics

    try:
        pop_objs = demographics.get_pop_objs(
            p.E,
            p.S,
            p.T,
            0,
            99,
            initial_data_year=p.start_year - 1,
            final_data_year=p.start_year,
            GraphDiag=False,
            download_path=demographic_data_path,
        )
        omega = pop_objs["omega"]
        g_n = pop_objs["g_n"]
    except (AssertionError, UnboundLocalError, OSError):
        if not hasattr(p, "omega") or not hasattr(p, "g_n"):
            raise
        warnings.warn(
            "Unable to build demographic objects with "
            "ogcore.demographics.get_pop_objs. Using p.omega and p.g_n "
            "instead.",
            RuntimeWarning,
            stacklevel=2,
        )
        omega = p.omega
        g_n = p.g_n

    return _demographic_moments_from_pop_path(p.E, p.S, omega, g_n)


def get_inequality_moments(
    income_source="cps",
    wealth_source="scf",
    income_year=None,
    scf_yrs_list=None,
    scf_web=False,
    scf_directory=None,
):
    """
    Compute moments that use income and wealth microdata.

    Computes the following moments:

        r"Before-tax income Gini"
        r"After-tax income Gini"
        r"Wealth Gini"

    Args:
        income_source (str): Source for income data. Currently supports
            "cps".
        wealth_source (str): Source for wealth data. Currently supports
            "scf".
        income_year (int): Year to use for Tax-Calculator CPS records. If
            None, use the CPS data start year.
        scf_yrs_list (list): SCF survey years to pool. If None, use the
            default years in wealth.get_wealth_data().
        scf_web (bool): If True, download SCF data from the web.
            Defaults to False and uses local trimmed CSVs in ogusa/data/SCF.
        scf_directory (str): Local SCF data directory when scf_web=False.
    """
    inequality_moments = {}
    income_source = income_source.lower()
    wealth_source = wealth_source.lower()

    if income_source == "cps":
        inequality_moments.update(_taxcalc_cps_income_ginis(income_year))
    else:
        raise ValueError(f"Unsupported income data source: {income_source}")

    if wealth_source == "scf":
        from ogusa import wealth

        if scf_yrs_list is None:
            scf = wealth.get_wealth_data(web=scf_web, directory=scf_directory)
        else:
            scf = wealth.get_wealth_data(
                scf_yrs_list=scf_yrs_list,
                web=scf_web,
                directory=scf_directory,
            )
        wealth_moments = wealth.compute_wealth_moments(
            scf.copy(), np.array([1.0])
        )
        inequality_moments["Gini coefficient, wealth"] = float(
            wealth_moments[-2]
        )
    else:
        raise ValueError(f"Unsupported wealth data source: {wealth_source}")

    return inequality_moments


def get_age_profile_moments(
    var,
    min_age=20,
    max_age=80,
    earnings_source="cps",
    hours_source="cps",
    wealth_source="scf",
    consumption_source="psid",
    cps=None,
    cps_years=(2023, 2022),
    cps_urls=None,
    cps_directory=None,
    cps_web=False,
    income_year=None,
    scf_yrs_list=None,
    scf_web=False,
    scf_directory=None,
    psid_path=None,
    psid_consumption_vars=None,
    psid_weight_col=None,
    psid_fallback=True,
):
    """
    Compute mean age profiles from household survey data.

    Args:
        var (str): Variable to compute. Must be one of "earnings",
            "hours", "wealth", or "consumption".
        min_age (int): Youngest age to include in the returned profile.
        max_age (int): Oldest age to include in the returned profile.
        earnings_source (str): Data source for earnings. Currently supports
            "cps" and "psid".
        hours_source (str): Data source for hours. Currently supports
            "cps" and "psid".
        wealth_source (str): Data source for wealth. Currently supports
            "scf".
        consumption_source (str): Data source for consumption expenditures.
            Currently supports "psid".
        cps (Pandas DataFrame): CPS hours data with age, hours, and
            optionally a weight column. If None and hours_source is "cps",
            download the NBER CPS ASEC person files.
        cps_years (tuple): CPS ASEC survey years to pool for hours.
        cps_urls (dict): Optional mapping from CPS ASEC survey year to zip
            file URL or local path.
        cps_directory (str): Local directory with trimmed CPS ASEC CSVs.
        cps_web (bool): If True, download CPS ASEC zip files when a local
            CSV is missing.
        income_year (int): Year to use for Tax-Calculator CPS earnings.
            If None, use the CPS data start year.
        scf_yrs_list (list): SCF survey years to pool. If None, use the
            default years in wealth.get_wealth_data().
        scf_web (bool): If True, download SCF data from the web.
            Defaults to False and uses local trimmed CSVs in ogusa/data/SCF.
        scf_directory (str): Local SCF data directory when scf_web=False.
        psid_path (str): Local PSID lifetime-income file path.
        psid_consumption_vars (list): PSID columns to add for the
            consumption measure. If None, use food_out_expend and
            food_in_expend, the expenditure fields currently packaged in
            the PSID data.
        psid_weight_col (str): Optional PSID weight column. The default is
            None because the packaged PSID sample is the SRC sample used as
            representative in psid_data_setup.py.
        psid_fallback (bool): If True, use PSID hours when a supplied CPS
            dataframe is not usable.

    Returns:
        profile (Pandas Series): Mean value by model age, indexed from age
            20 through 99. Ages outside min_age and max_age are NaN.
    """
    var = var.lower()
    if var not in {"earnings", "hours", "wealth", "consumption"}:
        raise ValueError(
            'var must be one of "earnings", "hours", "wealth", '
            'or "consumption".'
        )

    if var == "earnings":
        earnings_source = earnings_source.lower()
        if earnings_source == "cps":
            earnings = _taxcalc_cps_earnings_by_age(
                min_age, max_age, income_year=income_year
            )
            return _model_age_profile(earnings, min_age, max_age)
        if earnings_source == "psid":
            earnings = _psid_person_profile(
                "earnings",
                min_age,
                max_age,
                psid_path=psid_path,
                weight_col=psid_weight_col,
            )
            return _model_age_profile(earnings, min_age, max_age)
        raise ValueError(f"Unsupported earnings source: {earnings_source}")

    if var == "hours":
        hours_source = hours_source.lower()
        if hours_source == "cps":
            if cps is not None:
                try:
                    hours = _cps_hours_by_age(cps, min_age, max_age)
                except ValueError:
                    if not psid_fallback:
                        raise
                    warnings.warn(
                        "Supplied CPS hours data are not usable. Using "
                        "PSID hours data instead.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    hours = _psid_person_profile(
                        "hours",
                        min_age,
                        max_age,
                        psid_path=psid_path,
                        weight_col=psid_weight_col,
                    )
            else:
                hours = _nber_cps_hours_by_age(
                    min_age,
                    max_age,
                    cps_years=cps_years,
                    cps_urls=cps_urls,
                    cps_directory=cps_directory,
                    cps_web=cps_web,
                )
            hours = hours / ((24 - 8) * 7)  # scale so fraction of a waking day
            return _model_age_profile(hours, min_age, max_age)
        if hours_source == "psid":
            hours = _psid_person_profile(
                "hours",
                min_age,
                max_age,
                psid_path=psid_path,
                weight_col=psid_weight_col,
            )
            hours = hours / ((24 - 8) * 7 * 50)  # scale so fraction of
            # a waking day, noting that PSID hours are annual hours worked
            return _model_age_profile(hours, min_age, max_age)

        raise ValueError(f"Unsupported hours source: {hours_source}")

    if var == "wealth":
        wealth_source = wealth_source.lower()
        if wealth_source == "scf":
            from ogusa import wealth

            if scf_yrs_list is None:
                scf = wealth.get_wealth_data(
                    web=scf_web,
                    directory=scf_directory,
                    include_age=True,
                )
            else:
                scf = wealth.get_wealth_data(
                    scf_yrs_list=scf_yrs_list,
                    web=scf_web,
                    directory=scf_directory,
                    include_age=True,
                )
            wealth_profile = _weighted_mean_by_age(
                scf, "networth_infadj", "wgt", min_age, max_age
            )
            return _model_age_profile(wealth_profile, min_age, max_age)
        raise ValueError(f"Unsupported wealth source: {wealth_source}")

    consumption_source = consumption_source.lower()
    if consumption_source == "psid":
        consumption = _psid_consumption_by_age(
            min_age,
            max_age,
            psid_path=psid_path,
            consumption_vars=psid_consumption_vars,
            weight_col=psid_weight_col,
        )
        return _model_age_profile(consumption, min_age, max_age)

    raise ValueError(
        f"Unsupported consumption source: {consumption_source}. "
        "CPS and SCF do not contain a broad consumption expenditure "
        "measure comparable to PSID or CEX consumption data."
    )
