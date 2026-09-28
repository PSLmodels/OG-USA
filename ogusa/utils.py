import io
import os
import urllib.error
import urllib.request
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import requests

CBO_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;"
        "q=0.9,image/avif,image/webp,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.cbo.gov/",
}
CBO_FORECAST_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "CBO_forecasts")
)


def _url_exists(url, headers):
    request = urllib.request.Request(url, headers=headers, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status == 200
    except urllib.error.HTTPError:
        return False


def _cbo_forecast_file(filename):
    """
    Return the path to a locally saved CBO forecast workbook.
    """
    return os.path.join(CBO_FORECAST_DIR, filename)


def _fetch_excel(url):
    """Download a CBO Excel file using a session to acquire cookies first."""
    session = requests.Session()
    session.get("https://www.cbo.gov/", headers=CBO_HEADERS, timeout=30)
    response = session.get(url, headers=CBO_HEADERS, timeout=30)
    response.raise_for_status()
    return io.BytesIO(response.content)


def _excel_source(source):
    """
    Return a file-like object for URLs and a local path for saved files.
    """
    source = os.fspath(source)
    if source.startswith(("http://", "https://")):
        return _fetch_excel(source)
    return os.path.expanduser(source)


def get_cpi_monthly_data(start_year=2022, start_month=7):
    """
    Return a DataFrame of monthly CPI core index value data from FRED, starting
    from the specified year and month to the most recent data.

    Args:
        start_year (int): The year to start the data from.
        start_month (int): The month to start the data from.

    Returns:
        df (pd.DataFrame): A DataFrame with the monthly CPI core index values.
    """
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPILFESL"
    df = pd.read_csv(url)
    # FRED renamed the CSV date column from "DATE" to "observation_date", so
    # identify it by position rather than by name.
    df.rename(columns={df.columns[0]: "date"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    df = df[
        df.index >= pd.Timestamp(year=start_year, month=start_month, day=1)
    ]
    df.reset_index(inplace=True)
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    df["CPILFESL"] = df["CPILFESL"].astype(float)

    return df


def read_cbo_forecast(
    lt_econ_url=_cbo_forecast_file("57054-2025-03-LTBO-econ.xlsx"),
    lt_budget_url=_cbo_forecast_file("51119-2025-03-LTBO-budget.xlsx"),
    ten_year_budget_url=_cbo_forecast_file(
        "51118-2025-01-Budget-Projections.xlsx"
    ),
    ten_year_macro_url=_cbo_forecast_file(
        "51135-2025-01-Economic-Projections.xlsx"
    ),
    lt_start_year=1995,
    lt_end_year=2055,
    st_start_year=2024,
    st_end_year=2035,
):
    """
    This function reads the CBO Long-Term Budget Projections document
    from https://www.cbo.gov/about/products/budget-economic-data#1
    and then formats the relevant data for use with OG-Core.

    Warning: CBO spreadsheets are not consistent across years so you may
    run into errors passing different URLs to this function.
    """
    CBO_VAR_NAMES = {
        "Real GDP (trillions of 2017 dollars)": "Y",
        "Real rates": "r",
        "Growth of real earnings per worker": "w_growth",
        "Growth of total hours worked": "L_growth",
        "Hours of All Persons (nonfarm business sector)": "L",
        "Personal consumption expenditures": "C",
        "Gross private domestic investment": "I_total",
        "Government Consumption Expenditures and Gross Investment": "G",
        "Old-Age and Survivors Insurance": "agg_pension_outlays",
        "Individual income taxes": "iit_revenue",
        "Payroll taxes": "payroll_tax_revenue",
        "Corporate income taxes": "business_tax_revenue",
        "U.S. wage and salary disbursements\n (trillions of dollars)": "wL",
    }

    # Econ data in levels
    # Read in data
    df = pd.read_excel(
        _excel_source(lt_econ_url),
        sheet_name="3. Econ Vars_Annual Levels",
        skiprows=6,
        nrows=62,
    )
    # replace column names with full variable names
    df.rename(columns=CBO_VAR_NAMES, inplace=True)
    # keep only variables that map to model variables
    df.set_index("Year", inplace=True)
    df_levels = df.loc[:, df.columns.isin(CBO_VAR_NAMES.values())]
    df_levels.reset_index(inplace=True)

    # Econ data in rates
    # Read in data
    df = pd.read_excel(
        _excel_source(lt_econ_url),
        sheet_name="1. Econ Vars_Annual Rates",
        skiprows=7,
        nrows=39,
    )
    df = df[df["Unnamed: 0"].notna()]
    df.rename(columns={"Unnamed: 0": "variable"}, inplace=True)
    df["var_name"] = df["variable"].replace(CBO_VAR_NAMES)
    # keep just variables of interest
    df.drop(
        columns=["variable"],
        inplace=True,
    )
    df = df[df["var_name"].isin(CBO_VAR_NAMES.values())]
    # Keep just real interest rate (not nominal)
    # Note that real interest rate comes first in table
    df.drop_duplicates(subset="var_name", inplace=True)
    # reshape so that variable names down column
    df = pd.melt(
        df,
        id_vars="var_name",
        value_vars=[i for i in range(lt_start_year, lt_end_year + 1)],
    )
    df_rates = df.pivot(index="variable", columns="var_name", values="value")
    df_rates.reset_index(inplace=True)
    df_rates.rename(columns={"variable": "Year"}, inplace=True)

    # add debt forecast
    df_fiscal = pd.read_excel(
        _excel_source(lt_budget_url),
        sheet_name="1. Summary Ext Baseline",
        skiprows=9,
        nrows=32,
    )
    df_fiscal = df_fiscal[
        ["Fiscal year", "Total", "Federal debt held by the public"]
    ]
    # rename Total to "Revenues"
    df_fiscal.rename(columns={"Total": "Revenues"}, inplace=True)
    # merge to macro levels data
    df_lt = df_fiscal.merge(
        df_levels, left_on="Fiscal year", right_on="Year", how="left"
    )
    # merge to macro rates data
    df_lt = df_lt.merge(
        df_rates, left_on="Fiscal year", right_on="Year", how="left"
    )
    df_lt.rename(
        columns={"Federal debt held by the public": "D/Y"}, inplace=True
    )
    df_lt["D"] = df_lt["Y"] * df_lt["D/Y"]
    # drop Year_x, Year_y columns
    df_lt.drop(columns=["Year_x", "Year_y"], inplace=True)
    # rename Fiscal year to year
    df_lt.rename(columns={"Fiscal year": "year"}, inplace=True)
    # %%
    #  10 year budget
    df = pd.read_excel(
        _excel_source(ten_year_budget_url),
        sheet_name="Table B-1",
        skiprows=7,
        nrows=7,
    )
    df.rename(
        columns={
            "Unnamed: 0": "variable",
            "Actual, 2024": st_start_year,
        },
        inplace=True,
    )
    df.drop(columns=["2026–2030", "2026–2035"], inplace=True)
    df1 = df[
        ~(
            (pd.isnull(df.variable))
            | (df.variable == "Other")
            | (df.variable == "Revenues")
        )
    ]
    # cast all year columns to float
    df1.iloc[:, 1:] = df1.iloc[:, 1:].astype(float)
    # cast all year column names to int
    df1.columns = [
        int(i) if isinstance(i, str) and i.isdigit() else i
        for i in df1.columns
    ]
    # data from other table
    df = pd.read_excel(
        _excel_source(ten_year_budget_url),
        sheet_name="Table B-4",
        skiprows=8,
        nrows=18,
    )
    df.rename(
        columns={
            "Unnamed: 0": "variable",
            "Actual, 2024": st_start_year,
        },
        inplace=True,
    )
    df.drop(columns=["2026–2030", "2026–2035"], inplace=True)
    df.drop_duplicates(subset="variable", keep="last", inplace=True)
    df2 = df[~pd.isnull(df.variable)]
    # cast all year columns to float
    df2.iloc[:, 1:] = df2.iloc[:, 1:].astype(float)
    # cast all year column names to int
    df2.columns = [
        int(i) if isinstance(i, str) and i.isdigit() else i
        for i in df2.columns
    ]

    # %%
    # 10 year macro forecast
    df = pd.read_excel(
        _excel_source(ten_year_macro_url),
        sheet_name="2. Calendar Year",
        skiprows=6,
        nrows=131,
    )
    df.rename(columns={"Unnamed: 0": "variable"}, inplace=True)
    # Note that real values come second (after nominal values)
    df.drop_duplicates(subset="variable", keep="last", inplace=True)
    df.drop(columns=["Units"], inplace=True)
    df3 = df[~pd.isnull(df.variable)]
    # cast all year columns to float
    df3.iloc[:, 1:] = df3.iloc[:, 1:].astype(float)
    # cast all year column names to int
    df3.columns = [
        int(i) if isinstance(i, str) and i.isdigit() else i
        for i in df3.columns
    ]
    # it's creating a lot of NaN values in the final dataframe
    df_st = pd.concat([df1, df2, df3], sort=False, ignore_index=True)

    df_st["var_name"] = df_st["variable"].replace(CBO_VAR_NAMES)
    df_st = df_st[~pd.isnull(df_st.var_name)]
    df_st.drop(columns=["variable"], inplace=True)
    # reshape so each row a year
    df_st = pd.melt(
        df_st,
        id_vars="var_name",
        value_vars=[i for i in range(st_start_year, st_end_year + 1)],
    )
    df_st = df_st.pivot(
        index="variable", columns="var_name", values="value"
    ).reset_index()
    df_st.rename(columns={"variable": "year"}, inplace=True)

    # %%
    # merge with long term data
    df_cbo = df_lt.merge(
        df_st, how="outer", on="year", suffixes=("_lt", "_st")
    )
    # replace * with 0
    df_cbo.replace(to_replace="*", value=0.0, inplace=True)

    return df_cbo


# Will need to do some smoothing with a KDE when estimate the matrix...
def MVKDE(
    S,
    J,
    proportion_matrix,
    zaxis_label="Received proportion of total transfers",
    filename=None,
    plot=False,
    bandwidth=0.25,
):
    """
    Generates a Multivariate Kernel Density Estimator and returns a
    matrix representing a probability distribution according to given
    age categories, and ability type categories.

    Args:
        S (scalar): the number of age groups in the model
        J (scalar): the number of ability type groups in the model.
        proportion_matrix (Numpy array): SxJ shaped array that
            represents the proportions of the total going to each
            (s,j) combination
        filename (str): the file name  to save image to
        plot (bool): whether or not to save a plot of the probability
            distribution generated by the kde or the proportion matrix
        bandwidth (scalar):  used in the smoothing of the kernel. Higher
            bandwidth creates a smoother kernel.

    Returns:
        estimator_scaled (Numpy array): SxJ shaped array that
            that represents the smoothed distribution of proportions
            going to each (s,j)

    """
    proportion_matrix_income = np.sum(proportion_matrix, axis=0)
    proportion_matrix_age = np.sum(proportion_matrix, axis=1)
    age_probs = np.random.multinomial(70000, proportion_matrix_age)
    income_probs = np.random.multinomial(70000, proportion_matrix_income)
    age_frequency = np.array([])
    income_frequency = np.array([])
    age_mesh = complex(str(S) + "j")
    income_mesh = complex(str(J) + "j")
    j = 18
    """creating a distribution of age values"""
    for i in age_probs:
        listit = np.ones(i)
        listit *= j
        age_frequency = np.append(age_frequency, listit)
        j += 1

    k = 1
    """creating a distribution of ability type values"""
    for i in income_probs:
        listit2 = np.ones(i)
        listit2 *= k
        income_frequency = np.append(income_frequency, listit2)
        k += 1

    freq_mat = np.vstack((age_frequency, income_frequency)).T
    density = gaussian_kde(freq_mat.T, bw_method=bandwidth)
    age_min, income_min = freq_mat.min(axis=0)
    age_max, income_max = freq_mat.max(axis=0)
    agei, incomei = np.mgrid[
        age_min:age_max:age_mesh, income_min:income_max:income_mesh
    ]
    coords = np.vstack([item.ravel() for item in [agei, incomei]])
    estimator = density(coords).reshape(agei.shape)
    estimator_scaled = estimator / float(np.sum(estimator))
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.plot_surface(agei, incomei, estimator_scaled, rstride=5)
        ax.set_xlabel("Age")
        ax.set_ylabel("Ability Types")
        ax.set_zlabel(zaxis_label)
        plt.savefig(filename)
    return estimator_scaled
