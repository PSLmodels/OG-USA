import pandas as pd
import numpy as np
from scipy.stats import kde
import matplotlib.pyplot as plt
import requests
import urllib3
import ssl


def read_cbo_forecast():
    """
    This function reads the CBO Long-Term Budget Projections document
    from https://www.cbo.gov/about/products/budget-economic-data#1
    and then formats the relevant data for use with OG-Core
    """
    CBO_LT_URL = (
        "https://www.cbo.gov/system/files/2020-09/51119-2020-09-ltbo_0.xlsx"
    )
    # Read in data
    df = pd.read_excel(
        CBO_LT_URL, sheet_name="3. Economic Vars", skiprows=7, nrows=45
    )
    df.drop(columns=["Unnamed: 3", "Unnamed: 4"], inplace=True)
    df[
        ~(
            (pd.isnull(df["Unnamed: 0"]))
            & (pd.isnull(df["Unnamed: 1"]))
            & (pd.isnull(df["Unnamed: 2"]))
        )
    ]
    df.fillna(value="", inplace=True)
    df["full_var_name"] = (
        df["Unnamed: 0"] + df["Unnamed: 1"] + df["Unnamed: 2"]
    )
    CBO_VAR_NAMES = {
        "Real GDP (Billions of 2019 dollars) ": "Y",
        "On 10-year Treasury notes and the OASDI trust funds": "r",
        "Growth of Real Earnings per Worker": "w_growth",
        "Growth of Total Hours Worked": "L_growth",
        "Hours of All Persons (Nonfarm Business Sector)": "L",
        "Personal Consumption Expenditures": "C",
        "Gross Private Domestic Investment": "I_total",
        "Government Consumption Expenditures and Gross Investment": "G",
        "Old-Age and Survivors Insurance": "agg_pension_outlays",
        "Individual income taxes": "iit_revenue",
        "Payroll taxes": "payroll_tax_revenue",
        "Corporate income taxes": "business_tax_revenue",
        "Wages and Salaries": "wL",
    }
    df["var_name"] = df["full_var_name"].replace(CBO_VAR_NAMES)
    # keep just variables of interest
    df.drop(
        columns=["Unnamed: 0", "Unnamed: 1", "Unnamed: 2", "full_var_name"],
        inplace=True,
    )
    df = df[df["var_name"].isin(CBO_VAR_NAMES.values())]
    # Keep just real interest rate (not nominal)
    # Note that real interest rate comes first in table
    df.drop_duplicates(subset="var_name", inplace=True)
    # reshape so that variable names down column
    df = pd.melt(
        df, id_vars="var_name", value_vars=[i for i in range(1990, 2051)]
    )
    df = df.pivot(index="variable", columns="var_name", values="value")
    df.reset_index(inplace=True)
    df.rename(columns={"variable": "year"}, inplace=True)
    # add debt forcast
    df_fiscal = pd.read_excel(
        CBO_LT_URL,
        sheet_name="1. Summary Extended Baseline",
        skiprows=9,
        nrows=32,
    )
    df_fiscal = df_fiscal[
        ["Fiscal Year", "Revenues", "Federal Debt Held by the Public"]
    ]
    df_lt = df.merge(
        df_fiscal, left_on="year", right_on="Fiscal Year", how="left"
    )
    df_lt.rename(
        columns={"Federal Debt Held by the Public": "D/Y"}, inplace=True
    )
    df_lt["D"] = df_lt["Y"] * df_lt["D/Y"]

    CBO_10yr_budget_URL = (
        "https://www.cbo.gov/system/files/2021-02/51118-2021-02-11-"
        + "budgetprojections.xlsx"
    )
    df = pd.read_excel(
        CBO_10yr_budget_URL, sheet_name="Table 1-1", skiprows=8, nrows=7
    )
    df.rename(
        columns={"Unnamed: 0": "variable", "Actual, \n2020": 2020},
        inplace=True,
    )
    df.drop(columns=["2026.1", "2031.1"], inplace=True)
    df1 = df[~((pd.isnull(df.variable)) | (df.variable == "Other"))]

    df = pd.read_excel(
        CBO_10yr_budget_URL, sheet_name="Table 1-3", skiprows=9, nrows=22
    )
    df.rename(columns={"Unnamed: 0": "variable"}, inplace=True)
    df.drop(columns=["2026.1", "2031.1"], inplace=True)
    df.drop_duplicates(subset="variable", keep="last", inplace=True)
    df2 = df[~pd.isnull(df.variable)]

    CBO_10yr_macro_URL = (
        "https://www.cbo.gov/system/files/2021-02/51135-2021-02-"
        + "economicprojections.xlsx"
    )
    df = pd.read_excel(
        CBO_10yr_macro_URL,
        sheet_name="2. Calendar Year",
        skiprows=6,
        nrows=131,
    )
    df.rename(columns={"Unnamed: 1": "variable"}, inplace=True)
    df.drop(columns=["Unnamed: 0", "Unnamed: 2", "Units"], inplace=True)
    # Note that real values come second (after nominal values)
    df.drop_duplicates(subset="variable", keep="last", inplace=True)
    df3 = df[~pd.isnull(df.variable)]
    df_st = pd.concat([df1, df2, df3], sort=False, ignore_index=True)
    # df_st = df1.append(df2, sort=False, ignore_index=True).append(
    #     df3, sort=False, ignore_index=True
    # )
    df_st["var_name"] = df_st["variable"].replace(CBO_VAR_NAMES)
    df_st = df_st[~pd.isnull(df_st.var_name)]
    df_st.drop(columns=["variable"], inplace=True)
    # reshape so each row a year
    df_st = pd.melt(
        df_st, id_vars="var_name", value_vars=[i for i in range(2017, 2031)]
    )
    df_st = df_st.pivot(
        index="variable", columns="var_name", values="value"
    ).reset_index()
    df_st.rename(columns={"variable": "year"}, inplace=True)

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
    density = kde.gaussian_kde(freq_mat.T, bw_method=bandwidth)
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


class CustomHttpAdapter(requests.adapters.HTTPAdapter):
    """
    The UN Data Portal server doesn't support "RFC 5746 secure renegotiation". This causes and error when the client is using OpenSSL 3, which enforces that standard by default.
    The fix is to create a custom SSL context that allows for legacy connections. This defines a function get_legacy_session() that should be used instead of requests().
    """

    # "Transport adapter" that allows us to use custom ssl_context.
    def __init__(self, ssl_context=None, **kwargs):
        self.ssl_context = ssl_context
        super().__init__(**kwargs)

    def init_poolmanager(self, connections, maxsize, block=False):
        self.poolmanager = urllib3.poolmanager.PoolManager(
            num_pools=connections,
            maxsize=maxsize,
            block=block,
            ssl_context=self.ssl_context,
        )


def get_legacy_session():
    ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
    ctx.options |= 0x4  # OP_LEGACY_SERVER_CONNECT  #in Python 3.12 you will be able to switch from 0x4 to ssl.OP_LEGACY_SERVER_CONNECT.
    session = requests.session()
    session.mount("https://", CustomHttpAdapter(ctx))
    return session
