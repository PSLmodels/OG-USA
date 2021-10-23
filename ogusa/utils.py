import pandas as pd


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
    df.drop(
        columns=["Unnamed: 15", "Unnamed: 16", "2026.1", "2031.1"],
        inplace=True,
    )
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
    df.drop(
        columns=[
            "Unnamed: 0",
            "Unnamed: 2",
            "Units",
            "Unnamed: 19",
            "Unnamed: 20",
            "Unnamed: 21",
            "Unnamed: 22",
            "Unnamed: 23",
            "Unnamed: 24",
        ],
        inplace=True,
    )
    # Note that real values come second (after nominal values)
    df.drop_duplicates(subset="variable", keep="last", inplace=True)
    df3 = df[~pd.isnull(df.variable)]
    df_st = df1.append(df2, sort=False, ignore_index=True).append(
        df3, sort=False, ignore_index=True
    )
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
