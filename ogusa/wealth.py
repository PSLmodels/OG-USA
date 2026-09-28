import os
import datetime
import io
import zipfile
import pickle
import numpy as np
import pandas as pd
import urllib.error
import urllib.request
from matplotlib import pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter
from scipy.ndimage import gaussian_filter1d
from scipy.stats import gaussian_kde
from ogcore import utils as ogcore_utils
from ogusa import utils as ogusa_utils

# Set paths
CUR_PATH = os.path.split(os.path.abspath(__file__))[0]
CUR_DIR = os.path.dirname(os.path.realpath(__file__))
scf_data_dir = os.path.abspath(os.path.join(CUR_DIR, "data", "SCF"))

# Set some global parameters
scf_url = "https://www.federalreserve.gov/econres/files/scfp{year}s.zip"
scf_headers = {"User-Agent": "Mozilla/5.0"}
MIN_AGE, MAX_AGE = 21, 100
DIST_AGE_BIN_WIDTH = 10  # age bins for the output wealth distributions
# OG-Core default lifetime income group population shares
scf_default_lambdas = np.array([0.25, 0.25, 0.2, 0.1, 0.1, 0.09, 0.01])

# Define functions


def find_recent_scf_vintages(
    num_vintages=2, first_scf_year=1989, scf_url=scf_url, headers=scf_headers
):
    """
    Return the years of the most recent SCF vintages available on the
    Federal Reserve website, newest first.
    """
    this_year = datetime.date.today().year
    latest = this_year - (this_year - first_scf_year) % 3
    years = []
    for year in range(latest, first_scf_year - 1, -3):
        if ogusa_utils._url_exists(
            scf_url.format(year=year), headers=headers
        ):
            years.append(year)
        if len(years) == num_vintages:
            break
    return years


def load_scf(
    year, web=False, data_dir=scf_data_dir, scf_url=scf_url,
    headers=scf_headers
):
    """
    Download (or read the cached copy of) the SCF summary extract for a
    given survey year and return it as a DataFrame.
    """
    zip_path = os.path.join(data_dir, f"scfp{year}s.zip")
    if web:
        os.makedirs(data_dir, exist_ok=True)
        if not os.path.exists(zip_path):
            print(f"Downloading SCF {year} summary extract...")
            request = urllib.request.Request(
                scf_url.format(year=year), headers=headers
            )
            with urllib.request.urlopen(request, timeout=300) as response:
                with open(zip_path, "wb") as f:
                    f.write(response.read())
    with zipfile.ZipFile(zip_path) as zf:
        dta_name = [n for n in zf.namelist() if n.endswith(".dta")][0]
        df = pd.read_stata(io.BytesIO(zf.read(dta_name)))
    df.columns = df.columns.str.lower()
    return df


def compute_wealth_var(
    df_scf, foreign_equity_share=0.0, foreign_bond_share=0.0,
    exclude_durables=False
):
    """
    Add columns for domestic government debt holdings (D_d), domestic
    private asset holdings (K_d), and their sum (wealth) to the DataFrame.
    """
    df = df_scf.copy()
    gov_debt = (
        df["savbnd"] + df["govtbnd"] + df["notxbnd"] +
        df["gbmutf"] + df["tfbmutf"]
    )
    # EQUITY counts half of combination funds as equity, so the other half
    # is counted as bonds here
    foreign = foreign_equity_share * df["equity"] + foreign_bond_share * (
        df["obnd"] + df["obmutf"] + 0.5 * df["comutf"]
    )
    wealth = df["networth"] - foreign
    if exclude_durables:
        wealth = wealth - df["vehic"] - df["othnfin"]
    df["D_d"] = gov_debt
    df["K_d"] = wealth - gov_debt
    df["wealth"] = wealth
    return df


def assign_lifetime_income_group(
    df_scf, min_age=21, age_bin_width=5, lambdas=scf_default_lambdas
):
    """
    Assign each household to lifetime income group j = 0, ..., J-1 by its
    weighted percentile of normal income within its age bin.
    """
    df = df_scf.copy()
    # Drop all observations with df["age"] < min_age
    df = df[df["age"] >= min_age]
    df["age_bin"] = (df["age"] - min_age) // age_bin_width
    cutoffs = np.cumsum(lambdas)[:-1]
    df["j"] = 0
    for _, group in df.groupby("age_bin"):
        group = group.sort_values("norminc", kind="mergesort")
        weights = group["wgt"].to_numpy()
        pctile = (np.cumsum(weights) - 0.5 * weights) / weights.sum()
        df.loc[group.index, "j"] = np.searchsorted(
            cutoffs, pctile, side="right"
        )
    return df


def merge_infladj_mult_scf_years(df_scf_list, scf_years):
    """
    Consolidate multiple SCF years DataFrames into a single DataFrame, and
    adding inflation adjusted variables for D_d, K_d, and wealth to most recent
    CPILFESL monthly price levels.
    """
    df_cpi = ogusa_utils.get_cpi_monthly_data(start_year=min(scf_years))
    # Set cpi_cur to be the final monthly CPI value in df_cpi
    cpi_cur = df_cpi["CPILFESL"].iloc[-1]
    print("cpi_cur =", cpi_cur)
    df_scf_mult_year = pd.DataFrame()
    for year, df_scf in zip(scf_years, df_scf_list):
        df_scf["year"] = year
        # df_cpi["date"] holds "%Y-%m-%d" strings, so match July directly
        cpi_scf_year = (
            df_cpi.loc[df_cpi["date"] == f"{year}-07-01", "CPILFESL"]
        ).item()
        print("cpi_scf_year =", cpi_scf_year)
        df_scf["D_d_infladj"] = df_scf["D_d"] * (cpi_cur / cpi_scf_year)
        df_scf["K_d_infladj"] = df_scf["K_d"] * (cpi_cur / cpi_scf_year)
        df_scf["wealth_infladj"] = df_scf["wealth"] * (cpi_cur / cpi_scf_year)
        df_scf_mult_year = pd.concat(
            [df_scf_mult_year, df_scf], ignore_index=True
        )
    return df_scf_mult_year


def wealth_distributions(
    df, min_age=21, max_age=100, age_bin_width=10, lambdas=scf_default_lambdas
):
    """
    Return the share of total wealth held by households by age bin of the
    head of household and lifetime income group j as an (age bins x J)
    DataFrame, an (age bins,) Series by age bin, and a (J,) Series by
    lifetime income group. Return the same three distributions of the
    weighted population of households, the Gini coefficient of wealth, and
    the variance of the log of positive wealth.
    """
    age_bin_starts = range(min_age, max_age + 1, age_bin_width)
    age_labels = [f"{a}-{a + age_bin_width - 1}" for a in age_bin_starts]
    age_bin = (df["age"] - min_age) // age_bin_width

    def joint_dist(values):
        """
        Return the (age bins x J) DataFrame of each cell's share of the sum
        of values.
        """
        # Group the Series by the two key Series directly. Adding the keys
        # as columns instead would copy the (already fragmented) input
        # DataFrame.
        cell_sum = (
            values.groupby([age_bin, df["j"]]).sum().unstack(fill_value=0.0)
        )
        dist_2d = (
            cell_sum.reindex(
                index=range(len(age_labels)),
                columns=range(len(lambdas)),
                fill_value=0.0,
            )
            / values.sum()
        )
        dist_2d.index = pd.Index(age_labels, name="age")
        dist_2d.columns = pd.Index(
            [f"j={j}" for j in dist_2d.columns], name="lifetime income group"
        )
        return dist_2d

    # Distributions are (age bins x J) with rows as age bins and columns as
    # lifetime income groups, so summing across the columns (axis=1)
    # marginalizes to age bins and summing down the rows (axis=0)
    # marginalizes to lifetime income groups.
    # Wealth distributions
    w_dist_sj = joint_dist(df["wgt"] * df["wealth_infladj"])
    w_dist_s = w_dist_sj.sum(axis=1)
    w_dist_j = w_dist_sj.sum(axis=0)
    # Population distributions (p_dist_j should be close to lambdas)
    p_dist_sj = joint_dist(df["wgt"])
    p_dist_s = p_dist_sj.sum(axis=1)
    p_dist_j = p_dist_sj.sum(axis=0)

    # Compute gini coeff
    df_sorted = df.sort_values(by="wealth_infladj", ascending=True)
    wgt = df_sorted["wgt"].to_numpy()
    # Set negative wealth to zero
    wealth = df_sorted["wealth_infladj"].clip(lower=0.0).to_numpy()
    weighted_wealth =  wgt * wealth
    # Create vector of cummulative percent of population
    p = np.concatenate(([0.0], wgt.cumsum() / wgt.sum()))
    # Create vector of cumulative percent of wealth for each cumulative sum of
    # population, starting at 0
    nu = np.concatenate(
        ([0.0], weighted_wealth.cumsum() / weighted_wealth.sum())
    )
    gini_coef = (nu[1:] * p[:-1]).sum() - (nu[:-1] * p[1:]).sum()

    # Compute the variance of the log of wealth
    pos = df["wealth_infladj"] > 0.0
    ln_w = np.log(df.loc[pos, "wealth_infladj"])
    wgt_pos = df.loc[pos, "wgt"]
    mean_ln_w = (wgt_pos * ln_w).sum() / wgt_pos.sum()
    var_ln_w = (wgt_pos * (ln_w - mean_ln_w) ** 2).sum() / wgt_pos.sum()

    return (
        w_dist_sj, w_dist_s, w_dist_j, p_dist_sj, p_dist_s, p_dist_j,
        gini_coef, var_ln_w
    )


def smooth_wealth_dist_age(w_dist_sj, sigma_s=3.0):
    """
    Smooth an (S, J) joint wealth distribution by smoothing wealth per
    household across the age dimension of the data (s) using gaussian filter
    blurring, and rescale each j column to match the original share of wealth.
    """
    w = np.asarray(w_dist_sj, dtype=float)
    w_smooth = gaussian_filter1d(w, sigma=sigma_s, axis=0, mode="reflect")
    # Rescale each column to keep that group's raw share of total wealth
    return w_smooth * w.sum(axis=0) / w_smooth.sum(axis=0)
