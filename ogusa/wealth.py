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


def smooth_joint_dist(
    w_dist_sj, p_dist_sj, s_min, lambdas, sigma_s=3.0, sigma_j=0.5, var_str="",
    save_mat=False, plot=False, plot_type="bar", plot_title=True,
    save_plot=False
):
    """
    Smooth an (S, J) joint wealth distribution by smoothing wealth per
    household across neighboring (s, j) cells with a Gaussian kernel, using a
    separate bandwidth for each axis, and rescale it to sum to one.

    Args:
        w_dist_sj (array_like S x J): share of total wealth in each cell
        p_dist_sj (array_like S x J): share of total population in each cell
        s_min (int): Minimum or starting age in the age distribution of S
        lambdas (vector, len-J): percentiles associated with each lifetime
                    income group
        sigma_s (scalar >= 0): kernel standard deviation across ages, in
            number of age bins
        sigma_j (scalar >= 0): kernel standard deviation across lifetime
            income groups, in number of groups
        var_str (string): String of variable name--either "",
            "wealth $b_{j,s}$", or "bequests $bq_{j,s}$"
        save_mat (bool): Save the proportion matrix if =True
        plot (bool): Plot the KDE smoothed distributions if =True.
        plot_type (string): Plot type: either "bar" or "surface"
        plot_title (bool): Include plot title if =True
        save_plot (bool): Save the plot if =True

    Returns:
        w_dist_sj_smooth_scaled (array S x J): smoothed share of total wealth
            in each cell, summing to one
    """
    w_dist = np.asarray(w_dist_sj, dtype=float)
    p_dist = np.asarray(p_dist_sj, dtype=float)
    kernel = {"sigma": (sigma_s, sigma_j), "mode": "nearest"}
    S, J = w_dist.shape
    # Wealth per household relative to the average household (1 = average),
    # averaged over neighboring cells weighted by their population. Smoothing
    # the numerator and denominator separately avoids dividing by the small
    # or zero population of sparse cells.
    rel_wealth = gaussian_filter(w_dist, **kernel) / gaussian_filter(
        p_dist, **kernel
    )
    # Wealth share of each cell = population share x relative wealth
    w_dist_sj_smooth = p_dist * rel_wealth
    w_dist_sj_smooth_scaled = w_dist_sj_smooth / w_dist_sj_smooth.sum()
    if save_mat:
        save_dir = os.path.join(CUR_DIR, "data", "SCF")
        if var_str == "wealth":
            orig_path = os.path.join(save_dir, "w_dist_sj_orig_dict.pkl")
            smooth_path = os.path.join(save_dir, "w_dist_sj_smooth_dict.pkl")
        elif var_str == "bequests":
            orig_path = os.path.join(save_dir, "bq_dist_sj_orig_dict.pkl")
            smooth_path = os.path.join(save_dir, "bq_dist_sj_smooth_dict.pkl")
        w_dist_sj_dict = {
            "w_dist_sj": w_dist_sj,
            "p_dist_sj": p_dist_sj
        }
        w_dist_sj_smooth_scaled_dict = {
            "w_dist_sj_smooth_scaled": w_dist_sj_smooth_scaled,
            "p_dist_sj": p_dist_sj
        }
        pickle.dump(w_dist_sj_dict, open(orig_path, "wb"))
        pickle.dump(w_dist_sj_smooth_scaled_dict, open(smooth_path, "wb"))
    if plot:
        ages = np.arange(s_min, s_min + S)
        # Color gradation: Blues trimmed to its 0.25-1.0 range
        blues_trunc = colors.ListedColormap(
            cm.Blues(np.linspace(0.25, 1.0, 256))
        )
        if var_str=="":
            latex_title_str = ""
            latex_axis_str = ""
        elif var_str == "wealth":
            latex_title_str = r"$b_{j,s,1}$"
            latex_axis_str = r"$B_1$"
        elif var_str == "bequests":
            latex_title_str = r"$bq_{j,s,1}$"
            latex_axis_str = r"$BQ_1$"
        fig1 = plt.figure(figsize=(10, 7))
        ax1 = fig1.add_subplot(projection="3d")
        if plot_type == "bar":
            # One bar per (age, j) cell: x = age, y = j, height = wealth share
            A, Jg = np.meshgrid(ages, np.arange(J), indexing="ij")
            x = A.ravel() - 0.4
            y = Jg.ravel() - 0.4
            dz = w_dist_sj_smooth_scaled.ravel()
            # Color bars by height with a single-hue sequential colormap
            norm = colors.Normalize(vmin=min(dz.min(), 0.0), vmax=dz.max())
            bar_colors = cm.Blues(0.25 + 0.75 * norm(dz))
            ax1.bar3d(
                x, y, np.zeros_like(dz), 0.8, 0.8, dz, color=bar_colors,
                shade=True, linewidth=0
            )
        elif plot_type=="surface":
            # Grid of (age, j) points, height = wealth share
            A, Jg = np.meshgrid(ages, np.arange(J), indexing="ij")  # (80, 7)
            Z = w_dist_sj_smooth_scaled
            # Same gradation as the bar chart: Blues trimmed to range 0.25-1.0
            norm = colors.Normalize(vmin=min(Z.min(), 0.0), vmax=Z.max())
            surf = ax1.plot_surface(
                A, Jg, Z,
                cmap=blues_trunc, norm=norm,
                rstride=1, cstride=1,  # use every data point (no downsampling)
                linewidth=0, antialiased=False,
            )
        # Label j axis with the lifetime income percentile ranges
        cum = np.concatenate(([0], np.cumsum(lambdas))) * 100
        ax1.set_yticks(np.arange(J))
        ax1.set_yticklabels(
            [f"{cum[k]:.0f}-{cum[k + 1]:.0f}%" for k in range(J)]
        )
        ax1.set_xlabel(r"Age $s$")
        ax1.set_ylabel(r"Lifetime income group $j$")
        ax1.set_zlabel("Percent of total " + var_str + " " + latex_axis_str)
        ax1.view_init(elev=15, azim=-65)
        plt.tight_layout()
        if plot_title:
            plot_title1_str = (
                "Smoothed distribution of initial household " + var_str + " " +
                latex_title_str
            )
            ax1.set_title(plot_title1_str)
        if save_plot:
            if var_str == "wealth":
                smooth_save_path = os.path.join(
                    CUR_DIR, "data", "SCF", "w_dist_sj_smooth.png"
                )
            elif var_str == "bequests":
                smooth_save_path = os.path.join(
                    CUR_DIR, "data", "SCF", "bq_dist_sj_smooth.png"
                )
            plt.savefig(smooth_save_path, dpi=300)

        plt.show()
        plt.close()

        fig2 = plt.figure(figsize=(10, 7))
        ax2 = fig2.add_subplot(projection="3d")
        if plot_type == "bar":
            # One bar per (age, j) cell: x = age, y = j, height = wealth share
            dz2 = w_dist_sj.ravel()
            # Color bars by height with a single-hue sequential colormap
            norm2 = colors.Normalize(vmin=min(dz2.min(), 0.0), vmax=dz2.max())
            bar_colors = cm.Blues(0.25 + 0.75 * norm2(dz2))
            ax2.bar3d(
                x, y, np.zeros_like(dz2), 0.8, 0.8, dz2, color=bar_colors,
                shade=True, linewidth=0
            )
        elif plot_type=="surface":
            # Grid of (age, j) points, height = wealth share
            Z2 = w_dist_sj
            # Same gradation as the bar chart: Blues trimmed to range 0.25-1.0
            norm2 = colors.Normalize(vmin=min(Z2.min(), 0.0), vmax=Z2.max())
            surf = ax2.plot_surface(
                A, Jg, Z2,
                cmap=blues_trunc, norm=norm2,
                rstride=1, cstride=1,  # use every data point (no downsampling)
                linewidth=0, antialiased=False,
            )
        # Label j axis with the lifetime income percentile ranges
        cum = np.concatenate(([0], np.cumsum(lambdas))) * 100
        ax2.set_yticks(np.arange(J))
        ax2.set_yticklabels(
            [f"{cum[k]:.0f}-{cum[k + 1]:.0f}%" for k in range(J)]
        )
        ax2.set_xlabel(r"Age $s$")
        ax2.set_ylabel(r"Lifetime income group $j$")
        ax2.set_zlabel("Percent of total " + var_str + " " + latex_axis_str)
        ax2.view_init(elev=15, azim=-65)
        plt.tight_layout()
        if plot_title:
            plot_title2_str = (
                "Original distribution of initial household " + var_str + " " +
                latex_title_str
            )
            ax2.set_title(plot_title2_str)
        if save_plot:
            if var_str == "wealth":
                orig_save_path = os.path.join(
                    CUR_DIR, "data", "SCF", "w_dist_sj_orig.png"
                )
            elif var_str == "bequests":
                orig_save_path = os.path.join(
                    CUR_DIR, "data", "SCF", "bq_dist_sj_orig.png"
                )
            plt.savefig(orig_save_path, dpi=300)

        plt.show()
        plt.close()

    return w_dist_sj_smooth_scaled
