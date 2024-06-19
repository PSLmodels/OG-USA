import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from ogusa.utils import MVKDE
from ogusa.constants import CODE_PATH


def get_transfer_matrix(
    J=7,
    lambdas=np.array([0.25, 0.25, 0.2, 0.1, 0.1, 0.09, 0.01]),
    data_path=None,
    output_path=None,
):
    """
    Compute SxJ matrix representing the distribution of aggregate
    government transfers by age and lifetime income group.

    Args:
        J (int): number of lifetime income groups
        lambdas (Numpy array): length J array of lifetime income group
            proportions
        data_path (str): path to PSID data
        output_path (str): path to save output plots and data

    Returns:
        kde_matrix (Numpy array): SxJ shaped array that represents the
            smoothed distribution of proportions going to each (s,j)
    """
    # Read in PSID data
    if data_path is None:
        # Read data file shipped with OG-USA package
        df = pd.read_csv(
            os.path.join(CODE_PATH, "psid_lifetime_income.csv.gz")
        )
    else:
        # This is the case when running this from a branch of the OG-USA repo
        df = pd.read_csv(data_path)

    # Do some tabs with data file...
    df["total_transfers"] = (
        df["head_and_spouse_transfer_income"]
        + df["other_familyunit_transfer_income"]
    )

    df["sum_transfers"] = (
        # df["other_familyunit_ssi_prior_year"] # don't include SSI since OG-USA models separately
        df["head_other_welfare_prior_year"]
        + df["spouse_other_welfare_prior_year"]
        + df["other_familyunit_other_welfare_prior_year"]
        + df["head_unemp_inc_prior_year"]
        + df["spouse_unemp_inc_prior_year"]
        + df["other_familyunit_unemp_inc_prior_year"]
    )

    if output_path is not None:
        # Create plot path directory if it doesn't already exist
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        # Total total_transfers by year
        df.groupby("year_data").mean(numeric_only=True).plot(
            y="total_transfers"
        )
        plt.savefig(os.path.join(output_path, "total_transfers_year.png"))
        df.groupby("year_data").mean(numeric_only=True).plot(y="sum_transfers")
        plt.savefig(os.path.join(output_path, "sum_transfers_year.png"))
        # note that the sum of transfer categories is much lower than the
        # "total transfers" variable.  The transfers variable goes more to high income
        # and old, even though it says it excludes social security
        # because of this, we'll use the "sum transfers" variable

        # Fraction of total_transfers in a year by age
        # line plot
        df[df["year_data"] >= 1988].groupby("age").mean(
            numeric_only=True
        ).plot(y="total_transfers")
        plt.savefig(os.path.join(output_path, "total_transfers_age.png"))

        # total_transfers by lifetime income group
        # bar plot
        df[df["year_data"] >= 1988].groupby("li_group").mean(
            numeric_only=True
        ).plot.bar(y="total_transfers")
        plt.savefig(os.path.join(output_path, "total_transfers_li.png"))

        # lifecycle plots with line for each ability type
        pd.pivot_table(
            df[df["year_data"] >= 1988],
            values="total_transfers",
            index="age",
            columns="li_group",
            aggfunc="mean",
        ).plot(legend=True)
        plt.savefig(os.path.join(output_path, "total_transfers_age_li.png"))

        pd.pivot_table(
            df[df["year_data"] >= 1988],
            values="sum_transfers",
            index="age",
            columns="li_group",
            aggfunc="mean",
        ).plot(legend=True)
        plt.savefig(os.path.join(output_path, "sum_transfers_age_li.png"))

    # Matrix Fraction of sum_transfers in a year by age and lifetime_inc
    transfers_matrix = pd.pivot_table(
        df[df["year_data"] >= 1988],
        values="sum_transfers",
        index="age",
        columns="li_group",
        aggfunc="sum",
    )
    # replace NaN with zero
    transfers_matrix.fillna(value=0, inplace=True)
    transfers_matrix = transfers_matrix / transfers_matrix.sum().sum()
    # total_transfers_matrix.to_csv(os.path.join(
    #     output_dir, 'transfer_matrix.csv'))

    # estimate kernel density of transfers
    if output_path is not None:
        filename = os.path.join(output_path, "sum_transfers_kde.png")
    else:
        filename = None
    kde_matrix = MVKDE(
        80,
        7,
        transfers_matrix.to_numpy(),
        filename=filename,
        plot=(output_path is not None),
        bandwidth=0.5,
    )

    if (J == 10) and np.array_equal(
        np.squeeze(lambdas[:6]), np.array([0.25, 0.25, 0.2, 0.1, 0.1, 0.09])
    ):
        kde_matrix_new = np.zeros((80, J))
        kde_matrix_new[:, :6] = kde_matrix[:, :6]
        kde_matrix_new[:, 6:] = (
            kde_matrix[:, 6:].sum(axis=1).reshape(80, 1)
            * np.tile(np.reshape(lambdas[6:], (1, 4)), (80, 1))
            / lambdas[6:].sum()
        )
        kde_matrix = kde_matrix_new

    if output_path is not None:
        np.savetxt(
            os.path.join(output_path, "sum_transfers_kde.csv"),
            kde_matrix,
            delimiter=",",
        )

    return kde_matrix
