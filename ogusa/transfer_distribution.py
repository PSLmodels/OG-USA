import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from ogusa.utils import MVKDE

CURDIR = os.path.split(os.path.abspath(__file__))[0]


def get_transfer_matrix(
    J=7, lambdas=np.array([0.25, 0.25, 0.2, 0.1, 0.1, 0.09, 0.01]), graphs=True
):
    """
    Compute SxJ matrix representing the distribution of aggregate
    government transfers by age and lifetime income group.
    """
    # Create directory if output directory does not already exist
    CURDIR = os.path.split(os.path.abspath(__file__))[0]
    output_fldr = "csv_output_files"
    output_dir = os.path.join(CURDIR, output_fldr)
    if not os.access(output_dir, os.F_OK):
        os.makedirs(output_dir)
    image_fldr = "images"
    image_dir = os.path.join(CURDIR, image_fldr)
    if not os.access(image_dir, os.F_OK):
        os.makedirs(image_dir)

    # Define a lambda function to compute the weighted mean:
    # wm = lambda x: np.average(
    #     x, weights=df.loc[x.index, "fam_smpl_wgt_core"])

    # Read in dataframe of PSID data
    # df = ogcore.utils.safe_read_pickle(
    #     os.path.join(CURDIR, "data", "PSID", "psid_lifetime_income.pkl")
    # )
    df = pd.read_csv(
        os.path.join(CURDIR, "..", "data", "PSID", "psid_lifetime_income.csv")
    )

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

    if graphs:
        # Total total_transfers by year
        df.groupby("year_data").mean(numeric_only=True).plot(
            y="total_transfers"
        )
        plt.savefig(os.path.join(image_dir, "total_transfers_year.png"))
        df.groupby("year_data").mean(numeric_only=True).plot(y="sum_transfers")
        plt.savefig(os.path.join(image_dir, "sum_transfers_year.png"))
        # note that the sum of transfer categories is much lower than the
        # "total transfers" variable.  The transfers variable goes more to high income
        # and old, even though it says it excludes social security
        # because of this, we'll use the "sum transfers" variable

        # Fraction of total_transfers in a year by age
        # line plot
        df[df["year_data"] >= 1988].groupby("age").mean(
            numeric_only=True
        ).plot(y="total_transfers")
        plt.savefig(os.path.join(image_dir, "total_transfers_age.png"))

        # total_transfers by lifetime income group
        # bar plot
        df[df["year_data"] >= 1988].groupby("li_group").mean(
            numeric_only=True
        ).plot.bar(y="total_transfers")
        plt.savefig(os.path.join(image_dir, "total_transfers_li.png"))

        # lifecycle plots with line for each ability type
        pd.pivot_table(
            df[df["year_data"] >= 1988],
            values="total_transfers",
            index="age",
            columns="li_group",
            aggfunc="mean",
        ).plot(legend=True)
        plt.savefig(os.path.join(image_dir, "total_transfers_age_li.png"))

        pd.pivot_table(
            df[df["year_data"] >= 1988],
            values="sum_transfers",
            index="age",
            columns="li_group",
            aggfunc="mean",
        ).plot(legend=True)
        plt.savefig(os.path.join(image_dir, "sum_transfers_age_li.png"))

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
    kde_matrix = MVKDE(
        80,
        7,
        transfers_matrix.to_numpy(),
        filename=os.path.join(image_dir, "sum_transfers_kde.png"),
        plot=True,
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

    np.savetxt(
        os.path.join(output_dir, "sum_transfers_kde.csv"),
        kde_matrix,
        delimiter=",",
    )

    return kde_matrix
