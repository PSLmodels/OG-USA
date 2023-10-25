import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from ogusa.utils import MVKDE

CURDIR = os.path.split(os.path.abspath(__file__))[0]


def get_bequest_matrix(
    J=7, lambdas=np.array([0.25, 0.25, 0.2, 0.1, 0.1, 0.09, 0.01]), graphs=True
):
    """
    Returns S x J matrix representing the fraction of aggregate
    bequests that go to each household by age and lifetime income group.

    """
    # Create directory if output directory does not already exist
    CURDIR = os.path.split(os.path.abspath(__file__))[0]
    output_fldr = "csv_output_files"
    output_dir = os.path.join(CURDIR, "..", "data", output_fldr)
    if not os.access(output_dir, os.F_OK):
        os.makedirs(output_dir)
    image_fldr = "images"
    image_dir = os.path.join(CURDIR, "..", "data", image_fldr)
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
    # 'net_wealth', 'inheritance', 'value_inheritance_1st',
    # 'value_inheritance_2nd', 'value_inheritance_3rd's
    # inheritance available from 1988 onwards...
    # Note that the resulting distribution is very different from what
    # Rick has found with the SCF

    df["sum_inherit"] = (
        df["value_inheritance_1st"]
        + df["value_inheritance_2nd"]
        + df["value_inheritance_3rd"]
    )
    # print(df[['sum_inherit', 'inheritance']].describe())

    if graphs:
        # Total inheritances by year
        df.groupby("year_data").mean(numeric_only=True).plot(y="inheritance")
        plt.savefig(os.path.join(image_dir, "inheritance_year.png"))
        df.groupby("year_data").mean(numeric_only=True).plot(y="sum_inherit")
        plt.savefig(os.path.join(image_dir, "sum_inherit_year.png"))
        # not that summing up inheritances gives a much larger value than
        # taking the inheritance variable

        # Fraction of inheritances in a year by age
        # line plot
        df[df["year_data"] >= 1988].groupby("age").mean(
            numeric_only=True
        ).plot(y="net_wealth")
        plt.savefig(os.path.join(image_dir, "net_wealth_age.png"))
        df[df["year_data"] >= 1988].groupby("age").mean(
            numeric_only=True
        ).plot(y="inheritance")
        plt.savefig(os.path.join(image_dir, "inheritance_age.png"))

        # Inheritances by lifetime income group
        # bar plot
        df[df["year_data"] >= 1988].groupby("li_group").mean(
            numeric_only=True
        ).plot.bar(y="net_wealth")
        plt.savefig(os.path.join(image_dir, "net_wealth_li.png"))
        df[df["year_data"] >= 1988].groupby("li_group").mean(
            numeric_only=True
        ).plot.bar(y="inheritance")
        plt.savefig(os.path.join(image_dir, "inheritance_li.png"))

        # lifecycle plots with line for each ability type
        pd.pivot_table(
            df[df["year_data"] >= 1988],
            values="net_wealth",
            index="age",
            columns="li_group",
            aggfunc="mean",
        ).plot(legend=True)
        plt.savefig(os.path.join(image_dir, "net_wealth_age_li.png"))
        pd.pivot_table(
            df[df["year_data"] >= 1988],
            values="inheritance",
            index="age",
            columns="li_group",
            aggfunc="mean",
        ).plot(legend=True)
        plt.savefig(os.path.join(image_dir, "inheritance_age_li.png"))

    # Matrix Fraction of inheritances in a year by age and lifetime_inc
    inheritance_matrix = pd.pivot_table(
        df[df["year_data"] >= 1988],
        values="inheritance",
        index="age",
        columns="li_group",
        aggfunc="sum",
    )
    # replace NaN with zero
    inheritance_matrix.fillna(value=0, inplace=True)
    inheritance_matrix = inheritance_matrix / inheritance_matrix.sum().sum()
    # inheritance_matrix.to_csv(os.path.join(
    #     output_dir, 'bequest_matrix.csv'))

    # estimate kernel density of bequests
    kde_matrix = MVKDE(
        80,
        7,
        inheritance_matrix.to_numpy(),
        filename=os.path.join(image_dir, "inheritance_kde.png"),
        plot=graphs,
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
        os.path.join(output_dir, "bequest_matrix_kde.csv"),
        kde_matrix,
        delimiter=",",
    )

    return kde_matrix
