import pandas as pd
import numpy as np
import microdf as mdf
import matplotlib.pyplot as plt
import statsmodels.api as sm

cps = pd.read_csv(
    "https://github.com/PSLmodels/taxdata/raw/master/data/cps.csv.gz",
    usecols=[
        "age_head",
        "age_spouse",
        "elderly_dependents",
        "nu18",
        "n1820",
        "n21",
        "s006",
        "e00200"  # Wages, salaries, and tips.
        # TODO: Switch to AGI or expanded_income,
        # from taxcalc output.
    ],
)


# Add n65 and n1864 to CPS data NOT INCLUDING HEAD
# (CPS already has nu18)
cps["n65"] = cps.elderly_dependents + (cps.age_spouse >= 65)
cps["n1864"] = cps.n1820 + cps.n21 - cps.n65

# Initialize a new empty DataFrame with cps structure,
# and stack the version with quantiles on each.
cps2 = pd.DataFrame(columns=cps.columns)
for age in np.arange(20, 80):
    temp = cps[cps.age_head == age].copy()
    mdf.add_weighted_quantiles(temp, "e00200", "s006")
    cps2 = pd.concat([cps2, temp], ignore_index=True)
cps2["income_bin"] = pd.cut(
    cps2["e00200_percentile_exact"], [0, 25, 50, 70, 90, 99, 100], right=False
)

cps2.drop(
    [
        "e00200_percentile_exact",
        "e00200_percentile",
        "e00200_2percentile",
        "e00200_ventile",
        "e00200_decile",
        "e00200_quintile",
        "e00200_quartile",
    ],
    axis=1,
    inplace=True,
)

sj = (
    cps2[cps2.age_head.between(20, 80)]
    .groupby(["income_bin", "age_head"])
    .apply(
        lambda x: pd.Series(
            {
                "nu18": mdf.weighted_mean(x, "nu18", "s006"),
                "n1864": mdf.weighted_mean(x, "n1864", "s006"),
                "n65": mdf.weighted_mean(x, "n65", "s006"),
            }
        )
    )
    .reset_index()
)


def smooth(x, y, frac=0.4):
    """ Produces LOESS smoothed data.
    """
    return pd.Series(sm.nonparametric.lowess(y, x, frac=frac)[:, 1], index=x)


def smooth_all(data):
    """ Return smoothed versions of nu18, n1864, n65.
    """
    return data.groupby(["income_bin", "age_group"]).apply(
        lambda x: smooth(x.age_head, x.n)
    )


cps_long = sj.melt(
    ["age_head", "income_bin"], var_name="age_group", value_name="n"
)
smoothed_wide = smooth_all(cps_long).reset_index()
smoothed_long = smoothed_wide.melt(
    ["age_group", "income_bin"], var_name="age_head", value_name="n"
)

# Stack with raw.
cps_long["smoothed"] = False
smoothed_long["smoothed"] = True
combined_long = pd.concat([cps_long, smoothed_long])

# Add the household head. NB: age_head starts at 20 so no need to do for nu18.
combined_long["add_head"] = (
    # n1864 and head age between 18 and 64.
    (
        (combined_long.age_group == "n1864")
        & combined_long.age_head.between(18, 64)
    )
    |
    # n65 and head age exceeds 64.
    ((combined_long.age_group == "n65") & (combined_long.age_head > 64))
)
combined_long.n += combined_long.add_head


def plot(data):
    """ Produces and exports a plot of household size by age_head, with lines for each income bin.
        The title and filename reflect the age group and whether the data is smoothed based on the first record.
    """
    age_group = data.age_group.iloc[0]
    smoothed = data.smoothed.iloc[0]
    title = "Average number of people aged "
    fname = "../images/hh_composition/cps_" + age_group
    # Label according to age group.
    if age_group == "nu18":
        title += "0 to 17"
    elif age_group == "n1864":
        title += "18 to 64"
    else:
        title += "65 or older"
    # Label and export depending on smoothed/raw.
    if smoothed:
        title += " (smoothed)"
        fname += "_smoothed"
    else:
        title += " (raw)"
        fname += "_raw"
    # Plot one line per income bin.
    data.pivot_table("n", "age_head", "income_bin").plot()
    plt.title(title)
    plt.savefig(fname + ".png")


# Create and export all plots.
combined_long.groupby(["age_group", "smoothed"]).apply(plot)

# Export csv.
combined_long.to_csv("../data/hh_composition/cps.csv")
