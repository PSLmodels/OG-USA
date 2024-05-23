import pandas as pd
import numpy as np
import pickle
import os
from constants import CODE_PATH


# Create directory if output directory does not already exist
cur_path = os.path.split(os.path.abspath(__file__))[0]
output_fldr = "csv_output_files"
output_dir = os.path.join(cur_path, "..", "data", "PSID", output_fldr)
if not os.access(output_dir, os.F_OK):
    os.makedirs(output_dir)

# Read in dictionary of regression results
first_stage_results = pickle.load(
    open(os.path.join(cur_path, "first_stage_reg_results.pkl"), "rb")
)

# Read in PSID data
df = pd.read_csv("psid_lifetime_income.csv.gz")

# save psid to stata data file
stata_df = df.copy()
stata_df["li_group"] = stata_df.li_group.astype(str)
stata_df["li_decile"] = stata_df.li_decile.astype(str)
stata_df.to_csv("psid_lifetime_income.csv", index=False)

# Put regression results in a table to be read into excel for formatting
first_stage_reg = pd.DataFrame.from_dict(first_stage_results)
first_stage_reg.to_csv(
    os.path.join(output_dir, "FirstStageRegressionTable.csv"), header=True
)

# Compute summary stats and save to tables


# /* Summary Statistics for full sample */
df["wage_21"] = np.nan
df["wage_50"] = np.nan
df["wage_80"] = np.nan
df["earn_21"] = np.nan
df["earn_50"] = np.nan
df["earn_80"] = np.nan
df.loc[df.age == 21, "wage_21"] = df["wage_rate"]
df.loc[df.age == 50, "wage_50"] = df["wage_rate"]
df.loc[df.age == 80, "wage_80"] = df["wage_rate"]
df.loc[df.age == 21, "earn_21"] = df["earn_rate"]
df.loc[df.age == 50, "earn_50"] = df["earn_rate"]
df.loc[df.age == 80, "earn_80"] = df["earn_rate"]

vars_to_collapse = [
    "wage_21",
    "wage_50",
    "wage_80",
    "earn_21",
    "earn_50",
    "earn_80",
    "wage_rate",
    "earn_rate",
    "businc_hh",
    "lifetime_inc",
    "earninc_hh",
    "earnhours_hh",
    "incwage_hh",
    "married",
    "avg_earn_inc",
    "avg_wage_inc",
    "head_age",
    "singlefemale",
    "singlemale",
    "marriedmalehead",
    "marriedfemalehead",
]
df.singlefemale = df.singlefemale.astype(int)
df.singlemale = df.singlemale.astype(int)
df.marriedfemalehead = df.marriedfemalehead.astype(int)
df.marriedmalehead = df.marriedmalehead.astype(int)

mean_df = df[["earninc_hh", "incwage_hh"]].groupby("hh_id").mean()
mean_df.rename(
    columns={"earninc_hh": "avg_earn_inc", "incwage_hh": "avg_wage_inc"},
    inplace=True,
)
df = df.merge(mean_df, how="left", on="hh_id", copy=True)
fullsample_means = df[vars_to_collapse].mean()
fullsample_sds = df[vars_to_collapse].std()
fullsample_counts = df[vars_to_collapse].count()
fullsample_means.to_csv(
    os.path.join(output_dir, "full_sample_means.csv"), header=False
)
fullsample_sds.to_csv(
    os.path.join(output_dir, "full_sample_sds.csv"), header=False
)
fullsample_counts.to_csv(
    os.path.join(output_dir, "full_sample_counts.csv"), header=False
)

# Summary Stats for full sample - but with one obs per filer
one_per = df.groupby("hh_id").mean()
fullsample1_means = one_per[vars_to_collapse].mean()
fullsample1_sds = one_per[vars_to_collapse].std()
fullsample1_counts = one_per[vars_to_collapse].count()
fullsample1_means.to_csv(
    os.path.join(output_dir, "full_sample_means_one_per_filer.csv"),
    header=False,
)
fullsample1_sds.to_csv(
    os.path.join(output_dir, "full_sample_sds_one_per_filer.csv"), header=False
)
fullsample1_counts.to_csv(
    os.path.join(output_dir, "full_sample_counts_one_per_filer.csv"),
    header=False,
)

# Summary Stats for full sample by lifetime income decile
decile_means = df[vars_to_collapse + ["li_decile"]].groupby("li_decile").mean()
decile_sds = df[vars_to_collapse + ["li_decile"]].groupby("li_decile").std()
decile_counts = (
    df[vars_to_collapse + ["li_decile"]].groupby("li_decile").count()
)
decile_means.to_csv(
    os.path.join(output_dir, "means_by_li_decile.csv"), header=False
)
decile_sds.to_csv(
    os.path.join(output_dir, "sds_by_li_decile.csv"), header=False
)
decile_counts.to_csv(
    os.path.join(output_dir, "counts_by_li_decile.csv"), header=False
)

# Table of descriptive statistics by lifetime income percentile group
group_means = df[vars_to_collapse + ["li_group"]].groupby("li_group").mean()
group_sds = df[vars_to_collapse + ["li_group"]].groupby("li_group").std()
group_counts = df[vars_to_collapse + ["li_group"]].groupby("li_group").count()
group_means.to_csv(
    os.path.join(output_dir, "means_by_li_group.csv"), header=False
)
group_sds.to_csv(os.path.join(output_dir, "sds_by_li_group.csv"), header=False)
group_counts.to_csv(
    os.path.join(output_dir, "counts_by_li_group.csv"), header=False
)

# Put together formatted descriptive stats tables
cats_pct = ["0-25", "26-50", "51-70", "71-80", "81-90", "91-99", "100"]
summ_stats_by_group = {
    "Lifetime Income Percentile:": cats_pct + ["All"],
    "Observations": [],
    "Fraction Single": [""] * (len(cats_pct) + 1),
    "   Females": [],
    "   Males": [],
    "Fraction Married": [""] * (len(cats_pct) + 1),
    "   Female Head": [],
    "   Male Head": [],
    "Means:": [""] * (len(cats_pct) + 1),
    "   Age, Head": [],
    "   Hourly Wage": [],
    "   Annual Wages": [],
    "   Lifetime Income": [],
    "   Hours": [],
    "   Hourly Wage, Age 21": [],
    "   Hourly Wage, Age 50": [],
    "   Hourly Wage, Age 80": [],
}
# Append values for each group
for i, v in enumerate(cats_pct):
    summ_stats_by_group["Observations"].append(
        group_counts.loc[i, "lifetime_inc"]
    )
    summ_stats_by_group["   Females"].append(
        group_means.loc[i, "singlefemale"]
    )
    summ_stats_by_group["   Males"].append(group_means.loc[i, "singlemale"])
    summ_stats_by_group["   Female Head"].append(
        group_means.loc[i, "marriedfemalehead"]
    )
    summ_stats_by_group["   Male Head"].append(
        group_means.loc[i, "marriedmalehead"]
    )
    summ_stats_by_group["   Age, Head"].append(group_means.loc[i, "head_age"])
    summ_stats_by_group["   Hourly Wage"].append(
        group_means.loc[i, "earn_rate"]
    )
    summ_stats_by_group["   Annual Wages"].append(
        group_means.loc[i, "earninc_hh"]
    )
    summ_stats_by_group["   Lifetime Income"].append(
        group_means.loc[i, "lifetime_inc"]
    )
    summ_stats_by_group["   Hours"].append(group_means.loc[i, "earnhours_hh"])
    summ_stats_by_group["   Hourly Wage, Age 21"].append(
        group_means.loc[i, "earn_21"]
    )
    summ_stats_by_group["   Hourly Wage, Age 50"].append(
        group_means.loc[i, "earn_50"]
    )
    summ_stats_by_group["   Hourly Wage, Age 80"].append(
        group_means.loc[i, "earn_80"]
    )
# Append averages
summ_stats_by_group["Observations"].append(fullsample_counts["lifetime_inc"])
summ_stats_by_group["   Females"].append(fullsample_means["singlefemale"])
summ_stats_by_group["   Males"].append(fullsample_means["singlemale"])
summ_stats_by_group["   Female Head"].append(
    fullsample_means["marriedfemalehead"]
)
summ_stats_by_group["   Male Head"].append(fullsample_means["marriedmalehead"])
summ_stats_by_group["   Age, Head"].append(fullsample_means["head_age"])
summ_stats_by_group["   Hourly Wage"].append(fullsample_means["earn_rate"])
summ_stats_by_group["   Annual Wages"].append(fullsample_means["earninc_hh"])
summ_stats_by_group["   Lifetime Income"].append(
    fullsample_means["lifetime_inc"]
)
summ_stats_by_group["   Hours"].append(fullsample_means["earnhours_hh"])
summ_stats_by_group["   Hourly Wage, Age 21"].append(
    fullsample_means["earn_21"]
)
summ_stats_by_group["   Hourly Wage, Age 50"].append(
    fullsample_means["earn_50"]
)
summ_stats_by_group["   Hourly Wage, Age 80"].append(
    fullsample_means["earn_80"]
)
summ_group = pd.DataFrame.from_dict(summ_stats_by_group)
print(summ_group)
summ_group.to_csv(os.path.join(output_dir, "summ_stats_by_group.csv"))

decile_list = list(range(1, 11))
summ_stats_by_decile = {
    "Lifetime Income Decile:": decile_list + ["All"],
    "Observations": [],
    "Fraction Single": [""] * (len(decile_list) + 1),
    "   Females": [],
    "   Males": [],
    "Fraction Married": [""] * (len(decile_list) + 1),
    "   Female Head": [],
    "   Male Head": [],
    "Means:": [""] * (len(decile_list) + 1),
    "   Age, Head": [],
    "   Hourly Wage": [],
    "   Annual Wages": [],
    "   Lifetime Income": [],
    "   Hours": [],
    "   Hourly Wage, Age 21": [],
    "   Hourly Wage, Age 50": [],
    "   Hourly Wage, Age 80": [],
}
# Append values for each decile
for i, v in enumerate(decile_list):
    summ_stats_by_decile["Observations"].append(
        decile_counts.loc[i, "lifetime_inc"]
    )
    summ_stats_by_decile["   Females"].append(
        decile_means.loc[i, "singlefemale"]
    )
    summ_stats_by_decile["   Males"].append(decile_means.loc[i, "singlemale"])
    summ_stats_by_decile["   Female Head"].append(
        decile_means.loc[i, "marriedfemalehead"]
    )
    summ_stats_by_decile["   Male Head"].append(
        decile_means.loc[i, "marriedmalehead"]
    )
    summ_stats_by_decile["   Age, Head"].append(
        decile_means.loc[i, "head_age"]
    )
    summ_stats_by_decile["   Hourly Wage"].append(
        decile_means.loc[i, "earn_rate"]
    )
    summ_stats_by_decile["   Annual Wages"].append(
        decile_means.loc[i, "earninc_hh"]
    )
    summ_stats_by_decile["   Lifetime Income"].append(
        decile_means.loc[i, "lifetime_inc"]
    )
    summ_stats_by_decile["   Hours"].append(
        decile_means.loc[i, "earnhours_hh"]
    )
    summ_stats_by_decile["   Hourly Wage, Age 21"].append(
        decile_means.loc[i, "earn_21"]
    )
    summ_stats_by_decile["   Hourly Wage, Age 50"].append(
        decile_means.loc[i, "earn_50"]
    )
    summ_stats_by_decile["   Hourly Wage, Age 80"].append(
        decile_means.loc[i, "earn_80"]
    )
# Append averages
summ_stats_by_decile["Observations"].append(fullsample_counts["lifetime_inc"])
summ_stats_by_decile["   Females"].append(fullsample_means["singlefemale"])
summ_stats_by_decile["   Males"].append(fullsample_means["singlemale"])
summ_stats_by_decile["   Female Head"].append(
    fullsample_means["marriedfemalehead"]
)
summ_stats_by_decile["   Male Head"].append(
    fullsample_means["marriedmalehead"]
)
summ_stats_by_decile["   Age, Head"].append(fullsample_means["head_age"])
summ_stats_by_decile["   Hourly Wage"].append(fullsample_means["earn_rate"])
summ_stats_by_decile["   Annual Wages"].append(fullsample_means["earninc_hh"])
summ_stats_by_decile["   Lifetime Income"].append(
    fullsample_means["lifetime_inc"]
)
summ_stats_by_decile["   Hours"].append(fullsample_means["earnhours_hh"])
summ_stats_by_decile["   Hourly Wage, Age 21"].append(
    fullsample_means["earn_21"]
)
summ_stats_by_decile["   Hourly Wage, Age 50"].append(
    fullsample_means["earn_50"]
)
summ_stats_by_decile["   Hourly Wage, Age 80"].append(
    fullsample_means["earn_80"]
)

summ_decile = pd.DataFrame.from_dict(summ_stats_by_decile)
print(summ_decile)
summ_decile.to_csv(os.path.join(output_dir, "summ_stats_by_decile.csv"))
