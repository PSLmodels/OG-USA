import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from linearmodels import PanelOLS
import ogcore  # import just for MPL style file

# Create directory if output directory does not already exist
cur_path = os.path.split(os.path.abspath(__file__))[0]
output_fldr = "csv_output_files"
output_dir = os.path.join(cur_path, "..", "data", output_fldr)
if not os.access(output_dir, os.F_OK):
    os.makedirs(output_dir)


def estimate_profiles(graphs=False):
    """
    Function to estimate deterministic lifecycle profiles of hourly
    earnings.  Follows methodology of Fullerton and Rogers (1993).

    Args:
        graphs (bool): whether to create graphs of profiles

    Returns:
        reg_results (Pandas DataFrame): regression model coefficients
            for lifetime earnings profiles

    """
    # Read in dataframe of PSID data
    df = ogcore.utils.safe_read_pickle(
        os.path.join(
            cur_path, "..", "data", "PSID", "psid_lifetime_income.pkl"
        )
    )

    model_results = {
        "Names": [
            "Constant",
            "",
            "Head Age",
            "",
            "Head Age^2",
            "",
            "Head Age^3",
            "",
            "R-Squared",
            "Observations",
        ]
    }
    cats_pct = ["0-25", "26-50", "51-70", "71-80", "81-90", "91-99", "100"]
    long_model_results = {
        "Lifetime Income Group": [],
        "Constant": [],
        "Age": [],
        "Age^2": [],
        "Age^3": [],
        "Observations": [],
    }
    for i, group in enumerate(cats_pct):
        data = df[df[group] == 1].copy()
        data["ones"] = np.ones(len(data.index))
        mod = PanelOLS(
            data.ln_earn_rate, data[["ones", "age", "age2", "age3"]]
        )
        res = mod.fit(cov_type="clustered", cluster_entity=True)
        # print('Summary for lifetime income group ', group)
        # print(res.summary)
        # Save model results to dictionary
        model_results[group] = [
            res.params["ones"],
            res.std_errors["ones"],
            res.params["age"],
            res.std_errors["age"],
            res.params["age2"],
            res.std_errors["age2"],
            res.params["age3"],
            res.std_errors["age3"],
            res.rsquared,
            res.nobs,
        ]
        long_model_results["Lifetime Income Group"].extend([cats_pct[i], ""])
        long_model_results["Constant"].extend(
            [res.params["ones"], res.std_errors["ones"]]
        )
        long_model_results["Age"].extend(
            [res.params["age"], res.std_errors["age"]]
        )
        long_model_results["Age^2"].extend(
            [res.params["age2"], res.std_errors["age2"]]
        )
        long_model_results["Age^3"].extend(
            [res.params["age3"], res.std_errors["age3"]]
        )
        long_model_results["Observations"].extend([res.nobs, ""])

    reg_results = pd.DataFrame.from_dict(model_results)
    reg_results.to_csv(
        os.path.join(output_dir, "DeterministicProfileRegResults.csv")
    )
    long_reg_results = pd.DataFrame.from_dict(model_results)
    long_reg_results.to_csv(
        os.path.join(output_dir, "DeterministicProfileRegResults_long.csv")
    )

    if graphs:
        # Plot lifecycles of hourly earnings from processes estimated above
        age_vec = np.arange(20, 81, step=1)
        for i, group in enumerate(cats_pct):
            earn_profile = (
                model_results[group][0]
                + model_results[group][2] * age_vec
                + model_results[group][4] * age_vec**2
                + model_results[group][6] * age_vec**3
            )
            plt.plot(age_vec, earn_profile, label=group)
        plt.title(
            "Estimated Lifecycle Earnings Profiles by Lifetime Income Group"
        )
        plt.legend()

        plt.savefig(
            os.path.join(output_dir, "lifecycle_earnings_profiles.png")
        )

        # Plot of lifecycles of hourly earnings from processes from data
        pd.pivot_table(
            df,
            values="ln_earn_rate",
            index="age",
            columns="li_group",
            aggfunc="mean",
        ).plot(legend=True)
        plt.title(
            "Empirical Lifecycle Earnings Profiles by Lifetime Income Group"
        )

        plt.savefig(
            os.path.join(output_dir, "lifecycle_earnings_profiles_data.png")
        )

        # Plot of lifecycle profiles of hours by lifetime income group
        # create variable from fraction of time endowment work
        df["labor_supply"] = df["earnhours_hh"] / (
            24 * 5 * (df["married"] + 1) * 50
        )
        pd.pivot_table(
            df,
            values="labor_supply",
            index="age",
            columns="li_group",
            aggfunc="mean",
        ).plot(legend=True)
        plt.title("Lifecycle Profiles of Hours by Lifetime Income Group")

        plt.savefig(os.path.join(output_dir, "lifecycle_laborsupply.png"))

    return reg_results
