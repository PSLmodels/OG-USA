# %%
# Import packages
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, colors
from scipy.ndimage import gaussian_filter
from scipy.ndimage import gaussian_filter1d
from scipy.stats import gaussian_kde
import ogusa.wealth as wlth
from ogcore.parameters import Specifications
from ogusa import utils as ogusa_utils
from ogcore import utils as ogcore_utils

# Set paths
main_dir = "/Users/richardevans/Docs/Economics/OSE/OG/OG-USA"
script_dir = os.path.join(main_dir, "ogusa")
data_dir = os.path.join(script_dir, "data", "SCF")
base_dir = os.path.join(main_dir, "examples", "WealthSCF")

# %%
# Create the wealth data

# Instantiate parameters
p = Specifications(
    baseline=True,
    baseline_dir=base_dir,
    output_base=base_dir,
)

df_scf_2019 = wlth.load_scf(2019, web=True)
df_scf_2019_upd = wlth.compute_wealth_var(df_scf_2019)
df_scf_2019_upd_j = wlth.assign_lifetime_income_group(
    df_scf_2019_upd, age_bin_width=1, lambdas=p.lambdas
)
# df_scf_2019_upd_j.describe()

df_scf_2022 = wlth.load_scf(2022, web=True)
df_scf_2022_upd = wlth.compute_wealth_var(df_scf_2022)
df_scf_2022_upd_j = wlth.assign_lifetime_income_group(
    df_scf_2022_upd, age_bin_width=1, lambdas=p.lambdas
)
# df_scf_2022_upd_j.describe()

df_scf_2019_2022 = wlth.merge_infladj_mult_scf_years(
    [df_scf_2019_upd_j, df_scf_2022_upd_j], [2019, 2022]
)
# df_scf_2019_2022.describe()

df_scf_2022_2022 = wlth.merge_infladj_mult_scf_years(
    [df_scf_2022_upd_j], [2022]
)
# df_scf_2022_2022.describe()

# Generate wealth momoment matrices and vectors for age_bins = 1, 5, 10
age_bin_width_lst = [1, 4, 5, 10]
dist_moms_lst = []
for age_bin_width in age_bin_width_lst:
    (
        w_dist_sj, w_dist_s, w_dist_j, p_dist_sj, p_dist_s, p_dist_j, gini_w,
        var_ln_w
    ) = wlth.wealth_distributions(
        df_scf_2022_2022, age_bin_width=age_bin_width
    )
    print(f"w_dist_sj (age_bin_width={age_bin_width})")
    print(w_dist_sj)
    print(f"w_dist_s (age_bin_width={age_bin_width})")
    print(w_dist_s)
    print(f"w_dist_j (age_bin_width={age_bin_width})")
    print(w_dist_j)
    print(
        f"Gini coefficient on wealth (age_bin_width={age_bin_width}) " +
        f"= {gini_w}"
    )
    print(
        f"Variance of log wealth (age_bin_width={age_bin_width}) " +
        f"= {var_ln_w}"
    )
    dist_moms_dict = {
        "age_bin_width": age_bin_width,
        "scf_data_yrs": [2022],
        "w_dist_sj": w_dist_sj,
        "w_dist_s": w_dist_s,
        "w_dist_j": w_dist_j,
        "p_dist_sj": p_dist_sj,
        "p_dist_s": p_dist_s,
        "p_dist_j": p_dist_j,
        "gini_w": gini_w,
        "var_ln_w": var_ln_w
    }
    dist_moms_lst.append(dist_moms_dict)

    # Plot wealth moments matrix in 3D
    w_dist_sj_arr = w_dist_sj.to_numpy()
    print("w_dist_sj_arr.shape =", w_dist_sj_arr.shape)

    S, J = w_dist_sj_arr.shape
    if age_bin_width == 1:
        age_bins = np.arange(21, 21 + S)  # 21, ..., 100
        ages = np.arange(21, 21 + S)  # 21, ..., 100
    else:
        # Set the ages as the midpoints of the bins
        age_bins = np.arange(S)
        ages = np.linspace(
            p.E + (age_bin_width / 2),
            p.E + p.S - (age_bin_width / 2),
            int(p.S / age_bin_width)
        )
    if age_bin_width == 4:
        age_tick_lst = [
            "21-24", "25-28", "29-32", "33-36", "37-40", "41-44", "45-48",
            "49-52", "53-56", "57-60", "61-64", "65-68", "69-72", "73-76",
            "77-80", "81-84", "85-88", "89-92", "93-96", "97-100"
        ]
    if age_bin_width == 5:
        age_tick_lst = [
            "21-25", "26-30", "31-35", "36-40", "41-45", "46-50", "51-55",
            "56-60", "61-65", "66-70", "71-75", "76-80", "81-85", "86-90",
            "91-95", "96-100"
        ]
    elif age_bin_width == 10:
        age_tick_lst = [
            "21-30", "31-40", "41-50", "51-60", "61-70", "71-80", "81-90",
            "91-100"
        ]
    # One bar per (age, j) cell: x = age, y = j, height = wealth share
    A, Jg = np.meshgrid(age_bins, np.arange(J), indexing="ij")
    x = A.ravel() - 0.4
    y = Jg.ravel() - 0.4
    dz = w_dist_sj_arr.ravel()

    # Color bars by height with a single-hue sequential colormap
    norm = colors.Normalize(vmin=min(dz.min(), 0.0), vmax=dz.max())
    bar_colors = cm.Blues(0.25 + 0.75 * norm(dz))

    fig1 = plt.figure(figsize=(10, 7))
    ax1 = fig1.add_subplot(projection="3d")
    ax1.bar3d(
        x, y, np.zeros_like(dz), 0.8, 0.8, dz, color=bar_colors,
        shade=True, linewidth=0
    )

    if age_bin_width != 1:
        ax1.set_xticklabels(age_tick_lst)
    # Label j axis with the lifetime income percentile ranges
    cum = np.concatenate(([0], np.cumsum(p.lambdas))) * 100
    ax1.set_yticks(np.arange(J))
    ax1.set_yticklabels([f"{cum[k]:.0f}-{cum[k + 1]:.0f}%" for k in range(J)])
    ax1.set_xlabel(r"Age $s$")
    ax1.set_ylabel(r"Lifetime income group $j$")
    ax1.set_zlabel(r"Share of total wealth")
    ax1.set_title(
        r"Original distribution of initial household wealth $b_{s,j,1}$"
    )
    ax1.view_init(elev=25, azim=-55)
    plt.tight_layout()
    plt.savefig(
        os.path.join(data_dir, f"w_dist_sj_orig_3D_sbin{age_bin_width}.png"),
        dpi=300
    )
    plt.show()
    plt.close()

    # Plot wealth moments matrix in 2D
    fig2 = plt.figure()
    for j in range(p.J):
        label_str = f"{cum[j]:.0f}-{cum[j + 1]:.0f}%"
        plt.plot(ages, w_dist_sj_arr[:, j], label=f"j={label_str}")
    plt.legend()
    plt.xlabel(r"Age $s$")
    plt.ylabel(r"Share of total wealth")
    plt.title(
        r"Original distribution of initial household wealth $b_{s,j,1}$"
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(data_dir, f"w_dist_sj_orig_2D_sbin{age_bin_width}.png"),
        dpi=300
    )
    plt.show()
    plt.close()

# print(dist_moms_lst)

# %%
df_scf_2022_2022.describe()

# %%
# Generate and plot smoothed wealth distribution
w_dist_sj = dist_moms_lst[0]["w_dist_sj"]
w_dist_sj_smooth = wlth.smooth_wealth_dist_age(w_dist_sj, sigma_s=3.0)
ages = np.arange(p.E + 1, p.E + p.S + 1)  # 21, ..., 100
# One bar per (age, j) cell: x = age, y = j, height = wealth share
A, Jg = np.meshgrid(ages, np.arange(J), indexing="ij")
x = A.ravel() - 0.4
y = Jg.ravel() - 0.4
dz = w_dist_sj_smooth.ravel()

# Color bars by height with a single-hue sequential colormap
norm = colors.Normalize(vmin=min(dz.min(), 0.0), vmax=dz.max())
bar_colors = cm.Blues(0.25 + 0.75 * norm(dz))

fig9 = plt.figure(figsize=(10, 7))
ax9 = fig9.add_subplot(projection="3d")
ax9.bar3d(
    x, y, np.zeros_like(dz), 0.8, 0.8, dz, color=bar_colors,
    shade=True, linewidth=0
)

# Label j axis with the lifetime income percentile ranges
cum = np.concatenate(([0], np.cumsum(p.lambdas))) * 100
ax9.set_yticks(np.arange(J))
ax9.set_yticklabels([f"{cum[k]:.0f}-{cum[k + 1]:.0f}%" for k in range(J)])
ax9.set_xlabel(r"Age $s$")
ax9.set_ylabel(r"Lifetime income group $j$")
ax9.set_zlabel(r"Share of total wealth")
ax9.set_title(
    r"Smoothed distribution of initial household wealth $b_{s,j,1}$"
)
ax9.view_init(elev=25, azim=-55)
plt.tight_layout()
plt.savefig(os.path.join(data_dir, "w_dist_sj_smth_3D_sbin1.png"), dpi=300)
plt.show()
plt.close()

# Plot wealth moments matrix in 2D
fig10 = plt.figure()
for j in range(p.J):
    label_str = f"{cum[j]:.0f}-{cum[j + 1]:.0f}%"
    plt.plot(ages, w_dist_sj_smooth[:, j], label=f"j={label_str}")
plt.legend()
plt.xlabel(r"Age $s$")
plt.ylabel(r"Share of total wealth")
plt.title(
    r"Original distribution of initial household wealth $b_{s,j,1}$"
)
plt.tight_layout()
plt.savefig(os.path.join(data_dir, "w_dist_sj_smth_2D_sbin1.png"), dpi=300)
plt.show()
plt.close()

# %%
