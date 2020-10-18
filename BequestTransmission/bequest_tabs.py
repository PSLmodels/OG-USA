import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import ogusa # import just for MPL style file   

# Create directory if output directory does not already exist
cur_path = os.path.split(os.path.abspath(__file__))[0]
output_fldr = 'csv_output_files'
output_dir = os.path.join(cur_path, output_fldr)
if not os.access(output_dir, os.F_OK):
    os.makedirs(output_dir)
image_fldr = 'images'
image_dir = os.path.join(cur_path, image_fldr)
if not os.access(image_dir, os.F_OK):
    os.makedirs(image_dir)

# Define a lambda function to compute the weighted mean:
wm = lambda x: np.average(
    x, weights=df.loc[x.index, "fam_smpl_wgt_core"])

# Read in dataframe of PSID data
df = pickle.load(open(os.path.join(
    cur_path, '..', 'EarningsProcesses', 'psid_data_files',
    'psid_lifetime_income.pkl'), 'rb'))

# Do some tabs with data file...
# 'net_wealth', 'inheritance', 'value_inheritance_1st',
# 'value_inheritance_2nd', 'value_inheritance_3rd'
# inheritance available from 1988 onwards...

# Total inheritances by year
# line plot
# df.groupby('year_data').apply(wm).plot(y='net_wealth')
# plt.savefig(os.path.join(image_dir, 'net_wealth_year.png'))
df.groupby('year_data').mean().plot(y='inheritance')
plt.savefig(os.path.join(image_dir, 'inheritance_year.png'))

# Fraction of inheritances in a year by age
# line plot
df[df['year_data'] >= 1988].groupby('age').mean().plot(y='net_wealth')
plt.savefig(os.path.join(image_dir, 'net_wealth_age.png'))
df[df['year_data'] >= 1988].groupby('age').mean().plot(y='inheritance')
plt.savefig(os.path.join(image_dir, 'inheritance_age.png'))

# Inheritances by lifetime income group
# bar plot
df[df['year_data'] >= 1988].groupby('li_group').mean().plot.bar(
    y='net_wealth')
plt.savefig(os.path.join(image_dir, 'net_wealth_li.png'))
df[df['year_data'] >= 1988].groupby('li_group').mean().plot.bar(
    y='inheritance')
plt.savefig(os.path.join(image_dir, 'inheritance_li.png'))

# lifecycle plots with line for each ability type
pd.pivot_table(df[df['year_data'] >= 1988], values='net_wealth', index='age',
               columns='li_group', aggfunc='mean').plot(legend=True)
plt.savefig(os.path.join(image_dir, 'net_wealth_age_li.png'))
pd.pivot_table(df[df['year_data'] >= 1988], values='inheritance', index='age',
               columns='li_group', aggfunc='mean').plot(legend=True)
plt.savefig(os.path.join(image_dir, 'inheritance_age_li.png'))

# Matrix Fraction of inheritances in a year by age and lifetime_inc
inheritance_matrix = pd.pivot_table(
    df[df['year_data'] >= 1988], values='inheritance', index='age',
    columns='li_group', aggfunc='sum')
inheritance_matrix = inheritance_matrix / inheritance_matrix.sum().sum()
inheritance_matrix.to_csv(os.path.join(
    output_dir, 'bequest_matrix.csv'))

# Will need to do some smoothing with a KDE when estimate the matrix...
