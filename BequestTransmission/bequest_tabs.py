import pandas as pd
import numpy as np
import pickle
import os

# Create directory if output directory does not already exist
cur_path = os.path.split(os.path.abspath(__file__))[0]
output_fldr = 'csv_output_files'
output_dir = os.path.join(cur_path, output_fldr)
if not os.access(output_dir, os.F_OK):
    os.makedirs(output_dir)

# Read in dataframe of PSID data
df = pickle.load(open(os.path.join(
    cur_path, '..', 'EarningsProcesses', 'psid_data_files',
    'psid_lifetime_income.pkl', 'rb')))


# Do some tabs with data file...
# 'net_wealth', 'inheritance', 'value_inheritance_1st',
# 'value_inheritance_2nd', 'value_inheritance_3rd'

# Total inheritances by year
# line plot

# Fraction of inheritances in a year by age
# line plot

# Fraction of inheritances in a year by lifetime income group
# bar plot

# Matrix Fraction of inheritances in a year by age*lifetime_inc
# matrix
# lifecycle plots with line for each ability type
