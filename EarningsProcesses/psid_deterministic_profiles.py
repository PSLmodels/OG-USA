import numpy as np
import pandas as pd
import pickle
import os
from linearmodels import PanelOLS

# Create directory if output directory does not already exist
cur_path = os.path.split(os.path.abspath(__file__))[0]
output_fldr = 'csv_output_files'
output_dir = os.path.join(cur_path, output_fldr)
if not os.access(output_dir, os.F_OK):
    os.makedirs(output_dir)

# estimate lifetime wage profile for each percentile group
# Read in dataframe of PSID data
df = pickle.load(open('psid_lifetime_income.pkl', 'rb'))

list_of_dfs_with_fitted_vals = []
model_results = {
    'Names': ['Head Age', '', 'Head Age^2', '', 'Head Age^3', '',
              'R-Squared', 'Observations']}
cats_pct = ['0-25', '26-50', '51-70', '71-80', '81-90', '91-99', '100']
long_model_results = {'Lifetime Income Group': [],
                      'Age': [], 'Age^2': [], 'Age^3': [],
                      'Observations': []}
for i, group in enumerate(cats_pct):
    data = df[df[group] == 1]
    mod = PanelOLS(data.ln_wage_rate,
                   data[['age', 'age2', 'age3']])
    res = mod.fit(cov_type='clustered', cluster_entity=True)
    print('Summary for lifetime income group ', group)
    print(res.summary)
    # Save model results to dictionary
    model_results[group] = [
        res.params['age'], res.std_errors['age'],
        res.params['age2'], res.std_errors['age2'],
        res.params['age3'], res.std_errors['age3'],
        res.rsquared, res.nobs]
    long_model_results['Lifetime Income Group'].extend([cats_pct[i], ''])
    long_model_results['Age'].extend([res.params['age'],
                                      res.std_errors['age']])
    long_model_results['Age^2'].extend([res.params['age2'],
                                        res.std_errors['age2']])
    long_model_results['Age^3'].extend([res.params['age3'],
                                        res.std_errors['age3']])
    long_model_results['Observations'].extend([res.nobs, ''])

reg_results = pd.DataFrame.from_dict(model_results)
reg_results.to_csv(os.path.join(
    output_dir, 'DeterministicProfileRegResults.csv'))
long_reg_results = pd.DataFrame.from_dict(model_results)
long_reg_results.to_csv(os.path.join(
    output_dir, 'DeterministicProfileRegResults_long.csv'))
