import numpy as np
import pandas as pd
import math
import pickle
from pandas_datareader import data as web
import datetime
from linearmodels import PanelOLS
from rpy2.robjects import r
from rpy2.robjects import pandas2ri
pandas2ri.activate()
pd.options.mode.chained_assignment = 'raise'

'''
This script takes PSID data created from psid_download.R and:
1) Creates variables at the "tax filing unit" (equal to family unit in
   PSID since there is no info on the filing status chosen).
2) Selects a sample of observations to work with (e.g., dropping very
   old, very low income, etc.).
3) Computes a measure of lifetime income and places each household into
   a lifetime income percentile group
'''

# Read data from R into pandas dataframe
r['load']("./psid_data_files/psid1968to2017.RData")
raw_df = r('psid_df')

# Create unique identifier for each household
# note that will define a new household if head or spouse changes
# keep only current heads
# before 1983, head is relation.head == 1, 1983+ head is given by
# relation.head == 10

# Select just those in the SRC sampe, which is representitive of the
# population and so will not require the use of sampling weights
# SRC sample families have 1968 family interview numbers less than 3000
raw_df = raw_df[raw_df['ID1968'] < 3000].copy()

raw_df['relation.head'][(raw_df['year'] < 1983) &
                        (raw_df['relation.head'] == 1)] = 10
raw_df['relation.head'][(raw_df['year'] < 1983) &
                        (raw_df['relation.head'] == 2)] = 20
head_df = raw_df.loc[raw_df.index[(raw_df['relation.head'] == 10) &
                                  (raw_df['sequence'] == 1)], :]
head_df.rename(columns={"pid": "head_id"}, inplace=True)
# keep legal spouse or long term partners
spouse_df = raw_df.loc[raw_df.index[
    (raw_df['relation.head'] >= 20) &
    (raw_df['relation.head'] <= 22) & (raw_df['sequence'] == 2)],
                       ['pid', 'ID1968', 'year', 'interview_number']]
spouse_df.rename(columns={"pid": "spouse_id"}, inplace=True)
psid_df = head_df.merge(spouse_df, how='left', on=['ID1968', 'year',
                                                   'interview_number'])
# create unique household id for combination of head and a specific spouse
psid_df['hh_id'] = ((psid_df['head_id'] * 1000000) +
                    psid_df['spouse_id'].fillna(0))

# Fix ages to increment by one (or two) between survey waves.  They do
# not always do this because the survey may be asked as different times
# of year
min_age_df = psid_df.groupby('hh_id').agg(['min'])['head_age']
min_age_df.rename(columns={"min": "min_age"}, inplace=True)
min_year_df = psid_df.groupby('hh_id').agg(['min'])['year']
min_year_df.rename(columns={"min": "min_year"}, inplace=True)
psid_df = psid_df.merge(min_age_df, on='hh_id', how='left')
psid_df = psid_df.merge(min_year_df, on='hh_id', how='left')
psid_df.sort_values(by=['hh_id', 'year'], inplace=True)
psid_df['age'] = psid_df['year'] - psid_df['min_year'] + psid_df['min_age']


# Deflate nominal variables
# because surveys ask about prior year
psid_df['year_data'] = psid_df['year'] - 1

# create spouse labor income, since not consistent variable name
# across time
psid_df['spouse_labor_inc'] = (psid_df['spouse_labor_inc_pre1993'] +
                               psid_df['spouse_labor_inc_post1993'])
psid_df.loc[psid_df['year'] == 1993, 'spouse_labor_inc'] =\
    psid_df['spouse_labor_inc_post1993']

# set beginning and end dates for data
start = datetime.datetime(1968, 1, 1)
end = datetime.datetime(2015, 1, 1)
# pull series of interest using pandas_datareader
fred_data = web.DataReader(["CPIAUCSL"], "fred", start, end)
# Make data annual by averaging over months in year
fred_data = fred_data.resample('A').mean()
fred_data['year_data'] = fred_data.index.year
psid_df2 = psid_df.merge(fred_data, how='left', on='year_data')
psid_df = psid_df2
cpi_2010 = fred_data.loc[datetime.datetime(2010, 12, 31), 'CPIAUCSL']
nominal_vars = ['head_labor_inc', 'spouse_labor_inc',
                'head_whether_receive_afdc_prior_year',
                'spouse_whether_receive_afdc_prior_year',
                'head_ssi_prior_year', 'spouse_ssi_prior_year',
                'other_familyunit_ssi_prior_year',
                'head_other_welfare_prior_year',
                'spouse_other_welfare_prior_year',
                'other_familyunit_other_welfare_prior_year',
                'head_unemp_inc_prior_year',
                'spouse_unemp_inc_prior_year',
                'other_familyunit_unemp_inc_prior_year',
                'head_workers_comp_prior_year',
                'spouse_workers_comp_prior_year',
                'other_familyunit_workers_comp_prior_year',
                'head_vet_pen_prior_year',
                'spouse_vet_pen_prior_year',
                'other_familyunit_vet_pen_prior_year',
                'head_spouse_taxable_inc',
                'other_familyunit_taxable_inc',
                'head_spouse_tax_table', 'food_out_expend',
                'food_in_expend', 'other_familyunit_asset_inc',
                'head_dividend_inc', 'spouse_dividend_inc',
                'head_interest_inc', 'spouse_interest_inc',
                'head_rent_inc', 'spouse_rent_inc', 'family_total_inc',
                'head_and_spouse_transfer_income',
                'other_familyunit_transfer_income',
                'head_socsec_income', 'spouse_socsec_income',
                'other_familyunit_socsec_income',
                'head_noncorp_bus_asset_income',
                'spouse_noncorp_bus_asset_income',
                'head_noncorp_bus_labor_income',
                'spouse_noncorp_bus_labor_income', 'noncorp_businc',
                'net_wealth', 'inheritance', 'value_inheritance_1st',
                'value_inheritance_2nd', 'value_inheritance_3rd']
for item in nominal_vars:
    psid_df[item] = (psid_df[item] * cpi_2010) / psid_df['CPIAUCSL']
# remove intermediate dataframes
del raw_df, spouse_df, head_df, fred_data, psid_df2

# Fill in  missing values with zeros
psid_df[nominal_vars] = psid_df[nominal_vars].fillna(0)
psid_df[['head_annual_hours', 'spouse_annual_hours']] =\
    psid_df[['head_annual_hours', 'spouse_annual_hours']].fillna(0)

# Construct family ("filing unit") level variables
psid_df['incwage_hh'] = (psid_df['head_labor_inc'] +
                         psid_df['spouse_labor_inc'])
psid_df['earninc_hh'] = (psid_df['incwage_hh'] +
                         psid_df['head_noncorp_bus_labor_income'] +
                         psid_df['spouse_noncorp_bus_labor_income'])
psid_df['businc_hh'] = (psid_df['head_noncorp_bus_labor_income'] +
                        psid_df['spouse_noncorp_bus_labor_income'])
# note that PSID doesn't separate hours towards employed and business
# work
psid_df['earnhours_hh'] = (psid_df['head_annual_hours'] +
                           psid_df['spouse_annual_hours'])
psid_df['wage_rate'] = psid_df['incwage_hh'] / psid_df['earnhours_hh']
psid_df['earn_rate'] = psid_df['earninc_hh'] / psid_df['earnhours_hh']
with np.errstate(divide='ignore'):
    psid_df['ln_wage_rate'] = np.log(psid_df['wage_rate'])
    psid_df['ln_earn_rate'] = np.log(psid_df['earn_rate'])
psid_df['singlemale'] = ((psid_df['head_gender'] == 1) &
                         (psid_df['marital_status'] != 1))
psid_df['singlefemale'] = ((psid_df['head_gender'] == 2) &
                           (psid_df['marital_status'] != 1))
psid_df['marriedmalehead'] = ((psid_df['head_gender'] == 1) &
                              (psid_df['marital_status'] == 1))
psid_df['marriedfemalehead'] = ((psid_df['head_gender'] == 2) &
                                (psid_df['marital_status'] == 1))
psid_df['married'] = (psid_df['marital_status'] == 1).astype(int)

# sample selection
# drop very young, very old, those with very low earnings, and any
# outliers with very high earnings, those working at least 200 hrs
# should check to see if we want to drop any particular years... (e.g.,
# I think some data is missing before 1970)
psid_df.query('age >= 20 & age <= 80 & incwage_hh >= 5' +
              ' & wage_rate >= 5 & wage_rate <= 25000' +
              ' & earnhours_hh > 200', inplace=True)
# Indicator for obs beign from PSID not interpolated value
# used to make drops later
psid_df.sort_values(by=['hh_id', 'year'], inplace=True)
psid_df[['head_id', 'spouse_id', 'hh_id', 'head_age', 'age',
         'spouse_age', 'ID1968', 'year',
         'interview_number', 'head_marital_status',
         'marital_status']].to_csv('psid_to_check.csv')
# The next several lines try to identify and then drop from the sample
# hh_ids that report more than one type of marital status
# there are 179 of these, 26 are men who report being married and not at
# different times, even when a spouse id is not present
marriedmale_df = psid_df.groupby('hh_id').agg(['max'])['marriedmalehead']
singlemale_df = psid_df.groupby('hh_id').agg(['max'])['singlemale']
marriedfemale_df = psid_df.groupby('hh_id').agg(['max'])['marriedfemalehead']
singlefemale_df = psid_df.groupby('hh_id').agg(['max'])['singlefemale']
marriedmale_df.rename(columns={"max": "m_marriedmalehead"}, inplace=True)
singlemale_df.rename(columns={"max": "m_singlemale"}, inplace=True)
marriedfemale_df.rename(columns={"max": "m_marriedfemalehead"}, inplace=True)
singlefemale_df.rename(columns={"max": "m_singlefemale"}, inplace=True)
merged_df = marriedmale_df.join([singlemale_df, marriedfemale_df,
                                 singlefemale_df], how='outer',
                                sort=True)
merged_df['sum_status'] = (
    merged_df['m_singlemale'].astype(int) +
    merged_df['m_singlefemale'].astype(int) +
    merged_df['m_marriedfemalehead'].astype(int) +
    merged_df['m_marriedmalehead'].astype(int))
merged_df_to_list = merged_df[merged_df['sum_status'] > 1]
merged_df_to_list.to_csv('hh_id_two_statuses.csv')
hhid_to_drop = merged_df_to_list.copy()
hhid_to_drop['keep'] = False
psid_df = psid_df.merge(hhid_to_drop, on='hh_id', how='left')
psid_df['keep'].fillna(True, inplace=True)
psid_df = psid_df[psid_df['keep']].copy()
psid_df['in_psid'] = True
# print number of obs by year
print('Number of obs by year = ',
      psid_df['hh_id'].groupby([psid_df.year]).agg('count'))
num_obs_psid = psid_df.shape[0]
psid_df.sort_values(by=['hh_id', 'year'], inplace=True)
test_psid_df = psid_df.copy()

# "fill in" observations - so have observation for each household
# from age 20-80
# note that do this before running regression, but that's ok since
# wages missing here so these obs don't affect regression
uid = psid_df['hh_id'].unique()
all_ages = list(range(20, 81))  # for list of ages 20 to 80
ids_full = np.array([[x]*len(all_ages) for x in list(uid)]).flatten()
ages = all_ages * len(uid)
balanced_panel = pd.DataFrame({'hh_id': ids_full, 'age': ages})
rebalanced_data = balanced_panel.merge(psid_df, how='left',
                                       on=['hh_id', 'age'])
# Backfill and then forward fill variables that are constant over time
# within hhid
constant_vars = ['head_race', 'head_gender', 'singlemale',
                 'singlefemale', 'marriedmalehead', 'marriedfemalehead',
                 'ID1968', 'pernum']
rebalanced2 = rebalanced_data
for item in constant_vars:
    rebalanced_data[item] = rebalanced_data.groupby(
        'hh_id')[item].fillna(method='bfill')
    rebalanced_data[item] = rebalanced_data.groupby(
        'hh_id')[item].fillna(method='ffill')

### NOTE: we seem to get some cases where the marital status is not constant
# despite trying to set up the indentifcation of a household such that it
# has to be.  Why this is happening needs to be checked.


# Fill in year by doing a cumulative counter within each hh_id and then
# using the difference between age and this counter to infer what the
# year should be'
rebalanced_data.sort_values(['hh_id', 'age'], inplace=True)
rebalanced_data['counter'] = rebalanced_data.groupby('hh_id').cumcount()
rebalanced_data['diff'] = rebalanced_data['year'] - rebalanced_data['counter']
rebalanced_data['diff'].fillna(0, inplace=True)  # because NaNs if year missing
max_df = rebalanced_data.groupby('hh_id').agg(['max'])['diff']
rebalanced_data = rebalanced_data.join(max_df, how='left', on=['hh_id'])
rebalanced_data['year'] = rebalanced_data['max'] + rebalanced_data['counter']


### Check that there are 61 obs for each hh_id

# create additional variables for first stage regressions
df = rebalanced_data.reset_index()
df['age2'] = df['age'] ** 2
df['age3'] = df['age'] ** 3
df['age_smale'] = df['age'] * df['singlemale']
df['age_sfemale'] = df['age'] * df['singlefemale']
df['age_mmale'] = df['age'] * df['marriedmalehead']
df['age_mfemale'] = df['age'] * df['marriedfemalehead']
df['age_smale2'] = df['age2'] * df['singlemale']
df['age_sfemale2'] = df['age2'] * df['singlefemale']
df['age_mmale2'] = df['age2'] * df['marriedmalehead']
df['age_mfemale2'] = df['age2'] * df['marriedfemalehead']
df['age_smale3'] = df['age3'] * df['singlemale']
df['age_sfemale3'] = df['age3'] * df['singlefemale']
df['age_mmale3'] = df['age3'] * df['marriedmalehead']
df['age_mfemale3'] = df['age3'] * df['marriedfemalehead']

# run regressions to impute wages for years not observed in sample
df.set_index(['hh_id', 'year'], inplace=True)
list_of_statuses = ['Single Males', 'Single Females',
                    'Married, Male Head', 'Married, Female Head']
list_of_dfs = [df[df['singlemale']].copy(),
               df[df['singlefemale']].copy(),
               df[df['marriedmalehead']].copy(),
               df[df['marriedfemalehead']].copy()]
list_of_dfs_with_fitted_vals = []
first_stage_model_results = {'Names': ['Head Age', '', 'Head Age^2', '',
                                       'Head Age^3', '', 'R-Squared',
                                       'Observations', 'Households'],
                             'Single Males': [], 'Single Females': [],
                             'Married, Male Head': [],
                             'Married, Female Head': []}
for i, data in enumerate(list_of_dfs):
    # Note that including entity and time effects leads to a collinearity
    # I think this is because there are some years at begin and end of
    # sample with just one person
    # mod = PanelOLS(data.ln_wage_rate,
    #                data[['age', 'age2', 'age3']],
    #                weights=data.fam_smpl_wgt_core,
    #                entity_effects=True, time_effects=True)
    mod = PanelOLS(data.ln_wage_rate,
                   data[['age', 'age2', 'age3']],
                   entity_effects=True)
    res = mod.fit(cov_type='clustered', cluster_entity=True)
    print('Summary for ', list_of_statuses[i])
    print(res.summary)
    # Save model results to dictionary
    first_stage_model_results[list_of_statuses[i]] = [
        res.params['age'], res.std_errors['age'],
        res.params['age2'], res.std_errors['age2'],
        res.params['age3'], res.std_errors['age3'],
        res.rsquared, res.nobs, res.entity_info['total']]
    fit_values = res.predict(fitted=True, effects=True, missing=True)
    fit_values['predictions'] = (fit_values['fitted_values'] +
                                 fit_values['estimated_effects'])
    list_of_dfs_with_fitted_vals.append(data.join(
        fit_values, how='left', on=['hh_id', 'year']))

df_w_fit = list_of_dfs_with_fitted_vals[0].append(
    list_of_dfs_with_fitted_vals[1].append(
        list_of_dfs_with_fitted_vals[2].append(
            list_of_dfs_with_fitted_vals[3])))
df_w_fit.rename(columns={"predictions": "ln_fillin_wage"}, inplace=True)
print('Descritpion of data coming out of estimation: ', df_w_fit.describe())
# Seems to be the same as going into estimation

# Compute lifetime income for each filer
int_rate = 0.04  # assumed interest rate to compute NPV of lifetime income
time_endow = 4000  # assumed time endowment - set at 4000 hours !!! May want to change this to be different for single households than married !!!
df_w_fit['time_wage'] = np.exp(df_w_fit['ln_fillin_wage']) * time_endow
df_w_fit['lifetime_inc'] = (df_w_fit['time_wage'] *
                            ((1 / (1 + int_rate)) **
                            (df_w_fit['age'] - 20)))
li_df = (df_w_fit[['lifetime_inc']].groupby(['hh_id']).sum()).copy()
# find percentile in distrubtion of lifetime income
li_df['li_percentile'] = li_df.lifetime_inc.rank(pct=True)
# Put in bins
groups = [0.0, 0.25, 0.5, 0.7, 0.8, 0.9, 0.99, 1.0]
cats_pct = ['0-25', '26-50', '51-70', '71-80', '81-90', '91-99', '100']
li_df = li_df.join(pd.get_dummies(pd.cut(li_df['li_percentile'], groups,
                                         labels=cats_pct))).copy()
li_df['li_group'] = pd.cut(li_df['li_percentile'], groups)
deciles = list(np.arange(0.0, 1.1, 0.10))
cats_10 = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10']
li_df = li_df.join(pd.get_dummies(pd.cut(li_df['li_percentile'], deciles,
                                         labels=cats_10))).copy()
li_df['li_decile'] = pd.cut(li_df['li_percentile'], deciles)

# Merge lifetime income to panel
df_w_fit.drop(columns='lifetime_inc', inplace=True)
df_fit2 = df_w_fit.join(li_df, how='left', on=['hh_id'],
                        lsuffix='_x', rsuffix='_y')
# Drop of from balanced panel that were not in original panel
df_fit2['in_psid'].fillna(False, inplace=True)
panel_li = (df_fit2[df_fit2['in_psid']]).copy()

# Save dictionary of regression results
pickle.dump(first_stage_model_results,
            open('first_stage_reg_results.pkl', 'wb'))

# Save dataframe
pickle.dump(panel_li, open('psid_lifetime_income.pkl', 'wb'))

# Do some checks on the data
# Check that number of obs in final data equals what in psid after
# sample selection
if panel_li.shape[0] != num_obs_psid:
    print('Number of observations in final data set is not right')
    print('Obs in PSID after selection = ', num_obs_psid)
    print('Obs in final panel = ', panel_li.shape[0])
    assert False

# Check that have at least 1000 obs in each year
panel_li.sort_values(by=['hh_id', 'year'], inplace=True)
var_list = nominal_vars + constant_vars
for item in var_list:
        print('Checking ', item)
        try:
            assert np.allclose(panel_li[item], test_psid_df[item], atol=1e-5)
        except TypeError:
            print('Had to skip ', item)

# check everyone has a group and decile and that fraction in each is
# correct. Note that can't check the latter with final unbalanced panel.
print('Checking counts of percentile groupings: ')
for item in cats_10 + cats_pct:
    assert panel_li[item].count() == panel_li.shape[0]
print('Checking percentile groupings: ')
for d in cats_10:
    assert math.isclose(li_df[d].mean(), 0.1, rel_tol=0.03)
for i, g in enumerate(cats_pct):
    percent_in_g = groups[i + 1] - groups[i]
    assert math.isclose(li_df[g].mean(), percent_in_g, rel_tol=0.03)
