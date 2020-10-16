'''
This script uses data from FRED to find values for parameters for the
OG-USA model that rely on macro data for calibration.
'''

# imports
import pandas_datareader.data as web
import pandas as pd
import numpy as np
import datetime
from linearmodels import PanelOLS
import statsmodels.api as sm

# set beginning and end dates for data
# format is year (1940),month (1),day (1)
start = datetime.datetime(1947, 1, 1)
end = datetime.date.today()  # go through today
baseline_date = datetime.datetime(2018, 1, 1)

variable_dict = {
    'GDP Per Capita': 'A939RX0Q048SBEA',
    'Labor share': 'LABSHPUSA156NRUG',
    'Debt held by foreigners': 'FDHBFIN',
    'Debt held by public': 'FYGFDPUN',
    'BAA Corp Bond Rates': 'DBAA', '10 year treasury rate': 'DGS10',
    'Total gov transfer payments': 'B087RC1Q027SBEA',
    'Social Security payments': 'W823RC1',
    'Gov expenditures': 'W068RCQ027SBEA',
    'Gov interest payments': 'A091RC1Q027SBEA', 'Real GDP': 'GDPC1',
    'Nominal GDP': 'GDP'
    }

# pull series of interest using pandas_datareader
fred_data = web.DataReader(variable_dict.values(), 'fred', start, end)
fred_data.rename(columns=dict((y,x) for x,y in variable_dict.items()),
                 inplace=True)
print(fred_data.head(n=10))

# Separate quartely, monthly, and annual dataseries
# fred_data_q = fred_data.resample('Q').mean()
# fred_data_m = fred_data.resample('Q').mean()
# fred_data = fred_data[:-1] # drop last quarter since not all data available

# # Create pct change from a year ago
# fred_data['pct_gdp'] = fred_data['GDPC1'].pct_change(periods=4, freq='Q')
# fred_data['pct_emp'] = fred_data['PAYEMS'].pct_change(periods=4, freq='Q')
# fred_data['pct_cpi'] = fred_data['CPIAUCSL'].pct_change(periods=4, freq='Q')
# fred_data['pct_m1'] = fred_data['M1SL'].pct_change(periods=4, freq='Q')

# # De-trend GDP series
# fred_data['dt_gdp'] = (fred_data['GDPC1'].pct_change(periods=4, freq='Q') -
#                        np.mean(fred_data['GDPC1'].pct_change(periods=4,
#                                                              freq='Q')))

# initialize a dictionary of parameters
macro_parameters = {}

# print(fred_data.loc(str(baseline_date)))
# find initial_debt_ratio
macro_parameters['initial_debt_ratio'] = pd.Series(
    fred_data['Debt held by public'] / fred_data['Nominal GDP'])#.loc(baseline_date)
print(macro_parameters['initial_debt_ratio'])

# find alpha_T
macro_parameters['alpha_T'] = (
    (fred_data['Total gov transfer payments'] -
     fred_data['Social Security payments']) / fred_data['Nominal GDP'])

# find alpha_G
macro_parameters['alpha_G'] = (
    (fred_data['Gov expenditures'] -
     fred_data['Total gov transfer payments'] -
     fred_data['Gov interest payments']) / fred_data['Nominal GDP'])#.iloc(baseline_date)

# find gamma
macro_parameters['gamma'] = 1 - fred_data['Labor share'].mean()

# find g_y
macro_parameters['g_y'] = fred_data['GDP Per Capita'].pct_change(periods=4, freq='Q').mean()

# # estimate r_gov_shift and r_gov_scale
rate_data = fred_data[['10 year treasury rate', 'BAA Corp Bond Rates']].dropna()
rate_data['constant'] = np.ones(len(rate_data.index))
# mod = PanelOLS(fred_data['10 year treasury rate'],
#                fred_data[['constant', 'BAA Corp Bond Rates']])
mod = sm.OLS(rate_data['10 year treasury rate'],
               rate_data[['constant', 'BAA Corp Bond Rates']])
res = mod.fit()
print('Summary of interest rate regression:')
print(res.summary)
macro_parameters['r_gov_shift'] = res.params['BAA Corp Bond Rates']
macro_parameters['r_gov_scale'] = res.params['constant']

# # create dataframe of parameters and print to screen
df = pd.DataFrame.from_dict(macro_parameters)
print(df)
