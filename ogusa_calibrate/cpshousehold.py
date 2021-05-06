import pandas as pd
import numpy as np
import microdf as mdf
import matplotlib.pyplot as plt
import statsmodels.api as sm
lowess = sm.nonparametric.lowess

cps = pd.read_csv('https://github.com/PSLmodels/taxdata/raw/master/data/cps.csv.gz',
                  usecols=['age_head', 'age_spouse', 'elderly_dependents',
                           'nu18', 'n1820', 'n21', 's006',
                           'e00200'  # Wages, salaries, and tips. 
                           # TODO: Switch to AGI or expanded_income,
                           # from taxcalc output.
                           ])

def add65(age_spouse):
  count = 0
  if(age_spouse>=65):
    count += 1
  return count
age65 = cps.apply(lambda x: add65(x['age_spouse']), axis=1)
cps['n65'] = cps['elderly_dependents'] + age65

def add1864(n1820, n21, n65):
  if(n65>=n21):
    return 0
  else:
    return n1820+n21-n65-1
cps['n1864'] = cps.apply(lambda x: add1864(x['n1820'],x['n21'],x['n65']),axis=1)

isAge = cps['age_head']==21
temp = cps[isAge].copy(deep=True)
mdf.add_weighted_quantiles(temp, 'e00200', 's006')
cols = temp.columns

cps2 = pd.DataFrame(columns=cols)
for age in cps.age_head.unique():
  isAge = cps['age_head']==age
  temp = cps[isAge].copy(deep=True)
  mdf.add_weighted_quantiles(temp, 'e00200', 's006')
  cps2 = pd.concat([cps2,temp], ignore_index=True)
cps2['income_bin'] = pd.cut(cps2['e00200_percentile_exact'],
                            [0, 25, 50, 70, 90, 99, 100], right=False)

cps2 = cps2.drop(['e00200_percentile_exact',
           'e00200_percentile', 'e00200_2percentile', 'e00200_ventile',
           'e00200_decile', 'e00200_quintile', 'e00200_quartile'],
          axis=1)

cps2 = cps2.groupby(['income_bin', 'age_head']).apply(
    lambda x: pd.Series({
        'nu18': mdf.weighted_mean(x, 'nu18', 's006'),
        'n1864': mdf.weighted_mean(x, 'n1864', 's006'),
        'n65': mdf.weighted_mean(x, 'n65', 's006'),
        }))
cps2.reset_index(inplace = True)
cps2[cps2['age_head'].between(20,80)]

smoothed18 = []
for group in cps2['income_bin'].unique():
    temp = cps2[cps2['income_bin']==group]
    x = temp["age_head"]
    y = temp["nu18"]
    # plt.plot(x, y, label=group)
    z18 = lowess(y, x, frac=0.4)
    plt.plot(z18[:,0], z18[:,1], label = group)
    smoothed18.extend(z18[:,1])
cps2 = cps2.assign(nu18smooth = smoothed18)
plt.legend()
plt.title("Smoothed CPS <18")
plt.savefig('data/images/smoothcps18.png')
plt.close()

smoothed1864 = []
for group in cps2['income_bin'].unique():
    temp = cps2[cps2['income_bin']==group]
    x = temp["age_head"]
    y = temp["n1864"]
    z1864 = lowess(y, x, frac=0.4)
    #plt.plot(x, y, label=group)
    z1864[:,1] =(z1864[:,0]<65).astype('uint8')+z1864[:,1]
    plt.plot(z1864[:,0], z1864[:,1], label = group)
    smoothed1864.extend(z1864[:,1])
cps2 = cps2.assign(nu1864smooth = smoothed1864)
plt.legend()
plt.title("Smoothed CPS 18 to 64")
plt.savefig('data/images/smoothcps1864.png')
plt.close()

smoothed65 = []
for group in cps2['income_bin'].unique():
    temp = cps2[cps2['income_bin']==group]
    x = temp["age_head"]
    y = temp["n65"]
    z65 = lowess(y, x, frac=0.4)
    z65[:,1] =(z65[:,0]>65).astype('uint8')+z65[:,1]
    plt.plot(z65[:,0], z65[:,1], label = group)
    smoothed65.extend(z65[:,1])
cps2 = cps2.assign(nu65smooth = smoothed65)
plt.legend()
plt.title("Smoothed CPS >64")
plt.savefig('data/images/smoothcps65.png')
plt.close()

cps2['n1864'] = cps2['n1864'] + cps2['age_head'].apply(lambda x : 1 if x < 65 else 0)
cps2['n65'] = cps2['n65'] + cps2['age_head'].apply(lambda x : 1 if x >= 65 else 0)

for group in cps2['income_bin'].unique():
    temp = cps2[cps2['income_bin']==group]
    x = temp["age_head"]
    y = temp["nu18"]
    plt.plot(x, y, label=group)
plt.legend()
plt.title("Raw CPS <18")
plt.savefig('data/images/rawcps18.png')
plt.close()

for group in cps2['income_bin'].unique():
    temp = cps2[cps2['income_bin']==group]
    x = temp["age_head"]
    y = temp["n1864"]
    plt.plot(x, y, label=group)
plt.legend()
plt.title("Raw CPS 18 to 64")
plt.savefig('data/images/rawcps1864.png')
plt.close()

for group in cps2['income_bin'].unique():
    temp = cps2[cps2['income_bin']==group]
    x = temp["age_head"]
    y = temp["n65"]
    plt.plot(x, y, label=group)
plt.legend()
plt.title("Raw CPS >64")
plt.savefig('data/images/rawcps65.png')
plt.close()

cps2.to_csv('data/CPS/cps.csv')