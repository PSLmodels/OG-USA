import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
lowess = sm.nonparametric.lowess

psid = pickle.load(open('psid_lifetime_income.pkl', 'rb')) #created by psid_data_setup.py
psid = psid[['head_age','spouse_age','li_group','num_children_under18', 'num_family']]
 
psid['n65'] = np.where(psid.head_age > 64, 1, 0) + np.where(psid.spouse_age > 64, 1, 0)
psid['nu18'] = psid['num_children_under18'] #assumes only children can be <18
psid['n1864'] = psid.num_family - psid.n65 - psid.nu18

psid.reset_index(inplace=True)
psid = psid[(psid["head_age"] >= 20) & (psid["head_age"] < 81)]
psid["fams"] = 1

psid = psid.groupby(['li_group', 'head_age'])[["nu18", "n1864", "n65", "fams"]].sum().eval('avg18 = nu18 / fams').eval('avg1864 = n1864 / fams').eval('avg65 = n65 / fams').fillna(0)
psid = psid[["avg18", "avg1864", "avg65"]]
psid.rename(columns={'avg18': 'nu18', 'avg1864': 'n1864', 'avg65': 'n65'}, inplace=True)
psid.reset_index(inplace = True)

for group in psid['li_group'].unique():
    temp = psid[psid['li_group']==group]
    x = temp["head_age"]
    y = temp["nu18"]
    plt.plot(x, y, label=group)
plt.legend()
plt.title("Raw PSID <18")
plt.savefig('data/images/rawpsid18.png')
plt.close()

for group in psid['li_group'].unique():
    temp = psid[psid['li_group']==group]
    x = temp["head_age"]
    y = temp["n1864"]
    plt.plot(x, y, label=group)
plt.legend()
plt.title("Raw PSID 18 to 64")
plt.savefig('data/images/rawpsid1864.png')
plt.close()

for group in psid['li_group'].unique():
    temp = psid[psid['li_group']==group]
    x = temp["head_age"]
    y = temp["n65"]
    plt.plot(x, y, label=group)
plt.legend()
plt.title("Raw PSID >64")
plt.savefig('data/images/rawpsid65.png')
plt.close()

psid['n1864'] = psid['n1864'] + psid['head_age'].apply(lambda x : -1 if x < 65 else 0)
psid['n65'] = psid['n65'] + psid['head_age'].apply(lambda x : -1 if x >= 65 else 0)

smoothed18 = []
for group in psid['li_group'].unique():
    temp = psid[psid['li_group']==group]
    x = temp["head_age"]
    y = temp["nu18"]
    # plt.plot(x, y, label=group)
    z18 = lowess(y, x, frac=0.4)
    plt.plot(z18[:,0], z18[:,1], label = group)
    smoothed18.extend(z18[:,1])
psid = psid.assign(nu18smooth = smoothed18)
plt.legend()
plt.title("Smoothed PSID <18")
plt.savefig('data/images/smoothpsid18.png')
plt.close()

smoothed1864 = []
for group in psid['li_group'].unique():
    temp = psid[psid['li_group']==group]
    x = temp["head_age"]
    y = temp["n1864"]
    # plt.plot(x, y, label=group)
    z1864 = lowess(y, x, frac=0.4)
    z1864[:,1] =(z1864[:,0]<65).astype('uint8')+z1864[:,1]
    plt.plot(z1864[:,0], z1864[:,1], label = group)
    smoothed1864.extend(z1864[:,1])
psid = psid.assign(n1864smooth = smoothed1864)
plt.legend()
plt.title("Smoothed PSID 18 to 64")
plt.savefig('data/images/smoothpsid1864.png')
plt.close()

smoothed65 = []
for group in psid['li_group'].unique():
    temp = psid[psid['li_group']==group]
    x = temp["head_age"]
    y = temp["n65"]
    # plt.plot(x, y, label=group)
    z65 = lowess(y, x, frac=0.4)
    z65[:,1] =(z65[:,0]>=65).astype('uint8')+z65[:,1]
    plt.plot(z65[:,0], z65[:,1], label = group)
    smoothed65.extend(z65[:,1])
psid = psid.assign(n65smooth = smoothed65)
plt.legend()
plt.title("Smoothed PSID >64")
plt.savefig('data/images/smoothpsid65.png')
plt.close()

psid.to_csv('data/PSID/psid.csv')