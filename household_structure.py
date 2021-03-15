

import pickle
import pandas as pd
import numpy as np
import microdf as mdf
import matplotlib.pyplot as plt
from scipy.stats import kde

panel_li = pickle.load(open('psid_lifetime_income.pkl', 'rb')) #created by psid_data_setup.py

panel_li.insert(len(panel_li.columns),"weight",1) #create column of only 1s which is used as weights for taking microdf average
panel_li.insert(len(panel_li.columns),"nu18",0)
panel_li.insert(len(panel_li.columns),"n1864",0)
panel_li.insert(len(panel_li.columns),"n65",0)
#RData file and therefore dataset in psid_data_setup.py -> panel_li is assumed to have 'num_family' as follows the updated list of variables pulled

def add65(head_age,spouse_age):
  count = 0
  if(head_age >= 65):
    count += 1
  if(spouse_age>=65):
    count += 1
  return count #assumes only head or spouse of head can be 65+
panel_li['n65'] = panel_li.apply(lambda x: add65(x['head_age'],x['spouse_age']), axis=1)

def add1864(head_age,spouse_age,num_family,num_18):
  count = num_family-num_18
  if(head_age >= 65):
    count -= 1
  if(spouse_age>=65):
    count -= 1
  return count
panel_li['n1864'] = panel_li.apply(lambda x: add1864(x['head_age'],x['spouse_age'],x['num_family'],x['num_children_under18']), axis=1)

panel_li['nu18'] = panel_li.apply(lambda x: x['num_children_under18'], axis=1) #assumes only children can be <18

panel_li2 = panel_li.reset_index()
panel_li3 = panel_li2[panel_li2['head_age'] <= 80]
panel_li3 = panel_li3[panel_li3['head_age'] >= 20]
panel_li4 = panel_li3.groupby(['head_age', 'li_group']).apply(
    lambda x: pd.Series({
        'nu18': mdf.weighted_mean(x, 'nu18', 'weight')
        })).unstack() #rows become head age, columns income bracket, same for each 
panel_li4.dropna(inplace = True)

panel_li5 = panel_li3.groupby(['head_age', 'li_group']).apply(
    lambda x: pd.Series({
        'nu18': mdf.weighted_mean(x, 'n1864', 'weight')
        })).unstack()
panel_li5.dropna(inplace = True)
panel_li6 = panel_li3.groupby(['head_age', 'li_group']).apply(
    lambda x: pd.Series({
        'nu18': mdf.weighted_mean(x, 'n65', 'weight')
        })).unstack()
panel_li6.dropna(inplace = True)

temp1 = panel_li4.sum(axis=1)
panelFinal = temp1.sum()
temp18 = panel_li4[panel_li4.columns] * 1/panelFinal #scales down to sum of 1 for smoothing, same for each below

temp2 = panel_li5.sum(axis=1)
panelFinal2 = temp2.sum()
temp1864 = panel_li5[panel_li5.columns] * 1/panelFinal2 

temp3 = panel_li6.sum(axis=1)
panelFinal3 = temp3.sum()
temp65 = panel_li6[panel_li6.columns] * 1/panelFinal3 


def MVKDE(S, J, proportion_matrix, filename=None, plot=False, bandwidth=.25):
    '''
    Generates a Multivariate Kernel Density Estimator and returns a
    matrix representing a probability distribution according to given
    age categories, and ability type categories.
    Args:
        S (scalar): the number of age groups in the model
        J (scalar): the number of ability type groups in the model.
        proportion_matrix (Numpy array): SxJ shaped array that
            represents the proportions of the total going to each
            (s,j) combination
        filename (str): the file name  to save image to
        plot (bool): whether or not to save a plot of the probability
            distribution generated by the kde or the proportion matrix
        bandwidth (scalar):  used in the smoothing of the kernel. Higher
            bandwidth creates a smoother kernel.
    Returns:
        estimator_scaled (Numpy array): SxJ shaped array that
            that represents the smoothed distribution of proportions
            going to each (s,j)
    '''
    proportion_matrix_income = np.sum(proportion_matrix, axis=0)
    proportion_matrix_age = np.sum(proportion_matrix, axis=1)
    age_probs = np.random.multinomial(70000, proportion_matrix_age)
    income_probs = np.random.multinomial(70000, proportion_matrix_income)
    age_frequency = np.array([])
    income_frequency = np.array([])
    age_mesh = complex(str(S) + 'j')
    income_mesh = complex(str(J) + 'j')
    j = 18
    '''creating a distribution of age values'''
    for i in age_probs:
        listit = np.ones(i)
        listit *= j
        age_frequency = np.append(age_frequency, listit)
        j += 1

    k = 1
    '''creating a distribution of ability type values'''
    for i in income_probs:
        listit2 = np.ones(i)
        listit2 *= k
        income_frequency = np.append(income_frequency, listit2)
        k += 1

    freq_mat = np.vstack((age_frequency, income_frequency)).T
    density = kde.gaussian_kde(freq_mat.T, bw_method=bandwidth)
    age_min, income_min = freq_mat.min(axis=0)
    age_max, income_max = freq_mat.max(axis=0)
    agei, incomei = np.mgrid[age_min:age_max:age_mesh,
                             income_min:income_max:income_mesh]
    coords = np.vstack([item.ravel() for item in [agei, incomei]])
    estimator = density(coords).reshape(agei.shape)
    estimator_scaled = estimator/float(np.sum(estimator))
    if plot:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(agei,incomei, estimator_scaled, rstride=5)
        ax.set_xlabel("Age")
        ax.set_ylabel("Ability Types")
        ax.set_zlabel("Received proportion of total bequests")
        plt.savefig(filename)
    return estimator_scaled

result18 = MVKDE(60, 7, temp18)
result18 = result18*panelFinal #scale back up, same for each below

result1864 = MVKDE(60, 7, temp1864)
result1864 = result1864*panelFinal2

result65 = MVKDE(60, 7, temp65)
result65 = result65*panelFinal3

np.save('nu18', result18)
np.save('n1864', result1864)
np.save('n65', result65)
print(panel_li4)