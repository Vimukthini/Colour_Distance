import numpy as np
import pandas as pd
from scipy.stats import chi2
import scipy as sp

def colormapfunction(row):
    per = list(np.zeros(256))

    for item in row:
        per[int(item['x'])] = item['y']
    return per
    
def generate_noisy_data(noise_degree,data_to_generate,woodobjects):
    #noise_degree = 2
    np.random.seed(42)
    noisy_data = []
    for i in range(len(woodobjects)):
        #print(woodobjects.iloc[0,j])
        for j in range(data_to_generate):
            random_data = woodobjects.iloc[i]+np.random.uniform(-1,1,256)*noise_degree/100
            random_data = [ x if x>0 else 0 for x in random_data ]
            random_data = random_data/(np.ones(256)*sum(random_data))
            noisy_data.append(random_data)

    noisy_data = pd.DataFrame(noisy_data)
    noisy_data.columns = woodobjects.columns

    return noisy_data

def mahalanobis(x=None, data=None, cov=None):
    """Compute the Mahalanobis Distance between each row of x and the data  
    x    : vector or matrix of data with, say, p columns.
    data : ndarray of the distribution from which Mahalanobis distance of each observation of x is to be computed.
    cov  : covariance matrix (p x p) of the distribution. If None, will be computed from data.
    """
    x_minus_mu = x - np.mean(data)
    #print("x_minum_mu",x_minus_mu)
    if not cov:
        cov = np.cov(data.values.T)
    inv_covmat = sp.linalg.inv(cov)
    #print("inv_covmat",inv_covmat)
    left_term = np.dot(x_minus_mu, inv_covmat)
    #print("lt",left_term)
    mahal = np.dot(left_term, x_minus_mu.T)
    #print("mahal",mahal)
    return mahal.diagonal()

def get_distance_of(object_to_check,woodobjects):

    #Number of data points to generate
    data_to_generate = int(1000/len(woodobjects))

    noisy_data = generate_noisy_data(2,data_to_generate,woodobjects)
    woodobjects = pd.concat([object_to_check,woodobjects,noisy_data])
    woodobjects.reset_index(drop=True,inplace=True)

    woodobjects = woodobjects.loc[:,(woodobjects!=0).any(axis=0)]

    names = woodobjects.columns
    names = list(names)
    dof = len(woodobjects.columns)
    woodobjects['mahalanobis'] = mahalanobis(x=woodobjects, data=woodobjects[names])

    # Compute the P-Values
    woodobjects['p_value'] = 1 - chi2.cdf(woodobjects['mahalanobis'], dof)
    # Extreme values with a significance level of 0.01
    different_objects = woodobjects.loc[woodobjects.p_value < 0.001] 

    diff = 'yes' if 0 in list(different_objects.index) else 'no'
    distance = woodobjects['mahalanobis'][0]
    pvalue = woodobjects['p_value'][0]

    return diff, distance, pvalue
