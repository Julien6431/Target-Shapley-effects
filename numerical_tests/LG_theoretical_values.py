#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 31 15:55:57 2021

@author: Julien Demange-Chryst
"""

#%% Modules

import numpy as np
from scipy.stats import norm
from scipy.special import binom
import sys
sys.path.append("../shapley_estimators")
from utils import powerset


#%% Function

Phi = lambda x:norm.cdf(x)

def compute_theoretical_values(t,beta,mean_f,cov_f,N_sample=10**5,n_rep=10):
    """
    
    Returns the theoretical values of the target shapley effects in the Gaussian linear case

    Parameters
    ----------
    t : FLOAT
        Failure threshold.
    beta : 1D NUMPY ARRAY
        Coefficients of the function.
    mean_f : 1D NUMPY ARRAY
        Mean of the input distribution.
    cov_f : 2D NUMPY ARRAY
        Covariance matrix of the input distribution - symetric definite positive matrix.
    N_sample : INT, optional
        Size of each sample. The default is 10**5.
    n_rep : INT, optional
        Number of execution of the process. The default is 10.

    Returns
    -------
    1D NUMPY ARRAY
        Theoretical Shapley values.
    1D NUMPY ARRAY
        Standard deviation.

    """
    dim = len(beta)
    shap_theo = np.zeros((n_rep,dim))
    failure_prob = 1-Phi((t-np.sum(mean_f*beta))/np.sqrt(beta.T@cov_f@beta))
    var_tot = failure_prob*(1-failure_prob)
        
    for n in range(n_rep):
        X = np.random.multivariate_normal(mean_f,cov_f,size=N_sample)
        
        closed_Sobol = {}
        subsets = powerset(range(dim))
        for u in subsets:
            if len(u)==0:
                closed_Sobol[u] = 0
            elif len(u) == dim:
                closed_Sobol[u] = 1
                for i in range(len(u)):
                    shap_theo[n,u[i]] += (1 - closed_Sobol[tuple(np.delete(u,i))])
            else:
                u_arr = np.array(u)
                com_u = np.delete(np.arange(dim),u)
                
                if (beta[com_u] == 0).all() == True:
                    closed_Sobol[u] = 1
                else:
                    cov_matrix_u = cov_f[u_arr][:,u_arr]
                    inv_cov_matrix_u = np.linalg.inv(cov_matrix_u)
                    cov_matrix_com_u = cov_f[com_u][:,com_u]
                    cross_cov_matrix = cov_f[com_u][:,u_arr]
                    
                    cond_cov_matrix = cov_matrix_com_u - cross_cov_matrix@inv_cov_matrix_u@cross_cov_matrix.T
                    cond_mean = mean_f[com_u] + (cross_cov_matrix@inv_cov_matrix_u@((X[:,u_arr] - mean_f[u_arr]).T)).T
                    Y = Phi((t - np.sum(beta[u_arr]*X[:,u_arr],axis=1) - np.sum(beta[com_u]*cond_mean,axis=1))/np.sqrt(beta[com_u].T@cond_cov_matrix@beta[com_u]))
                    
                    closed_Sobol[u] = np.var(Y)/var_tot
        
                if len(u)==1:
                    shap_theo[n,u[0]] += closed_Sobol[u]
                
                else:
                    for i in range(len(u)):
                        shap_theo[n,u[i]] += 1/binom(dim-1,len(u)-1) * (closed_Sobol[u] - closed_Sobol[tuple(np.delete(u,i))])

    return np.mean(shap_theo/dim,axis=0), np.std(shap_theo/dim,axis=0)
