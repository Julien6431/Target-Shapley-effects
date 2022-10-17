# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 10:02:20 2021

@author: Julien Demange-Chryst
"""

#%% Modules

import numpy as np
from scipy.stats import multivariate_normal


#%% Compute closed Sobol without IS given-model


def compute_closed_sobol_MC(u,Nu,Ni,phi,t,mean_f,cov_f):
    """
    
    Estimation of the closed Sobol index associated to the subset u by double Monte Carlo without importance sampling when the input distribution is Gaussian.

    Parameters
    ----------
    u : ARRAY
        Subset associated to the index to estimate.
    Nu : INT
        Size of the outer double Monte Carlo loop.
    Ni : INT
        Size of the inner double Monte Carlo loop.
    phi : FUNCTION
        Functon of interest.
    t : FLOAT
        Failure threshold.
    mean_f : ARRAY
        Mean vector of the Gaussian input distribution, size (d,).
    cov_f : ARRAY
        Covariance matrix of the Gaussian input distribution, size (d,d).

    Returns
    -------
    FLOAT
         Estimated closed Sobol index associated to subset u.

    """
    
    dim = len(mean_f)
    com_u = np.delete(np.arange(dim),u)
    
    #Computation of the conditional Gaussian distribution
    cov_matrix_com_P_pi_j = cov_f[com_u][:,com_u]
    inv_cov_matrix_com_P_pi_j = np.linalg.inv(cov_matrix_com_P_pi_j)
    cov_matrix_P_pi_j = cov_f[u][:,u]
    cross_cov_matrix = cov_f[u][:,com_u]
                                
    sample_No = np.random.multivariate_normal(mean_f[com_u],cov_matrix_com_P_pi_j,size=Nu)
    cond_cov_matrix = cov_matrix_P_pi_j - cross_cov_matrix@inv_cov_matrix_com_P_pi_j@cross_cov_matrix.T
    
    #Double Monte Carlo loop
    samples = np.zeros((Nu,dim,Ni))
    for l in range(Nu):
        cond_mean = mean_f[u] + cross_cov_matrix@inv_cov_matrix_com_P_pi_j@(sample_No[l] - mean_f[com_u])
        sample_Ni = np.random.multivariate_normal(cond_mean, cond_cov_matrix,size=Ni)
        samples[l,:,:][com_u] = np.repeat(sample_No[l].reshape((-1,1)),Ni,axis=1)
        samples[l,:,:][u] = sample_Ni.T
          
    #Estimation of the index 
    Y_mix = np.apply_along_axis(lambda x:phi(x)>t,1,samples)
    var = np.var(Y_mix,axis=1)*Ni/(Ni-1)
    return np.mean(var)




def compute_closed_sobol_PF(u,Nu,phi,t,mean_f,cov_f):
    """
    
    Estimation of the closed Sobol index associated to the subset u by Pick-Freeze without importance sampling when the input distribution is Gaussian.

    Parameters
    ----------
    u : ARRAY
        Subset associated to the index to estimate.
    Nu : INT
        Size of the Pick-Freeze loop.
    phi : FUNCTION
        Functon of interest.
    t : FLOAT
        Failure threshold.
    mean_f : ARRAY
        Mean vector of the Gaussian input distribution, size (d,).
    cov_f : ARRAY
        Covariance matrix of the Gaussian input distribution, size (d,d).

    Returns
    -------
    FLOAT
         Estimated closed Sobol index associated to subset u.

    """
    dim = len(mean_f)
    com_u = np.delete(np.arange(dim),u)
              
    #Computation of the conditional Gaussian distribution
    cov_matrix_P_pi_j = cov_f[u][:,u]
    inv_cov_matrix_P_pi_j = np.linalg.inv(cov_matrix_P_pi_j)
    cov_matrix_com_P_pi_j = cov_f[com_u][:,com_u]
    cross_cov_matrix = cov_f[com_u][:,u]
                
    sample_No = np.random.multivariate_normal(mean_f[u],cov_matrix_P_pi_j,size=Nu)
    cond_cov_matrix = cov_matrix_com_P_pi_j - cross_cov_matrix@inv_cov_matrix_P_pi_j@cross_cov_matrix.T
                
    Y_lh = np.zeros((Nu,2))
         
    #Pick-Freeze loop       
    for l in range(Nu):
        cond_mean = mean_f[com_u] + cross_cov_matrix@inv_cov_matrix_P_pi_j@(sample_No[l] - mean_f[u])
        sample_Ni = np.random.multivariate_normal(cond_mean, cond_cov_matrix,size=2)
                    
        for h in range(2):   
            new_input = np.zeros(dim)
            new_input[u] = sample_No[l]
            new_input[com_u] = sample_Ni[h]
                                                
            Y_lh[l,h] = (phi(new_input)>t)
                     
    #Estimation of the index                                               
    PF = np.prod(Y_lh,axis=1)
    return np.mean(PF)

#%% Compute closed Sobol with IS given-model

def compute_closed_sobol_MC_IS(u,Nu,Ni,phi,t,input_distr,mean_g,cov_g):
    """
    
    Estimation of the closed Sobol index associated to the subset u by double Monte Carlo with importance sampling when the auxiliary distribution is Gaussian.

    Parameters
    ----------
    u : ARRAY
        Subset associated to the index to estimate.
    Nu : INT
        Size of the outer double Monte Carlo loop.
    Ni : INT
        Size of the inner double Monte Carlo loop.
    phi : FUNCTION
        Functon of interest.
    t : FLOAT
        Failure threshold.
    input_distr : OPENTURNS DISTRIBUTION
        Input distribution.
    mean_g : ARRAY
        Mean vector of the Gaussian auxiliary distribution, size (d,).
    cov_g : ARRAY
        Covariance matrix of the Gaussian auxiliary distribution, size (d,d).

    Returns
    -------
    FLOAT
         Estimated closed Sobol index associated to subset u.

    """
    
    dim = len(mean_g)
    
    com_u = np.delete(np.arange(dim),u)
    
    f = lambda x:input_distr.computePDF(x)
    f_minus_u = lambda x:input_distr.getMarginal(com_u.tolist()).computePDF(x)
                
    #Computation of the conditional Gaussian distribution
    cond_cov_matrix_g = cov_g[com_u][:,com_u]
    inv_cond_cov_matrix_g = np.linalg.inv(cond_cov_matrix_g)
    esti_cov_matrix_g = cov_g[u][:,u]
    cross_cov_matrix_g = cov_g[u][:,com_u]
    
    g = lambda x:multivariate_normal.pdf(x,mean_g,cov_g)                              
    g_minus_u = lambda x:multivariate_normal.pdf(x,mean_g[com_u],cond_cov_matrix_g)    
                
    outputs = np.zeros((Nu,Ni))
    sample_No = np.random.multivariate_normal(mean_g[com_u],cond_cov_matrix_g,size=Nu)
    new_cov_g = esti_cov_matrix_g - cross_cov_matrix_g@inv_cond_cov_matrix_g@cross_cov_matrix_g.T
    
    #Computation of the importance sampling weights
    w_minus_u = np.array(f_minus_u(sample_No)).flatten()/g_minus_u(sample_No)
    w_minus_u[w_minus_u==0] = np.inf
    
    samples = np.zeros((Nu,dim,Ni))
    
    #Dooble Monte Carlo loop
    for l in range(Nu):
        new_mean_g = mean_g[u] + cross_cov_matrix_g@inv_cond_cov_matrix_g@(sample_No[l] - mean_g[com_u])
        sample_Ni = np.random.multivariate_normal(new_mean_g, new_cov_g,size=Ni)
                    
        samples[l,:,:][com_u] = np.repeat(sample_No[l].reshape((-1,1)),Ni,axis=1)
        samples[l,:,:][u] = sample_Ni.T
                
        
    w_g_t = lambda x:(phi(x)>t)*f(x)/g(x)
    outputs = np.apply_along_axis(w_g_t,1,samples)
               
    #Estimation of the index         
    mean_out = np.mean(np.mean(outputs,axis=1)**2/w_minus_u)
    biais = np.mean((np.mean(outputs**2,axis=1) - np.mean(outputs,axis=1)**2)/w_minus_u)/(Ni-1)
    
    return mean_out - biais


def compute_closed_sobol_PF_IS(u,Nu,phi,t,input_distr,mean_g,cov_g):
    """
    
    Estimation of the closed Sobol index associated to the subset u by Pick-Freeze with importance sampling when the auxiliary distribution is Gaussian.

    Parameters
    ----------
    u : ARRAY
        Subset associated to the index to estimate.
    Nu : INT
        Size of the Pick-Freeze loop.
    phi : FUNCTION
        Functon of interest.
    t : FLOAT
        Failure threshold.
    input_distr : OPENTURNS DISTRIBUTION
        Input distribution.
    mean_g : ARRAY
        Mean vector of the Gaussian auxiliary distribution, size (d,).
    cov_g : ARRAY
        Covariance matrix of the Gaussian auxiliary distribution, size (d,d).

    Returns
    -------
    FLOAT
         Estimated closed Sobol index associated to subset u.

    """
    dim = len(mean_g)
    com_u = np.delete(np.arange(dim),u)
    
    f = lambda x:input_distr.computePDF(x)
    f_u = lambda x:input_distr.getMarginal(u.tolist()).computePDF(x)
    
    #Computation of the conditional Gaussian distribution
    cond_cov_matrix_g = cov_g[u][:,u]
    inv_cond_cov_matrix_g = np.linalg.inv(cond_cov_matrix_g)
    esti_cov_matrix_g = cov_g[com_u][:,com_u]
    cross_cov_matrix_g = cov_g[com_u][:,u]
    
    g = lambda x:multivariate_normal.pdf(x,mean_g,cov=cov_g)                            
    g_u = lambda x:multivariate_normal.pdf(x,mean_g[u],cond_cov_matrix_g) 
                
    samples = np.zeros((Nu,dim,2))
    sample_No = np.random.multivariate_normal(mean_g[u],cond_cov_matrix_g,size=Nu)
    
    #Computation of the importance sampling weights
    w_u = np.array(f_u(sample_No)).flatten()/g_u(sample_No)
    w_u[w_u==0] = np.inf
    
    
    new_cov_g = esti_cov_matrix_g - cross_cov_matrix_g@inv_cond_cov_matrix_g@cross_cov_matrix_g.T
    #Pick-Freeze loop    
    for l in range(Nu):
        new_mean_g = mean_g[com_u] + cross_cov_matrix_g@inv_cond_cov_matrix_g@(sample_No[l] - mean_g[u])
        sample_Ni = np.random.multivariate_normal(new_mean_g, new_cov_g,size=2)
            
        samples[l,:,:][u] = np.repeat(sample_No[l].reshape((-1,1)),2,axis=1)
        samples[l,:,:][com_u] = sample_Ni.T
    
    w_g_t = lambda x:(phi(x)>t)*f(x)/g(x)
    
    #Estimation of the index        
    outputs = np.apply_along_axis(w_g_t,1,samples)
    PF = np.prod(outputs,axis=1) / w_u
    
    return np.mean(PF)
