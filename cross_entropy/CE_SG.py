# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 14:00:16 2021

@author: Julien Demange-Chryst
"""

#%% Modules

import numpy as np
import openturns as ot

#%% Cross entropy with a signle Gaussian

def CEIS_SG(N,p,phi,t,distr):
    """
    
    Adaptive cross-entropy algorithm to compute a failure probability using a single Gaussian auxiliary distribution.

    Parameters
    ----------
    N : INT
        Number of points drawn at each step.
    p : FLOAT
        Quantile to set the new intermediate critical threshold, 0<=p0<=1.
    phi : FUNCTION
        Function to estimate the failure probability.
    t : FLOAT
        Failure threshold.
    distr : OPENTURS DISTRIBUTION
        Input distribution.

    Raises
    ------
    RuntimeError
        N*p and 1/p must be positive integers. Adjust N and p accordingly.

    Returns
    -------
    Pr : FLOAT
        Estimated failure probability.
    samples : LIST
        List containing the samples drawn at each iteration.
    aux_distr : OPENTURNS DISTRIBUTION
        Final auxiliary distribution.
    N_tot : INT
        Total number of calls to the function.

    """
    if (N * p != np.fix(N * p)) or (1 / p != np.fix(1 / p)):
        raise RuntimeError(
            "N*p and 1/p must be positive integers. Adjust N and p accordingly"
        )
    
    j = 0 #current iteration
    max_it = 50 #maximal number of iterations
    N_tot = 0 #number of calls to the function
    
    dim = distr.getDimension()
    
    #Initialisation
    mu_init = np.array(distr.getMean())#np.zeros(dim)
    sigma_init = np.diag(np.diag(np.array(distr.getCovariance())))#np.eye(dim)
    gamma_hat = np.zeros(max_it+1)
    
    mu_hat = mu_init
    sigma_hat = sigma_init
    gamma_hat[0]= 0
    
    samples = []
        
    #Adaptive algorithm
    for j in range(max_it):
        
        #Computation if the new auxiliary distribution
        ot_mu = ot.Point(mu_hat)
        ot_sigma = ot.CovarianceMatrix(sigma_hat)
        aux_distr = ot.Normal(ot_mu,ot_sigma)
        ot_X = aux_distr.getSample(N)
        X = np.array(ot_X)
        
        samples.append(X)
        Y = np.apply_along_axis(phi,1,X)
        N_tot += N
        
        h = np.array(aux_distr.computePDF(X))
        
        #Break the loop if the threshold is greater or equal to the real one
        if gamma_hat[j] >= t:
            break
        
        #Computation of the new threshold
        gamma_hat[j+1] = np.minimum(t,np.nanpercentile(Y,100*(1-p)))
        I = (Y>=gamma_hat[j+1])
        W = np.array(distr.computePDF(X))/h
        W = W.flatten()
        
        prod = np.matmul(W[I], X[I, :])
        summ = np.sum(W[I])
        mu_hat = (prod) / summ
        Xtmp = X[I, :] - mu_hat
        Xo = (Xtmp) * np.tile(np.sqrt(W[I]), (dim, 1)).T
        sigma_hat = np.matmul(Xo.T, Xo) / np.sum(W[I]) + 1e-6 * np.eye(dim)
        
    lv = j
    gamma_hat = gamma_hat[: lv + 1]

    #Estimation of the failure probability  
    W_final = np.array(distr.computePDF(X))/h
    W_final = W_final.flatten()
    I_final = (Y >= t)
    Pr = 1 / N * sum(I_final * W_final)

    return Pr, samples, aux_distr, N_tot

