# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 08:57:59 2021

@author: Julien Demange-Chryst
"""

#%% Modules

import numpy as np
import openturns as ot
from EMGM import EMGM
from sklearn.cluster import KMeans
from kneed import KneeLocator

#%% Cross entropy with a Gaussian mixture

def CEIS_GM(N,p,phi,t,distr,nb_gaussian_max = 10):
    """
    
    Adaptive cross-entropy algorithm to compute a failure probability using a Gaussian mixture auxiliary distribution.

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
    nb_gaussian_max : INT, optional
        Maximal number of Gaussian distributions in the mixture. The default is 10.

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
    k_fin : INT
        Number of Gaussian distribution in the final mixture aux_distr.

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
    mu_init = np.array(distr.getMean())*np.ones([1,dim])
    sigma_init = np.diag(np.diag(np.array(distr.getCovariance())))
    pi_init = np.array([1.0])
    gamma_hat = np.zeros([max_it+1])
    
    mu_hat = mu_init
    sigma_hat = sigma_init
    pi_hat = pi_init
    gamma_hat[0]= 0
    
    samples = []
    slope_Kmeans = []
    
    #Adaptive algorithm
    for j in range(max_it):
        
        #Computation if the new auxiliary distribution
        if len(pi_hat) == 1:
            if len(sigma_hat.shape) == 3:
                aux_distr = ot.Normal(ot.Point(mu_hat[0]),ot.CovarianceMatrix(sigma_hat[:,:,0]))
            else:
                aux_distr = ot.Normal(ot.Point(mu_hat[0]),ot.CovarianceMatrix(sigma_hat))
        else:
            collDist = [ot.Normal(ot.Point(mu_hat[i]),ot.CovarianceMatrix(sigma_hat[:,:,i])) for i in range(mu_hat.shape[0])]
            aux_distr = ot.Mixture(collDist,pi_hat)
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
                
        weights = W[I]/np.sum(W[I])
        weights = len(weights)*weights
                
        #Computation of the optimal number of components in the mixture        
        inertia = np.zeros(nb_gaussian_max)
        for nb_gaussian in range(1,nb_gaussian_max+1):
            Km = KMeans(n_clusters=nb_gaussian)
            Km.fit(X[I],sample_weight=weights)
            current_inertia = Km.inertia_
            inertia[nb_gaussian-1] = current_inertia
            if nb_gaussian == 2:
                slope_Kmeans.append(inertia[1] - inertia[0])
            
        if (j>0) and (gamma_hat[j+1]<t):
            power_10 = int(np.log10(-slope_Kmeans[0]))
            if abs((slope_Kmeans[-1] - np.mean(slope_Kmeans[:-1]))/10**power_10)<=1:
                best_k = 0
            else:
                best_k = KneeLocator(np.arange(len(inertia)), inertia, curve='convex', direction='decreasing').knee
        elif gamma_hat[j+1]==t:
            if k==1:
                best_k = 0
        else:
            best_k = KneeLocator(np.arange(len(inertia)), inertia, curve='convex', direction='decreasing').knee

        k = best_k  + 1       
                
        [mu_hat, sigma_hat, pi_hat] = EMGM(X[I, :].T, W[I], k)
        mu_hat = mu_hat.T
        k = len(pi_hat)
       
    
    level = j
    k_fin = k
    gamma_hat = gamma_hat[: level + 1]
    
    #Estimation of the failure probability    
    W_final = np.array(distr.computePDF(X))/h 
    W_final = W_final.flatten()
    I_final = (Y >= t)
    Pr = 1 / N * sum(I_final * W_final)
    
            
    return Pr, samples, aux_distr, N_tot, k_fin
