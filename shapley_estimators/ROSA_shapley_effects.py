# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 10:03:06 2021

@author: Julien Demange-Chryst
"""

#%% Modules

import numpy as np
import itertools
from scipy.stats import multivariate_normal
from scipy.special import binom
from utils import powerset
import openturns as ot
from compute_closed_sobol_ngd import *
from compute_closed_sobol_gd import *


#%% Estimation of the target Shapley effects in the given-model framework

def rosa_shapley_effects(phi,t,input_distr,Nv=10**4,Nu=1,Ni=3,m=10**4,withIS=False,aggregation="subset",type_estimator="dMC",**kwargs):
    """
    
    Estimation of the target Shapley effects in the given-model framework
    
    Parameters
    ----------
    phi : FUNCTION
        Function of interest.
    t : FLOAT
        Failure threshold.
    input_distr : OPENTURS DISTRIBUTION
        Input distribution. If "withIS=False", it has to be a Gaussian distribution.
    Nv : INT, optional
        Size of the sample to estimate the mean and the variance of the indicator of the failure event. The default is 10**4.
    Nu : INT, optional
        Size of the outer loop. The default is 1.
    Ni : INT, optional
        Size if the inner loop. The default is 3.
    m : INT, optional
        Number of random permutations picked. The default is 10**4.
    withIS : BOOLEAN, optional
        True to use importance sampling, else False. The default is False.
    aggregation : STRING, optional
        Aggregation procedure chosen: "subset" for subset procedure, "spr" for random permutation procedure. The default is "subset".
    type_estimator : STRING, optional
        Type of estimator to use: "dMC" dor double Monte Carlo, "PF" for Pick-Freeze. The default is "dMC".
    **kwargs : aux_distr : OPENTURNS DISTRIBUTION
        Importance sampling auxiliary distribution, it has to be Gaussian.

    Returns
    -------
    ARRAY
        Estimated target Shapley effects, size (d,).
    estimated_mean : FLOAT
        Estimated failure probaility.

    """
    
    dim = input_distr.getDimension()
    
    if withIS:
        if len(kwargs) == 1:
            aux_distr = kwargs['aux_distr']
            mean_g = np.array(aux_distr.getMean())
            cov_g = np.array(aux_distr.getCovariance())
        
            sample_MC = np.random.multivariate_normal(mean_g,cov_g,size=Nv)
            Y = np.apply_along_axis(lambda x:phi(x)>t,1,sample_MC)
        
            f = lambda x:input_distr.computePDF(x)
            g = lambda x:multivariate_normal.pdf(x,mean_g,cov_g)
            w = np.array(f(sample_MC)).flatten()/g(sample_MC)
            estimated_mean = np.mean(Y*w)
            E_2Y = estimated_mean**2 - (np.mean((Y*w)**2) - estimated_mean**2)/(Nv-1)
            estimated_var = estimated_mean - E_2Y
            
            if type_estimator=="dMC":
                get_closed_sobol = lambda u:estimated_mean - compute_closed_sobol_MC_IS(u,Nu,Ni,phi,t,input_distr,mean_g,cov_g)
            elif type_estimator=="PF":
                get_closed_sobol = lambda u:compute_closed_sobol_PF_IS(u,Nu,phi,t,input_distr,mean_g,cov_g) - E_2Y
            
        else:
            print("error !")
            return
    
    else:
        mean_f = np.array(input_distr.getMean())
        cov_f = np.array(input_distr.getCovariance())
        sample_MC = np.random.multivariate_normal(mean_f,cov_f,size=Nv)
        Y = np.apply_along_axis(lambda x:phi(x)>t,1,sample_MC)
        estimated_mean = np.mean(Y)
        estimated_var = np.sum((Y-estimated_mean)**2)/(Nv-1)
        
        if type_estimator=="dMC":
            get_closed_sobol = lambda u:compute_closed_sobol_MC(u,Nu,Ni,phi,t,mean_f,cov_f)
        elif type_estimator=="PF":
            get_closed_sobol = lambda u:compute_closed_sobol_PF(u,Nu,phi,t,mean_f,cov_f) - estimated_mean**2

    shapley = np.zeros(dim)
    
    if aggregation=="spr":
        nb_perm = np.math.factorial(dim)
        m = min(m,nb_perm)
        all_permutations = np.array(list(itertools.permutations(range(dim))))
        if m < nb_perm:
            indices = np.random.choice(nb_perm,replace=True,size=m)
            permutations = all_permutations[indices]
        else:
            permutations = all_permutations
            
        for perm in permutations:
            prev = 0
            for j in range(dim):
                if j==dim-1:
                    mean_out = estimated_var
                else:
                    P_pi_j = np.array(perm[:int(np.argwhere(perm==perm[j]))+1])       
                    
                    mean_out = get_closed_sobol(P_pi_j)
                                                
                delta_s = mean_out - prev
                shapley[perm[j]] += delta_s
                prev = mean_out
                
        return shapley/estimated_var/m, estimated_mean
            
    elif aggregation=="subset":
        closed_Sobol={}
        subsets = powerset(range(dim))
        subsets = list(subsets)
        nb_subsets = len(subsets)
    
        for n,u in enumerate(subsets):
            if len(u)==0:
                closed_Sobol[u] = 0
            elif len(u) == dim:
                closed_Sobol[u] = estimated_var
                for i in range(len(u)):
                    shapley[u[i]] += 1/binom(dim-1,len(u)-1) * (closed_Sobol[u] - closed_Sobol[tuple(np.delete(u,i))])
            else:
                u_arr = np.array(u)
                
                closed_Sobol[u] = get_closed_sobol(u_arr)
            
                if len(u)==1:
                    shapley[u[0]] += 1/binom(dim-1,len(u)-1) * (closed_Sobol[u] - closed_Sobol[()])
                
                else:
                    for i in range(len(u)):
                        shapley[u[i]] += 1/binom(dim-1,len(u)-1) * (closed_Sobol[u] - closed_Sobol[tuple(np.delete(u,i))])
    
        return shapley/estimated_var/dim, estimated_mean
    

#%% Estimation of the target Shapley effects in the given-data framework

def rosa_shapley_effects_gd(X,Y,Nu=1,Ni=3,m=10**4,withIS=False,aggregation="spr",type_estimator="dMC",standardisation=False,**kwargs):
    """
    
    Estimation of the target Shapley effects in the given-data framework

    Parameters
    ----------
    X : ARRAY
        Input data, size (N,d).
    Y : ARRAY
        Output data, size (N,).
    Nu : INT, optional
        Size of the outer loop. The default is 1.
    Ni : INT, optional
        Size if the inner loop. The default is 3.
    m : INT, optional
        Number of random permutations picked. The default is 10**4.
    withIS : BOOLEAN, optional
        True to use importance sampling, else False. The default is False.
    aggregation : STRING, optional
        Aggregation procedure chosen: "subset" for subset procedure, "spr" for random permutation procedure. The default is "subset".
    type_estimator : STRING, optional
        Type of estimator to use: "dMC" dor double Monte Carlo, "PF" for Pick-Freeze. The default is "dMC".
    standardisation : BOOLEAN, optional
        True to standardise the data to perform the nearest neighbour procedure, else False. The default is False.
    **kwargs : init_distr : OPENTURNS DISTRIBUTION
        Input distribution.
               aux_distr : OPENTURNS DISTRIBUTION
        Importance sampling auxiliary distribution.

    Returns
    -------
    ARRAY
        Estimated target Shapley effects, size (d,).
    estimated_mean : FLOAT
        Estimated failure probaility.

    """

    
    N_MC,dim = X.shape
    
    init_distr = None
        
    if len(kwargs)==1:
        init_distr = kwargs['init_distr']
    
    elif len(kwargs) == 2:
        init_distr,aux_distr = kwargs['init_distr'],kwargs['aux_distr']
    
    
    if withIS:
        f = lambda x:init_distr.computePDF(x)
        g = lambda x:aux_distr.computePDF(x)
        w = np.array(f(X))/np.array(g(X))
        w = w.flatten()
        estimated_mean = np.mean(Y*w)
        E_2Y = estimated_mean**2 - (np.mean((Y*w)**2) - estimated_mean**2)/(N_MC-1)
        estimated_var = estimated_mean - E_2Y
        
        if standardisation:
            marginal_esps = np.array(aux_distr.getMean())
            marginal_vars = np.diag(np.array(aux_distr.getCovariance()))
            marginal_stds = np.sqrt(marginal_vars)
            Z = (X - marginal_esps)/marginal_stds
        else:
            Z=X
            
        if type_estimator=="dMC":
            get_closed_sobol = lambda u:estimated_mean - compute_closed_sobol_MC_IS_gd(u,Nu,Ni,X,Y,w,init_distr,aux_distr,Z)
        elif type_estimator=="PF":
            get_closed_sobol = lambda u:compute_closed_sobol_PF_IS_gd(u,Nu,X,Y,w,init_distr,aux_distr,Z) - E_2Y
    
    else:
        estimated_mean = np.mean(Y)
        estimated_var = np.sum((Y-estimated_mean)**2)/(N_MC-1)
        if standardisation:
            if init_distr==None:
                marginal_esps = np.mean(X,axis=0)
                marginal_vars = np.var(X,axis=0)
            else:
                marginal_esps = np.array(init_distr.getMean())
                marginal_vars = np.diag(np.array(init_distr.getCovariance()))
            marginal_stds = np.sqrt(marginal_vars)
            Z = (X - marginal_esps)/marginal_stds
        else:
            Z=X
        
        if type_estimator=="dMC":
            get_closed_sobol = lambda u:compute_closed_sobol_MC_gd(u,Nu,Ni,Z,Y)
        elif type_estimator=="PF":
            get_closed_sobol = lambda u:compute_closed_sobol_PF_gd(u,Nu,Z,Y) - estimated_mean**2

    shapley = np.zeros(dim)
    
    if aggregation=="spr":
        nb_perm = np.math.factorial(dim)
        m = min(m,nb_perm)
        all_permutations = np.array(list(itertools.permutations(range(dim))))
        if m < nb_perm:
            indices = np.random.choice(nb_perm,replace=True,size=m)
            permutations = all_permutations[indices]
        else:
            permutations = all_permutations
            
        for perm in permutations:
            prev = 0
            for j in range(dim):
                if j==dim-1:
                    mean_out = estimated_var
                else:
                    P_pi_j = np.array(perm[:int(np.argwhere(perm==perm[j]))+1])  
                    
                    mean_out = get_closed_sobol(P_pi_j)
                    
                delta_s = mean_out - prev
                shapley[perm[j]] += delta_s
                prev = mean_out
                
        return shapley/estimated_var/m, estimated_mean
            
    elif aggregation=="subset":
        closed_Sobol={}
        subsets = powerset(range(dim))
    
        for u in subsets:
            if len(u)==0:
                closed_Sobol[u] = 0
            elif len(u) == dim:
                closed_Sobol[u] = estimated_var
                for i in range(len(u)):
                    shapley[u[i]] += 1/binom(dim-1,len(u)-1) * (closed_Sobol[u] - closed_Sobol[tuple(np.delete(u,i))])
            else:
                u_arr = np.array(u)
                
                closed_Sobol[u] = get_closed_sobol(u_arr)
            
                if len(u)==1:
                    shapley[u[0]] += 1/binom(dim-1,len(u)-1) * (closed_Sobol[u] - closed_Sobol[()])
                
                else:
                    for i in range(len(u)):
                        shapley[u[i]] += 1/binom(dim-1,len(u)-1) * (closed_Sobol[u] - closed_Sobol[tuple(np.delete(u,i))])
    
        return shapley/estimated_var/dim, estimated_mean