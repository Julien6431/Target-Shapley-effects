# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 10:01:41 2021

@author: Julien Demange-Chryst
"""

#%% Modules

import numpy as np
import openturns as ot
from sklearn.neighbors import NearestNeighbors

#%% Compute closed Sobol without IS given-data


def compute_closed_sobol_MC_gd(u,Nu,Ni,X,Y):
    """
    
    Estimation of the closed Sobol index associated to the subset u by double Monte Carlo without importance sampling.

    Parameters
    ----------
    u : ARRAY
        Subset associated to the index to estimate.
    Nu : INT
        Size of the outer double Monte Carlo loop.
    Ni : INT
        Size of the inner double Monte Carlo loop, number of nearest neighbours to find.
    X : ARRAY
        Input data, size (N,d).
    Y : ARRAY
        Output data, size (N,).

    Returns
    -------
    FLOAT
        Estimated closed Sobol index associated to subset u.

    """
    
    N_MC,dim = X.shape
    com_u = np.delete(np.arange(dim),u)

    #Nearest neighbour procedure
    indices = np.random.choice(range(N_MC),replace=True,size=Nu)
    nbrs = NearestNeighbors(n_neighbors=Ni, algorithm='ball_tree').fit(X[:,com_u])
    _, knn_indices = nbrs.kneighbors(X[:,com_u][indices])
        
    #Estimation of the index
    Y_knn = Y[knn_indices]
    means = np.mean(Y_knn,axis=1).reshape(-1,1)
    var = np.sum((Y_knn-means)**2,axis=1)/(Ni-1)
    return np.mean(var)


def compute_closed_sobol_PF_gd(u,Nu,X,Y):
    """
    
    Estimation of the closed Sobol index associated to the subset u by Pick-Freeze without importance sampling.

    Parameters
    ----------
    u : ARRAY
        Subset associated to the index to estimate.
    Nu : INT
        Size of the Pick-Freeze loop.
    X : ARRAY
        Input data, size (N,d).
    Y : ARRAY
        Output data, size (N,).

    Returns
    -------
    FLOAT
        Estimated closed Sobol index associated to subset u.

    """
    N_MC,dim = X.shape
    
    #Nearest neighbour procedure
    indices = np.random.choice(range(N_MC),replace=True,size=Nu)
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X[:,u])
    _, knn_indices = nbrs.kneighbors(X[:,u][indices])
        
    #Estimation of the index        
    Y_knn = Y[knn_indices]
    PF = np.prod(Y_knn,axis=1)
                
    return np.mean(PF)

#%% Compute closed Sobol with IS given-data

def compute_closed_sobol_MC_IS_gd(u,Nu,Ni,X,Y,w,init_distr,aux_distr,Z):
    """
    
    Estimation of the closed Sobol index associated to the subset u by double Monte Carlo with importance sampling.

    Parameters
    ----------
    u : ARRAY
        Subset associated to the index to estimate.
    Nu : INT
        Size of the outer double Monte Carlo loop.
    Ni : INT
        Size of the inner double Monte Carlo loop, number of nearest neighbours to find.
    X : ARRAY
        Input data, size (N,d).
    Y : ARRAY
        Output data, size (N,).
    w : ARRAY
        Importance sampling weights corresponding to the data, size (N,).
    init_distr : OPENTURNS DISTRIBUTION
        Input distribution.
    aux_distr : OPENTURNS DISTRIBUTION
        Importance sampling auxiliary distribution.
    Z : ARRAY
        Input data for which the nearest neighbour algorithm is performed, size (N,d).

    Returns
    -------
    FLOAT
        Estimated closed Sobol index associated to subset u.

    """
    
    N_MC,dim = X.shape
    com_u = np.delete(np.arange(dim),u)
           
    #Computation of the marginal importance sampling weights     
    f_minus_u = lambda x:init_distr.getMarginal(com_u.tolist()).computePDF(x)
    g_minus_u = lambda x:aux_distr.getMarginal(com_u.tolist()).computePDF(x)
                
    w_minus_u = np.array(f_minus_u(X[:,com_u]))/np.array(g_minus_u(X[:,com_u]))
    w_minus_u = w_minus_u.flatten()
    w_minus_u[w_minus_u==0] = np.inf
                
    #Nearest neighbour procedure
    indices = np.random.choice(range(N_MC),replace=True,size=Nu)
    nbrs = NearestNeighbors(n_neighbors=Ni, algorithm='ball_tree').fit(Z[:,com_u])
    _, knn_indices = nbrs.kneighbors(Z[:,com_u][indices])     
          
    #Estimation of the index      
    Y_knn = Y[knn_indices]
    inner_loop = np.mean(Y_knn*w[knn_indices],axis=1)
    outer_loop = np.mean(inner_loop**2 / w_minus_u[indices])
    
    #Estimation of the bias
    bias_inner = np.mean(Y_knn**2*w[knn_indices]**2,axis=1) - inner_loop**2
    bias_outer = 1/(Ni-1) * np.mean(bias_inner / w_minus_u[indices])
            
    return outer_loop - bias_outer 


def compute_closed_sobol_PF_IS_gd(u,Nu,X,Y,w,init_distr,aux_distr,Z):
    """
    
    Estimation of the closed Sobol index associated to the subset u by Pick-Freeze with importance sampling.

    Parameters
    ----------
    u : ARRAY
        Subset associated to the index to estimate.
    Nu : INT
        Size of the Pick-Freeze loop.
    X : ARRAY
        Input data, size (N,d).
    Y : ARRAY
        Output data, size (N,).
    w : ARRAY
        Importance sampling weights corresponding to the data, size (N,).
    init_distr : OPENTURNS DISTRIBUTION
        Input distribution.
    aux_distr : OPENTURNS DISTRIBUTION
        Importance sampling auxiliary distribution.
    Z : ARRAY
        Input data for which the nearest neighbour algorithm is performed, size (N,d).

    Returns
    -------
    FLOAT
        Estimated closed Sobol index associated to subset u.

    """
    N_MC,dim = X.shape
    
    #Computation of the marginal importance sampling weights     
    f_u = lambda x:init_distr.getMarginal(u.tolist()).computePDF(x)
    g_u = lambda x:aux_distr.getMarginal(u.tolist()).computePDF(x)
    
    w_u = np.array(f_u(X[:,u]))/np.array(g_u(X[:,u]))
    w_u = w_u.flatten()
    w_u[w_u==0] = np.inf
                
    #Nearest neighbour procedure
    indices = np.random.choice(range(N_MC),replace=True,size=Nu)
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(Z[:,u])
    _, knn_indices = nbrs.kneighbors(Z[:,u][indices])    
        
    #Estimation of the index              
    Y_knn = Y[knn_indices]
    PF = np.prod(Y_knn,axis=1) * np.prod(w[knn_indices],axis=1) / w_u[indices]
                
    return np.mean(PF)