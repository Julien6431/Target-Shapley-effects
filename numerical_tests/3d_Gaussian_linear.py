# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 10:16:10 2021

@author: Julien Demange-Chryst
"""

#%% Modules

import numpy as np
import openturns as ot
import sys
from tqdm import tqdm

from LG_theoretical_values import compute_theoretical_values
sys.path.append("../cross_entropy/")
from CE_SG import CEIS_SG
sys.path.append("../shapley_estimators/")
from ROSA_shapley_effects import rosa_shapley_effects,rosa_shapley_effects_gd

#%% Setting

def phi(x):
    return np.sum(x)

n_rep = 2*10**2
N = 2*10**4
Nv = 10**4
p=.25
dim = 3
No = 10**3
No_mc = int(Nv/(2**dim-2)//3)
No_pf = int(Nv/(2**dim-2)//2)

rho = -.3
beta = np.ones(3)
mean_f = np.zeros(3)
cov_f = np.array([[1,0,0],
                 [0,1,rho],
                 [0,rho,1]])
t = 4
input_distr = ot.Normal(ot.Point(dim),ot.CovarianceMatrix(cov_f))

theoretical_shapleys_3,stds_3 = compute_theoretical_values(t,beta,mean_f,cov_f,N_sample=10**6,n_rep=20)

#%% Failure probability and importance sampling auxiliary distribution: cross-entropy algorithm

N_ce = 5*10**4
Pr, samples, aux_distr, N_tot = CEIS_SG(N_ce,p,phi,t,input_distr)

#%% Boxplots given-data

shapleys_mc_gd_box = np.zeros((n_rep,dim))
shapleys_mc_is_gd_box = np.zeros((n_rep,dim))
shapleys_pf_gd_box = np.zeros((n_rep,dim))
shapleys_pf_is_gd_box = np.zeros((n_rep,dim))


for j in tqdm(range(n_rep)):
    Xf = np.array(input_distr.getSample(N))
    Yf = np.apply_along_axis(lambda x:phi(x)>t,1,Xf)
    shapleys_mc_gd_box[j,:],_ = rosa_shapley_effects_gd(Xf,Yf,Nu=No,Ni=3,aggregation='subset',type_estimator="dMC",standardisation=True)
    shapleys_pf_gd_box[j,:],_ = rosa_shapley_effects_gd(Xf,Yf,Nu=No,aggregation="subset",type_estimator="PF",standardisation=True)
        
    Xg = np.array(aux_distr.getSample(N))
    Yg = np.apply_along_axis(lambda x:phi(x)>t,1,Xg)
    shapleys_mc_is_gd_box[j,:],_ = rosa_shapley_effects_gd(Xg,Yg,Nu=No,Ni=3,aggregation="subset",withIS=True,type_estimator="dMC",standardisation=True,init_distr=input_distr,aux_distr=aux_distr)
    shapleys_pf_is_gd_box[j,:],_ = rosa_shapley_effects_gd(Xg,Yg,Nu=No,aggregation="subset",withIS=True,type_estimator="PF",standardisation=True,init_distr=input_distr,aux_distr=aux_distr)


#%% Boxplots given-model

shapleys_mc_box = np.zeros((n_rep,dim))
shapleys_mc_is_box = np.zeros((n_rep,dim))
shapleys_pf_box = np.zeros((n_rep,dim))
shapleys_pf_is_box = np.zeros((n_rep,dim))

for j in tqdm(range(n_rep)):
    shapleys_mc_box[j,:],_ = rosa_shapley_effects(phi,t,input_distr,Nv=Nv,Nu=No_mc,Ni=3,aggregation="subset",type_estimator="dMC")
    shapleys_pf_box[j,:],_ = rosa_shapley_effects(phi,t,input_distr,Nv=Nv,Nu=No_pf,aggregation="subset",type_estimator="PF")
    
    shapleys_mc_is_box[j,:],_ = rosa_shapley_effects(phi,t,input_distr,Nv=Nv,Nu=No_mc,Ni=3,aggregation="subset",type_estimator="dMC",withIS=True,aux_distr=aux_distr)
    shapleys_pf_is_box[j,:],_ = rosa_shapley_effects(phi,t,input_distr,Nv=Nv,Nu=No_pf,aggregation="subset",type_estimator="PF",withIS=True,aux_distr=aux_distr)

#%% Save data

np.savez("data/3d_Gaussian_linear_boxplots.npz",
         theo_values = theoretical_shapleys_3,
         dmc=shapleys_mc_box,
         pf=shapleys_pf_box,
         dmc_isSG=shapleys_mc_is_box,
         pf_isSG=shapleys_pf_is_box,
         dmc_gd=shapleys_mc_gd_box,
         pf_gd=shapleys_pf_gd_box,
         dmc_isSG_gd=shapleys_mc_is_gd_box,
         pf_isSG_gd=shapleys_pf_is_gd_box)
