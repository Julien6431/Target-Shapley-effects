# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 10:16:10 2021

@author: Julien Demange-Chryst
"""

#%% Modules

import numpy as np
import openturns as ot
import sys

sys.path.append("../cross_entropy/")
from CE_SG import CEIS_SG
sys.path.append("../shapley_estimators/")
from ROSA_shapley_effects import rosa_shapley_effects

#%% Setting

dim = 6
mu_FX,sigma_FX,_ = ot.LogNormalMuSigmaOverMu(556.8,0.08).evaluate()
distr_FX = ot.LogNormal(mu_FX,sigma_FX)
mu_FY,sigma_FY,_ = ot.LogNormalMuSigmaOverMu(453.6,0.08).evaluate()
distr_FY = ot.LogNormal(mu_FY,sigma_FY)
mu_E,sigma_E,_ = ot.LogNormalMuSigmaOverMu(200*10**9,0.06).evaluate()
distr_E = ot.LogNormal(mu_E,sigma_E)/(0.06*200*10**9)

distr_w = ot.Normal(0.062,0.1*0.062)
distr_t = ot.Normal(0.0987,0.1*0.0987)
distr_L = ot.Normal(4.29,0.1*4.29)

corr_matrix = ot.CorrelationMatrix(dim)
corr_matrix[3,4] = -0.55
corr_matrix[3,5] = 0.45
corr_matrix[4,5] = 0.45

Copule_normale = ot.NormalCopula(corr_matrix)
input_distr = ot.ComposedDistribution([distr_FX,distr_FY,distr_E,distr_w,distr_t,distr_L],Copule_normale)

def phi(x):
    FX,FY,E,w,t,L = x
    E = E*0.06*200*10**9
    D = 4*L**3/(E*w*t) * np.sqrt((FX/w**2)**2 + (FY/t**2)**2)
    return D

t = 0.066

#%% Failure probability and importance sampling auxiliary distribution: cross-entropy algorithm

N_ce = 5*10**4
p=0.25
Pr_SG, samples_SG, aux_distr_SG, N_tot_SG = CEIS_SG(N_ce,p,phi,t,input_distr)
print(f"Failure probability with CE-SG : {Pr_SG}")

#%% Boxplots given-model

n_rep = 2*10**2
N = 2*10**4
Nv = 10**4
No_mc = int(Nv/(2**dim-2)//3)
No_pf = int(Nv/(2**dim-2)//2)

shapleys_mc_isSG_box_cb = np.zeros((n_rep,dim))
shapleys_pf_isSG_box_cb = np.zeros((n_rep,dim))


for j in range(n_rep):
    shapleys_mc_isSG_box_cb[j,:],_ = rosa_shapley_effects(phi,t,input_distr,Nv=Nv,Nu=No_mc,Ni=3,m=10**4,withIS=True,aggregation="subset",type_estimator="dMC",aux_distr = aux_distr_SG)
    shapleys_pf_isSG_box_cb[j,:],_ = rosa_shapley_effects(phi,t,input_distr,Nv=Nv,Nu=No_pf,Ni=3,m=10**4,withIS=True,aggregation="subset",type_estimator="PF",aux_distr = aux_distr_SG)
    
    if (j+1)%(n_rep//10)==0:
        print("*",end="")
print(" - Ok non given-data")


#%% Save data

reference_values = np.load("data/ref_values_cantiler_beam.npz")

np.savez("data/Cantilever_beam_boxplots_ngd.npz",
         theo_values = reference_values['ref'],
         dmc_isSG=shapleys_mc_isSG_box_cb,
         pf_isSG=shapleys_pf_isSG_box_cb)
