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

sys.path.append("../cross_entropy/")
from CE_SG import CEIS_SG
from CE_GM import CEIS_GM
sys.path.append("../shapley_estimators/")
from ROSA_shapley_effects import rosa_shapley_effects,rosa_shapley_effects_gd

#%% Setting

dim = 6
mu_FX,sigma_FX,_ = ot.LogNormalMuSigmaOverMu(556.8,0.08).evaluate()
distr_FX = ot.LogNormal(mu_FX,sigma_FX)
mu_FY,sigma_FY,_ = ot.LogNormalMuSigmaOverMu(453.6,0.08).evaluate()
distr_FY = ot.LogNormal(mu_FY,sigma_FY)
mu_E,sigma_E,_ = ot.LogNormalMuSigmaOverMu(200*10**9,0.06).evaluate()
distr_E = ot.LogNormal(mu_E,sigma_E)

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
    D = 4*L**3/(E*w*t) * np.sqrt((FX/w**2)**2 + (FY/t**2)**2)
    return D

t = 0.066

#%% Failure probability and importance sampling auxiliary distribution: cross-entropy algorithm

N_ce = 5*10**4
p=0.25
Pr_SG, samples_SG, aux_distr_SG, N_tot_SG = CEIS_SG(N_ce,p,phi,t,input_distr)
print(f"Failure probability with CE-SG : {Pr_SG}")
Pr_GM, samples_GM, aux_distr_GM, N_tot_GM, k_fin = CEIS_GM(N_ce,p,phi,t,input_distr)
print(f"Failure probability with CE-GM : {Pr_GM}")
print(f"Number of Gaussian in the mixture : {k_fin}")


#%% Boxplots given-data

n_rep = 2*10**2
N = 2*10**4
No = 10**3

shapleys_mc_box = np.zeros((n_rep,dim))
shapleys_mc_isSG_box = np.zeros((n_rep,dim))
shapleys_mc_isGM_box = np.zeros((n_rep,dim))
shapleys_pf_box = np.zeros((n_rep,dim))
shapleys_pf_isSG_box = np.zeros((n_rep,dim))
shapleys_pf_isGM_box = np.zeros((n_rep,dim))


for j in tqdm(range(n_rep)):
    Xf = np.array(input_distr.getSample(N))
    Yf = np.apply_along_axis(lambda x:phi(x)>t,1,Xf)
    shapleys_mc_box[j,:],_ = rosa_shapley_effects_gd(Xf,Yf,Nu=No,Ni=3,aggregation='subset',type_estimator="dMC")
    shapleys_pf_box[j,:],_ = rosa_shapley_effects_gd(Xf,Yf,Nu=No,aggregation="subset",type_estimator="PF")
        
    Xg_SG = np.array(aux_distr_SG.getSample(N))
    Yg_SG = np.apply_along_axis(lambda x:phi(x)>t,1,Xg_SG)
    shapleys_mc_isSG_box[j,:],_ = rosa_shapley_effects_gd(Xg_SG,Yg_SG,Nu=No,Ni=3,aggregation="subset",withIS=True,type_estimator="dMC",init_distr=input_distr,aux_distr=aux_distr_SG)
    shapleys_pf_isSG_box[j,:],_ = rosa_shapley_effects_gd(Xg_SG,Yg_SG,Nu=No,aggregation="subset",withIS=True,type_estimator="PF",init_distr=input_distr,aux_distr=aux_distr_SG)
    
    Xg_GM = np.array(aux_distr_GM.getSample(N))
    Yg_GM = np.apply_along_axis(lambda x:phi(x)>t,1,Xg_GM)
    shapleys_mc_isGM_box[j,:],_ = rosa_shapley_effects_gd(Xg_SG,Yg_SG,Nu=No,Ni=3,aggregation="subset",withIS=True,type_estimator="dMC",init_distr=input_distr,aux_distr=aux_distr_GM)
    shapleys_pf_isGM_box[j,:],_ = rosa_shapley_effects_gd(Xg_SG,Yg_SG,Nu=No,aggregation="subset",withIS=True,type_estimator="PF",init_distr=input_distr,aux_distr=aux_distr_GM)


#%% Save data

reference_values = np.load("data/ref_values_cantiler_beam.npz")

np.savez("data/Cantilever_beam_boxplots.npz",
         theo_values = reference_values['ref'],
         dmc_gd=shapleys_mc_box,
         pf_gd=shapleys_pf_box,
         dmc_isSG_gd=shapleys_mc_isSG_box,
         pf_isSG_gd=shapleys_pf_isSG_box,
         dmc_isGM_gd=shapleys_mc_isGM_box,
         pf_isGM_gd=shapleys_pf_isGM_box)



#%% Boxplots given-data with standardisation

n_rep = 2*10**2
N = 2*10**4
No = 10**3

shapleys_mc_box_norm = np.zeros((n_rep,dim))
shapleys_mc_isSG_box_norm = np.zeros((n_rep,dim))
shapleys_mc_isGM_box_norm = np.zeros((n_rep,dim))
shapleys_pf_box_norm = np.zeros((n_rep,dim))
shapleys_pf_isSG_box_norm = np.zeros((n_rep,dim))
shapleys_pf_isGM_box_norm = np.zeros((n_rep,dim))


for j in tqdm(range(n_rep)):
    Xf = np.array(input_distr.getSample(N))
    Yf = np.apply_along_axis(lambda x:phi(x)>t,1,Xf)
    shapleys_mc_box_norm[j,:],_ = rosa_shapley_effects_gd(Xf,Yf,Nu=No,Ni=3,aggregation='subset',type_estimator="dMC",standardisation=True)
    shapleys_pf_box_norm[j,:],_ = rosa_shapley_effects_gd(Xf,Yf,Nu=No,aggregation="subset",type_estimator="PF",standardisation=True)
        
    Xg_SG = np.array(aux_distr_SG.getSample(N))
    Yg_SG = np.apply_along_axis(lambda x:phi(x)>t,1,Xg_SG)
    shapleys_mc_isSG_box_norm[j,:],_ = rosa_shapley_effects_gd(Xg_SG,Yg_SG,Nu=No,Ni=3,aggregation="subset",withIS=True,type_estimator="dMC",standardisation=True,init_distr=input_distr,aux_distr=aux_distr_SG)
    shapleys_pf_isSG_box_norm[j,:],_ = rosa_shapley_effects_gd(Xg_SG,Yg_SG,Nu=No,aggregation="subset",withIS=True,type_estimator="PF",standardisation=True,init_distr=input_distr,aux_distr=aux_distr_SG)
    
    Xg_GM = np.array(aux_distr_GM.getSample(N))
    Yg_GM = np.apply_along_axis(lambda x:phi(x)>t,1,Xg_GM)
    shapleys_mc_isGM_box_norm[j,:],_ = rosa_shapley_effects_gd(Xg_SG,Yg_SG,Nu=No,Ni=3,aggregation="subset",withIS=True,type_estimator="dMC",standardisation=True,init_distr=input_distr,aux_distr=aux_distr_GM)
    shapleys_pf_isGM_box_norm[j,:],_ = rosa_shapley_effects_gd(Xg_SG,Yg_SG,Nu=No,aggregation="subset",withIS=True,type_estimator="PF",standardisation=True,init_distr=input_distr,aux_distr=aux_distr_GM)


#%% Save data with standardisation


reference_values = np.load("data/ref_values_cantiler_beam.npz")

np.savez("data/Cantilever_beam_boxplots_std.npz",
         theo_values = reference_values['ref'],
         dmc_gd=shapleys_mc_box_norm,
         pf_gd=shapleys_pf_box_norm,
         dmc_isSG_gd=shapleys_mc_isSG_box_norm,
         pf_isSG_gd=shapleys_pf_isSG_box_norm,
         dmc_isGM_gd=shapleys_mc_isGM_box_norm,
         pf_isGM_gd=shapleys_pf_isGM_box_norm)


#%% Boxplots given-model

n_rep = 2*10**2
N = 2*10**4
Nv = 10**4
No_mc = int(Nv/(2**dim-2)//3)
No_pf = int(Nv/(2**dim-2)//2)

shapleys_mc_isSG_box_gm = np.zeros((n_rep,dim))
shapleys_pf_isSG_box_gm = np.zeros((n_rep,dim))


for j in range(n_rep):
    shapleys_mc_isSG_box_gm[j,:],_ = rosa_shapley_effects(phi,t,input_distr,Nv=Nv,Nu=No_mc,Ni=3,m=10**4,withIS=True,aggregation="subset",type_estimator="dMC",aux_distr = aux_distr_SG)
    shapleys_pf_isSG_box_gm[j,:],_ = rosa_shapley_effects(phi,t,input_distr,Nv=Nv,Nu=No_pf,Ni=3,m=10**4,withIS=True,aggregation="subset",type_estimator="PF",aux_distr = aux_distr_SG)
    
    if (j+1)%(n_rep//10)==0:
        print("*",end="")
print(" - Ok non given-data")


#%% Save data

reference_values = np.load("data/ref_values_cantiler_beam.npz")

np.savez("data/Cantilever_beam_boxplots_gm.npz",
         theo_values = reference_values['ref'],
         dmc_isSG=shapleys_mc_isSG_box_gm,
         pf_isSG=shapleys_pf_isSG_box_gm)
