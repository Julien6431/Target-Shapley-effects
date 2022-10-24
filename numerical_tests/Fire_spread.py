# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 09:08:25 2022

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

#%% Conversion

def lb2kg(x):
    return 0.4535924*x

def kg2lb(x):
    return 2.204622*x

def ft2m(x):
    return 0.3047995*x

def m2ft(x):
    return 3.28083*x

#%% Input

dim = 10

distr_delta = ot.LogNormal(2.19,0.517)
distr_sigma = ot.TruncatedDistribution(ot.LogNormal(3.31,0.294),3/0.6,ot.TruncatedDistribution.LOWER)
distr_h = ot.LogNormal(8.48,0.063)
distr_rho_p = ot.LogNormal(-0.592,0.219)
distr_m_l = ot.TruncatedDistribution(ot.Normal(1.18,0.377),0,ot.TruncatedDistribution.LOWER)
distr_m_d = ot.Normal(0.19,0.047)
distr_S_t = ot.TruncatedDistribution(ot.Normal(0.049,0.011),0,ot.TruncatedDistribution.LOWER)
distr_U = 6.9*ot.LogNormal(1.0174,0.5569)
distr_tan_phi = ot.TruncatedDistribution(ot.Normal(0.38,0.186),0,ot.TruncatedDistribution.LOWER)
distr_P = ot.TruncatedDistribution(ot.LogNormal(-2.19,0.64),1,ot.TruncatedDistribution.UPPER)

corr_matrix = ot.CorrelationMatrix(dim)
corr_matrix[5,7] = -0.8

Copule_normale = ot.NormalCopula(corr_matrix)
input_distr = ot.ComposedDistribution([distr_delta,distr_sigma,distr_h,distr_rho_p,distr_m_l,
                                      distr_m_d,distr_S_t,distr_U,distr_tan_phi,distr_P],Copule_normale)

#%% Fire spread model

def phi(x,input_distr=input_distr):
    
    if input_distr.computePDF(x) == 0:
        return 0
    
    delta,sigma,h,rho_p,m_l,m_d,S_t,U,tan_phi,P = x
    
    if delta<=0:
        return 0
    if h <= 0:
        return 0
    if rho_p <= 0:
        return 0
    if S_t>=1:
        return 0
    if U<=0:
        return 0
    if P<=0:
        return 0
    
    w_0 = 4.8/4.8824 * 1/(1+np.exp((15-delta)/3.5))
    w_0 = kg2lb(w_0)*0.09290272
    
    
    delta = m2ft(delta/100)
    sigma = 1/m2ft(0.01/sigma)
    rho_p = kg2lb(rho_p)*28.3167
    h = 4184*h/1055.87 * 0.4235924
    U = m2ft(U*10**3)/60
    
    Gamma_max = sigma**(3/2)/(495+0.0594*sigma**(3/2))
    beta_op = 3.348*sigma**(-0.8189)
    A = 133*sigma**(-0.7913)
    theta_star = (301.4 - 305.87*(m_l-m_d) + 2260*m_d)/m_l/2260
    theta = min(1,max(theta_star,0))
    mu_m = np.exp(-7.3*P*m_d - (7.3*theta+2.13)*(1-P)*m_l)
    mu_S = min(0.174*S_t**(-0.19),1)
    C = 7.47*np.exp(-0.133*sigma**(0.55))
    B = 0.02526*sigma**(0.54)
    E = 0.715*np.exp(-3.59*10**(-4)* sigma)
    w_n = w_0*(1-S_t)
    rho_b = w_0/delta
    epsilon = np.exp(-138/sigma)
    Q_ig = 130.87 + 1054.43*m_d
    beta = rho_b/rho_p
    Gamma = Gamma_max * (beta/beta_op) ** A * np.exp(A*(1 - beta/beta_op))
    
    in_exp = (0.792+0.681*sigma**(0.5))*(beta+0.1)
    in_exp = min(in_exp,709.78)
    
    ksi = np.exp(in_exp) / (192 + 0.2595*sigma)
    phi_W = C*U**B * (beta/beta_op)**(-E)
    phi_S = 5.275*beta**(-0.3)*tan_phi**2
    I_R = Gamma*w_n*h*mu_m*mu_S
    
    return ft2m((I_R*ksi*(1+phi_W+phi_S))/(rho_b*epsilon*Q_ig))*100/60


t = 60

#%% Failure probability and importance sampling auxiliary distribution: cross-entropy algorithm

N_ce = 10**5
p=0.25

Pr_SG, samples_SG, aux_distr_SG, N_tot_SG = CEIS_SG(N_ce,p,phi,t,input_distr)
print(f"Failure probability with CE-SG : {Pr_SG}")

Pr_GM, samples_GM, aux_distr_GM, N_tot_GM, k_fin = CEIS_GM(N_ce,p,phi,t,input_distr)
print(f"Failure probability with CE-GM : {Pr_GM}")
print(f"Number of Gaussian in the mixture : {k_fin}")


#%% Boxplots given-data with standardisation

n_rep = 2*10**2
N = 2*10**4
No = 10**3

shapleys_mc_isSG_box_norm = np.zeros((n_rep,dim))
shapleys_mc_isGM_box_norm = np.zeros((n_rep,dim))
shapleys_pf_isSG_box_norm = np.zeros((n_rep,dim))
shapleys_pf_isGM_box_norm = np.zeros((n_rep,dim))


for j in tqdm(range(n_rep)):
    Xg_SG = np.array(aux_distr_SG.getSample(N))
    Yg_SG = np.apply_along_axis(lambda x:phi(x)>t,1,Xg_SG)
    shapleys_mc_isSG_box_norm[j,:],_ = rosa_shapley_effects_gd(Xg_SG,Yg_SG,Nu=No,Ni=2,aggregation="subset",withIS=True,type_estimator="dMC",standardisation=True,init_distr=input_distr,aux_distr=aux_distr_SG)
    shapleys_pf_isSG_box_norm[j,:],_ = rosa_shapley_effects_gd(Xg_SG,Yg_SG,Nu=No,aggregation="subset",withIS=True,type_estimator="PF",standardisation=True,init_distr=input_distr,aux_distr=aux_distr_SG)

    
    Xg_GM = np.array(aux_distr_GM.getSample(N))
    Yg_GM = np.apply_along_axis(lambda x:phi(x)>t,1,Xg_GM)
    shapleys_mc_isGM_box_norm[j,:],_ = rosa_shapley_effects_gd(Xg_GM,Yg_GM,Nu=No,Ni=2,aggregation="subset",withIS=True,type_estimator="dMC",standardisation=True,init_distr=input_distr,aux_distr=aux_distr_GM)
    shapleys_pf_isGM_box_norm[j,:],_ = rosa_shapley_effects_gd(Xg_GM,Yg_GM,Nu=No,aggregation="subset",withIS=True,type_estimator="PF",standardisation=True,init_distr=input_distr,aux_distr=aux_distr_GM)

#%% Save data

reference_values = np.load("data/ref_values_fire_spread.npz")

np.savez("data/Fire_spread_boxplots.npz",
         theo_values = reference_values['ref'],
         dmc_isSG_gd=shapleys_mc_isSG_box_norm,
         pf_isSG_gd=shapleys_pf_isSG_box_norm,
         dmc_isGM_gd=shapleys_mc_isGM_box_norm,
         pf_isGM_gd=shapleys_pf_isGM_box_norm)



#%% Boxplots given-model

n_rep = 1*10**2
N = 2*10**4
Nv = 10**4
No_mc = int(np.floor(Nv/(3*(2**dim-2))))
No_pf = int(np.floor(Nv/(2*(2**dim-2))))

shapleys_mc_isSG_box_gm = np.zeros((n_rep,dim))
shapleys_pf_isSG_box_gm = np.zeros((n_rep,dim))


for j in tqdm(range(n_rep)):
    shapleys_mc_isSG_box_gm[j,:],_ = rosa_shapley_effects(phi,t,input_distr,Nv=Nv,Nu=No_mc,Ni=3,m=10**4,withIS=True,aggregation="subset",type_estimator="dMC",aux_distr = aux_distr_SG)
    shapleys_pf_isSG_box_gm[j,:],_ = rosa_shapley_effects(phi,t,input_distr,Nv=Nv,Nu=No_pf,Ni=3,m=10**4,withIS=True,aggregation="subset",type_estimator="PF",aux_distr = aux_distr_SG)


#%% Save data

reference_values = np.load("data/ref_values_fire_spread.npz")

np.savez("data/Fire_spread_boxplots_gm.npz",
         theo_values = reference_values['ref'],
         dmc_isSG=shapleys_mc_isSG_box_gm,
         pf_isSG=shapleys_pf_isSG_box_gm)