"""
    Genereates all the data related to pions
"""
import numpy as np
from scipy.optimize import minimize_scalar
from numpy import pi

from functions import *
from parameters import *


###########################
#! OPTMIZATION, 1st ORDER #
###########################

a0_list = np.zeros_like(mu_list)

for i, mu in enumerate(mu_list):
    F = lambda alpha: F_0_2(mu, alpha)
    minF = minimize_scalar(F, bounds=(0, pi), method="bounded")
    if not minF["success"]:
        raise Exception("Minimization failed")
    a0_list[i] = minF["x"]

np.save("data/alpha_lo_numerical", a0_list)


################
#! FREE ENERGY #
################

def gen_F1_surf():
    F1 = np.zeros_like(MU)
    for i, mu in enumerate(mu_list):
        for j, a in enumerate(a_list):
            F1[j, i] = F_0_2(mu, a)

    np.save("data/F_0_2", F1)

def gen_F2_surf():
    F2 = np.zeros_like(MU)
    for i, mu in enumerate(mu_list):
        for j, a in enumerate(a_list):
            F2[j, i] = F_0_4(mu, a)

    np.save("data/F_0_4", F2)

def gen_F3_surf():
    F3 = np.zeros_like(MU)
    for i, mu in enumerate(mu_list):
        for j, a in enumerate(a_list):
            F3[j, i] = F_fin(mu, a)

    np.save("data/F_fin", F3)

def gen_F_afo_mu_lo():
    """ generates F_0_2 using the lowest order approximation of alpha """
    a_list = np.load("data/alpha_lo_numerical.npy")
    F1 = F_0_2(mu_list, a_list)
    np.save("data/F_0_2_lo_a", F1)


###########################
#! OPTMIZATION, 2nd ORDER #
###########################

a1_list = np.zeros_like(mu_list)

for i, mu in enumerate(mu_list):
    F1 = np.load("data/F_0_2.npy")
    F2 = np.load("data/F_0_4.npy")
    F3 = np.load("data/F_fin.npy")
    F = lambda alpha: F_0_2(mu, alpha)
    minF = minimize_scalar(F, bounds=(0, pi), method="bounded")
    if not minF["success"]:
        raise Exception("Minimization failed")
    a0_list[i] = minF["x"]



if __name__ == "__main__":
    pass
    # gen_F1_surf()
    # gen_F2_surf()
    # gen_F3_surf()
    gen_F_afo_mu_lo()