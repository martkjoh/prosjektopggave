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

def gen_alpha_lo():
    a0_list = np.zeros_like(mu_list)

    for i, mu in enumerate(mu_list):
        F = lambda alpha: F_0_2(mu, alpha)
        minF = minimize_scalar(F, bounds=(0, pi), method="bounded")
        if not minF["success"]:
            raise Exception("Minimization failed")
        a0_list[i] = minF["x"]

    np.save("data/alpha_lo", a0_list)


########################
#! FREE ENERGY SURFACE #
########################

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



###########################
#! OPTMIZATION, 2nd ORDER #
###########################

def gen_alpha_nlo():
    a1_list = np.zeros_like(mu_list)
    a0_list = get_alpha_lo()
    for i, mu in enumerate(mu_list):
        F = lambda alpha: F_0_2(mu, alpha) + F_0_4(mu, alpha) + F_fin(mu, alpha)
        a0 = a0_list[i]
        minF = minimize_scalar(F, bounds=(a0, 1.01*a0), method="bounded")
        if not minF["success"]:
            raise Exception("Minimization failed")
        a1_list[i] = minF["x"]

    np.save("data/alpha_nlo.npy", a1_list)

########################
#! FREE ENERGY SURFACE #
########################

def gen_free_energy():
    # TODO: Get and use higer order alpha estimates
    alpha = np.load("data/alpha_lo.npy")
    F1, F2, F3 = np.empty_like(mu_list), np.empty_like(mu_list), np.empty_like(mu_list)
    # TODO: vectorize
    for i in range(len(mu_list)):
        mu, a = mu_list[i], alpha[i]
        F1[i] = F_0_2(mu, a)
        F2[i] = F_0_4(mu, a)
        F3[i] = F_fin(mu, a)

    np.save("data/F_0_2_lo_a.npy", F1)
    np.save("data/F_0_4_lo_a.npy", F2)
    np.save("data/F_fin_lo_a.npy", F3)


if __name__ == "__main__":
    pass
    gen_F1_surf()
    gen_F2_surf()
    gen_F3_surf()
    gen_free_energy()
    gen_alpha_lo()
    gen_alpha_nlo()