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

def gen_F1():
    F1 = np.zeros_like(MU)
    for i, mu in enumerate(mu_list):
        for j, a in enumerate(a_list):
            F1[j, i] = F_0_2(mu, a)

    np.save("data/F_0_2", F1)

def gen_F2():
    F2 = np.zeros_like(MU)
    for i, mu in enumerate(mu_list):
        for j, a in enumerate(a_list):
            F2[j, i] = F_0_4(mu, a)

    np.save("data/F_0_4", F2)

def gen_F3():
    F3 = np.zeros_like(MU)
    for i, mu in enumerate(mu_list):
        for j, a in enumerate(a_list):
            F3[j, i] = F_fin(mu, a)

    np.save("data/F_fin", F3)


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
    # gen_F1()
    # gen_F2()
    gen_F3()
