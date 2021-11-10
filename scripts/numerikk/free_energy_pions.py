"""
    Genereates all the data related to pions
"""
import numpy as np
from scipy.optimize import minimize_scalar, fsolve, root, brentq
from numpy import pi

from functions import *
from parameters import *


###########################
#! OPTMIZATION, 1st ORDER #
###########################

def gen_alpha_lo():
    """ Use the analytical result for ɑ_nlo """
    np.save("data/alpha_lo", alpha_0(mu_list))


########################
#! FREE ENERGY SURFACE #
########################

def gen_F_surf():
    """ generates a surface of free energy on the mu-alpha plane """
    np.save("data/F_0_2", F_0_2(MU, A))
    np.save("data/F_0_4", F_0_4(MU, A))
    np.save("data/F_fin", F_fin(MU, A))



###########################
#! OPTMIZATION, 2nd ORDER #
###########################

## Old version, using numerical minimization
# def gen_alpha_nlo():
#     a1_list = np.zeros_like(mu_list)
#     a0_list = get_alpha_lo()
#     for i, mu in enumerate(mu_list):
#         F = lambda alpha: F_0_2(mu, alpha) + F_0_4(mu, alpha) + F_fin(mu, alpha)
#         a0 = a0_list[i]
#         minF = minimize_scalar(F, bracket=(0, np.pi/2), method="brent")
#         print(minF)
#         if not minF["success"]:
#             raise Exception("Minimization failed")
#         a1_list[i] = minF["x"]

#     np.save("data/alpha_nlo.npy", a1_list)


# new version, using an expression for the derivative
def gen_alpha_nlo():
    """ find alpha that minimizes F, given a value of mu """
    a1_list = np.zeros_like(mu_list)
    a0_list = get_alpha_lo()
    for i, mu in enumerate(mu_list):
        print(i)
        F = lambda alpha: F_diff_a(mu, alpha)
        a0 = a0_list[i]
        minF = fsolve(F, a0, full_output=True)
        # print(minF)
        a1_list[i] = minF[0][0]

    np.save("data/alpha_nlo.npy", a1_list)


################
#! FREE ENERGY #
################

def gen_free_energy():
    alpha = np.load("data/alpha_nlo.npy")

    np.save("data/F_0_2_lo_a.npy", F_0_2(mu_list, alpha))

    alpha = np.load("data/alpha_nlo.npy")

    np.save("data/F_0_2_nlo_a.npy", F_0_2(mu_list, alpha))
    np.save("data/F_0_4_nlo_a.npy", F_0_4(mu_list, alpha))
    np.save("data/F_fin_nlo_a.npy", F_fin(mu_list, alpha))


if __name__ == "__main__":
    # Run this script to generate all data used
    pass
    gen_alpha_lo()
    gen_alpha_nlo()
    gen_F_surf()
    gen_free_energy()
