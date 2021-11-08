import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

from parameters import *
from functions import *


# Makes the plots pretty
plt.rcParams['mathtext.fontset'] = 'cm'
font = {
    'family' : 'serif', 
    'size': 16
}
plt.rc('font', **font)
plt.rc('lines', lw=2)


# plt.plot(mu_list, a0_list)
# plt.plot(mu_list, alpha_0(mu_list), "k--")
# plt.show()


#################
# !SURFACE PLOT #
#################


fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

F1 = np.load("data/F_0_2.npy")
F2 = np.load("data/F_0_4.npy")
F3 = np.load("data/F_fin.npy")

# a0_list = np.load("data/alpha_lo_numerical.npy")
# plt.plot(mu_list, a0_list, np.min(F1), "k--")

# ax.plot_surface(MU, A, F1, cmap=cm.viridis, alpha=0.5)
# ax.scatter3D(MU, A, F2)
# ax.scatter3D(MU, A, F3)

# plt.show()
# print(F3)

# fig, ax = plt.subplots()

# p = np.linspace(0, 100, 10_000)

# dF = dF_fin(p, 1, 1.5)
# plt.plot(p, dF)
# plt.show()
# print(dF_fin(0, 1, 1.5))

def plot_masses():
    fig, ax = plt.subplots()
    mu_list = np.linspace(0, 2.5, 100)
    alpha_list = alpha_0(mu_list)

    plt.plot(mu_list, sqrt(m0sq(mu_list, alpha_list)))
    plt.plot(mu_list, sqrt(mp_sq(mu_list, alpha_list)))
    plt.plot(mu_list, sqrt(mm_sq(mu_list, alpha_list)))
    plt.savefig("plots/masses.pdf")


plot_masses()