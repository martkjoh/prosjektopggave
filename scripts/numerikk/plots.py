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


def plot_alpha0():
    a0_list = np.load("data/alpha_lo_numerical.npy")
    plt.plot(mu_list, a0_list, "k--")
    # plt.plot(mu_list, alpha_0(mu_list), "k--")
    plt.savefig("plots/alpha0.pdf")


def plot_masses():
    fig, ax = plt.subplots()
    mu_list = np.linspace(0, 2.5, 100)
    alpha_list = alpha_0(mu_list)

    plt.plot(mu_list, sqrt(m0sq(mu_list, alpha_list)))
    plt.plot(mu_list, sqrt(mp_sq(mu_list, alpha_list)))
    plt.plot(mu_list, sqrt(mm_sq(mu_list, alpha_list)))

    plt.savefig("plots/masses.pdf")


def plot_free_energy_surface():
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    F1 = np.load("data/F_0_2.npy")
    F2 = np.load("data/F_0_4.npy")
    F3 = np.load("data/F_fin.npy")

    a0_list = np.load("data/alpha_lo_numerical.npy")
    plt.plot(mu_list, a0_list, np.min(F1), "k--")


    ax.plot_surface(MU, A, F1, cmap=cm.viridis, alpha=0.7)
    ax.scatter3D(MU, A, F2)
    ax.scatter3D(MU, A, F3)

    ax.azim=-60
    ax.elev=30

    plt.savefig("plots/free_energy_surface.pdf")


def plot_free_energy():
    fig, ax = plt.subplots()
    F = np.load("data/F_0_2_lo_a.npy")
    plt.plot(mu_list, F)
    plt.savefig("plots/free_energy_a_lo.pdf")

# plot_alpha0()
# plot_masses()
# plot_free_energy_surface()

# MU
# A
# F3 = np.load("data/F_fin.npy")

# print(F3)
# print(dF_fin(0, 2.5, 1))
# print(Em_sq(0, 1.5, 1))
plot_free_energy()