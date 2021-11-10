import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

from parameters import *
from functions import *
from free_energy_pions import *

# Makes the plots pretty

plt.rcParams['mathtext.fontset'] = 'cm'
font = {
    'family' : 'serif', 
    'size': 16
}
plt.rc('font', **font)
plt.rc('lines', lw=1)



# Make plots

def plot_alpha():
    a_lo_list = get_alpha_lo()
    a_nlo_list = get_alpha_nlo()
    i = np.where(mu_list>=1)
    print(mu_list)
    plt.plot(mu_list[i], a_lo_list[i], "r-")
    plt.plot(mu_list[i], a_nlo_list[i], "k-.")

    plt.savefig("plots/alpha.pdf")


def plot_masses():
    fig, ax = plt.subplots()
    mu_list = np.linspace(0, 2.5, 100)
    alpha_list = alpha_0(mu_list)
    m0 = lambda x, y : sqrt(lambdify((mu, a), m0_sq, "numpy")(x, y))
    mp = lambda x, y : sqrt(lambdify((mu, a), mp_sq, "numpy")(x, y))
    mm = lambda x, y : sqrt(lambdify((mu, a), mm_sq, "numpy")(x+0j, y+0j))

    assert not np.sum(np.abs(mm(mu_list, alpha_list).imag) > 1e-6)
    
    plt.plot(mu_list, m0(mu_list, alpha_list), label=r"$\pi_0$")
    plt.plot(mu_list, mp(mu_list, alpha_list), label=r"$\pi_+$")
    plt.plot(mu_list, mm(mu_list, alpha_list).real, label=r"$\pi_-$")

    plt.legend()
    plt.savefig("plots/masses.pdf")


def plot_free_energy_surface():
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    FLO, FNLO = get_free_energy_surface()
    a_lo_list = get_alpha_lo()

    plt.plot(mu_list, a_lo_list, np.min(FLO), "k--")
    # I = np.argmin(np.abs(A[:, :] - a_lo_list[None, :]), axis=0)
    plt.plot(mu_list, a_lo_list, F_0_2(mu_list, a_lo_list), "k--")
    ax.plot_surface(MU, A, FLO, cmap=cm.viridis, alpha=0.7)

    ax.azim=-55
    ax.elev=30

    plt.savefig("plots/free_energy_surface.pdf")
    

def plot_free_energy_surface_NLO():
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    FLO, FNLO = get_free_energy_surface()

    ax.plot_surface(MU[0:50], A[0:50], FNLO[0:50], cmap=cm.viridis, alpha=0.7)

    ax.azim=-60
    ax.elev=30

    plt.savefig("plots/free_energy_surface_NLO.pdf")


def plot_free_energy():
    fig, ax = plt.subplots()
    FLO, FNLO = get_free_energy()

    FLO_label = r"$\mathcal{F}^{\mathrm{LO}}$"
    FNLO_label = r"$\mathcal{F}^{\mathrm{NLO}}$"
    ax.plot(mu_list, FLO, "k--", label=FLO_label)
    ax.plot(mu_list, FNLO, "r-.", label=FNLO_label)

    plt.legend()
    plt.savefig("plots/free_energy_a_lo.pdf")


def plot_pressure():
    fig, ax = plt.subplots()
    PLO, PNLO = get_pressure()

    i =  np.where(mu_list>=1)[0][0]

    PLO_label = r"$\mathcal{P}^{\mathrm{LO}}$"
    PNLO_label = r"$\mathcal{P}^{\mathrm{NLO}}$"
    ax.plot(mu_list[i::], PLO[i::], "k--", label=PLO_label)
    ax.plot(mu_list[i::], PNLO[i::], "r--", label=PNLO_label)

    plt.legend()
    plt.savefig("plots/pressure_a_lo.pdf")


def plot_isospin_density():
    fig, ax = plt.subplots()
    
    nILO, nINLO, mu_list_diff = get_isospin_density()

    nILO_label = r"$n_I^{\mathrm{LO}}$"
    nINLO_label = r"$n_I^{\mathrm{NLO}}$"

    ax.plot(mu_list_diff, nILO, "k--", label=nILO_label)
    ax.plot(mu_list_diff, nINLO, "r-.", label=nINLO_label)
    ax.set_ylim(-0.1, 1.5)

    plt.legend()
    plt.savefig("plots/isospin_density_lo.pdf")

def plot_energy_density():
    fig, ax = plt.subplots()
    
    ELO, ENLO, mu_list_diff = get_energy_density()
    PLO, PNLO = get_pressure()

    ELO_label = r"$\mathcal{E}^{\mathrm{LO}}$"
    ENLO_label = r"$\mathcal{E}^{\mathrm{NLO}}$"

    ax.plot(PLO[1:-1], ELO, "k--", label=ELO_label)
    ax.plot(PNLO[1:-1], ENLO, "r-.", label=ENLO_label)

    plt.legend()
    plt.savefig("plots/energy_density_lo.pdf")



# plot_alpha()
# plot_masses()
plot_free_energy_surface()
# plot_free_energy_surface_NLO()
# plot_free_energy()
# plot_pressure()
# plot_isospin_density()
# plot_energy_density()
