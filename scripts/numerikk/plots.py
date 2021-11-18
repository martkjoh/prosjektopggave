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
plt.rc('lines', lw=1.5)



# Make plots

def plot_alpha():
    fig, ax = plt.subplots()
    a_lo_list = get_alpha_lo()
    a_nlo_list = get_alpha_nlo()
    i = np.where(mu_list>=0)

    ax.plot(mu_list[i], a_nlo_list[i], "k-.", label=r"$\mathrm{NLO}$")
    ax.plot(mu_list[i], a_lo_list[i], "r--", label=r"$\mathrm{LO}$")
    plt.xlabel(r"$\mu_I/m_\pi$")
    plt.ylabel(r"$\alpha$")

    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/alpha.pdf")


def plot_masses():
    fig, ax = plt.subplots()
    mu_list = np.linspace(0, 2.5, 100)
    alpha_list = alpha_0(mu_list)
    m0 = lambda x, y : sqrt(lambdify((mu, a), lo(m0_sq), "numpy")(x, y))
    mp = lambda x, y : sqrt(lambdify((mu, a), lo(mp_sq), "numpy")(x, y))
    mm = lambda x, y : sqrt(lambdify((mu, a), lo(mm_sq), "numpy")(x+0j, y+0j))

    assert not np.sum(np.abs(mm(mu_list, alpha_list).imag) > 1e-6)
    
    plt.plot(mu_list, m0(mu_list, alpha_list), "-", color="tab:blue", label=r"$m_{0}$")
    plt.plot(mu_list, mp(mu_list, alpha_list), "r-.", label=r"$m_{+}$")
    plt.plot(mu_list, mm(mu_list, alpha_list).real, "k--",  label=r"$m_{-}$")

    plt.xlabel(r"$\mu_I/m_\pi$")
    plt.ylabel(r"$m/m_\pi$")

    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/masses.pdf")


def plot_free_energy_surface():
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(6, 6))

    N = 30
    d = 0.6
    a = np.linspace(-d, np.pi + d, N)
    mu = np.linspace(0, 2.5, N)
    a_lo_list = alpha_0(mu)
    MU, A = np.meshgrid(mu, a)
    FLO = F_0_2_lo(MU, A)
    X, Y, Z = MU, A, FLO

    ax.plot(mu, a_lo_list, F_0_2_lo(mu, a_lo_list) + 0.01, "-k", lw=2, alpha=1, zorder=10)
    ax.plot(mu, a_lo_list, np.min(FLO), "k--")

    surf = ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.7)
    surf = ax.plot_wireframe(X, Y, Z, color="black", lw=0.2)

    ax.azim=-35
    ax.elev=25

    plt.xlabel(r"$\mu_I/m_\pi$")
    ax.set_ylabel(r"$\alpha$")
    ax.set_zlabel(r"$\mathcal{F}/m_\pi^4$")
    ax.zaxis.set_tick_params(labelsize=10)
    ax.xaxis.set_tick_params(labelsize=10)
    ax.yaxis.set_tick_params(labelsize=10)

    plt.subplots_adjust(top=1, bottom=0, right=0.8, left=0, hspace=0, wspace=1)
    save_opt = dict(
        bbox_inches='tight',
        pad_inches = 0, 
        transparent=True, 
        dpi=300
    )
    plt.savefig("plots/free_energy_surface.pdf", **save_opt)


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

    PLO_label = r"${\mathrm{LO}}$"
    PNLO_label = r"${\mathrm{NLO}}$"
    ax.plot(mu_list[i::], PLO[i::], "k--", label=PLO_label)
    ax.plot(mu_list[i::], PNLO[i::], "r-.", label=PNLO_label)

    ax.set_xlabel(r"$\mu_I/m_\pi$")
    ax.set_ylabel(r"$P/m_\pi^4$")

    plt.legend()
    plt.tight_layout()

    plt.savefig("plots/pressure.pdf")


def plot_isospin_density():
    fig, ax = plt.subplots()    
    nILO_label = r"${\mathrm{LO}}$"
    nINLO_label = r"${\mathrm{NLO}}$"

    # nILO, nINLO, mu_list_diff = get_isospin_density()
    nILO, nINLO = get_isospin_density2()    
    ax.plot(mu_list, nILO, "k--", label=nILO_label)
    ax.plot(mu_list, nINLO, "r-.", label=nINLO_label)
    

    ax.set_xlabel(r"$\mu_I/m_\pi$")
    ax.set_ylabel(r"$n_I/m_\pi^3$")

    plt.legend()
    plt.tight_layout()

    plt.savefig("plots/isospin_density.pdf")

def plot_energy_density():
    fig, ax = plt.subplots()
    
    ELO, ENLO = get_energy_density()
    PLO, PNLO = get_pressure()

    ax.set_xlabel(r"$\mathcal{E}/m_\pi^4$")
    ax.set_ylabel(r"$P/m_\pi^4$")

    ELO_label = r"${\mathrm{LO}}$"
    ENLO_label = r"${\mathrm{NLO}}$"

    ax.plot(PLO, ELO, "k--", label=ELO_label)
    ax.plot(PNLO, ENLO, "r-.", label=ENLO_label)

    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/energy_density.pdf")



plot_alpha()
plot_masses()
plot_free_energy_surface()
plot_free_energy()
plot_pressure()
plot_isospin_density()
plot_energy_density()

# FLO, FNLO = get_free_energy()
# # plt.plot(mu_list, FLO - FNLO)
# n = 89
# # plt.plot(mu_list[n+1:], FNLO[n:-1], ".")
# # plt.plot(mu_list[n+1:], FNLO[n+1:], ".")

# plt.plot(mu_list[n:-1], FNLO[n] +  FNLO[n:-1] - FNLO[n+1:], ".")
# # plt.plot(mu_list[:-1], np.diff(FLO))
# # plt.plot(mu_list[:-1], np.diff(FNLO))
# plt.show()
