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
plt.rc('lines', lw=2)



# Make plots
fs = (6.5, 4)
adj = dict(top=0.96, bottom=0.15, right=0.99, left=0.13, hspace=0, wspace=0)

LO_label = r"${\mathrm{LO}}$"
LO_style = "k--"
NLO_label = r"${\mathrm{NLO}}$"
NLO_style = "r-."


def plot_alpha():
    fig, ax = plt.subplots(figsize=fs)
    a_lo_list = get_alpha_lo()
    a_nlo_list = get_alpha_nlo()
    i = np.where(mu_list>=0)

    ax.plot(mu_list[i], a_lo_list[i], LO_style, label=LO_label)
    ax.plot(mu_list[i], a_nlo_list[i], NLO_style, label=NLO_label)
    plt.xlabel(r"$\mu_I/m_\pi$")
    plt.ylabel(r"$\alpha$")
    ax.set_ylim(ax.set_ylim()[0], 1.5)
    
    fig.subplots_adjust(**adj)
    plt.legend()
    plt.savefig("plots/alpha.pdf")


def plot_masses():
    fig, ax = plt.subplots(figsize=fs)
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
    fig.subplots_adjust(**adj)
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
    fig, ax = plt.subplots(figsize=fs)
    FLO, FNLO = get_free_energy()

    ax.plot(mu_list, FLO, LO_style, label=LO_label)
    ax.plot(mu_list, FNLO, NLO_style, label=NLO_label)

    plt.legend()
    plt.savefig("plots/free_energy_a_lo.pdf")


def plot_pressure():
    fig, ax = plt.subplots(figsize=fs)
    PLO, PNLO = get_pressure()

    ax.plot(mu_list, PLO, LO_style, label=LO_label)
    ax.plot(mu_list, PNLO, NLO_style, label=NLO_label)

    ax.set_xlabel(r"$\mu_I/m_\pi$")
    ax.set_ylabel(r"$P/m_\pi^4$")

    plt.legend()
    fig.subplots_adjust(**adj)
    plt.savefig("plots/pressure.pdf")


def plot_isospin_density():
    fig, ax = plt.subplots(figsize=fs)

    nILO, nINLO = get_isospin_density2()    
    ax.plot(mu_list, nILO, LO_style, label=LO_label)
    ax.plot(mu_list, nINLO, NLO_style, label=NLO_label)
    

    ax.set_xlabel(r"$\mu_I/m_\pi$")
    ax.set_ylabel(r"$n_I/m_\pi^3$")

    plt.legend()
    fig.subplots_adjust(**adj)
    plt.savefig("plots/isospin_density.pdf")


def plot_eos():
    fig, ax = plt.subplots(figsize=fs)
    
    ELO, ENLO = get_energy_density()
    PLO, PNLO = get_pressure()

    ax.set_xlabel(r"$u/m_\pi^4$")
    ax.set_ylabel(r"$P/m_\pi^4$")

    ax.plot(PLO, ELO, LO_style, label=LO_label)
    ax.plot(PNLO, ENLO, NLO_style, label=NLO_label)

    plt.legend()
    fig.subplots_adjust(**adj)
    plt.savefig("plots/eos.pdf")


def plot_phase():
    N = 30
    a = np.linspace(-0.05, 0.8, N)
    
    fig, ax = plt.subplots(figsize=(12, 6))

    F0 = F_0_2_lo(0, 0)
    ax.plot(a, F_0_2_lo(0.9, a) - F0, "royalblue", label=r"$\mu_I<m_\pi$")
    ax.plot(a, F_0_2_lo(1, a) - F0, "k--", label=r"$\mu_I=m_\pi$")
    ax.plot(a, F_0_2_lo(1.1, a) - F0, "k", label=r"$\mu_I>m_\pi$")
    ax.scatter((0, 0.59), (0.0012, -0.0075), s=400)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$(\mathcal{F} - \mathcal{F}_0)/m_\pi$")

    plt.tight_layout()
    plt.legend()
    plt.savefig("plots/phase_transition.pdf")


def plot_free_energy_surface_wo_axis():
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(8, 8))

    N = 30
    d = 0.6
    a = np.linspace(-d, np.pi + d, N)
    mu = np.linspace(0, 2.5, N)
    a_lo_list = alpha_0(mu)
    MU, A = np.meshgrid(mu, a)
    FLO = F_0_2_lo(MU, A)
    X, Y, Z = MU, A, FLO

    ax.plot(mu, a_lo_list, F_0_2_lo(mu, a_lo_list) + 0.01, "-k", lw=2, alpha=1, zorder=10)
    surf = ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.7)
    surf = ax.plot_wireframe(X, Y, Z, color="black", lw=0.2)

    ax.azim=-35
    ax.elev=25

    # Hide grid lines
    ax.grid(False)
    ax.axis("off")

    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    save_opt = dict(
        bbox_inches='tight',
        pad_inches = -0.5,
        transparent=True, 
        dpi=300
    )
    plt.savefig("plots/free_energy_surface_wo_axis.pdf", **save_opt)


F = F_0_2_symb - 4/(24) * F_0_2_symb.diff(a, 4) * a**4
F = lambdify((mu, a), F.subs(m, 1.).subs(f, fpi), "numpy")

def plot_free_energy_surface2():
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10, 6))

    N = 500
    a = np.linspace(-0.1, 0.8, N)
    mu = np.linspace(0.9, 1.1, N)
    a_lo_list = alpha_0(mu)
    MU, A = np.meshgrid(mu, a)
    FLO = F(MU, A)
    X, Y, Z = MU, A, FLO

    mx = np.argmin(Z, axis=0)

    skip = 20
    surf = ax.plot_surface(X[::skip, ::skip], Y[::skip, ::skip], Z[::skip, ::skip], cmap="viridis", alpha=0.7)
    surf = ax.plot_wireframe(X[::skip, ::skip], Y[::skip, ::skip], Z[::skip, ::skip], color="black", lw=0.2)
    zmin = [Z[m, i] for i, m in enumerate(mx)]
    ax.plot(X[0, :-skip], Y[mx[:-skip], 0], zmin[:-skip], "k.",  zorder=3, markersize=2)

    plt.xlabel(r"$\mu_I/m_\pi$")
    ax.set_ylabel(r"$\alpha$")
    ax.set_zlabel(r"$\mathcal{F}/m_\pi^4$")
    ax.zaxis.set_tick_params(labelsize=8)
    ax.xaxis.set_tick_params(labelsize=8)
    ax.yaxis.set_tick_params(labelsize=8)

    plt.subplots_adjust(top=1, bottom=0, right=0.9, left=0, hspace=0, wspace=1)
    save_opt = dict(
        bbox_inches='tight',
        pad_inches = 0, 
        transparent=True, 
        dpi=300,
    )
    plt.savefig("plots/free_energy_surface2.pdf", **save_opt)


# Self explanatory, really

plot_alpha()
plot_masses()
plot_free_energy_surface()
plot_free_energy()
plot_pressure()
plot_isospin_density()
plot_eos()
plot_phase()
plot_free_energy_surface_wo_axis()
plot_free_energy_surface2()


# Find largest deviance in NLO-result

ELO, ENLO = get_energy_density()
PLO, PNLO = get_pressure()
i =  np.where(mu_list>1)[0][0]
i = 0
ELO, ENLO = ELO[i::], ENLO[i::]
PLO, PNLO = PLO[i::], PNLO[i::]

deltaE = (ELO - ENLO)
deltaP = (PLO -  PNLO) 

i1 = np.argmax(np.abs(deltaE))
i2 = np.argmax(np.abs(deltaP))

print(deltaE[i1])
print(deltaP[i2])

print(PLO[-1])
print(PNLO[-1])

