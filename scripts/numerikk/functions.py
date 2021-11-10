import numpy as np
from scipy.integrate import quad, quadrature
from numpy import sin, cos, sqrt, pi, log, arccos
import sympy as sp
from sympy import lambdify

from parameters import *

# Everything is done in units of m_pi
mpi = sp.S.One
# This is usefull to write rational numbers
one = sp.S.One
a, mu, p = sp.symbols("a, mu, p", real=True)


# Mass-parameters
m1_sq = mpi**2 * sp.cos(a) - mu**2 * sp.cos(2 * a)
m2_sq = mpi**2 * sp.cos(a) - mu**2 * sp.cos(a)**2
m3_sq = mpi**2 * sp.cos(a) + mu**2 * sp.sin(a)**2
m12 = 2 * mu * sp.cos(a)

# square of the dispertion relations of the particles
E0_sq = p**2 + m2_sq
M_sq = (m1_sq + m2_sq + m12**2)
Ep_sq = p**2 + 1 / 2 * M_sq + 1/2 * sp.sqrt(4 * p**2 * m12**2 + M_sq**2 - 4*m1_sq*m2_sq)
Em_sq = p**2 + 1 / 2 * M_sq - 1/2 * sp.sqrt(4 * p**2 * m12**2 + M_sq**2 - 4*m1_sq*m2_sq)

# Tree-level masses. Should equal E(p=0)
m0_sq = m3_sq
mp_sq = 1 / 2 * M_sq + 1/2 * sp.sqrt(M_sq**2 - 4*m1_sq*m2_sq)
mm_sq = 1 / 2 * M_sq - 1/2 * sp.sqrt(M_sq**2 - 4*m1_sq*m2_sq)

# Mass/energies used for F_fin
mtilde1_sq = m1_sq + 1/4 * m12**2
mtilde2_sq = m2_sq + 1/4 * m12**2

E1_sq = p**2 + mtilde1_sq
E2_sq = p**2 + mtilde2_sq

F_0_2_symb = - f**2 * (mpi**2 * sp.cos(a) + 1/2*mu**2 * sp.sin(a)**2)

ls = [sp.symbols("l{}".format(i)) for i in range(0, 5)]
ls_numb = [0, l1, l2, l3, l4]

F_0_4_symb = \
    - one/2 * 1 / (4 * sp.pi)**2 * (
    one/3 * ( ls[1] + 2 * ls[2] + one*3/2 - 3*sp.log(m3_sq) ) * mu**4 * sp.sin(a)**4
    + one/2*(-ls[3] + 4*ls[4] + one*3/2 - 2*sp.log(m3_sq) - sp.log(mtilde2_sq) )*mpi**4*sp.cos(a)**2
    + 2 * (ls[4] + one/2 - sp.log(m3_sq)) * mpi**2 *mu**2 * sp.cos(a)*sp.sin(a)**2
)

# Much slower method
# integral = lambda f: (lambda *args: quad(f, 0, 100, args=args, epsabs=1e-6, epsrel=1e-6)[0])

integral = lambda f: (lambda *args: quadrature(f, 0, 100, args=args, maxiter=200, tol=1e-6)[0])


###################################################
# LO and NLO approximation to the free energy

dF_fin_symb = 1/2 * (4 * pi) / (2 * pi)**3 * p**2 * (
    sp.sqrt(Ep_sq) + sp.sqrt(Em_sq) - sp.sqrt(E1_sq) - sp.sqrt(E2_sq)
    )
dF_fin = lambdify((p, mu, a), dF_fin_symb, "numpy")

dF_fin_C = lambda p, m, a: dF_fin(p+0j, m+0j, a+0j)
dF_fin_R = lambda p, m, a: dF_fin_C(p, m, a).real
dF_fin_I = lambda p, m, a: dF_fin_C(p, m, a).imag
F_fin = integral(dF_fin_R)
F_fin_I = integral(dF_fin_I)
F_fin = np.vectorize(F_fin)

F_0_2 = lambdify((mu, a), F_0_2_symb, "numpy")

for l, ln in zip(ls, ls_numb):
    F_0_4_symb = F_0_4_symb.subs(l, ln)
F_0_4 = lambdify((mu, a), F_0_4_symb, "numpy")
F = lambda mu, alpha: F_0_2(mu, alpha) + F_0_4(mu, alpha) + F_fin(mu, alpha)


###################################################
# derivative of free energy with respect to alpha

dF_fin_diff_a = lambdify((p, mu, a), dF_fin_symb.diff(a), "numpy")
dF_fin_C_diff_a = lambda p, m, a: dF_fin_diff_a(p+0j, m+0j, a+0j)
dF_fin_abs_diff_a = lambda p, m, a: np.abs(dF_fin_C_diff_a(p, m, a))
F_fin_diff_a = integral(dF_fin_abs_diff_a)
F_fin_diff_a = np.vectorize(F_fin_diff_a)

F_0_2_diff_a = lambdify((mu, a), F_0_2_symb.diff(a), "numpy")

F_0_4_symb_diff_a = F_0_4_symb.diff(a)
for l, ln in zip(ls, ls_numb):
    F_0_4_symb_diff_a = F_0_4_symb_diff_a.subs(l, ln)
F_0_4_diff_a = lambdify((mu, a), F_0_4_symb_diff_a, "numpy")
F_diff_a = lambda mu, alpha: F_0_2_diff_a(mu, alpha) + F_0_4_diff_a(mu, alpha) + F_fin_diff_a(mu, alpha)


# first approx to alpha as a function of mu_I, analytical result
def alpha_0(mu):
    morethan_mpi = mu**2 > np.ones_like(mu)
    a = np.zeros_like(mu)
    a[morethan_mpi] = arccos((1/mu[morethan_mpi]**2))
    return a


# Load physcial quantities + derived quantities

def get_free_energy_surface():
    FLO = np.load("data/F_0_2.npy")
    FNLOa = np.load("data/F_0_4.npy")
    Ffin = np.load("data/F_fin.npy")

    FNLO = FLO + FNLOa + Ffin
    return FLO, FNLO

def get_alpha_lo():
    return np.load("data/alpha_lo.npy")

def get_alpha_nlo():
    return np.load("data/alpha_nlo.npy")

def get_alpha_nlo2():
    return np.load("data/alpha_nlo2.npy")

def get_free_energy():
    FLO = np.load("data/F_0_2_lo_a.npy")

    F1 = np.load("data/F_0_2_nlo_a.npy")
    F2 = np.load("data/F_0_4_nlo_a.npy")
    F3 = np.load("data/F_fin_nlo_a.npy")
    FNLO = F1 + F2 + F3

    return FLO, FNLO

def get_pressure():
    FLO, FNLO = get_free_energy()
    PLO = -(FLO - FLO[0])
    PNLO = -(FNLO - FNLO[0])

    return PLO, PNLO

def get_isospin_density():
    FLO, FNLO = get_free_energy()
    d = lambda x: x[2::] - x[:-2:]

    nILO = - d(FLO) / d(mu_list)
    nINLO = - d(FNLO) / d(mu_list)

    return nILO, nINLO, mu_list[1:-1:]

def get_energy_density():
    PLO, PNLO = get_pressure()
    nILO, nINLO, mu = get_isospin_density()
    ELO = -PLO[1:-1] + nILO * mu
    ENLO = -PNLO[1:-1] + nINLO * mu

    return ELO, ENLO, mu
    
