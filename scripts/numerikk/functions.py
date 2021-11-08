import numpy as np
from scipy.integrate import quad, quadrature
from mpmath import quad as mpquad
from numpy import sin, cos, sqrt, arccos, pi, log
from parameters import *

# Mass-parameters
m1_sq = lambda mu, alpha: mpi**2 * cos(alpha) - mu**2 * cos(2 * alpha)
m2_sq = lambda mu, alpha: mpi**2 * cos(alpha) - mu**2 * cos(alpha)**2
m3_sq = lambda mu, alpha: mpi**2 * cos(alpha) + mu**2 * sin(alpha)**2
m12 = lambda mu, alpha: 2 * mu * cos(alpha)

# square of the dispertion relations of the particles
E0_sq = lambda p, mu, alpha: p^2 + m2_sq(mu, alpha)

def Ep_sq(p, mu, alpha):
    M1sq = m1_sq(mu, alpha)
    M2sq = m2_sq(mu, alpha)
    M12sq = m12(mu, alpha)**2
    M_sq = (M1sq + M2sq + M12sq)
    return p**2 + 1 / 2 * M_sq + 1/2 * sqrt(4 * p**2 * M12sq + M_sq**2 - 4*M1sq*M2sq)
    
def Em_sq(p, mu, alpha):
    M1sq = m1_sq(mu, alpha)
    M2sq = m2_sq(mu, alpha)
    M12sq = m12(mu, alpha)**2
    M_sq = (M1sq + M2sq + M12sq)
    return p**2 + 1 / 2 * M_sq - 1/2 * sqrt(4 * p**2 * M12sq + M_sq**2 - 4*M1sq*M2sq)


# Tree-level masses. Should equal E(p=0)
m0sq = lambda mu, alpha: m3_sq(mu, alpha)

def mp_sq(mu, alpha):
    M1sq = m1_sq(mu, alpha)
    M2sq = m2_sq(mu, alpha)
    M12sq = m12(mu, alpha)**2
    M_sq = (M1sq + M2sq + M12sq)
    return 1 / 2 * M_sq + 1/2 * sqrt(M_sq**2 - 4*M1sq*M2sq)
    
def mm_sq(mu, alpha):
    M1sq = m1_sq(mu, alpha)
    M2sq = m2_sq(mu, alpha)
    M12sq = m12(mu, alpha)**2
    M_sq = (M1sq + M2sq + M12sq)
    return 1 / 2 * M_sq - 1/2 * sqrt(M_sq**2 - 4*M1sq*M2sq)


# Masss/energies used for F_fin
mtilde1_sq = lambda mu, alpha: m3_sq(mu, alpha)
mtilde2_sq = lambda mu, alpha: mpi**2 * cos(alpha)

E1_sq = lambda p, mu, alpha: p**2 + mtilde1_sq(mu, alpha)
E2_sq = lambda p, mu, alpha: p**2 + mtilde2_sq(mu, alpha)

func = lambda p, mu, alpha: (
    sqrt(Ep_sq(p, mu, alpha)) + sqrt(Em_sq(p, mu, alpha))
    - sqrt(E1_sq(p, mu, alpha)) - sqrt(E2_sq(p, mu, alpha))
)

F_0_2 = lambda mu, alpha: - f**2 * (mpi**2 * cos(alpha) + 1/2*mu**2 * sin(alpha)**2)

F_0_4 = lambda mu, alpha: - 1/2 * 1 / (4 * pi)**2 * (
    1/3 * ( l1 + 2 * l2 - 3/2 - 3*log(m3_sq(mu, alpha)) ) * mu**4 * sin(alpha)**2
    + 1/2 * (
        -l3 + 4*l4 - 3/2 - 2*log(m3_sq(mu, alpha)) - log(mtilde2_sq(mu, alpha))
        ) * mpi**4 * cos(alpha)**2
    + 2 * (l4 - 1/2 - log(m3_sq(mu, alpha))) * mpi**2 * cos(alpha)*sin(alpha)**2
)

# Integrand of F_fin
dF_fin = lambda p, mu, alpha: 4 / (4 * pi)**2 * p**2 * func(p, mu, alpha)

# F_fin = lambda mu, alpha: quad(dF_fin, 0, np.inf, args=(mu, alpha))[0]
# F_fin = lambda mu, alpha: quadrature(dF_fin, 0, 10, args=(mu, alpha), maxiter=200)[0]
def F_fin(mu, alpha):
    f = lambda x: dF_fin(x, mu, alpha)
    F = mpquad(f, [0, np.inf])
    return np.float(F.real) + 1j * np.float(F.imag)

# first approx to alpha as a function of mu_I, analytical result
def alpha_0(mu):
    morethan_mpi = mu**2 > np.ones_like(mu)
    a = np.zeros_like(mu)
    a[morethan_mpi] = arccos((1/mu[morethan_mpi]**2))
    return a
