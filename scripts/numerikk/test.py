import sympy as sp
from sympy import sin, cos, sqrt, pi, log
from sympy.utilities import lambdify
from scipy.integrate import quad, quadrature

from parameters import *

mpi = 1
a, mu, p = sp.symbols("a, mu, p", real=True)

# Mass-parameters
m1_sq = mpi**2 * cos(a) - mu**2 * cos(2 * a)
m2_sq = mpi**2 * cos(a) - mu**2 * cos(a)**2
m3_sq = mpi**2 * cos(a) + mu**2 * sin(a)**2
m12 = 2 * mu * cos(a)

# square of the dispertion relations of the particles
E0_sq = p**2 + m2_sq

M_sq = (m1_sq + m2_sq + m12**2)


Ep_sq = p**2 + 1 / 2 * M_sq + 1/2 * sqrt(4 * p**2 * m12**2 + M_sq**2 - 4*m1_sq*m2_sq)
Em_sq = p**2 + 1 / 2 * M_sq - 1/2 * sqrt(4 * p**2 * m12**2 + M_sq**2 - 4*m1_sq*m2_sq)

m0sq = m3_sq
mp_sq = 1 / 2 * M_sq + 1/2 * sqrt(M_sq**2 - 4*m1_sq*m2_sq)
mm_sq = 1 / 2 * M_sq - 1/2 * sqrt(M_sq**2 - 4*m1_sq*m2_sq)

# Masss/energies used for F_fin
mtilde1_sq = m3_sq
mtilde2_sq = mpi**2 * cos(a)

E1_sq = p**2 + mtilde1_sq
E2_sq = p**2 + mtilde2_sq

F_0_2 = - f**2 * (mpi**2 * cos(a) + 1/2*mu**2 * sin(a)**2)
F_0_4 = - 1/2 * 1 / (4 * pi)**2 * (
    1/3 * ( l1 + 2 * l2 - 3/2 - 3*log(m3_sq) ) * mu**4 * sin(a)**2
    + 1/2 * ( -l3 + 4*l4 - 3/2 - 2*log(m3_sq) - log(mtilde2_sq) ) * mpi**4 * cos(a)**2
    + 2 * (l4 - 1/2 - log(m3_sq)) * mpi**2 * cos(a)*sin(a)**2
)

# Integrand of F_fin
dF_fin = 4 / (4 * pi)**2 * p**2 * (sqrt(Ep_sq) + sqrt(Em_sq) - sqrt(E1_sq) - sqrt(E2_sq))
# dF_fin_l = lambdify(p, mu, a, dF_fin)
# F_fin = lambda mu, alpha: quadrature(dF_fin, 0, 10, args=(mu, alpha), maxiter=200)[0]

# print(dF_fin)

##

F_0_2_diff_a = lambdify((mu, a), F_0_2.diff(a).simplify())
F_0_4_diff_a = lambdify((mu, a), F_0_2.diff(a).simplify())
dF_fin_diff_a = lambdify((p, mu, a), dF_fin.diff(a).simplify())
F_fin_diff_a = lambda mu, alpha: quadrature(dF_fin_diff_a, 0, 10, args=(mu, alpha), maxiter=200)[0]

F_diff_a = lambda mu, alpha: F_0_2_diff_a(mu, alpha) + F_0_4_diff_a(mu, alpha) + F_fin_diff_a(mu, alpha)

print(F_diff_a(1, 0))

