# This file must be run once to generate the requisite files
import numpy as np
from numpy import pi

# The parameter space of mu and alpha
n = 101
mu_list = np.linspace(0., 2.5, n)
a_list = np.linspace(0, pi, n)

MU, A = np.meshgrid(mu_list, a_list)

# Everything is done in units of m_pi
mpi = 1

# LEO constants
MPI = 132
f = 84.9 / MPI
l1 = -0.4
l2 = 4.3
l3 = 2.9
l4 = 4.4

