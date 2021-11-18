import numpy as np
from numpy import pi, cos, sin
from matplotlib import pyplot as plt

plt.style.use("bmh")

m, l = 1, 1
r_max = 1.3

V = lambda r, theta : -1 / 2 * m**2 * r**2 + 1/ 4 * l * r**4

N = 100
r = np.linspace(0, r_max, N)
theta = np.linspace(0, 2*pi, N)

R, THETA = np.meshgrid(r, theta)
mask1 = THETA > np.pi/2
mask2 = THETA < np.pi
mask3 =  R > 1
mask = np.logical_and(mask1, mask2)
mask = np.logical_and(mask, mask3)
X, Y = R*cos(THETA+pi), R*sin(THETA+pi)
Z = V(R, THETA)

Z[mask] = np.nan

fig, ax = plt.subplots(
    subplot_kw={"projection" : "3d"},
    constrained_layout=True
)
# ax.set_axis_off()

ax.set_box_aspect(aspect = (1,1,1/3))

plot_opts = dict(
    antialiased=True, 
    rstride=2, 
    cstride=2, 
    cmap="cool", 
    lw=0.2, 
    shade=True,  
    edgecolors='k', 
    alpha=1, 
)
ax.plot_surface(X, Y, Z, **plot_opts)
# ax.plot_wireframe(X, Y, Z, color="black", lw=0.2)

print(Z)
plt.show()


