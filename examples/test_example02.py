"""Some tests on example 2 (debugging).

Author:
    Ilias Bilionis

Date:
    12/29/2022

"""

import matplotlib.pyplot as plt
import numpy as np
import sys

prefix = f"example02b_gamma={float(sys.argv[1]):1.2e}"

# Visualize weights
wspr = np.loadtxt(prefix + "_prior_ws.csv")
wsps = np.loadtxt(prefix + "_post_ws.csv")

fig, ax = plt.subplots(1,2, figsize=(5 * 2 * 1.59, 5))
ax[0].plot(wspr[:,1:])
ax[1].plot(wsps[:,1:])

# Visualize function
xs = np.loadtxt(prefix + "_prior_x.csv")
phipr = np.loadtxt(prefix + "_prior_phi.csv")
phips = np.loadtxt(prefix + "_post_phi.csv")

x_obs = np.loadtxt("example02_n=100_sigma=1.00e-04_0_x_obs.csv")
y_obs = np.loadtxt("example02_n=100_sigma=1.00e-04_0_y_obs.csv")

xst = np.loadtxt("example02_xs.csv")
yst = np.loadtxt("example02_ys.csv")

skip_prior = 90
skip = 90
end = -1
thin = 1

fig, ax = plt.subplots(1,2, figsize=(5 * 2 * 1.59, 5))
ax[0].plot(xs, phipr[skip_prior:end:thin,:].T, 'r', lw=0.5)
ax[0].plot(xst, yst, 'k', lw=1)
ax[0].plot(x_obs, y_obs, 'kx')
ax[1].plot(xs, phips[skip:end:thin,:].T, 'r', lw=0.5)
ax[1].plot(xst, yst, 'k', lw=1)
ax[1].plot(x_obs, y_obs, 'kx')


thetas = np.loadtxt(prefix + "_theta.csv")
fig, ax = plt.subplots()
ax.plot(thetas[:, 1])

plt.show()
