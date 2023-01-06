"""Some tests on example 2 (debugging).

Author:
    Ilias Bilionis

Date:
    1/2/2023

"""

import matplotlib.pyplot as plt
import numpy as np
import sys

beta = float(sys.argv[1])
n = int(sys.argv[2])
sigma = float(sys.argv[3])

obs_prefix = f"example02_n={n}_sigma={sigma:1.2e}_0"
x_file = obs_prefix + "_x_obs.csv"
y_file = obs_prefix + "_y_obs.csv"

prefix = f"example03_beta={beta:1.2e}_n={n}_sigma={sigma:1.2e}_0"

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

x_obs = np.loadtxt(x_file)
y_obs = np.loadtxt(y_file)

xst = np.loadtxt("example02_xs.csv")
yst = np.loadtxt("example02_ys.csv")

skip_prior = 0
skip = 0
end = 1000
thin = 1

fig, ax = plt.subplots(1,2, figsize=(5 * 2 * 1.59, 5))
ax[0].plot(xs, phipr[skip_prior:end:thin,:].T, 'r', lw=0.5)
ax[0].plot(xst, yst, 'k', lw=1)
ax[0].plot(x_obs, y_obs, 'kx')
ax[1].plot(xs, phips[skip:end:thin,:].T, 'r', lw=0.5)
ax[1].plot(xst, yst, 'k', lw=1)
ax[1].plot(x_obs, y_obs, 'kx')
plt.show()
# Visualize the source term
skip_s = 90
fig, ax = plt.subplots()
xs = np.loadtxt(prefix + "_source_x.csv")
fs = np.loadtxt(prefix + "_source_f.csv")
ax.plot(xs, fs[skip_s::, :].T, 'r', lw=0.1)
ax.plot(xs, np.cos(4.0 * xs), '--')

#thetas = np.loadtxt(prefix + "_theta.csv")
#fig, ax = plt.subplots()
#ax.plot(thetas[:, 1])

plt.show()
