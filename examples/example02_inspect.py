"""Some tests on example 2 (debugging).

Author:
    Ilias Bilionis

Date:
    1/2/2023

"""

import matplotlib.pyplot as plt
import numpy as np
import sys

gamma = float(sys.argv[1])
n = int(sys.argv[2])
sigma = float(sys.argv[3])

obs_prefix = f"example02_n={n}_sigma={sigma:1.2e}_0"
x_file = obs_prefix + "_x_obs.csv"
y_file = obs_prefix + "_y_obs.csv"

prefix = f"example02_newb_results/example02_gamma={gamma:1.2e}_n={n}_sigma={sigma:1.2e}_0"
#prefix = f"example02_gamma={gamma:1.2e}_n={n}_sigma={sigma:1.2e}_0"

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

skip_prior = 100
skip = 100
end = 1000
thin = 1

fig, ax = plt.subplots(1,2, figsize=(5 * 2 * 1.59, 5))
ax[0].plot(xs, phipr[skip_prior:end:thin,:].T, 'r', lw=0.5)
ax[0].plot(xst, yst, 'k', lw=1)
ax[0].plot(x_obs, y_obs, 'kx')
ax[1].plot(xs, phips[skip:end:thin,:].T, 'r', lw=0.5)
ax[1].plot(xst, yst, 'k', lw=1)
ax[1].plot(x_obs, y_obs, 'kx')


#thetas = np.loadtxt(prefix + "_theta.csv")
#fig, ax = plt.subplots()
#ax.plot(thetas[:, 1])

plt.show()
