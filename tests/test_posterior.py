"""Plots the results of ./test_posterior.

Author:
    Ilias Bilionis

Date:
    12/21/2022

"""

import matplotlib.pyplot as plt
import sys
import numpy as np
import pandas as pd

if not len(sys.argv) == 3:
    print(f"Usage:\n\t{sys.argv[0]} <BETA> <SIGMA>")
    quit()

beta = float(sys.argv[1])
assert beta > 0.0
sigma = float(sys.argv[2])
assert sigma > 0.0

prefix = f"posterior_beta={beta:.2e}_sigma={sigma:.2e}"
print(f"Working with {prefix}")
warmup_data = np.loadtxt(prefix + "_warmup.csv")
samples_data = np.loadtxt(prefix + "_samples.csv")
x_obs = np.loadtxt(prefix + "_x_obs.csv")
y_obs = np.loadtxt(prefix + "_y_obs.csv")
x = np.loadtxt(prefix + "_x.csv")
phi = np.loadtxt(prefix + "_phi.csv")
true_phi = np.loadtxt(prefix + "_true.csv")

# minus log posterior
fh = np.hstack([warmup_data[:, 0], samples_data[:, 0]])
fh = pd.Series(fh)
window_size = 1000 
fh_windows = fh.rolling(window_size)
fh_moving_average = fh_windows.mean()

fig, ax = plt.subplots()
ax.plot(fh_moving_average)

# the w's
ws = np.vstack([warmup_data[:, 1:], samples_data[:, 1:]])
fig, ax = plt.subplots()
ax.plot(ws)

# the statistic of the field
p_500, p_025, p_975 = np.percentile(phi, [50, 2.5, 97.5], axis=0)
fig, ax = plt.subplots()
ax.plot(x, p_500, "b-", lw=2)
ax.fill_between(x, p_025, p_975, color="blue", alpha=0.25)
ax.plot(x, phi[::100, :].T, "r", lw=0.5)
ax.plot(x, true_phi, "k--")
ax.plot(x_obs, y_obs, "kx")

plt.show()
