"""Generate observations for example 2 of the paper.

Author:
    Ilias Bilionis

Date:
    12/27/2022

"""

import numpy as np
from scipy.integrate import solve_bvp
import sys

num_times = 1
num_observations = 40
sigma=0.01

# Set up and solve the boundary value problem
D = 0.1
kappa = 1.0
f = lambda x: np.cos(4.0 * x)
rhs = lambda x, y: (y[1], (f(x) + kappa * y[0] ** 3) / D)
bc = lambda ya, yb: np.array([ya[0] - 0.0, yb[0] - 0.0])
xs = np.linspace(0, 1, 1000)
ys = np.zeros((2, 1000))
res = solve_bvp(rhs, bc, xs, ys)
assert res["success"]
f = lambda x: res.sol(x)[0, :]

prefix = "example02"

# Save the true solution
xs = np.linspace(0, 1, 100)
ys = f(xs)

np.savetxt(prefix + "_xs.csv", xs)
np.savetxt(prefix + "_ys.csv", ys)

# Same the sampled measured data
prefix += f"_n={num_observations}_sigma={sigma:1.2e}"
s = len(str(num_times))
ymin = 1e99
ymax = -1e99

for t in range(num_times):
    obs_prefix = prefix + f"_{t}"
    x_file = obs_prefix + "_x_obs.csv"
    y_file = obs_prefix + "_y_obs.csv"
    x_obs = np.linspace(0, 1, num_observations + 2)[1:-1]
    y_obs = f(x_obs) + sigma * np.random.randn(x_obs.shape[0])
    np.savetxt(x_file, x_obs)
    np.savetxt(y_file, y_obs)
