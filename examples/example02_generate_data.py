"""Generate observations for example 2 of the paper.

Author:
    Ilias Bilionis

Date:
    12/27/2022

"""


import jax.numpy as jnp
from jax.random import PRNGKey
import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_bvp
from options import *

parser = make_standard_option_parser()

parser.add_argument(
    "--num-times",
    dest="num_times",
    help="the number of times to repeat the data sampling process",
    type=int,
    default=10
)
parser.add_argument(
    "--num-observations",
    dest="num_observations",
    help="the number of observations to use in the example",
    type=int,
    default=10
)
parser.add_argument(
    "--sigma",
    dest="sigma",
    help="the scale of the observation noise.",
    type=float,
    default=0.01
)


args = parser.parse_args()

# Set up and solve the boundary value problem
D = 0.1
kappa = 0.0
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
prefix += f"_n={args.num_observations}_sigma={args.sigma:1.2e}"
s = len(str(args.num_times))
ymin = 1e99
ymax = -1e99

for t in range(args.num_times):
    obs_prefix = prefix + f"_{t}"
    x_file = obs_prefix + "_x_obs.csv"
    y_file = obs_prefix + "_y_obs.csv"
    x_obs = np.linspace(0, 1, args.num_observations)
    y_obs = f(x_obs) + args.sigma * np.random.randn(x_obs.shape[0])
    np.savetxt(x_file, x_obs)
    np.savetxt(y_file, y_obs)
