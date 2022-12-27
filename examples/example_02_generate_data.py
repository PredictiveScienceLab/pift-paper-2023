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

from problems import Diffusion1DGamma
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

example = Diffusion1DGamma(
    sigma=args.sigma,
    num_samples=args.num_observations
)

result = example.generate_experimental_data()
f = result["solution"]

prefix = "example02"

# Save the true solution
xs = np.linspace(0, 1, 100)
ys = f(xs)
np.savetxt(prefix + "_xs.csv", xs)
np.savetxt(prefix + "_ys.csv", ys)

# Same the sampled measured data
prefix += f"_n={args.num_observations}_sigma={args.sigma:1.2e}"
s = len(str(args.num_times))
for t in range(args.num_times):
    obs_prefix = prefix + f"_{t}"
    x_file = obs_prefix + "_x_obs.csv"
    y_file = obs_prefix + "_y_obs.csv"
    result = example.generate_experimental_data()
    x_obs, y_obs = result["x"], result["y"]
    np.savetxt(x_file, x_obs)
    np.savetxt(y_file, y_obs)

