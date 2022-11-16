"""Replicate Example 2 of the paper.

TODO: Alex, you must modify this to generate the examples of the file.
"""


import jax.numpy as jnp
from jax.random import PRNGKey
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import brentq

from pift import *
from problems import Diffusion1DGamma
from options import make_standard_option_parser

parser = make_standard_option_parser()
parser.add_option(
    "--beta-min",
    dest="beta_min",
    help="the minimum beta",
    type="float",
    default=1e-3
)
parser.add_option(
    "--beta-max",
    dest="beta_max",
    help="the maximum beta",
    type="float",
    default=1e7
)
parser.add_option(
    "--num-observations",
    dest="num_observations",
    help="the number of observations to use in the example",
    type="int",
    default=10
)
parser.add_option(
    "--sigma",
    dest="sigma",
    help="the scale of the observation noise.",
    type="float",
    default=0.01
)
parser.add_option(
    "--figure-format",
    dest="figure_format",
    help="the figure format",
    type="str",
    default="png"
)
(options, args) = parser.parse_args()

example = Diffusion1DGamma(
    sigma=options.sigma,
    num_samples=options.num_samples
)
print(example)

num_terms = options.num_terms

V = FunctionParameterization.from_basis(
    "psi",
    Fourier1DBasis(example.b, options.num_terms)
)

result = example.generate_experimental_data()
data = (result["x"], result["y"])

problem = example.make_pift_problem(V)
print(problem)

rng_key = PRNGKey(123456)

log_like = GradMinusLogMarginalLikelihood(
    problem=problem,
    data=data,
    rng_key=rng_key,
    disp=True,
    num_samples=options.num_samples,
    num_warmup=options.num_warmup,
    thinning=options.thinning,
    progress_bar=False,
)

func = lambda beta, gamma: log_like(jnp.array([beta, gamma]))[0]

out = []
for gamma in np.linspace(0.0, 1.0, 10):
    print(f"working on gamma={gamma:1.2f}")
    try:
        beta_star, r = brentq(
            func,
            options.beta_min,
            options.beta_max,
            args=(gamma,),
            xtol=1.0,
            disp=False,
            full_output=True
        )
    except:
        beta_star = options.beta_max
    print(f"result: {gamma:1.2f} {beta_star:1.0f}")
    out.append((gamma, beta_star))
out = np.array(out)

out_file = "example_02.csv"
print(f"saving results in `{out_file}`")
np.savetxt(out_file, out)

fig_name = "example_02" + "." + options.figure_format
print(f"saving figure `{fig_name}`")
fig, ax = plt.subplots()
ax.plot(out[:, 0], out[:, 1], 'bo')
ax.set_xlabel(r"$\gamma$")
ax.set_ylabel(r"$\beta$")
fig.savefig(fig_name)
