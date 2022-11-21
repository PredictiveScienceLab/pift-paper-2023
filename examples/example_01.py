"""Replicate Example 1 of the paper.

TODO: Alex, you must modify this to generate the examples of the file.
"""


import jax.numpy as jnp
from jax.random import PRNGKey
import numpy as np
import matplotlib.pyplot as plt

from pift import *
from problems import Diffusion1D
from options import make_standard_option_parser

parser = make_standard_option_parser()
parser.add_argument(
    "--beta",
    dest="beta",
    help="the beta you want to run the simulation on",
    type=float,
    default=0.0
)
parser.add_argument(
    "--figure-format",
    dest="figure_format",
    help="the figure format",
    type=str,
    default="png"
)

args = parser.parse_args()

example = Diffusion1D()

num_terms = args.num_terms

V = FunctionParameterization.from_basis(
    "psi",
    Fourier1DBasis(example.b, args.num_terms)
)

problem = example.make_pift_problem(V)

rng_key = PRNGKey(123456)

mcmc = MCMCSampler(
    problem.pyro_model,
    rng_key=rng_key,
    num_warmup=args.num_warmup,
    num_samples=args.num_samples,
    thinning=args.thinning,
    progress_bar=True
)

beta = args.beta

samples = mcmc.sample(
    theta=[example.kappa * beta, beta]
)

xs = np.linspace(example.a, example.b, 200)
ws = samples["w"]
samples_out = f"example_01_beta={beta:1.1f}" + "_ws.csv"
np.savetxt(samples_out, ws)

ys_pift = problem.vwphi(xs, ws)
ys_out = f"example_01_beta={beta:1.1f}" + "_ys.csv"
np.savetxt(ys_out, ys_pift)

f = example.solve()
ys_true = f(xs)
# TODO - Make the figures compatible with paper
import seaborn as sns
plt.style.use("bmh")


plt.rcParams.update({'font.size': 9})
sns.set_context("paper")
sns.set_style("white")

plt.figure(figsize = [5.1667, 5.1667])
# some samples
plt.plot(xs, ys_pift.T[:,0:10], "r", lw=0.5)
# exact solution
plt.plot(xs, ys_true, "k--", label='exact', lw=2.0)
# median of posterior
plt.plot(xs, jnp.quantile(ys_pift, q = 0.5, axis = 0), "b-.", label = 'median', lw=2.0)
# 95% predictive interval
plt.fill_between(xs, jnp.quantile(ys_pift, q = 0.025, axis=0), jnp.quantile(ys_pift, q = 0.975, axis=0),
                 alpha = 0.5, color = 'cornflowerblue')
# data points
plt.plot(example.a, example.yb[0], "kx", markersize = 7, markeredgewidth=2)
plt.plot(example.b, example.yb[1], "kx", markersize = 7, markeredgewidth=2)

plt.ylim([-0.1,1.5])
plt.xlim([0.0-.1,1.0+.1])

plt.xlabel("$x$", fontsize=10.0)
plt.ylabel(r"$\phi(x)$", fontsize = 10.0)
plt.title(f"$\\beta={beta:1.1f}$", fontsize=10.0)

plt.legend(loc = 'best', fontsize = 9.0)
plt.tight_layout()

out_file = f"example_01_beta={beta:1.1f}" + "." + args.figure_format
print(f"> writing: {out_file}")
#fig.savefig(out_file)
plt.savefig(out_file, dpi = 500, bbox_inches = 'tight')
