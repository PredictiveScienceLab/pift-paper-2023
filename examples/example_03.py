"""
Example 3 of the paper
"""

import jax.numpy as jnp
from jax.random import PRNGKey
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import brentq

from pift import *
from problems import Cubic1D
from options import *

parser = make_standard_option_parser()
add_nr_options(parser)
add_sgld_options(parser)

parser.add_argument(
    "--beta-fixed",
    dest="beta_fixed",
    help="fixed beta value",
    type=float,
    default=10.0
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
parser.add_argument(
    "--figure-format",
    dest="figure_format",
    help="the figure format",
    type=str,
    default="pdf"
)

# optimization options


args = parser.parse_args()

example = Cubic1D(
    sigma=args.sigma,
    num_samples=args.num_observations
)

num_terms = args.num_terms

V = FunctionParameterization.from_basis(
    "psi",
    Fourier1DBasis(example.b, args.num_terms)
)

result = example.generate_experimental_data()
data = (result["x"], result["y"])

#func = lambda beta, gamma: log_like(jnp.array([beta, gamma]))[0]

gamma = args.gamma

problem = example.make_pift_problem(V)

rng_key = PRNGKey(123456)

log_like = GradMinusLogMarginalLikelihood(
    problem=problem,
    data=data,
    rng_key=rng_key,
    disp=False,
    num_samples=args.num_samples,
    num_warmup=args.num_warmup,
    thinning=args.thinning,
    progress_bar=args.progress_bar,
    return_hessian=True
)

out_prefix = (
    f"example_03")

out_opt = out_prefix + ".opt"

with open(out_opt, "w") as fd:
#    res = brentq(func, 1.0, 1e6, args=(fd,), xtol=1)

    res = newton_raphson(
        log_like,
        theta0=jnp.array([args.beta_start]),
        alpha=args.nr_alpha,
        maxit=args.nr_maxit,
        tol=args.nr_tol,
        fd=fd
    )

    print("nr results:")
    print(res)

    beta_star = res[0]

    theta_samples = stochastic_gradient_langevin_dynamics(
        log_like,
        theta0=res[0],
        M=res[1],
        alpha=args.sgld_alpha,
        beta=args.sgld_beta,
        gamma=args.sgld_gamma,
        maxit=args.sgld_maxit,
        maxit_after_which_epsilon_is_fixed=args.sgld_fix_it,
        fd=fd
    )

out_mcmc = out_prefix = ".mcmc"
np.savetxt(out_mcmc, theta_samples)
