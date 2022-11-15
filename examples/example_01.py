"""Replicate example 1 of the paper."""


import jax.numpy as jnp
from jax.random import PRNGKey
import numpy as np
import matplotlib.pyplot as plt

from pift import *
from problems import Diffusion1D


example = Diffusion1D(
    kappa = 0.1,
    yb = (1.0, 0.0)
)

num_terms = 4

V = FunctionParameterization.from_basis(
    "psi",
    Fourier1DBasis(example.b, num_terms)
)

weight_mean = jnp.zeros((V.num_params,))
weight_scale= jnp.ones((V.num_params,))

problem = example.make_pift_problem(V)

rng_key = PRNGKey(123456)

mcmc = MCMCSampler(
    problem.pyro_model,
    rng_key=rng_key,
    num_warmup=100,
    num_samples=1000,
    thinning=10,
    progress_bar=True
)

beta = 0.0

samples = mcmc.sample(
    theta=[example.kappa * beta, beta],
    weight_mean=weight_mean,
    weight_scale=weight_scale
)

xs = np.linspace(example.a, example.b, 200)
ws = samples["w"]
ys_pift = problem.vwphi(xs, ws)

f = example.solve()
ys_true = f(xs)

fig, ax = plt.subplots()
ax.plot(xs, ys_pift.T, "r", lw=0.1)
ax.plot(xs, ys_true, "b--")
ax.plot(example.a, example.yb[0], "ko")
ax.plot(example.b, example.yb[1], "ko")
ax.set_xlabel("$x$")
ax.set_ylabel(r"$\phi(x)$")
ax.set_title(f"$\\beta={beta:1.1f}$")
plt.legend(loc="best")
plt.show()
