"""Unit tests to make sure that the C++ code is correct.

Author:
    Ilias Bilionis

Date:
    12/14/2022
"""

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
from jax import vmap, grad
from pift import *
from problems import *

num_terms = 4
V = FunctionParameterization.from_basis(
    "psi",
    Fourier1DBasis(1.0, num_terms)
)

psi = V.eval
psi_prime = grad(psi, argnums=0)
vpsi = vmap(psi, (0, None))
vpsi_prime = vmap(psi_prime, (0, None))
grad_w_psi = grad(psi, argnums=1)
vgrad_w_psi = vmap(grad_w_psi, (0, None))
grad_w_psi_prime = grad(psi_prime, argnums=1)
vgrad_w_psi_prime = vmap(grad_w_psi_prime, (0, None))

phi = enforce_1d_boundary_conditions(
    V.eval,
    0.0,
    1.0,
    1.0,
    1.0/10.0
)
phi_prime = grad(phi, argnums=0)
vphi = vmap(phi, (0, None))
vphi_prime = vmap(phi_prime, (0, None))
grad_w_phi = grad(phi, argnums=1)
vgrad_w_phi = vmap(grad_w_phi, (0, None))
grad_w_phi_prime = grad(phi_prime, argnums=1)
vgrad_w_phi_prime = vmap(grad_w_phi_prime, (0, None))

q = lambda x: jnp.exp(-x)
H = lambda x, w: 1000 * (
    0.5 * 0.25 * phi_prime(x, w) ** 2
    - phi(x, w) * q(x)
)
vH = vmap(H, (0, None))
grad_w_H = grad(H, argnums=1)
vgrad_w_H = vmap(grad_w_H, (0, None))

from problems import *
example = Diffusion1D()
f = example.solve()

#np.random.seed(123456)
#result = example.generate_experimental_data()
#x_obs, y_obs = (result["x"], result["y"])
#x_obs, f_obs = (result["x_source"], result["y_source"])
#np.savetxt("src/x_obs.csv", x_obs)
#np.savetxt("src/y_obs.csv", y_obs)

def plot_pred_dist(file):
    ws = np.loadtxt(file)
    fig, ax = plt.subplots()
    ax.plot(ws[:, 0])
    fig, ax = plt.subplots()
    ax.plot(ws[:, 1:])

    vwphi = vmap(vphi, (None, 0))

    xs = np.linspace(0, 1, 100)
    ys = vwphi(xs, ws[:, 1:])

    fig, ax = plt.subplots()
    ax.plot(xs, ys.T, "r", lw=0.1)
    ax.plot(xs, f(xs))
    #ax.plot(x_obs, y_obs, 'kx')

plot_pred_dist("src/foo_prior.csv")

plt.show()
