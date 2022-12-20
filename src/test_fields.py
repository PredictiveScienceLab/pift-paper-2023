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
    0.0,
    1.0,
    0.0
)
phi_prime = grad(phi, argnums=0)
vphi = vmap(phi, (0, None))
vphi_prime = vmap(phi_prime, (0, None))
grad_w_phi = grad(phi, argnums=1)
vgrad_w_phi = vmap(grad_w_phi, (0, None))
grad_w_phi_prime = grad(phi_prime, argnums=1)
vgrad_w_phi_prime = vmap(grad_w_phi_prime, (0, None))

q = lambda x: jnp.exp(-x)
H = lambda x, w, theta: 100 * (
    0.5 * 0.25 * phi_prime(x, w) ** 2
    - phi(x, w) * q(x)
)
vH = vmap(H, (0, None, None))
grad_w_H = grad(H, argnums=1)
vgrad_w_H = vmap(grad_w_H, (0, None, None))

prefix = "src/unit_test_"
x = np.loadtxt(prefix + "x.csv")
w = np.loadtxt(prefix + "w.csv")
theta = np.loadtxt(prefix + "theta.csv")

# Test psi
psi_prolong = np.loadtxt(prefix + "psi.csv")
psi_vals_cpp = psi_prolong[:, 0]
psi_vals = vpsi(x, w)

error = np.linalg.norm(psi_vals - psi_vals_cpp)
print(f"psi error: {error:1.5e}")

fig, ax = plt.subplots()
ax.set_title(r"Testing function parameterization")
ax.plot(x, psi_vals, label=r"Python $\psi(x,w)$")
ax.plot(x, psi_vals_cpp, "--", label=r"C++ $\psi(x,w)$")
plt.legend(loc="best")

# Test psi_prime
psi_prime_vals_cpp = psi_prolong[:, 1] 
psi_prime_vals = vpsi_prime(x, w)

error = np.linalg.norm(psi_prime_vals - psi_prime_vals_cpp)
print(f"psi_prime error: {error:1.5e}")

fig, ax = plt.subplots()
ax.set_title(r"Testing derivative of function parameterization")
ax.plot(x, psi_prime_vals, label=r"Python $\psi'(x,w)$")
ax.plot(x, psi_prime_vals_cpp, "--", label=r"C++ $\psi'(x,w)$")
plt.legend(loc="best")

# Test the grad_w of psi
grad_w_psi_prolong = np.loadtxt(prefix + "grad_w_psi.csv")

grad_w_psi_vals_cpp = grad_w_psi_prolong[:, :w.shape[0]]
grad_w_psi_vals = vgrad_w_psi(x, w)

error = np.linalg.norm(grad_w_psi_vals - grad_w_psi_vals_cpp)
print(f"grad_w_psi error: {error:1.5e}")

fig, ax = plt.subplots()
ax.set_title(r"Testing $\nabla_w$ of function parameterization")
ax.plot(x, grad_w_psi_vals, label=r"Python $\nabla_w\psi(x,w)$")
ax.plot(x, grad_w_psi_vals_cpp, "--", label=r"C++ $\nabla_w\psi(x,w)$")

# Test the grad_w of psi_prime
grad_w_psi_prime_vals_cpp =  grad_w_psi_prolong[:, w.shape[0]:]
grad_w_psi_prime_vals = vgrad_w_psi_prime(x, w)

error = np.linalg.norm(grad_w_psi_prime_vals - grad_w_psi_prime_vals_cpp)
print(f"grad_w_psi_prime error: {error:1.5e}")

fig, ax = plt.subplots()
ax.set_title(r"Testing $\nabla_w$ of derivative of function parameterization")
ax.plot(x, grad_w_psi_prime_vals, label=r"Python $\nabla_w\psi'(x,w)$")
ax.plot(x, grad_w_psi_prime_vals_cpp, "--", label=r"C++ $\nabla_w\psi'(x,w)$")

# Same as above for the constrained parameterization
phi_prolong = np.loadtxt(prefix + "phi.csv")
phi_vals_cpp = phi_prolong[:, 0]
phi_vals = vphi(x, w)

error = np.linalg.norm(phi_vals - phi_vals_cpp)
print(f"phi error: {error:1.5e}")

fig, ax = plt.subplots()
ax.set_title(r"Testing constrained function parameterization")
ax.plot(x, phi_vals, label=r"Python $\phi(x,w)$")
ax.plot(x, phi_vals_cpp, "--", label=r"C++ $\phi(x,w)$")
plt.legend(loc="best")

# Test phi_prime
phi_prime_vals_cpp = phi_prolong[:, 1]
phi_prime_vals = vphi_prime(x, w)

error = np.linalg.norm(phi_prime_vals - phi_prime_vals_cpp)
print(f"phi_prime error: {error:1.5e}")

fig, ax = plt.subplots()
ax.set_title(r"Testing derivative of constrained function parameterization")
ax.plot(x, phi_prime_vals, label=r"Python $\phi'(x,w)$")
ax.plot(x, phi_prime_vals_cpp, "--", label=r"C++ $\phi'(x,w)$")
plt.legend(loc="best")


# Test the grad_w of psi
grad_w_prolong = np.loadtxt(prefix + "grad_w_phi.csv")
grad_w_phi_vals_cpp = grad_w_prolong[:, :w.shape[0]]
grad_w_phi_vals = vgrad_w_phi(x, w)

error = np.linalg.norm(grad_w_phi_vals - grad_w_phi_vals_cpp)
print(f"grad_w_phi error: {error:1.5e}")

fig, ax = plt.subplots()
ax.set_title(r"Testing $\nabla_w$ of constrained function parameterization")
ax.plot(x, grad_w_phi_vals, label=r"Python $\nabla_w\phi(x,w)$")
ax.plot(x, grad_w_phi_vals_cpp, "--", label=r"C++ $\nabla_w\phi(x,w)$")

# Test the grad_w of psi_prime
grad_w_phi_prime_vals_cpp = grad_w_prolong[:, w.shape[0]:] 
grad_w_phi_prime_vals = vgrad_w_phi_prime(x, w)

error = np.linalg.norm(grad_w_phi_prime_vals - grad_w_phi_prime_vals_cpp)
print(f"grad_w_phi_prime error: {error:1.5e}")

fig, ax = plt.subplots()
ax.set_title(r"Testing $\nabla_w$ of derivative of constrained function parameterization")
ax.plot(x, grad_w_phi_prime_vals, label=r"Python $\nabla_w\phi'(x,w)$")
ax.plot(x, grad_w_phi_prime_vals_cpp, "--", label=r"C++ $\nabla_w\phi'(x,w)$")

# Test the Hamiltonian
H_vals_cpp = np.loadtxt(prefix + "H.csv")
H_vals = vH(x, w, theta)

error = np.linalg.norm(H_vals - H_vals_cpp)
print(f"H error: {error:1.5e}")

fig, ax = plt.subplots()
ax.set_title(r"Testing the Hamiltonian density")
ax.plot(x, H_vals, label=r"Python $h(x,w,\theta)$")
ax.plot(x, H_vals_cpp, "--", label=r"C++ $h(x,w,\theta)$")
plt.legend(loc="best")

# The the grad_w of the Hamiltonian
grad_w_H_vals_cpp = np.loadtxt(prefix + "grad_w_H.csv")
grad_w_H_vals = vgrad_w_H(x, w, theta)

error = np.linalg.norm(grad_w_H_vals - grad_w_H_vals_cpp)
print(f"grad_w H error: {error:1.5e}")

fig, ax = plt.subplots()
ax.set_title(r"Testing the $\nabla_w$ Hamiltonian density")
ax.plot(x, grad_w_H_vals, label=r"Python $\nabla_w h(x,w,\theta)$")
ax.plot(x, grad_w_H_vals_cpp, "--", label=r"C++ $\nabla_w h(x,w,\theta)$")

#from problems import *
#example = Cubic1D()
#f = example.solve()

#np.random.seed(123456)
#result = example.generate_experimental_data()
#x_obs, y_obs = (result["x"], result["y"])
#x_obs, f_obs = (result["x_source"], result["y_source"])
#np.savetxt("src/x_obs.csv", x_obs)
#np.savetxt("src/y_obs.csv", y_obs)


def plot_pred_dist(file):
    ws = np.loadtxt(file)
    print(ws.shape)
    fig, ax = plt.subplots()
    ax.plot(ws[:, 0])
    fig, ax = plt.subplots()
    ax.plot(ws[:, 1:])

    vwphi = vmap(vphi, (None, 0))

    xs = np.linspace(0, 1, 100)
    ys = vwphi(xs, ws[-10000::1000, 1:])

    fig, ax = plt.subplots()
    ax.plot(xs, ys.T, "r", lw=0.1)
    #ax.plot(xs, f(xs))
    #ax.plot(x_obs, y_obs, 'kx')

plot_pred_dist("src/foo_prior.csv")
#plot_pred_dist("src/foo_post.csv")

#thetas = np.loadtxt("src/theta.csv")
fig, ax = plt.subplots()
#ax.plot(thetas[:, 1:])

#fig, ax = plt.subplots()
#ax.hist(thetas[:, 1], bins=100, alpha=0.5, density=True)
#ax.hist(thetas[:, 2], bins=100, alpha=0.5, density=True)

plt.show()
