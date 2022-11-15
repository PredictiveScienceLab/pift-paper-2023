"""Generates data for the diffusion example.

Author:
    Alex Alberts
    Ilias Bilionis

Date:
    10/12/2022
"""


__all__ = ["Diffusion1D"]


from typing import NamedTuple, Callable
from numpy.typing import NDArray

import numpy as np
from jax import grad
import jax.numpy as jnp
from scipy.integrate import solve_bvp
from functools import partial

from pift import *

class Diffusion1D(NamedTuple):
    name: str   = "1D diffusion example"
    # The number of discretization points
    n: int      = 1000
    # The source term
    q: Callable = lambda x: jnp.exp(-x)
    # Parameter
    kappa: float = 0.25
    # Left boundary
    a: float    = 0.0
    # Right boundary
    b: float    = 1.0
    # Boundary conditions
    yb: float = (1.0, 0.0)
    # The measurement noise
    sigma: float = 0.1
    # The mumber of observations
    num_samples: int = 100

    def rhs(self, x: NDArray, y: NDArray) -> NDArray:
        """The right hand size of the boundary value problem."""
        return y[1], -1.0 / self.kappa * self.q(x)

    def bc(self, ya: NDArray, yb: NDArray) -> NDArray:
        return np.array([ya[0] - self.yb[0], yb[0] - self.yb[1]])

    def solve(self) -> Callable:
        """Returns the solution as a function."""
        xs = np.linspace(self.a, self.b, self.n)
        ys = np.zeros((2, self.n))
        res = solve_bvp(
            self.rhs,
            self.bc,
            xs,
            ys
        )
        assert res["success"]
        return lambda x: res.sol(x)[0, :]

    def generate_experimental_data(self) -> NDArray:
        f = self.solve()
        xr, yr = uniformly_sample_1d_function_w_Gaussian_noise(
            f,
            num_samples=self.num_sample,
            a=self.a,
            b=self.b,
            sigma=self.sigma
        )
        res = {
            "solution": f,
            "x": xr,
            "y": yr
        }
        return res

    def make_pift_problem(self, V: FunctionParameterization) -> PiftProblem:
        phi = enforce_1d_boundary_conditions(
            V.eval,
            self.a,
            self.yb[0],
            self.b,
            self.yb[1]
        )
        gphi = grad(phi, argnums=0)
        f = lambda x: jnp.exp(- x)
        hamiltonian_density = lambda x, w, theta: (
            0.5 * theta[0] * gphi(x, w) ** 2
            - theta[1] * phi(x, w) * f(x)
        )
        log_theta_prior = lambda theta: 0.0
        xs_all = np.linspace(self.a, self.b, 10000)
        return make_pyro_model(phi, hamiltonian_density, xs_all)
