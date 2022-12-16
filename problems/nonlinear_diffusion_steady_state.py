"""Generates data for the diffusion example.

Author:
    Alex Alberts
    Ilias Bilionis

Date:
    10/12/2022
"""


__all__ = ["Cubic1D"]


from typing import NamedTuple, Callable
from numpy.typing import NDArray

import scipy
import numpy as np
from jax import grad
import jax.numpy as jnp
from scipy.integrate import solve_bvp
from functools import partial

from pift import *

class Cubic1D(NamedTuple):
    name: str   = "1D cubic nonlinear example"
    # The number of discretization points
    n: int      = 1000
    # The ground truth source term
    q: Callable = lambda x: jnp.cos(4.0*x)
    # Parameter
    kappa: float = 1.0
    # Parameter
    D: float = .1
    # Left boundary
    a: float    = 0.0
    # Right boundary
    b: float    = 1.0
    # Boundary conditions
    yb: float = (0.0, 0.0)
    # The measurement noise
    sigma: float = 0.01
    # The mumber of observations
    num_samples: int = 20
    # The measurement noise for the source term
    sigma_source: float = 0.01
    num_samples_source: int = 5

    def rhs(self, x: NDArray, y: NDArray) -> NDArray:
        """The right hand size of the boundary value problem."""
        return y[1], (self.kappa * y[0] **3 + self.q(x))/self.D

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
            n=self.num_samples,
            a=self.a,
            b=self.b,
            sigma=self.sigma
        )
        x_source, y_source = uniformly_sample_1d_function_w_Gaussian_noise(
            self.q,
            n=self.num_samples_source,
            a=self.a,
            b=self.b,
            sigma=self.sigma_source
        )
        res = {
            "solution": f,
            "x": xr,
            "y": yr,
            "x_source": x_source,
            "y_source": y_source
        }
        return res

    def make_pift_problem(
        self,
        x_obs: NDArray,
        f_obs: NDArray,
        V1: FunctionParameterization,
        V2: FunctionParameterization,
        beta: float = 100.0,
        v_precision: float = 0.1,
    ) -> PiftProblem:
        phi = enforce_1d_boundary_conditions(
            V1.eval,
            self.a,
            self.yb[0],
            self.b,
            self.yb[1]
        )

        gphi = grad(phi, argnums=0)
        g2phi = grad(gphi, argnums=0)

        # def forcing_basis(x):
        #     tmp = jnp.pi*jnp.arange(1,4) * x
        #     return jnp.hstack([jnp.ones(()), jnp.cos(tmp), jnp.sin(tmp)])

        #f = V2.eval
        f = lambda x: jnp.cos(4 * x)
        # Theta is as follows:
        # theta = [D, kappa, v]
        # hamiltonian_density = lambda x, w, theta: theta[0] * (
        #     # 0.5 * theta[0] * gphi(x, w) ** 2
        #     # + 0.25 * theta[1] * phi(x, w) ** 4
        #     0.5 * theta[1] * gphi(x, w) ** 2
        #     + 0.25 * theta[2] * phi(x, w) ** 4
        #     # + phi(x, w) * f(x, theta[2:])
        #     + phi(x, w) * f(x)
        # )
        hamiltonian_density = lambda x, w, theta: beta * (
            0.5 * theta[0] * gphi(x, w) ** 2
            + 0.25 * theta[1] * phi(x, w) ** 4
            + phi(x, w) * f(x)
        )
        # hamiltonian_density = lambda x, w, theta: jnp.exp(theta[0]) * (
        #     jnp.exp(theta[1]) * g2phi(x, w) - jnp.exp(theta[2]) * phi(x, w) ** 3 - f(x)
        # ) ** 2

        # Calculate prior for f (after seeing the data x_obs, f_obs)
        # Design matrix
        K = V2.basis.dim
        Psi = np.array(V2.basis.veval(x_obs))
        Cinv = Psi.T @ Psi / self.sigma_source ** 2 + v_precision * np.eye(K)
        L = scipy.linalg.cholesky(Cinv, lower=True)
        mu = scipy.linalg.cho_solve((L, True), (Psi.T @ f_obs) / self.sigma_source ** 2)

        # # Verify that the predictions for the source term look okay
        # import matplotlib.pyplot as plt
        # xs = np.linspace(self.a, self.b, 100)
        # fig, ax = plt.subplots()
        # ax.plot(xs, self.q(xs))
        # pq = V2.veval(xs, mu)
        # ax.plot(xs, pq, 'r--')
        # ax.plot(x_obs, f_obs, 'kx')
        # S = 10
        # for _ in range(S):
        #     w = mu + scipy.linalg.cho_solve((L, True), np.random.randn(K))
        #     qS = V2.veval(xs, w)
        #     ax.plot(xs, qS, 'r', lw=0.1)
        # plt.show()
        # quit()

        L = jnp.array(L)
        mu = jnp.array(mu)


        def log_theta_prior(theta):
            return 0.0
            #return -jnp.sum(jnp.log(theta))
            #D = jnp.log(jnp.exp(-theta[0])*jnp.heaviside(theta[0],1.)+1e-8) # picking exp(1.)
            #kappa = jnp.log(jnp.exp(-theta[1])*jnp.heaviside(theta[1],1.)+1e-8) # picking exp(1.)
            #return 0.0
            v = theta[2:]
            r = -0.5 * jnp.sum( (L.T @ (mu - v)) ** 2)
            return r
            #return r - jnp.log(theta[0]) - jnp.log(theta[1])
            # Alex add to r the log of the prior of the other stuff
            #return r - D - kappa

        xs_all = np.linspace(self.a, self.b, 10000)
        problem = make_pyro_model(
            phi,
            hamiltonian_density,
            xs_all,
            log_theta_prior,
            weight_mean=jnp.zeros((V1.num_params,)),
            weight_scale=100.0 * jnp.ones((V1.num_params,))
        )
        return problem, mu, L
