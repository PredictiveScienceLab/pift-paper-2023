"""Some learning algorithms."""


__all__ = [
    "learn",
    "GradMinusLogMarginalLikelihood",
    "adam",
    "cg",
    "newton_raphson",
    "stochastic_gradient_langevin_dynamics",
    "stochastic_newton_langevin_dynamics"
]

import sys
from typing import Callable, Any
from numpy.typing import NDArray
import numpy as np
from jax import random, grad, vmap, hessian, jacobian, jit
from functools import partial


from .infer import *


def learn(
    hamiltonian: Callable,
    log_prior: Callable,
    model: Callable,
    data: Any,
    theta: NDArray,
    init_params: NDArray,
    out_file: str,
    rng_key: random.PRNGKey,
    xs: NDArray,
    do_nr: bool = True,
    nr_params = {},
    do_sgld: bool = True,
    sgld_params = {},
    mcmc_params = {}
):
    """Trains the model doing first NR and the SGLD."""
    func = GradMinusLogMarginalLikelihood(
        hamiltonian,
        log_prior,
        model,
        data=data,
        return_hessian=True,
        init_params=init_params,
        rng_key=rng_key,
        **mcmc_params
    )

    with open(out_file, "w") as fd:
        if do_nr:
            theta, H = newton_raphson(
                func,
                theta,
                fd=fd,
                **nr_params
            )

        if do_sgld:
            theta, H = stochastic_gradient_langevin_dynamics(
                func,
                theta,
                M=H,
                fd=fd,
                **sgld_params
            )


class GradMinusLogMarginalLikelihood:

    def __init__(
        self,
        problem: PiftProblem,
        data: Any,
        rng_key: random.PRNGKey,
        init_params: NDArray = None,
        xs: NDArray = np.linspace(0.0, 1.0, 1000),
        return_hessian: bool = False,
        disp: bool = False,
        **kwargs
    ) -> None:
        model = problem.pyro_model
        hamiltonian = problem.hamiltonian
        log_prior = problem.log_theta_prior

        prior_key, post_key = random.split(rng_key)
        model_prior = partial(model, data=None)
        model_post = partial(model, data=data)
        self.prior_mcmc = MCMCSampler(
            model_prior,
            rng_key=prior_key,
            init_params=init_params,
            **kwargs
        )
        self.post_mcmc = MCMCSampler(
            model_post,
            rng_key=post_key,
            init_params=init_params,
            **kwargs
        )
        self.xs = xs
        self.hamiltonian = lambda w, theta: hamiltonian(self.xs, w, theta)
        self.vhamiltonian = vmap(self.hamiltonian, (0, None))

        self.grad_hamiltonian = grad(self.hamiltonian, argnums=1)
        # Vectorize these with respect to w
        self.vgrad_hamiltonian = vmap(self.grad_hamiltonian, (0, None))
        self.expect_grad_H = lambda ws, theta: (
            np.mean(self.vgrad_hamiltonian(ws, theta), axis=0)
        )
        self.log_prior = log_prior
        self.grad_log_prior = grad(log_prior)

        def func_base(theta):
            if disp:
                sys.stdout.write(f"evaluating likelihood at {theta}...")
                sys.stdout.flush()
            self.prior_samples = self.prior_mcmc.sample(theta=theta)
            prior_ws = self.prior_samples["w"]
            self.post_samples = self.post_mcmc.sample(theta=theta)
            post_ws = self.post_samples["w"]
            grad_H_prior = self.vgrad_hamiltonian(prior_ws, theta)
            grad_H_post = self.vgrad_hamiltonian(post_ws, theta)
            grad_U_post = np.mean(grad_H_post, axis=0)
            grad_U_prior = np.mean(grad_H_prior, axis=0)
            grad_U = grad_U_post - grad_U_prior
            grad_minus_log_p = grad_U - self.grad_log_prior(theta)

            grad_U_post_v = np.var(grad_H_post, axis=0) / post_ws.shape[0]
            grad_U_prior_v = np.var(grad_H_prior, axis=0) / prior_ws.shape[0]
            grad_minus_log_p_v = grad_U_post_v + grad_U_prior_v

            grad_U_post_c = np.cov(grad_H_post, rowvar=False) / post_ws.shape[0]
            grad_U_prior_c = np.cov(grad_H_prior, rowvar=False) / prior_ws.shape[0]
            grad_minus_log_p_c = grad_U_post_c + grad_U_prior_c
            if disp:
                sys.stdout.write(f"-> {grad_minus_log_p}\n")
                sys.stdout.flush()
            return (
                prior_ws, post_ws,
                grad_H_prior, grad_H_post,
                grad_U,
                grad_minus_log_p_c,
                grad_minus_log_p
            )

        if return_hessian:
            self.grad2_hamiltonian = hessian(self.hamiltonian, argnums=1)
            self.vgrad2_hamiltonian = jit(
                vmap(self.grad2_hamiltonian, (0, None))
            )
            self.grad2_log_prior = hessian(self.log_prior)

            def func(theta):
                (
                    prior_ws, post_ws,
                    grad_H_prior, grad_H_post,
                    grad_U,
                    grad_minus_log_p_v,
                    grad_minus_log_p
                ) = (
                    func_base(theta)
                )
                grad2_H_prior = self.vgrad2_hamiltonian(prior_ws, theta)
                grad2_H_post = self.vgrad2_hamiltonian(post_ws, theta)
                grad3_H_prior = self.vgrad3_hamiltonian(prior_ws, theta)
                grad3_H_post = self.vgrad3_hamiltonian(post_ws, theta)
                grad2_U = (
                    np.cov(grad_H_prior, rowvar=False)
                    -
                    np.cov(grad_H_post, rowvar=False)
                    +
                    np.mean(grad2_H_post, axis=0)
                    +
                    np.mean(grad2_H_prior, axis=0)
                )
                grad2_minus_log_p = grad2_U - self.grad2_log_prior(theta)
                grad2_minus_log_p_inv = np.linalg.inv(grad2_U)
                return grad_minus_log_p, grad2_minus_log_p_inv, grad_minus_log_p_v
        else:
            func = lambda theta: func_base(theta)[-1]
        self.func = func

    def __call__(self, theta):
        return self.func(theta)


def print_progress(
    disp: bool,
    i: int,
    theta: NDArray,
    g: NDArray,
    prefix: str="",
    fd: Any = sys.stdout
):
    if disp:
        s = prefix + f" {i:d}\t"
        for t in theta:
            s += f"{t:1.3f} "
        for gg in g:
            s += f"{gg:1.6e} "
        print(s, file=fd, flush=True)


def adam(
    grad_f: Callable,
    theta0: NDArray,
    alpha: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    maxit: int = 1000,
    disp: bool = True
) -> NDArray:
    assert theta0.ndim == 1
    assert alpha > 0.0
    assert beta1 > 0.0 and beta1 < 1.0
    assert beta2 > 0.0 and beta2 < 1.0
    assert epsilon > 0.0
    assert maxit >= 0

    d = theta0.shape[0]
    m = np.zeros((d,))
    v = np.zeros((d,))
    t = 0
    theta = theta0
    while t <= maxit:
        t += 1
        g = grad_f(theta)
        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * g ** 2
        m_hat = m / (1.0 - beta1 ** t)
        v_hat = v / (1.0 - beta2 ** t)
        theta = theta - alpha * m_hat / (np.sqrt(v_hat) + epsilon)
        print_progress(disp, theta, g)
    return theta


def cg(
    A: NDArray,
    b: NDArray,
    t: int,
    x0: NDArray
) -> NDArray:
    """Perform t steps of the Conjugate-Gradient algorithm starting at x0."""
    assert A.ndim == 2
    assert b.ndim == 1
    assert A.shape[0] == A.shape[1]
    assert b.shape[0] == A.shape[0]
    assert t >= 1
    assert x0.ndim == 1
    assert x0.shape[0] == A.shape[0]

    r = b - np.dot(A, x0)
    p = r
    rTr = np.dot(r, r)
    x = x0
    for k in range(t):
        Ap = np.dot(A, p)
        a = np.dot(r, r) / np.dot(p, Ap)
        x = x + a * p
        r = r - a * Ap
        new_rTr = np.dot(r, r)
        b = new_rTr / rTr
        rTr = new_rTr
        p = r + b * p
    return x


def newton_raphson(
    func: Callable,
    theta0: NDArray,
    alpha: float = 1.0,
    maxit: int = 1000,
    disp: bool = True,
    max_beta: float = 10000.0,
    beta_index: int = -1,
    fd: Any = sys.stdout
) -> NDArray:
    i = 0
    theta = theta0
    while i <= maxit:
        i += 1
        g, H, C, _ = func(theta)
        # Check if the step is problematic
        step = H @ g
        C_step = H @ C @ H
        next_beta = theta[beta_index] - alpha * step[beta_index]
        if next_beta >= max_beta:
            if disp:
                s = "*** Stopping NR because beta became too big {theta[-1]:1.3e}"
                print(s)
                break
        theta = theta - alpha * step
        print_progress(disp, i, theta, g, prefix="nr", fd=fd)
    return theta, H


def stochastic_gradient_langevin_dynamics(
    func: Callable,
    theta0: NDArray,
    alpha: float = 0.1,
    beta: float = 0.0,
    gamma: float = 0.51,
    maxit: int = 1000,
    maxit_after_which_epsilon_is_fixed=500,
    M: NDArray = None,
    disp: bool = True,
    fd: Any = sys.stdout
) -> NDArray:
    """Implements Welling and Teh, 2011."""
    if M is None:
        M = np.eye(theta0.shape[0])
    L = np.linalg.cholesky(M)
    i = 0
    theta = theta0
    prev_epsilon = 0.0
    while i <= maxit:
        i += 1
        if i >= maxit_after_which_epsilon_is_fixed:
            epsilon = prev_epsilon
        else:
            epsilon = alpha / (beta + i) ** gamma
            prev_epsilon = epsilon
        g = func(theta)[0]
        eta = np.sqrt(2.0 * epsilon) * L @ np.random.randn(*theta.shape)
        theta = theta - epsilon * (M @ g) + eta
        print_progress(disp, i, theta, g, prefix="sgld", fd=fd)
    return theta


def stochastic_newton_langevin_dynamics(
    func: Callable,
    theta0: NDArray,
    alpha: float = .01,
    beta: float = 10.0,
    gamma: float = 0.51,
    t: int = 2,
    maxit: int = 1000,
    disp: bool = True
) -> NDArray:
    """This is an algorithm I came up with. I am not sure that it converges."""
    i = 0
    theta = theta0
    while i <= maxit:
        i += 1
        epsilon = alpha / (beta + i) ** gamma
        g, grad2_U_inv, Gamma = func(theta)
        # H += np.eye(*H.shape) * 1e-3
        L = np.linalg.cholesky(grad2_U_inv)
        d = grad2_U_inv @ g + Gamma
        #L[L <= 0.0] = 0.0
        w = L @ np.random.randn(*theta.shape)
        theta = theta - epsilon * d + np.sqrt(2.0 * epsilon) * w
        print_progress(disp, theta, g)
    return theta

# def hessian_approximated_mcmc()
