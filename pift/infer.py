"""Inference methods, i.e., sample from prior or sample from posterior
conditional on parameters.

Inference methods assume that the numpyro model has the following form:

def model(data: Any, **kwargs) -> None:
    pass

with the convention that when data is None, we have the prior.

Any physical parameters to the model can be passed when you are running the
MCMCSampler.

"""


from typing import Any, Callable, NamedTuple
from numpy.typing import NDArray

import jax.numpy as jnp
from jax import random, vmap, grad
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, HMCECS, HMC


__all__ = ["PiftProblem", "make_pyro_model", "MCMCSampler"]




class PiftProblem(NamedTuple):
    phi: Callable
    vphi: Callable
    vwphi: Callable
    hamiltonian: Callable
    pyro_model: Callable
    xs: NDArray
    log_theta_prior: Callable


def make_pyro_model(
    phi: Callable,
    hamiltonian_density: Callable,
    xs_all: NDArray,
    log_theta_prior: Callable,
    weight_mean=None,
    weight_scale=None,
    sigma_rate=1.0,
    sigma=None
) -> Callable:
    vphi = vmap(phi, (0, None))
    vwphi = vmap(vphi, (None, 0))
    v_hamiltonian_density = vmap(
        hamiltonian_density,
        (0, None, None)
    )

    hamiltonian = lambda xs, w, theta: (
        jnp.mean(v_hamiltonian_density(xs, w, theta))
    )

    unormalized_log_field_prior = lambda xs, w, theta: -hamiltonian(xs, w, theta)

    def model(
            data=None,
            theta= None,
            weight_mean=weight_mean,
            weight_scale=weight_scale,
            sigma_rate=sigma_rate,
            sigma=sigma,
            xs_all: NDArray = xs_all,
            mini_batch: int = 1000
        ) -> None:

            assert theta is not None
            assert weight_mean is not None
            assert weight_scale is not None

            # Subsample
            with numpyro.plate("batched_space", len(xs_all), mini_batch):
                xs = numpyro.subsample(xs_all, event_dim=0)

            w = numpyro.sample(
                "w",
                dist.Normal(
                    weight_mean,
                    weight_scale
                )
            )

            numpyro.factor(
                "unormalized_log_prior",
                unormalized_log_field_prior(xs, w, theta)
            )

            if data is not None:
                if sigma_rate is not None:
                    s = numpyro.sample(
                        "s",
                        dist.Exponential(
                            sigma_rate
                        )
                    )
                else:
                    assert sigma is not None
                    s = sigma
                xr, yr = data
                phir = vphi(xr, w)
                with numpyro.plate("observed_data", len(yr)):
                    return numpyro.sample(
                        "predictions",
                        dist.Normal(
                            loc=phir,
                            scale=s
                        ),
                        obs=yr
                    )

    return PiftProblem(phi, vphi, vwphi, hamiltonian, model, xs_all, log_theta_prior)


class MCMCSampler:
    """Makes an MCMC sampler to sample from the prior or the psoterior.

    Arguments:
    model - the model to sample from.
    init_params - the initial parameters for MCMC.
    rng_key - a random generator.
    continue_init_params - the next time you are called, start from the last
                           parameters.
    """

    def __init__(
        self,
        pyro_model: Callable,
        rng_key: random.PRNGKey,
        **kwargs
    ) -> None:
        self.pyro_model = pyro_model
        self.mcmc_kwargs = kwargs
        self.rng_key = rng_key

    def sample(self, **kwargs) -> None:
        rng_key, self.rng_key = random.split(self.rng_key)
        self.mcmc = MCMC(HMCECS(NUTS(self.pyro_model)), **self.mcmc_kwargs)
        self.mcmc.run(
            rng_key=rng_key,
            **kwargs
        )
        samples = self.mcmc.get_samples()
        return samples
