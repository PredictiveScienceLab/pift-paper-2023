"""Some classes to facilitate initializing weights of functional approximations.

Author:
    Ilias Bilionis

Date:
    10/13/2022
"""

__all__ = [
    "mvn",
    "mvn_fourier"
]

from typing import Callable, NewType, List
from numpy.typing import NDArray

import jax
import jax.numpy as jnp
import numpy

PRNGKey = NewType("PRNGKey", jax.random.PRNGKey)


def mvn(
    mean: NDArray,
    cov: NDArray,
    dtype: "ScalarType" = jnp.float32
) -> Callable:
    """This can be used to initialize vectors of a known size."""
    normal_initializer = jax.nn.initializers.normal(stddev=1.0, dtype=dtype)
    shape = (mean.shape[0], 1)
    if cov.ndim == 1:
        sigma = jnp.sqrt(cov)
        def initializer(prng_key, shape_in=shape, dtype=dtype):
            z = normal_initializer(prng_key, shape, dtype).flatten()
            return mean + sigma * z
    elif cov.ndim == 2:
        L = np.linalg.cholesky(cov)
        def initializer(prng_key, shape_in=shape, dtype=dtype):
            assert shape_in == shape
            return mean + jnp.dot(
                            L,
                            normal_initializer(prng_key, shape, dtype)
                          ).flatten()
    else:
        raise ValueError
    return initializer


def mvn_fourier(
    cov: NDArray,
    mean: NDArray = None,
    dtype: "ScalarType" = jnp.float32
) -> Callable:
    if mean is None:
        mean = jnp.zeros((cov.shape[0],))
    full_mean = jnp.hstack([mean, mean[1:]])
    if cov.ndim == 1:
        full_cov = jnp.hstack([cov, cov[1:]])
    elif cov.ndim == 2:
        full_cov = jax.scipy.linalg.block_diag(cov, cov[1:, 1:])
    return mvn(full_mean, full_cov, dtype)
