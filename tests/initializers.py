"""Test the initializer functions."""

import jax
import jax.numpy as jnp

from pift import mvn_fourier

key = jax.random.PRNGKey(45)
spectrum = jnp.ones((10,)) / jnp.arange(1, 11) ** 2
initializer = mvn_fourier(spectrum)
w = initializer(key)

print(w)
