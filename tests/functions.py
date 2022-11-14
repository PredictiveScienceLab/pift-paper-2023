"""Test the FucntionParameterization class."""

import jax.numpy as jnp
import matplotlib.pyplot as plt

from pift import Fourier1DBasis, FunctionParameterization, mvn_fourier

L = 1.0
num_terms = 100
spectrum = jnp.ones((num_terms,)) / jnp.arange(1, num_terms + 1) ** 4

f = FunctionParameterization.from_basis("f",
                                Fourier1DBasis(L, num_terms),
                                weight_initializer=mvn_fourier(spectrum))

xs = jnp.linspace(0.0, L, 100)
ys = f(xs)

fig, ax = plt.subplots()
ax.plot(xs, ys)
plt.show()
