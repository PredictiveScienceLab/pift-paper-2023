"""Implementation of a Fourier basis in Jax.

Authors:
    Alex Alberts
    Ilias Bilionis

Date:
    10/12/2022

"""


__all__ = [
    "Basis",
    "Fourier1DBasis",
    "Fourier2DBasis"
]


from typing import Callable, NamedTuple, Tuple, Union
from numpy.typing import NDArray
import jax
import jax.numpy as jnp
import math


class Basis:
    """A class representing the concept of a basis."""

    name: str
    in_dim: int
    dim: int
    eval: Callable
    veval: Callable

    def __init__(
        self,
        name: str,
        in_dim: int,
        dim: int,
        eval: Callable,
        veval: Callable = None
    ):
        self.name = name
        self.in_dim = in_dim
        self.dim = dim
        self.eval = eval
        if veval is None:
            veval = jax.vmap(self.eval)
        self.veval = veval

    def __call__(self, x: NDArray) -> NDArray:
        return self.veval(x)


class Fourier1DBasis(Basis):
    """Makes a Fourier basis in the domain [0, L]."""

    L: float

    def __init__(self, L: float, num_terms: int, name: str = "Fourier 1D"):
        self.L = L
        self._coeffs = jnp.pi * jnp.arange(1, num_terms) / L
        def f(x):
            tmp = self._coeffs * x
            return jnp.hstack([jnp.ones(()), jnp.cos(tmp), jnp.sin(tmp)])
        dim = 1 + (num_terms - 1) * 2
        super().__init__(name, 1, dim, f)


class Fourier2DBasis(Basis):
    """Makes a Fourier basis in the domain [0, L1] x [0, L2]."""

    L: Tuple[float, float]

    def __init__(
        self,
        L: Union[float,Tuple[float,float]],
        num_terms_per_dim: Union[int,Tuple[int,int]],
        name: str = "Fourier 2D"
    ):
        if isinstance(L, float):
            L = (L, L)
        self.L = L
        if isinstance(num_terms_per_dim, int):
            num_terms_per_dim = (num_terms_per_dim, num_terms_per_dim)
        basis_per_dim = tuple(
            Fourier1DBasis(l, n) for l, n in zip(L, num_terms_per_dim)
        )
        self._basis_per_dim = basis_per_dim
        def f(x):
            return jnp.kron(*(b.eval(xc) for b, xc in zip(basis_per_dim, x)))
        dim = math.prod((b.dim for b in basis_per_dim))
        super().__init__(name, 2, dim, f)
