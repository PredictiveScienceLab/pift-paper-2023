"""Various utilities for defining function spaces.

Authors:
    Alex Alberts
    Ilias Bilionis

Date:
    10/12/2022
"""


__all__ = [
    "FunctionParameterization"
]


from typing import Callable, NewType
from numpy.typing import NDArray

import jax
import jax.numpy as jnp

from . import Basis

PyTree = NewType("PyTree", jax.tree_util.PyTreeDef)
PRNGKey = NewType("PRNGKey", jax.random.PRNGKey)


class FunctionParameterization:
    name: str
    in_dim: int
    out_dim: int
    params: PyTree
    eval: Callable
    veval: Callable

    @property
    def num_params(self):
        return len(self.params)

    def __init__(
        self,
        name: str,
        in_dim: int,
        out_dim: int,
        params: PyTree,
        eval: Callable,
        veval: Callable = None
    ):
        if veval is None:
            if params is None:
                veval = jax.vmap(eval)
            else:
                veval = jax.vmap(eval, (0, None))
        self.name = name
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.params = params
        self.eval = eval
        self.veval = veval

    def __call__(self, x, params=None):
        if params is None:
            if self.params is None:
                return self.veval(x)
            else:
                params = self.params
        return self.veval(x, params)

    @staticmethod
    def from_basis(
        name: str,
        basis: Basis,
        out_dim: int = 1,
        prng_key: PRNGKey = jax.random.PRNGKey(65462),
        weight_initializer: Callable = jax.nn.initializers.glorot_normal(),
        dtype=jnp.float32
    ) -> "FunctionParameterization":
        in_dim = basis.in_dim
        num_weights = basis.dim
        if out_dim == 1:
            params = weight_initializer(prng_key, (num_weights,1), dtype)
            params = params.flatten()
            def f(x, w):
                return jnp.dot(basis.eval(x), w)
        else:
            params = weight_initializer(prng_key, (num_weights, out_dim), dtype)
            def f(x, W):
                return jnp.dot(basis.eval(x), W)
        def vf(x, w):
            return jnp.dot(basis(x), w)
        obj = FunctionParameterization(
            name,
            in_dim,
            out_dim,
            params,
            f,
            vf
        )
        obj.basis = basis
        return obj
