"""Some common utility functions.

Author:
    Ilias Bilionis

Date:
    10/12/2022

"""


__all__ = [
    "sample_1d_function",
    "uniformly_sample_1d_function",
    "uniformly_sample_1d_function_w_Gaussian_noise",
    "eval_on_grid"
]


from typing import Callable, NewType, Tuple, List
from numpy.typing import NDArray
import scipy.stats as st
import numpy as np


RV = NewType("RV", st.rv_continuous)


def sample_1d_function(
    f: Callable,
    X: RV,
    n: int,
    eps: RV=None
) -> Tuple[NDArray, NDArray]:
    """Randomly samples a 1D function.

    Arguments

        f       --      A function from R to R.
        X       --      A random variable to sample from.
        n       --      The number of times to sample.
        eps     --      A random variable (scipy.stats) from which to sample
                        the noise. If not specified, nothing is added.
    """
    if eps is None:
        noise = lambda _n: np.zeros((_n,))
    else:
        noise = lambda _n: eps.rvs(_n)
    xs = X.rvs(n)
    ys = f(xs) + noise(n)
    return xs, ys

def uniformly_sample_1d_function(
    f: Callable,
    n: int,
    a: float = 0.0,
    b: float = 1.0,
    eps: RV=None
) -> Tuple[NDArray, NDArray]:
    X = st.uniform(loc=a, scale=b - a)
    return sample_1d_function(f, X, n, eps)

def uniformly_sample_1d_function_w_Gaussian_noise(
    f: Callable,
    n: int,
    a: float = 0.0,
    b: float = 1.0,
    sigma: float=0.1
) -> Tuple[NDArray, NDArray]:
    eps = st.norm(0.0, sigma)
    return uniformly_sample_1d_function(f, n, a, b, eps)


def eval_on_grid(f: Callable, X: List[NDArray]) -> NDArray:
    """Evaluate function on a grid."""
    in_flat = np.hstack(
        [x.flatten()[:, None] for x in X]
    )
    out_flat = f(in_flat)
    if out_flat.ndim == 1:
        return out_flat.reshape(X[0].shape)
    else:
        assert out_flat.ndim == 2
        _, m = out_flat.shape
        return out_flat.reshape(X[0].shape + (m, ))
