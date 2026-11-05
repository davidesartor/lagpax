"""The separable squared-exponential kernel.

    k(x, x') = exp(-sum_j (x_j - x'_j)^2 / theta_j)

`theta` is a *squared* lengthscale (laGP's `d`). The conventional lengthscale of
`exp(-0.5 * sum ((x - x') / l)^2)` is `l = sqrt(theta / 2)`; `Prior` can put its bounds and
Gamma prior on either one.
"""

from jaxtyping import Array, Float, Scalar

import jax.numpy as jnp


def sqdist(
    X1: Float[Array, "n1 m"],
    X2: Float[Array, "n2 m"] | None = None,
) -> Float[Array, "n1 n2"]:
    """Squared euclidean distance between the rows of X1 and X2 (X1 itself by default)."""
    if X2 is None:
        X2 = X1
    return jnp.sum(jnp.square(X1[:, None, :] - X2[None, :, :]), axis=-1)


def correlation(
    X1: Float[Array, "n1 m"],
    X2: Float[Array, "n2 m"],
    theta: Float[Array, "m"],
) -> Float[Array, "n1 n2"]:
    """K[i, j] = exp(-sum_k (X1[i, k] - X2[j, k])^2 / theta[k])."""
    return jnp.exp(
        -jnp.sum(jnp.square(X1[:, None, :] - X2[None, :, :]) / theta, axis=-1)
    )


def correlation_matrix(
    X: Float[Array, "n m"],
    theta: Float[Array, "m"],
    nugget: Scalar,
) -> Float[Array, "n n"]:
    """correlation(X, X, theta) with the nugget added to the diagonal."""
    K = correlation(X, X, theta)
    return jnp.fill_diagonal(K, 1.0 + nugget, inplace=False)


def dcorrelation(
    X1: Float[Array, "n1 m"],
    X2: Float[Array, "n2 m"],
    theta: Float[Array, "m"],
    K: Float[Array, "n1 n2"],
) -> Float[Array, "m n1 n2"]:
    """dK/dtheta_k, given K = correlation(X1, X2, theta)."""
    sqdiff = jnp.square(X1[:, None, :] - X2[None, :, :])
    return jnp.einsum("ij,ijk->kij", K, sqdiff / jnp.square(theta))
