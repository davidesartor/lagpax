"""Separable power-exponential covariance, port of laGP's covar_sep.c."""

from jaxtyping import Array, Float, Scalar

import jax.numpy as jnp


def distance(
    X1: Float[Array, "n1 m"],
    X2: Float[Array, "n2 m"] | None = None,
) -> Float[Array, "n1 n2"]:
    """Squared euclidean distance matrix, matching laGP's `distance`."""
    if X2 is None:
        X2 = X1
    return jnp.sum(jnp.square(X1[:, None, :] - X2[None, :, :]), axis=-1)


def covar_sep(
    d: Float[Array, "m"],
    X1: Float[Array, "n1 m"],
    X2: Float[Array, "n2 m"],
) -> Float[Array, "n1 n2"]:
    """K[i,j] = exp(-sum_k (X1[i,k] - X2[j,k])^2 / d[k])."""
    return jnp.exp(-jnp.sum(jnp.square(X1[:, None, :] - X2[None, :, :]) / d, axis=-1))


def covar_sep_symm(
    d: Float[Array, "m"],
    X: Float[Array, "n m"],
    g: Scalar,
) -> Float[Array, "n n"]:
    """covar_sep(d, X, X) with the nugget g added to the diagonal."""
    K = covar_sep(d, X, X)
    return jnp.fill_diagonal(K, 1.0 + g, inplace=False)


def diff_covar_sep(
    d: Float[Array, "m"],
    X1: Float[Array, "n1 m"],
    X2: Float[Array, "n2 m"],
    K: Float[Array, "n1 n2"],
) -> Float[Array, "m n1 n2"]:
    """dK/dd_k, matching laGP's diff_covar_sep (diagonal of the symmetric case is 0)."""
    sqdiff = jnp.square(X1[:, None, :] - X2[None, :, :])
    return jnp.einsum("ij,ijk->kij", K, sqdiff / jnp.square(d))
