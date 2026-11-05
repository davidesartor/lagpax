"""The separable squared-exponential kernel, k(x, x') = exp(-sum_j (x_j - x'_j)^2 / s_j).

`s` holds squared lengthscales: the conventional lengthscale l of exp(-(x - x')^2 / (2 l^2)) is
sqrt(s / 2).
"""

from jaxtyping import Array, Float, Scalar

import jax.numpy as jnp


def squared_distances(
    left: Float[Array, "n d"], right: Float[Array, "k d"] | None = None
) -> Float[Array, "n k"]:
    right = left if right is None else right
    return jnp.sum(jnp.square(left[:, None, :] - right[None, :, :]), axis=-1)


def correlation(
    left: Float[Array, "n d"],
    right: Float[Array, "k d"],
    squared_lengthscales: Float[Array, "d"],
) -> Float[Array, "n k"]:
    scaled_squared_differences = (
        jnp.square(left[:, None, :] - right[None, :, :]) / squared_lengthscales
    )
    return jnp.exp(-jnp.sum(scaled_squared_differences, axis=-1))


def correlation_with_nugget(
    inputs: Float[Array, "n d"],
    squared_lengthscales: Float[Array, "d"],
    nugget: Scalar,
) -> Float[Array, "n n"]:
    within = correlation(inputs, inputs, squared_lengthscales)
    return jnp.fill_diagonal(within, 1.0 + nugget, inplace=False)
