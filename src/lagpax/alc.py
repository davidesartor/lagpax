"""Active learning Cohn criterion, port of laGP's alcGPsep (gp_sep.c + ieci.c)."""

from jaxtyping import Array, Float

import jax.numpy as jnp

from .covar import covar_sep
from .gp import SDEPS, GPsep


def alc_gpsep(
    gp: GPsep,
    Xcand: Float[Array, "ncand m"],
    Xref: Float[Array, "nref m"],
) -> Float[Array, "ncand"]:
    """Expected reduction in predictive variance at Xref from adding each candidate."""
    k = jnp.where(gp.mask, covar_sep(gp.d, Xref, gp.X), 0.0)  # (nref, n)
    kx = jnp.where(gp.mask, covar_sep(gp.d, Xcand, gp.X), 0.0)  # (ncand, n)
    kxy = covar_sep(gp.d, Xcand, Xref)  # (ncand, nref)

    gvec = kx @ gp.Ki
    mui = 1.0 + gp.g - jnp.sum(kx * gvec, axis=-1)
    gvec = -gvec / mui[:, None]

    kg = gvec @ k.T  # (ncand, nref)
    ktKikx = (
        jnp.square(kg) * mui[:, None] + 2.0 * kg * kxy + jnp.square(kxy) / mui[:, None]
    )

    df = gp.n
    alc = jnp.mean(gp.phi * ktKikx / df * (df / (df - 2.0)), axis=-1)
    return jnp.where(mui <= SDEPS, -jnp.inf, alc)
