"""Design criteria: score candidate points against a reference set.

Every criterion has the same signature, `(gp, Xcand, Xref) -> Float[Array, "ncand"]`, so
anything with that shape can be handed to `Local(criterion=...)`. A criterion that is
better when smaller carries `minimize = True`; the greedy search reads that attribute
rather than hardcoding a direction. Criteria needing configuration are factories.
"""

from typing import Callable, Optional
from jaxtyping import Array, Float

import jax.numpy as jnp
from jax.scipy.special import betainc, gammaln

from .covar import covar
from .gp import SDEPS, GP

Criterion = Callable[
    [GP, Float[Array, "ncand m"], Float[Array, "nref m"]], Float[Array, "ncand"]
]


def _cond_var(
    gp: GP,
    Xcand: Float[Array, "ncand m"],
    Xref: Float[Array, "nref m"],
) -> tuple[Float[Array, "ncand nref"], Float[Array, "ncand"]]:
    """Rank-1 variance reduction at each Xref from adding each candidate, and each mui.

    `ktKikx` is laGP's quantity in alcGPsep/ieciGPsep: the amount by which the predictive
    variance at a reference location drops once the candidate joins the design.
    """
    k = jnp.where(gp.mask, covar(gp.d, Xref, gp.X), 0.0)  # (nref, n)
    kx = jnp.where(gp.mask, covar(gp.d, Xcand, gp.X), 0.0)  # (ncand, n)
    kxy = covar(gp.d, Xcand, Xref)  # (ncand, nref)

    gvec = kx @ gp.Ki
    mui = 1.0 + gp.g - jnp.sum(kx * gvec, axis=-1)
    gvec = -gvec / mui[:, None]

    kg = gvec @ k.T  # (ncand, nref)
    ktKikx = (
        jnp.square(kg) * mui[:, None] + 2.0 * kg * kxy + jnp.square(kxy) / mui[:, None]
    )
    return ktKikx, mui


def alc(
    gp: GP,
    Xcand: Float[Array, "ncand m"],
    Xref: Float[Array, "nref m"],
) -> Float[Array, "ncand"]:
    """Expected reduction in predictive variance at Xref from adding each candidate."""
    ktKikx, mui = _cond_var(gp, Xcand, Xref)
    df = gp.df
    out = jnp.mean(gp.phi * ktKikx / df * (df / (df - 2.0)), axis=-1)
    return jnp.where(mui <= SDEPS, -jnp.inf, out)


def nn(
    gp: GP,
    Xcand: Float[Array, "ncand m"],
    Xref: Float[Array, "nref m"],
) -> Float[Array, "ncand"]:
    """Nearest neighbour: minus the distance to the closest reference location."""
    return -jnp.min(
        jnp.sum(jnp.square(Xcand[:, None, :] - Xref[None]), axis=-1), axis=-1
    )


def _student_t_ei(
    mean: Float[Array, "..."],
    s2: Float[Array, "..."],
    df: Float[Array, ""],
    fmin: Float[Array, ""],
) -> Float[Array, "..."]:
    """E[max(fmin - Y, 0)] under a Student-t(df) with location `mean` and scale^2 `s2`.

    The predictive here is Student-t, not Gaussian; using the Gaussian formula gives
    answers that look almost right and are not.
    """
    sd = jnp.sqrt(jnp.maximum(s2, 0.0))
    sd = jnp.where(sd > 0, sd, 1.0)
    z = (fmin - mean) / sd

    log_pdf = (
        gammaln(0.5 * (df + 1.0))
        - gammaln(0.5 * df)
        - 0.5 * jnp.log(jnp.pi * df)
        - 0.5 * (df + 1.0) * jnp.log1p(jnp.square(z) / df)
    )
    pdf = jnp.exp(log_pdf)
    cdf = _student_t_cdf(z, df)
    return sd * ((fmin - mean) / sd * cdf + (df + jnp.square(z)) / (df - 1.0) * pdf)


def _student_t_cdf(z: Float[Array, "..."], df: Float[Array, ""]) -> Float[Array, "..."]:
    """Student-t CDF via the regularized incomplete beta, valid for df > 0."""
    x = df / (df + jnp.square(z))
    tail = 0.5 * betainc(0.5 * df, 0.5, x)
    return jnp.where(z <= 0, tail, 1.0 - tail)


def ieci(
    fmin: Float[Array, ""],
    w: Optional[Float[Array, "nref"]] = None,
    nugget: bool = True,
) -> Criterion:
    """Integrated expected conditional improvement, port of ieciGPsep. Lower is better.

    Scores a candidate by how much adding it improves the prospect of finding an optimum
    at the reference locations, rather than by how much it reduces variance. `fmin` is
    the incumbent best value and changes every step of a BO loop, so it stays an argument.
    """

    def criterion(gp, Xcand, Xref):
        ktKikx, mui = _cond_var(gp, Xcand, Xref)
        ref = gp.predict(Xref, lite=True, nugget=nugget)
        df = gp.df

        # adding a candidate sharpens the variance at Xref; the mean is unchanged because
        # the response at the candidate is unknown
        s2 = ref.s2 - gp.phi * ktKikx / df
        eci = _student_t_ei(ref.mean, jnp.maximum(s2, 0.0), df, fmin)

        # laGP averages over the reference set and lets `w` scale it, rather than
        # normalizing by sum(w) -- so weights change the magnitude, not only the balance
        eci = eci if w is None else w * eci
        out = jnp.mean(eci, axis=-1)
        return jnp.where(mui <= SDEPS, jnp.inf, out)

    criterion.minimize = True
    return criterion


def _named(criterion) -> Criterion:
    """Resolve a criterion name to a function; pass callables through unchanged."""
    if callable(criterion):
        return criterion
    if criterion == "ieci":
        raise ValueError(
            "ieci needs an incumbent: pass criterion=lagpax.ieci(fmin=...)"
        )
    try:
        return {"alc": alc, "nn": nn}[criterion]
    except KeyError:
        raise ValueError(f"unknown criterion {criterion!r}") from None
