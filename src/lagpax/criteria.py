"""Design criteria: score candidate points against a reference set.

A criterion is a small frozen dataclass registered as a JAX pytree. Calling one has the
signature `(gp, Xcand, Xref) -> Float[Array, "ncand"]`. Two class attributes describe it
to the greedy loop: `minimize` (lower is better) and `cached` (it accepts a precomputed
`kx = correlation(Xcand, gp.X, gp.theta)`, so the loop can carry that matrix forward one column per
step instead of rebuilding it).

Configuration is split the JAX way. Settings that change the program (`IECI.nugget`) are
static fields; settings that are numbers (`IECI.fmin`, `IECI.w`) are pytree leaves, so a
Bayesian-optimization loop that moves `fmin` every step reuses one compiled program. `alc`
and `nn` are ready-made instances for direct calls.
"""

from collections.abc import Callable
from typing import ClassVar, Optional
from jaxtyping import Array, Float

import jax.numpy as jnp
from jax.scipy.special import betainc, gammaln

from ._config import pytree_config, static
from .gp import SDEPS, GP, _moments
from .kernel import correlation


class Criterion:
    """Base class. Subclasses are `@pytree_config` dataclasses implementing `__call__`."""

    minimize: ClassVar[bool] = False
    cached: ClassVar[bool] = False

    def __call__(
        self,
        gp: GP,
        Xcand: Float[Array, "ncand m"],
        Xref: Float[Array, "nref m"],
        kx: Optional[Float[Array, "ncand n"]] = None,
    ) -> Float[Array, "ncand"]:
        raise NotImplementedError


def _cond_var(
    gp: GP,
    Xcand: Float[Array, "ncand m"],
    Xref: Float[Array, "nref m"],
    kx: Optional[Float[Array, "ncand n"]] = None,
) -> tuple[Float[Array, "ncand nref"], Float[Array, "ncand"]]:
    """Rank-1 variance reduction at each Xref from adding each candidate, and each mui.

    `ktKikx` is laGP's quantity in alcGPsep/ieciGPsep: the amount by which the predictive
    variance at a reference location drops once the candidate joins the design.

    `kx` is `correlation(Xcand, gp.X, gp.theta)` and may be supplied precomputed: the greedy loop
    adds one design point per step, so it can carry the columns forward instead of
    rebuilding the whole cross-covariance every step.
    """
    k = jnp.where(gp.mask, correlation(Xref, gp.X, gp.theta), 0.0)  # (nref, n)
    kx = correlation(Xcand, gp.X, gp.theta) if kx is None else kx  # (ncand, n)
    kx = jnp.where(gp.mask, kx, 0.0)
    kxy = correlation(Xcand, Xref, gp.theta)  # (ncand, nref)

    gvec = kx @ gp.Ki
    mui = 1.0 + gp.nugget - jnp.sum(kx * gvec, axis=-1)
    gvec = -gvec / mui[:, None]

    kg = gvec @ k.T  # (ncand, nref)
    ktKikx = (
        jnp.square(kg) * mui[:, None] + 2.0 * kg * kxy + jnp.square(kxy) / mui[:, None]
    )
    return ktKikx, mui


def _alc(
    gp: GP,
    Xcand: Float[Array, "ncand m"],
    Xref: Float[Array, "nref m"],
    kx: Optional[Float[Array, "ncand n"]] = None,
) -> Float[Array, "ncand"]:
    """Expected reduction in predictive variance at Xref from adding each candidate."""
    ktKikx, mui = _cond_var(gp, Xcand, Xref, kx)
    df = gp.dof
    out = jnp.mean(gp.phi * ktKikx / df * (df / (df - 2.0)), axis=-1)
    return jnp.where(mui <= SDEPS, -jnp.inf, out)


def _nn(
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


def _ieci(gp, Xcand, Xref, kx, fmin, w, nugget):
    """IECI's score; see `IECI`."""
    ktKikx, mui = _cond_var(gp, Xcand, Xref, kx)
    ref_mean, ref_scale = _moments(gp, Xref, nugget)
    df = gp.dof

    # adding a candidate sharpens the variance at Xref; the mean is unchanged because
    # the response at the candidate is unknown
    s2 = ref_scale - gp.phi * ktKikx / df
    eci = _student_t_ei(ref_mean, jnp.maximum(s2, 0.0), df, fmin)

    # laGP averages over the reference set and lets `w` scale it, rather than
    # normalizing by sum(w) -- so weights change the magnitude, not only the balance
    eci = eci if w is None else w * eci
    out = jnp.mean(eci, axis=-1)
    return jnp.where(mui <= SDEPS, jnp.inf, out)


@pytree_config
class ALC(Criterion):
    """Expected reduction in predictive variance at Xref, port of alcGPsep. laGP's default."""

    cached: ClassVar[bool] = True

    def __call__(self, gp, Xcand, Xref, kx=None):
        return _alc(gp, Xcand, Xref, kx)


@pytree_config
class NN(Criterion):
    """Nearest neighbour: minus the distance to the closest reference location."""

    def __call__(self, gp, Xcand, Xref, kx=None):
        return _nn(gp, Xcand, Xref)


@pytree_config
class IECI(Criterion):
    """Integrated expected conditional improvement, port of ieciGPsep. Lower is better.

    Scores a candidate by how much adding it improves the prospect of finding an optimum
    at the reference locations, rather than by how much it reduces variance. `fmin` is
    the incumbent best value; it and the weights `w` are pytree leaves, so changing them
    between calls does not recompile.
    """

    fmin: Float[Array, ""]
    w: Optional[Float[Array, "nref"]] = None
    nugget: bool = static(default=True)
    minimize: ClassVar[bool] = True
    cached: ClassVar[bool] = True

    def __call__(self, gp, Xcand, Xref, kx=None):
        return _ieci(gp, Xcand, Xref, kx, self.fmin, self.w, self.nugget)


@pytree_config
class Custom(Criterion):
    """Wrap any `(gp, Xcand, Xref) -> scores` function as a criterion.

    `fn` is static, so it is part of the compiled program and keyed by identity. A plain
    callable handed to `LocalGP(criterion=...)` is wrapped this way, reading `minimize` and
    `cached` off the function if it carries them.
    """

    fn: Callable = static()
    minimize: bool = static(default=False)
    cached: bool = static(default=False)

    def __call__(self, gp, Xcand, Xref, kx=None):
        if self.cached and kx is not None:
            return self.fn(gp, Xcand, Xref, kx=kx)
        return self.fn(gp, Xcand, Xref)


alc = ALC()
nn = NN()

_NAMES = {"alc": ALC, "nn": NN}


def as_criterion(criterion) -> Criterion:
    """A Criterion from a name, a Criterion, or a plain scoring function."""
    if isinstance(criterion, Criterion):
        return criterion
    if isinstance(criterion, str):
        if criterion == "ieci":
            raise ValueError(
                "ieci needs an incumbent: pass criterion=lagpax.IECI(fmin=...)"
            )
        if criterion not in _NAMES:
            raise ValueError(f"unknown criterion {criterion!r}")
        return _NAMES[criterion]()
    if callable(criterion):
        return Custom(
            criterion,
            minimize=bool(getattr(criterion, "minimize", False)),
            cached=bool(getattr(criterion, "cached", False)),
        )
    raise TypeError(f"not a criterion: {criterion!r}")
