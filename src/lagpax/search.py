"""How the greedy loop looks for the next design point, given a criterion to score with.

`criterion x search` is a grid, not a list of methods: laGP spells the three combinations
it ships as `alcGPsep` / `alcrayGPsep` / `alcoptGPsep`, and adding criteria multiplies the
names. Every search here has the same signature and reads the criterion's `minimize`
attribute for its direction, so any criterion works with any search.

`ray` and `continuous` are the scalability searches: their cost per greedy step is set by
`numrays` / `numstart`, not by the candidate count, which is what makes the local scheme
usable when the candidate set is huge.
"""

from typing import Callable
from jaxtyping import Array, Bool, Float, Int, Scalar

import jax
import jax.numpy as jnp

from .covar import distance
from .criteria import Criterion
from .gp import GP
from .optim import lbfgsb

Search = Callable[..., Int[Array, ""]]

GOLDEN = (5.0**0.5 - 1.0) / 2.0


def _pick(
    score: Float[Array, "ncand"],
    taken: Bool[Array, "ncand"],
    minimize: bool,
) -> Int[Array, ""]:
    """Best untaken candidate, with non-finite scores treated as unusable."""
    bad = jnp.inf if minimize else -jnp.inf
    score = jnp.where(taken | ~jnp.isfinite(score), bad, score)
    return jnp.argmin(score) if minimize else jnp.argmax(score)


def _nearest(
    Xstar: Float[Array, "1 m"],
    Xcand: Float[Array, "ncand m"],
    taken: Bool[Array, "ncand"],
) -> Int[Array, ""]:
    """Snap a point in the input space to the closest candidate still available."""
    dst = distance(Xstar, Xcand)[0]
    return jnp.argmin(jnp.where(taken, jnp.inf, dst))


def exhaustive(
    criterion: Criterion,
    gp: GP,
    Xcand: Float[Array, "ncand m"],
    Xref: Float[Array, "nref m"],
    taken: Bool[Array, "ncand"],
    t: Scalar,
    rect: Float[Array, "2 m"],
) -> Int[Array, ""]:
    """Score every candidate and take the best, matching laGP's default search."""
    return _pick(
        criterion(gp, Xcand, Xref), taken, getattr(criterion, "minimize", False)
    )


def _golden_max(
    f: Callable[[Float[Array, "k"]], Float[Array, "k"]],
    lo: Float[Array, "k"],
    hi: Float[Array, "k"],
    iters: int,
) -> tuple[Float[Array, "k"], Float[Array, "k"]]:
    """Golden-section maximization of `f` on each of k intervals at once.

    Fixed iteration count rather than a tolerance: the loop has to be a `fori_loop` with
    static shapes, and the interval shrinks by 0.618 per step regardless of the function.
    """

    def probes(a, b):
        c = b - GOLDEN * (b - a)
        d = a + GOLDEN * (b - a)
        return c, d, f(c), f(d)

    def body(_, state):
        a, b, c, d, fc, fd = state
        left = fc > fd
        a, b = jnp.where(left, a, c), jnp.where(left, d, b)
        return (a, b, *probes(a, b))

    state = jax.lax.fori_loop(0, iters, body, (lo, hi, *probes(lo, hi)))
    a, b, c, d, fc, fd = state
    best = jnp.where(fc > fd, c, d)
    return best, jnp.maximum(fc, fd)


def _ray_ends(
    Xref: Float[Array, "1 m"],
    Xstart: Float[Array, "k m"],
    rect: Float[Array, "2 m"],
) -> Float[Array, "k m"]:
    """laGP's ray.end: cast 10x away from Xref, then pull back to the bounding box.

    R shrinks the ray one violated coordinate at a time in a while loop; the first
    boundary the ray crosses is the same point, so this computes it directly.
    """
    step = 10.0 * (Xstart - Xref)
    room = jnp.where(step > 0, rect[1] - Xstart, rect[0] - Xstart)
    safe = jnp.where(step != 0.0, step, 1.0)
    s = jnp.min(jnp.where(step != 0.0, room / safe, jnp.inf), axis=-1)
    return Xstart + jnp.clip(s, 0.0, 1.0)[:, None] * step


def ray(
    criterion: Criterion,
    gp: GP,
    Xcand: Float[Array, "ncand m"],
    Xref: Float[Array, "nref m"],
    taken: Bool[Array, "ncand"],
    t: Scalar,
    rect: Float[Array, "2 m"],
    *,
    numrays: int = 0,
    pool: int = 0,
    iters: int = 16,
) -> Int[Array, ""]:
    """Search along rays leaving Xref instead of scoring every candidate, per alcrayGPsep.

    Cost per step is `numrays * iters` criterion evaluations on a handful of points, so it
    barely grows with the candidate count. `numrays` defaults to the input dimension, as
    laGP's does.

    **Divergence from laGP:** laGP draws the ray starting points with `sample()`, which
    makes the whole local design depend on R's RNG state. Here step `t` takes a fixed
    consecutive block of `numrays` candidates from the `pool` nearest, rotating with `t` —
    deterministic, jittable, and covering the same near neighbourhood.
    """
    ncand, m = Xcand.shape
    numrays = min(numrays or m, ncand)
    pool = min(pool or 10 * numrays, ncand)
    minimize = getattr(criterion, "minimize", False)

    offset = (t * numrays) % pool
    Xstart = Xcand[(offset + jnp.arange(numrays)) % pool]
    Xend = _ray_ends(Xref[:1], Xstart, rect)

    def along(s: Float[Array, "k"]) -> Float[Array, "k"]:
        pts = Xstart + s[:, None] * (Xend - Xstart)
        score = criterion(gp, pts, Xref)
        return -score if minimize else score

    zeros = jnp.zeros(numrays, Xcand.dtype)
    s, val = _golden_max(along, zeros, zeros + 1.0, iters)
    r = jnp.argmax(jnp.where(jnp.isfinite(val), val, -jnp.inf))
    Xstar = (Xstart + s[:, None] * (Xend - Xstart))[r][None]
    return _nearest(Xstar, Xcand, taken)


def continuous(
    criterion: Criterion,
    gp: GP,
    Xcand: Float[Array, "ncand m"],
    Xref: Float[Array, "nref m"],
    taken: Bool[Array, "ncand"],
    t: Scalar,
    rect: Float[Array, "2 m"],
    *,
    numstart: int = 1,
    maxit: int = 100,
) -> Int[Array, ""]:
    """Optimize the criterion over the box, then snap to a candidate, per alcoptGPsep.

    Gradients come from `jax.grad`, so this works for any differentiable criterion rather
    than only the ones laGP hand-differentiated.
    """
    ncand = Xcand.shape[0]
    numstart = min(numstart, ncand)
    minimize = getattr(criterion, "minimize", False)

    def objective(x):
        score = criterion(gp, x[None], Xref)[0]
        return score if minimize else -score

    offset = (t * numstart) % ncand
    starts = Xcand[(offset + jnp.arange(numstart)) % ncand]

    def one(x0):
        res = lbfgsb(objective, x0, rect[0], rect[1], maxiter=maxit)
        return _nearest(res.x[None], Xcand, taken)

    ws = jnp.stack([one(starts[i]) for i in range(numstart)])
    if numstart == 1:
        return ws[0]

    # the snapped candidates are what actually gets added, so rank those, not the optima
    score = criterion(gp, Xcand[ws], Xref)
    best = _pick(score, taken[ws], minimize)
    return ws[best]


def _named(search) -> Search:
    """Resolve a search name to a function; pass callables through unchanged."""
    if callable(search):
        return search
    try:
        return {"exhaustive": exhaustive, "ray": ray, "continuous": continuous}[search]
    except KeyError:
        raise ValueError(f"unknown search {search!r}") from None
