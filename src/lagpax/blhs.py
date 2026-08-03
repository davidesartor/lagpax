"""Block Latin hypercube subsampling, port of laGP's blhs / blhs.loop.

Fitting one global lengthscale to a huge design is the expensive part of a global-then-local
workflow. blhs cuts the input box into `m^k` blocks, keeps only the points falling inside a
Latin hypercube of `m` of them, and fits that instead; repeating it `K` times and taking the
per-dimension median is the bootstrap. The result plugs into `Local.condition(prior=...)`.

Host-side numpy for the same reason `prior.py` is: every draw has a different number of rows,
so the `K` fits are different shapes and there is nothing to batch.
"""

from typing import Optional
from jaxtyping import Array, Bool, Float, Int

import numpy as np

from .gp import GP
from .mle import mle_d
from .prior import Prior, Range


def latin(
    m: int, k: int, rng: Optional[np.random.Generator] = None
) -> Int[Array, "m k"]:
    """An (m, k) Latin hypercube of 1-based block labels: one permutation per dimension."""
    rng = np.random.default_rng() if rng is None else rng
    return np.stack([rng.permutation(m) + 1 for _ in range(k)], axis=1)


def _code(B: Int[Array, "n k"], m: int) -> Int[Array, "n"]:
    """laGP's Global_block: mixed-radix, but with the first digit left un-offset.

    The stray `+1` on the leading digit is in laGP and is applied to the block labels too,
    so it cancels; it only matters that both sides are coded the same way.
    """
    return B[:, 0] + ((B[:, 1:] - 1) * m ** np.arange(1, B.shape[1])).sum(-1)


def blocks(
    X: Float[Array, "n k"],
    m: int,
    labels: Int[Array, "m k"],
) -> Bool[Array, "n"]:
    """Which rows of X fall in the `m` blocks named by `labels`.

    laGP assigns bin `i` on the *open* interval `((i-1)/m, i/m)` and leaves anything on a
    boundary in bin 1 -- including `x = 0` and `x = 1`, so the top bin quietly loses its
    upper face to the bottom one. Reproduced as written; it matters only on gridded designs.
    """
    X = np.atleast_2d(np.asarray(X, float))
    bins = np.ones(X.shape, int)
    for i in range(1, m + 1):
        bins = np.where((X > (i - 1) / m) & (X < i / m), i, bins)
    return np.isin(_code(bins, m), _code(np.asarray(labels, int), m))


def subsample(
    X: Float[Array, "n k"],
    Z: Float[Array, "n"],
    m: int,
    rng: Optional[np.random.Generator] = None,
) -> tuple[Float[Array, "ns k"], Float[Array, "ns"]]:
    """One blhs draw: the rows of (X, Z) inside a random Latin hypercube of m blocks."""
    X = np.atleast_2d(np.asarray(X, float))
    k = X.shape[1]
    if k < 2:
        raise ValueError("blhs needs at least 2 input dimensions")
    if m <= 0:
        raise ValueError("m must be positive")
    if m > len(X) ** (1.0 / (k - 1)):
        raise ValueError("must have m <= n^(1/(k-1))")

    sel = blocks(X, m, latin(m, k, rng))
    return X[sel], np.asarray(Z, float)[sel]


def bootstrap(
    X: Float[Array, "n k"],
    Z: Float[Array, "n"],
    m: int,
    K: int = 10,
    prior: Optional[Prior] = None,
    g: float = 1e-3,
    maxit: int = 100,
    rng: Optional[np.random.Generator] = None,
) -> Prior:
    """Median of K lengthscale MLEs on block-bootstrapped subsamples, per blhs.loop.

    Returns a Prior carrying that median as `d.start`. laGP's own use is as a *fixed* global
    lengthscale, which here is `bootstrap(...).at(d=Range(estimate=False))`.
    """
    X = np.atleast_2d(np.asarray(X, float))
    Z = np.asarray(Z, float)
    prior = Prior.from_design(X, Z, rng=rng) if prior is None else prior
    ab = tuple(float(v) for v in prior.d.ab)

    # laGP widens the upper bound tenfold for the subsample fits, and for those only
    dmax = 10.0 * float(prior.d.max)

    fitted = [
        np.asarray(
            mle_d(
                GP(*subsample(X, Z, m, rng=rng), prior.d.start, g),
                float(prior.d.min),
                dmax,
                ab=ab,
                maxit=maxit,
            ).d
        )
        for _ in range(K)
    ]
    return prior.at(d=Range(start=np.median(np.stack(fitted), axis=0)))
