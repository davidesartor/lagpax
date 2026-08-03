"""Local approximate GPs, port of laGP's laGPsep and aGPsep."""

from functools import partial
from typing import Any, Literal, NamedTuple, Optional
from jaxtyping import Array, Float, Int, Scalar

import jax
import jax.numpy as jnp
import numpy as np

from .alc import alc_gpsep
from .covar import distance
from .gp import GPsep, new_gpsep, pred_gpsep
from .mle import mle_switch
from .prior import Darg, Garg, darg, garg

Method = Literal["alc", "nn"]


class LaGP(NamedTuple):
    """laGPsep output for a single reference set."""

    mean: Float[Array, "nref"]
    s2: Float[Array, "..."]
    df: Scalar
    llik: Scalar
    Xi: Int[Array, "end"]
    mle: dict[str, Any]


class AGP(NamedTuple):
    """aGPsep output; `var` carries laGP's df/(df-2) Student-t scaling."""

    mean: Float[Array, "nn"]
    var: Float[Array, "nn"]
    llik: Float[Array, "nn"]
    Xi: Int[Array, "nn end"]
    mle: dict[str, Any]


def _default_close(end: int, n: int) -> int:
    return int(min(1000 + end, n))


@partial(
    jax.jit,
    static_argnames=(
        "start",
        "end",
        "close",
        "method",
        "lite",
        "d_mle",
        "g_mle",
        "maxit",
    ),
)
def _lagpsep_core(
    Xref: Float[Array, "nref m"],
    X: Float[Array, "n m"],
    Z: Float[Array, "n"],
    d_start: Float[Array, "m"],
    g_start: Scalar,
    drange: tuple[Scalar, Scalar],
    grange: tuple[Scalar, Scalar],
    dab: tuple[Scalar, Scalar],
    gab: tuple[Scalar, Scalar],
    *,
    start: int,
    end: int,
    close: int,
    method: Method,
    lite: bool,
    d_mle: bool,
    g_mle: bool,
    maxit: int,
) -> LaGP:
    """Greedy local design followed by hyperparameter estimation and prediction."""
    # order the full design by distance to the reference set and keep the `close` nearest
    # top_k, not a full argsort: only the `close` nearest are ever used, and it ties-breaks
    # on the lower index the same way a stable sort does
    dst = jnp.min(distance(Xref, X), axis=0)
    sel = jax.lax.top_k(-dst, close)[1]
    Xloc, Zloc = X[sel], Z[sel]

    if method == "nn":
        pos = jnp.arange(end)
    else:
        pos = jnp.concatenate([jnp.arange(start), jnp.zeros(end - start, int)])
        taken = jnp.arange(close) < start
        mask = jnp.arange(end) < start

        def greedy(t, state):
            pos, mask, taken = state
            gp = new_gpsep(Xloc[pos], Zloc[pos], d_start, g_start, mask)
            als = jnp.where(taken, -jnp.inf, alc_gpsep(gp, Xloc, Xref))
            w = jnp.argmax(als)
            return pos.at[t].set(w), mask.at[t].set(True), taken.at[w].set(True)

        pos, _, _ = jax.lax.fori_loop(start, end, greedy, (pos, mask, taken))

    gp = new_gpsep(Xloc[pos], Zloc[pos], d_start, g_start)
    gp, mle = mle_switch(gp, d_mle, g_mle, drange, grange, dab, gab, maxit=maxit)
    pred = pred_gpsep(gp, Xref, lite=lite)
    return LaGP(
        mean=pred.mean, s2=pred.s2, df=pred.df, llik=pred.llik, Xi=sel[pos], mle=mle
    )


def _resolve_args(
    X: np.ndarray,
    Z: np.ndarray,
    d: Any,
    g: Any,
    rng: Optional[np.random.Generator],
) -> tuple[Darg, Garg]:
    da = darg(d, X, rng=rng)
    ga = garg(g, Z)
    if ga.start is None:
        raise ValueError("g$start should be scalar")
    return da, ga


def _ranges(da: Darg, ga: Garg, m: int):
    d_start = jnp.broadcast_to(jnp.asarray(da.start, jnp.float64), (m,))
    return (
        d_start,
        jnp.asarray(ga.start),
        (jnp.asarray(da.min), jnp.asarray(da.max)),
        (jnp.asarray(ga.min), jnp.asarray(ga.max)),
        (jnp.asarray(da.ab[0]), jnp.asarray(da.ab[1])),
        (jnp.asarray(ga.ab[0]), jnp.asarray(ga.ab[1])),
    )


def lagpsep(
    Xref: Float[Array, "nref m"],
    X: Float[Array, "n m"],
    Z: Float[Array, "n"],
    start: int = 6,
    end: int = 50,
    d: Any = None,
    g: Any = 1 / 10000,
    method: Method = "alc",
    close: Optional[int] = None,
    lite: bool = True,
    maxit: int = 100,
    rng: Optional[np.random.Generator] = None,
) -> LaGP:
    """Local approximate GP at one reference set, matching laGPsep."""
    Xref = jnp.atleast_2d(jnp.asarray(Xref, jnp.float64))
    X, Z = jnp.asarray(X, jnp.float64), jnp.asarray(Z, jnp.float64)
    n, m = X.shape
    if not (6 <= start < end):
        raise ValueError("must have 6 <= start < end")
    if n <= end:
        raise ValueError("nrow(X) <= end, so nothing to do")

    close = _default_close(end, n) if close is None else int(close)
    if not (end < close <= n):
        raise ValueError("must have end < close <= n")

    da, ga = _resolve_args(np.asarray(X), np.asarray(Z), d, g, rng)
    d_start, g_start, drange, grange, dab, gab = _ranges(da, ga, m)
    return _lagpsep_core(
        Xref,
        X,
        Z,
        d_start,
        g_start,
        drange,
        grange,
        dab,
        gab,
        start=start,
        end=end,
        close=close,
        method=method,
        lite=lite,
        d_mle=da.mle,
        g_mle=ga.mle,
        maxit=maxit,
    )


def agpsep(
    X: Float[Array, "n m"],
    Z: Float[Array, "n"],
    XX: Float[Array, "nn m"],
    start: int = 6,
    end: int = 50,
    d: Any = None,
    g: Any = 1 / 10000,
    method: Method = "alc",
    close: Optional[int] = None,
    maxit: int = 100,
    batch: Optional[int] = 64,
    rng: Optional[np.random.Generator] = None,
) -> AGP:
    """One local approximate GP per predictive location, matching aGPsep.

    `batch` bounds how many locations are vmapped at once; it trades peak memory for
    throughput and does not change results.
    """
    X, Z = jnp.asarray(X, jnp.float64), jnp.asarray(Z, jnp.float64)
    XX = jnp.atleast_2d(jnp.asarray(XX, jnp.float64))
    n, m = X.shape
    if XX.shape[-1] != m:
        raise ValueError("mismatch XX and X cols")
    if len(Z) != n:
        raise ValueError("length(Z) != nrow(X)")

    close = _default_close(end, n) if close is None else int(close)
    da, ga = _resolve_args(np.asarray(X), np.asarray(Z), d, g, rng)
    d_start, g_start, drange, grange, dab, gab = _ranges(da, ga, m)

    def one(xref: Float[Array, "m"]) -> LaGP:
        return _lagpsep_core(
            xref[None, :],
            X,
            Z,
            d_start,
            g_start,
            drange,
            grange,
            dab,
            gab,
            start=start,
            end=end,
            close=close,
            method=method,
            lite=True,
            d_mle=da.mle,
            g_mle=ga.mle,
            maxit=maxit,
        )

    out = jax.lax.map(one, XX, batch_size=batch)
    df = out.df
    return AGP(
        mean=out.mean[:, 0],
        var=out.s2[:, 0] * df / (df - 2.0),
        llik=out.llik,
        Xi=out.Xi,
        mle=out.mle,
    )
