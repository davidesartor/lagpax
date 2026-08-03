"""Lengthscale/nugget priors and ranges, port of laGP's prior.R (darg/garg)."""

from typing import NamedTuple, Optional

import numpy as np
import scipy as sp

SDEPS = float(np.sqrt(np.finfo(float).eps))


class Darg(NamedTuple):
    """laGP `darg` output: starting value, box, whether to MLE, and Gamma prior (a, b)."""

    start: np.ndarray
    min: float
    max: float
    mle: bool
    ab: tuple[float, float]


class Garg(NamedTuple):
    """laGP `garg` output: starting value, box, whether to MLE, and Gamma prior (a, b)."""

    start: float
    min: float
    max: float
    mle: bool
    ab: tuple[float, float]


def igamma_inv(a: float, y: float, lower: bool = False) -> float:
    """laGP's Igamma.inv: the Gamma(a, 1) quantile at probability y / gamma(a)."""
    p = y / sp.special.gamma(a)
    return float(
        sp.special.gammaincinv(a, p) if lower else sp.special.gammainccinv(a, p)
    )


def get_ds(
    X: np.ndarray,
    p: float = 0.1,
    samp_size: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> tuple[float, float, float]:
    """laGP's getDs: (start, min, max) from the positive off-diagonal squared distances."""
    X = np.atleast_2d(np.asarray(X, float))
    if X.shape[0] > samp_size:
        rng = np.random.default_rng() if rng is None else rng
        X = X[rng.choice(X.shape[0], samp_size, replace=False)]

    D = np.sum(np.square(X[:, None, :] - X[None, :, :]), axis=-1)
    D = D[np.triu_indices_from(D, k=1)]
    D = D[D > 0]
    return float(np.quantile(D, p)), float(D.min()), float(D.max())


def darg(
    d: float | np.ndarray | dict | None,
    X: np.ndarray,
    samp_size: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> Darg:
    """Default lengthscale start/range/prior from the design, matching laGP's darg."""
    d = (
        {}
        if d is None
        else ({"start": d} if np.isscalar(d) or isinstance(d, np.ndarray) else dict(d))
    )
    mle = d.get("mle", True)

    needs_ds = "start" not in d or (
        mle and ("max" not in d or "min" not in d or "ab" not in d)
    )
    start_q, dmin_q, dmax_q = (
        get_ds(X, samp_size=samp_size, rng=rng) if needs_ds else (0.0, 0.0, 0.0)
    )

    start = np.atleast_1d(np.asarray(d.get("start", start_q), float))
    dmax = float(d["max"]) if "max" in d else (dmax_q if mle else float(start.max()))
    if "min" in d:
        dmin = float(d["min"])
    else:
        dmin = max(dmin_q / 2.0 if mle else float(start.min()), SDEPS)

    ab = (
        tuple(d["ab"])
        if "ab" in d
        else (
            (0.0, 0.0)
            if not mle
            else (
                1.5,
                igamma_inv(1.5, 0.95 * sp.special.gamma(1.5), lower=True) / dmax_q,
            )
        )
    )

    out = Darg(
        start=start, min=dmin, max=dmax, mle=bool(mle), ab=(float(ab[0]), float(ab[1]))
    )
    _check_arg(out)
    return out


def garg(g: float | dict | None, y: np.ndarray) -> Garg:
    """Default nugget start/range/prior from the responses, matching laGP's garg."""
    g = {} if g is None else ({"start": g} if np.isscalar(g) else dict(g))
    mle = g.get("mle", False)

    y = np.asarray(y, float)
    r2s = np.square(y - y.mean())

    start = float(g["start"]) if "start" in g else float(np.quantile(r2s, 0.025))
    gmax = float(g["max"]) if "max" in g else (float(r2s.max()) if mle else start)
    gmin = float(g["min"]) if "min" in g else SDEPS

    if "ab" in g:
        ab = tuple(g["ab"])
    elif not mle:
        ab = (0.0, 0.0)
    else:
        ab = (
            1.5,
            igamma_inv(1.5, 0.95 * sp.special.gamma(1.5), lower=True)
            / float(r2s.mean()),
        )

    out = Garg(
        start=start, min=gmin, max=gmax, mle=bool(mle), ab=(float(ab[0]), float(ab[1]))
    )
    _check_arg(out)
    return out


def _check_arg(a: Darg | Garg) -> None:
    """Port of laGP's check.arg validation."""
    if a.max <= 0:
        raise ValueError("max should be a positive scalar")
    if a.min < 0 or a.min > a.max:
        raise ValueError("min should be a positive scalar < max")
    if np.any(np.asarray(a.start) < a.min) or np.any(np.asarray(a.start) > a.max):
        raise ValueError("start should lie in [min, max]")
    if len(a.ab) != 2 or any(v < 0 for v in a.ab):
        raise ValueError("ab should be a positive 2-vector")
