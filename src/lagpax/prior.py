"""Design-derived lengthscale/nugget ranges and priors, port of laGP's prior.R.

The only host-side numpy/scipy step in the package. It runs once per pipeline, and R's
`quantile` type 7 is numpy's default, so it is deliberately not jitted.
"""

from typing import Mapping, NamedTuple, Optional, Union

import numpy as np
import scipy as sp

SDEPS = float(np.sqrt(np.finfo(float).eps))


class Range(NamedTuple):
    """Start value, box, Gamma prior `(a, b)`, and whether to estimate the parameter.

    Fields left as `None` are filled in by `Prior.at`, which is what makes a partial
    `Range(start=1e-6, estimate=False)` a usable override.
    """

    start: Union[float, np.ndarray, None] = None
    min: Optional[float] = None
    max: Optional[float] = None
    ab: Optional[tuple[float, float]] = None
    estimate: Optional[bool] = None

    def merge(self, other: "Range") -> "Range":
        """`other` wins field by field, ignoring the fields it leaves unset."""
        return Range(*(o if o is not None else s for s, o in zip(self, other)))

    def check(self) -> None:
        """Port of laGP's check.arg validation."""
        if self.max <= 0:
            raise ValueError("max should be a positive scalar")
        if self.min < 0 or self.min > self.max:
            raise ValueError("min should be a positive scalar < max")
        start = np.asarray(self.start)
        if np.any(start < self.min) or np.any(start > self.max):
            raise ValueError("start should lie in [min, max]")
        if len(self.ab) != 2 or any(v < 0 for v in self.ab):
            raise ValueError("ab should be a positive 2-vector")


Spec = Union[float, np.ndarray, Mapping, Range, None]


class Prior(NamedTuple):
    """Lengthscale and nugget ranges together, since every call site needs both."""

    d: Range
    g: Range

    @classmethod
    def from_design(
        cls,
        X: np.ndarray,
        Z: np.ndarray,
        d: Spec = None,
        g: Spec = 1 / 10000,
        samp_size: int = 1000,
        rng: Optional[np.random.Generator] = None,
    ) -> "Prior":
        """laGP's darg + garg. `g` defaults to laGPsep/aGPsep's nugget, not to garg's."""
        return cls(d=darg(d, X, samp_size=samp_size, rng=rng), g=garg(g, Z))

    def at(self, d: Spec = None, g: Spec = None) -> "Prior":
        """Override fields of either range, leaving everything unset as it was."""
        return Prior(
            d=self.d.merge(_as_range(d)) if d is not None else self.d,
            g=self.g.merge(_as_range(g)) if g is not None else self.g,
        )


def _as_range(spec: Spec) -> Range:
    """Accept a Range, a mapping of field names, or a bare starting value."""
    if isinstance(spec, Range):
        return spec
    if isinstance(spec, Mapping):
        return Range(**{("estimate" if k == "mle" else k): v for k, v in spec.items()})
    return Range(start=spec)


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


def _gamma_ab(scale: float) -> tuple[float, float]:
    """laGP's default Gamma(1.5, b), with b placing 95% of the mass below `scale`."""
    return 1.5, igamma_inv(1.5, 0.95 * sp.special.gamma(1.5), lower=True) / scale


def darg(
    d: Spec,
    X: np.ndarray,
    samp_size: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> Range:
    """Default lengthscale start/range/prior from the design, matching laGP's darg."""
    spec = _as_range(d)
    estimate = True if spec.estimate is None else bool(spec.estimate)

    needs_ds = spec.start is None or (
        estimate and (spec.max is None or spec.min is None or spec.ab is None)
    )
    start_q, dmin_q, dmax_q = (
        get_ds(X, samp_size=samp_size, rng=rng) if needs_ds else (0.0, 0.0, 0.0)
    )

    start = np.atleast_1d(
        np.asarray(start_q if spec.start is None else spec.start, float)
    )
    if spec.max is not None:
        dmax = float(spec.max)
    else:
        dmax = dmax_q if estimate else float(start.max())
    if spec.min is not None:
        dmin = float(spec.min)
    else:
        dmin = max(dmin_q / 2.0 if estimate else float(start.min()), SDEPS)
    if spec.ab is not None:
        ab = tuple(float(v) for v in spec.ab)
    else:
        ab = _gamma_ab(dmax_q) if estimate else (0.0, 0.0)

    out = Range(start=start, min=dmin, max=dmax, ab=ab, estimate=estimate)
    out.check()
    return out


def garg(g: Spec, Z: np.ndarray) -> Range:
    """Default nugget start/range/prior from the responses, matching laGP's garg."""
    spec = _as_range(g)
    estimate = False if spec.estimate is None else bool(spec.estimate)

    r2s = np.square(np.asarray(Z, float) - np.asarray(Z, float).mean())
    start = float(np.quantile(r2s, 0.025) if spec.start is None else spec.start)
    if spec.max is not None:
        gmax = float(spec.max)
    else:
        gmax = float(r2s.max()) if estimate else start
    gmin = SDEPS if spec.min is None else float(spec.min)
    if spec.ab is not None:
        ab = tuple(float(v) for v in spec.ab)
    else:
        ab = _gamma_ab(float(r2s.mean())) if estimate else (0.0, 0.0)

    out = Range(start=start, min=gmin, max=gmax, ab=ab, estimate=estimate)
    out.check()
    return out
