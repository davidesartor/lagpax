"""Hyperparameter bounds, starting values and priors, with data-driven defaults.

A `Param` describes one hyperparameter: where the fit starts, its box, an optional
Gamma(shape, rate) prior, and whether it is learned at all. A `Prior` bundles the kernel's
`Param` with the nugget's.

The kernel parameter can be described in either of two spaces. `Prior(theta=...)` puts the
bounds and the Gamma prior on theta itself, the squared lengthscale of
`exp(-sum (x - x')^2 / theta)` -- laGP's parameterization, so fits match laGP's exactly.
`Prior(lengthscale=...)` puts them on the conventional lengthscale `l = sqrt(theta / 2)`
instead. The two give the same likelihood but different MAP estimates, because a Gamma
density on `l` is not a Gamma density on `theta`.

`Prior.from_data` sets every default from the design and responses, following laGP's
`darg`/`garg`. It runs on the host in numpy (R's `quantile` type 7 is numpy's default), once
per model; nothing downstream of it does.
"""

from dataclasses import fields, replace
from typing import Literal, Optional, Union
from jaxtyping import Array

import jax
import jax.numpy as jnp
import numpy as np
import scipy as sp

from ._config import pytree_config, static

SDEPS = float(np.sqrt(np.finfo(float).eps))

Space = Literal["theta", "lengthscale"]


@pytree_config
class Param:
    """One hyperparameter: start, box, Gamma(shape, rate) prior, and whether to learn it.

    Fields left as `None` mean "use the default" when handed to `Prior.from_data` or
    `Prior.replace`, so `Param(init=1e-6, learn=False)` is a usable partial override. A
    pytree: the numbers are leaves; `learn` is static, since it decides which solver runs.
    A `gamma` of `None` or `(0, 0)` means no prior.
    """

    init: Union[float, np.ndarray, None] = None
    lower: Optional[float] = None
    upper: Optional[float] = None
    gamma: Optional[tuple[float, float]] = None
    learn: Optional[bool] = static(default=None)

    def merge(self, other: "Param") -> "Param":
        """`other` wins field by field, ignoring the fields it leaves unset."""
        return Param(
            **{
                f.name: getattr(self, f.name)
                if getattr(other, f.name) is None
                else getattr(other, f.name)
                for f in fields(Param)
            }
        )

    def check(self) -> None:
        if self.upper <= 0:
            raise ValueError("upper should be positive")
        if self.lower < 0 or self.lower > self.upper:
            raise ValueError("lower should be in [0, upper]")
        init = np.asarray(self.init)
        if np.any(init < self.lower) or np.any(init > self.upper):
            raise ValueError("init should lie in [lower, upper]")
        if len(self.gamma) != 2 or any(v < 0 for v in self.gamma):
            raise ValueError("gamma should be a non-negative (shape, rate) pair")


Spec = Union[float, np.ndarray, Param, None]


def _as_param(spec: Spec) -> Param:
    """A Param, or a bare starting value."""
    return spec if isinstance(spec, Param) else Param(init=spec)


@pytree_config
class Prior:
    """The kernel's and the nugget's `Param`. Set exactly one of `theta` and `lengthscale`."""

    nugget: Param
    theta: Optional[Param] = None
    lengthscale: Optional[Param] = None

    def __post_init__(self):
        if (self.theta is None) == (self.lengthscale is None):
            raise ValueError("a Prior takes exactly one of theta= and lengthscale=")

    @property
    def space(self) -> Space:
        return "theta" if self.theta is not None else "lengthscale"

    @property
    def kernel(self) -> Param:
        """The kernel parameter's Param, in whichever space it was given."""
        return self.theta if self.theta is not None else self.lengthscale

    def to_theta(self, v):
        """A value in this prior's kernel space, as theta."""
        return v if self.space == "theta" else 2.0 * jnp.square(v)

    def from_theta(self, theta):
        """theta, as a value in this prior's kernel space."""
        return theta if self.space == "theta" else jnp.sqrt(0.5 * theta)

    @property
    def theta_gamma(self) -> tuple:
        """Gamma (shape, rate) on theta itself; zeros when the prior is on the lengthscale."""
        return _gamma_or_zero(self.theta) if self.space == "theta" else (0.0, 0.0)

    @property
    def nugget_gamma(self) -> tuple:
        return _gamma_or_zero(self.nugget)

    def kernel_log_prior(self, theta) -> jax.Array:
        """Log-prior on the kernel parameter that `theta_gamma` does not already cover."""
        if self.space == "theta":
            return jnp.zeros((), jnp.result_type(theta))
        a, b = _gamma_or_zero(self.lengthscale)
        return _gamma_log_prior(self.from_theta(theta), a, b)

    def replace(self, theta: Spec = None, lengthscale: Spec = None, nugget: Spec = None):
        """Override fields of the given Params, leaving everything unset as it was."""
        out = self
        if nugget is not None:
            out = replace(out, nugget=out.nugget.merge(_as_param(nugget)))
        if theta is not None:
            out = replace(out, theta=_require(out.theta, "theta").merge(_as_param(theta)))
        if lengthscale is not None:
            base = _require(out.lengthscale, "lengthscale")
            out = replace(out, lengthscale=base.merge(_as_param(lengthscale)))
        return out

    @classmethod
    def from_data(
        cls,
        X: np.ndarray,
        y: np.ndarray,
        theta: Spec = None,
        nugget: Spec = 1e-4,
        space: Space = "theta",
        sample: int = 1000,
        key: Optional[Array] = None,
    ) -> "Prior":
        """Defaults from the data, following laGP's darg/garg.

        theta: start at the 10% quantile of the pairwise squared distances, search between
        half the smallest and the largest, Gamma(1.5, b) prior with 95% of its mass below the
        largest. Nugget: fixed at 1e-4 unless `nugget=Param(learn=True)`, in which case it is
        bounded by the largest squared residual. `space="lengthscale"` returns the same
        defaults mapped to `l = sqrt(theta / 2)`, with the prior recentered on `l`; a `theta`
        spec is then read in lengthscale units.

        A design over `sample` rows is subsampled to compute the distances, and that draw
        needs a `key`; without one it is an error, not a hidden draw.
        """
        if space == "theta":
            return cls(theta=darg(theta, X, sample, key), nugget=garg(nugget, y))
        if space != "lengthscale":
            raise ValueError(f"unknown space {space!r}")

        spec = _as_param(theta)
        # resolve in theta units, then map back: init and the box are monotone in l
        sq = lambda v: None if v is None else 2.0 * np.square(np.asarray(v, float))
        as_theta = Param(sq(spec.init), sq(spec.lower), sq(spec.upper), (0.0, 0.0), spec.learn)
        t = darg(as_theta, X, sample, key)
        root = lambda v: np.sqrt(0.5 * np.asarray(v, float))
        upper = float(root(t.upper))
        gamma = spec.gamma
        if gamma is None:
            gamma = _gamma_ab(upper) if t.learn else (0.0, 0.0)
        ell = Param(root(t.init), float(root(t.lower)), upper, tuple(gamma), t.learn)
        ell.check()
        return cls(lengthscale=ell, nugget=garg(nugget, y))


def _require(p: Optional[Param], name: str) -> Param:
    if p is None:
        raise ValueError(f"this prior has no {name}; it is in the other space")
    return p


def _gamma_or_zero(p: Param) -> tuple:
    return (0.0, 0.0) if p.gamma is None else tuple(p.gamma)


def _gamma_log_prior(x, a, b):
    """sum log Gamma(shape a, rate b) at x; zero when either parameter is non-positive."""
    on = (a > 0) & (b > 0)
    a_, b_ = jnp.where(on, a, 1.0), jnp.where(on, b, 1.0)
    logpdf = (
        (a_ - 1.0) * jnp.log(x) - b_ * x + a_ * jnp.log(b_) - jax.scipy.special.gammaln(a_)
    )
    return jnp.where(on, jnp.sum(logpdf), 0.0)


def igamma_inv(a: float, y: float, lower: bool = False) -> float:
    """laGP's Igamma.inv: the Gamma(a, 1) quantile at probability y / gamma(a)."""
    p = y / sp.special.gamma(a)
    return float(
        sp.special.gammaincinv(a, p) if lower else sp.special.gammainccinv(a, p)
    )


def get_ds(
    X: np.ndarray,
    p: float = 0.1,
    sample: int = 1000,
    key: Optional[Array] = None,
) -> tuple[float, float, float]:
    """(p-quantile, min, max) of the positive pairwise squared distances; laGP's getDs."""
    X = np.atleast_2d(np.asarray(X, float))
    if X.shape[0] > sample:
        X = X[_choice(key, X.shape[0], sample)]

    D = np.sum(np.square(X[:, None, :] - X[None, :, :]), axis=-1)
    D = D[np.triu_indices_from(D, k=1)]
    D = D[D > 0]
    return float(np.quantile(D, p)), float(D.min()), float(D.max())


def _choice(key: Optional[Array], n: int, k: int) -> np.ndarray:
    """k of n row indices without replacement, drawn with an explicit key."""
    if key is None:
        raise ValueError(
            f"the default prior subsamples a design of {n} rows down to {k}, which is "
            "random: pass key=jax.random.key(...)"
        )
    return np.asarray(jax.random.choice(key, n, (k,), replace=False))


def _gamma_ab(scale: float) -> tuple[float, float]:
    """Gamma(1.5, b), with b placing 95% of the mass below `scale`; laGP's default."""
    return 1.5, igamma_inv(1.5, 0.95 * sp.special.gamma(1.5), lower=True) / scale


def darg(
    theta: Spec,
    X: np.ndarray,
    sample: int = 1000,
    key: Optional[Array] = None,
) -> Param:
    """Default theta Param from the design, matching laGP's darg."""
    spec = _as_param(theta)
    learn = True if spec.learn is None else bool(spec.learn)

    needs_ds = spec.init is None or (
        learn and (spec.upper is None or spec.lower is None or spec.gamma is None)
    )
    start_q, dmin_q, dmax_q = (
        get_ds(X, sample=sample, key=key) if needs_ds else (0.0, 0.0, 0.0)
    )

    init = np.atleast_1d(np.asarray(start_q if spec.init is None else spec.init, float))
    if spec.upper is not None:
        upper = float(spec.upper)
    else:
        upper = dmax_q if learn else float(init.max())
    if spec.lower is not None:
        lower = float(spec.lower)
    else:
        lower = max(dmin_q / 2.0 if learn else float(init.min()), SDEPS)
    if spec.gamma is not None:
        gamma = tuple(float(v) for v in spec.gamma)
    else:
        gamma = _gamma_ab(dmax_q) if learn else (0.0, 0.0)

    out = Param(init=init, lower=lower, upper=upper, gamma=gamma, learn=learn)
    out.check()
    return out


def garg(nugget: Spec, y: np.ndarray) -> Param:
    """Default nugget Param from the responses, matching laGP's garg."""
    spec = _as_param(nugget)
    learn = False if spec.learn is None else bool(spec.learn)

    r2s = np.square(np.asarray(y, float) - np.asarray(y, float).mean())
    init = float(np.quantile(r2s, 0.025) if spec.init is None else spec.init)
    if spec.upper is not None:
        upper = float(spec.upper)
    else:
        upper = float(r2s.max()) if learn else init
    lower = SDEPS if spec.lower is None else float(spec.lower)
    if spec.gamma is not None:
        gamma = tuple(float(v) for v in spec.gamma)
    else:
        gamma = _gamma_ab(float(r2s.mean())) if learn else (0.0, 0.0)

    out = Param(init=init, lower=lower, upper=upper, gamma=gamma, learn=learn)
    out.check()
    return out
