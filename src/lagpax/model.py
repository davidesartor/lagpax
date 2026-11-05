"""Local approximate GPs: one small GP per prediction location, on a greedily chosen design.

`LocalGP` holds the settings. `LocalGP.fit(X, y)` resolves the prior -- the one host-side
step -- and returns an `Emulator`; calling the emulator on new inputs builds, for each row, a
GP on `neighbors` training points chosen by `criterion` and `search`, fits its hyperparameters
and predicts.

Both are pytrees. In `LocalGP` the sizes, `trend` and `maxiter` are static (they shape the
program) and the criterion and search are children; an `Emulator` adds the data and the prior
as leaves. So an emulator crosses `jax.jit` as an argument, `jax.grad` runs through it with
respect to its data, and `Emulator.local` -- the per-location primitive -- can be mapped with
`vmap` as readily as `__call__` maps it with `lax.map`.
"""

import dataclasses
from functools import partial
from collections.abc import Sequence
from typing import Any, NamedTuple, Optional
from jaxtyping import Array, Float, Int, Scalar

import jax
import jax.numpy as jnp
import numpy as np

from ._config import pytree_config, static
from .criteria import NN, as_criterion
from .fit import fit
from .gp import GP, Prediction, Trend, _floating
from .kernel import correlation, sqdist
from .prior import Prior
from .search import Continuous, Ray, Search, as_search


def _default_candidates(neighbors: int, n: int, search: Search) -> int:
    """laGP's default pool: wider when the search does not score every candidate."""
    wide = isinstance(search, (Ray, Continuous))
    return int(min((1000 + neighbors) * (10 if wide else 1), n))


def _buckets(start: int, end: int) -> list[tuple[int, int, int]]:
    """Greedy steps grouped by padded capacity: `(lo, hi, c)` runs steps lo..hi-1 on c rows.

    Step `t` has t live rows and needs a static shape, so it pays for whatever it is padded
    to. Padding every step to `end` costs `(end - start) * end^2`; padding to the next power
    of two instead is ~2x less and still traces a logarithmic number of loop bodies rather
    than one per step.
    """
    out, t = [], start
    while t < end:
        c = min(1 << max(t - 1, 0).bit_length(), end)
        hi = min(c + 1, end)
        out.append((t, hi, c))
        t = hi
    return out


def _design(
    model: "LocalGP",
    Xloc: Float[Array, "close m"],
    Zloc: Float[Array, "close"],
    Xref: Float[Array, "nref m"],
    d0: Float[Array, "m"],
    g0: Scalar,
) -> Int[Array, "end"]:
    """Grow the local design one point at a time, returning positions into Xloc.

    `d0`/`g0` are the starting theta and nugget: candidates are scored at them, not at the
    fitted values, which is what `Emulator.refine` exists to improve on.
    """
    start, end = model.init, model.neighbors
    criterion, search = model.criterion, model.search

    # nearest neighbour needs no search: the candidates are already distance-ordered
    if isinstance(criterion, NN):
        return jnp.arange(end)

    # laGP takes the box from the candidates only, after the starting design is removed
    rect = jnp.stack([jnp.min(Xloc[start:], axis=0), jnp.max(Xloc[start:], axis=0)])
    pos = jnp.concatenate([jnp.arange(start), jnp.zeros(end - start, int)])
    mask = jnp.arange(end) < start
    taken = jnp.arange(model.candidates) < start

    # column j of `kxc` is correlation(Xloc, Xloc[pos[j]], d0), so a step appends one column
    # instead of rebuilding the whole candidate/design cross-covariance
    cached = criterion.cached and search.cached
    kxc = (
        jnp.zeros((model.candidates, end), Xloc.dtype)
        .at[:, :start]
        .set(correlation(Xloc, Xloc[:start], d0))
        if cached
        else None
    )

    # `pos`/`mask` stay full length -- only the GP is built at the bucket's capacity
    def greedy(capacity: int):
        def body(t: Scalar, state):
            pos, mask, taken, kxc = state
            gp = GP(
                Xloc[pos[:capacity]],
                Zloc[pos[:capacity]],
                d0,
                g0,
                mask[:capacity],
                trend=model.trend,
            )
            kw = {} if kxc is None else {"kx": kxc[:, :capacity]}
            w = search(criterion, gp, Xloc, Xref, taken, t - start, rect, **kw)
            if kxc is not None:
                kxc = kxc.at[:, t].set(correlation(Xloc, Xloc[w][None], d0)[:, 0])
            return pos.at[t].set(w), mask.at[t].set(True), taken.at[w].set(True), kxc

        return body

    state = (pos, mask, taken, kxc)
    for lo, hi, capacity in _buckets(start, end):
        state = jax.lax.fori_loop(lo, hi, greedy(capacity), state)
    return state[0]


@partial(jax.jit, static_argnames=("full_cov",))
def _core(
    emu: "Emulator",
    Xref: Float[Array, "nref m"],
    theta0: Float[Array, "m"],
    *,
    full_cov: bool,
) -> Prediction:
    """One local GP: choose its design, fit it, predict at Xref."""
    model, X, y = emu.model, emu.X, emu.y
    g0 = jnp.asarray(emu.prior.nugget.init, X.dtype)

    # only the `candidates` nearest points are ever considered, and top_k breaks ties on the
    # lower index the way a stable sort does
    dst = jnp.min(sqdist(Xref, X), axis=0)
    sel = jax.lax.top_k(-dst, model.candidates)[1]
    Xloc, yloc = X[sel], y[sel]

    pos = _design(model, Xloc, yloc, Xref, theta0, g0)
    gp = GP(Xloc[pos], yloc[pos], theta0, g0, trend=model.trend)
    gp, info = fit(gp, emu.prior, maxiter=model.maxiter)

    pred = gp.predict(Xref, full_cov=full_cov)
    return pred._replace(neighbors=sel[pos], fit=info)


class Emulator(NamedTuple):
    """A fitted `LocalGP`: the settings, the training data and the resolved prior.

    Call it on new inputs to predict. A pytree, so `emu._replace(y=ynew)` is the same model
    on new responses (no prior re-resolution), and functions of an emulator jit and
    differentiate like any other JAX code.
    """

    model: "LocalGP"
    X: Float[Array, "n m"]
    y: Float[Array, "n"]
    prior: Prior

    def _theta0(self, theta0, shape) -> Float[Array, "..."]:
        """The starting theta: the prior's, or a caller's, clipped into the prior's box."""
        dtype, p = self.X.dtype, self.prior
        if theta0 is None:
            return jnp.broadcast_to(p.to_theta(jnp.asarray(p.kernel.init, dtype)), shape)
        v = p.from_theta(jnp.broadcast_to(jnp.asarray(theta0, dtype), shape))
        lo = jnp.asarray(p.kernel.lower, dtype)
        hi = jnp.asarray(p.kernel.upper, dtype)
        return p.to_theta(jnp.clip(v, lo, hi))

    def local(
        self,
        Xref: Float[Array, "nref m"],
        theta0: Optional[Float[Array, "m"]] = None,
        full_cov: bool = False,
    ) -> Prediction:
        """One local GP for a whole reference set, and its prediction there.

        The per-location primitive: `__call__` maps it over rows with `lax.map`, and
        `jax.vmap(lambda x: emu.local(x[None]))(Xnew)` is the same thing batched by vmap.
        `theta0` overrides the starting theta (clipped into the prior's box).
        """
        Xref = jnp.atleast_2d(jnp.asarray(Xref, self.X.dtype))
        if Xref.shape[-1] != self.X.shape[-1]:
            raise ValueError("Xref and X have different numbers of columns")
        theta0 = self._theta0(theta0, (self.X.shape[-1],))
        return _core(self, Xref, theta0, full_cov=full_cov)

    def __call__(
        self,
        Xnew: Float[Array, "nn m"],
        batch: Optional[int] = 256,
        theta0: Optional[Float[Array, "nn m"]] = None,
    ) -> Prediction:
        """Predict at every row of Xnew, each from its own local GP.

        `batch` bounds how many locations are in flight at once (`lax.map`'s batch size); it
        trades peak memory for throughput and does not change results.
        """
        Xnew = jnp.atleast_2d(jnp.asarray(Xnew, self.X.dtype))
        if Xnew.shape[-1] != self.X.shape[-1]:
            raise ValueError("Xnew and X have different numbers of columns")
        T0 = self._theta0(theta0, Xnew.shape)

        def one(loc):
            x, t = loc
            return self.local(x[None, :], theta0=t)

        out = jax.lax.map(one, (Xnew, T0), batch_size=batch)
        return out._replace(mean=out.mean[:, 0], variance=out.variance[:, 0])

    def refine(
        self,
        Xnew: Float[Array, "nn m"],
        passes: Sequence[Any] = ("nn", "alc"),
        batch: Optional[int] = 256,
    ) -> Prediction:
        """Predict repeatedly, each pass starting every location from its last fitted theta.

        Designs are scored at the *starting* theta, so a pass that inherits per-location
        fits chooses neighbourhoods that respect which inputs matter. The first pass only
        supplies those fits, so a cheap `"nn"` there does as well as `"alc"`.

        A Python loop, not `lax.scan`: passes may use different criteria, which compile to
        different functions. It unrolls at trace time and still jits as one graph.
        """
        if not passes:
            raise ValueError("passes must name at least one pass")
        Xnew = jnp.atleast_2d(jnp.asarray(Xnew, self.X.dtype))
        theta = self._theta0(None, Xnew.shape)
        for criterion in passes:
            emu = self._replace(model=self.model.replace(criterion=criterion))
            out = emu(Xnew, batch=batch, theta0=theta)
            theta = jnp.where(jnp.isfinite(out.fit.theta), out.fit.theta, theta)
        return out


@pytree_config
class LocalGP:
    """Settings for local approximate GP prediction.

    neighbors: points in each local design. init: how many of them are simply the nearest
    (the rest are chosen greedily by `criterion`). candidates: how many nearest points the
    greedy search may choose from; the default follows laGP (1000 + neighbors, ten times
    that for `Ray`/`Continuous`). trend: "zero" (laGP's), "constant" or "linear".
    maxiter: iteration cap for each hyperparameter solve.

    `criterion` and `search` accept names ("alc", "nn"; "exhaustive", "ray", "continuous")
    or objects (`ALC()`, `IECI(fmin=...)`, `Ray(numrays=8)`), normalized on construction.
    """

    neighbors: int = static(default=50)
    init: int = static(default=6)
    candidates: Optional[int] = static(default=None)
    criterion: Any = "alc"
    search: Any = "exhaustive"
    trend: Trend = static(default="zero")
    maxiter: int = static(default=100)

    def __post_init__(self):
        object.__setattr__(self, "criterion", as_criterion(self.criterion))
        object.__setattr__(self, "search", as_search(self.search))

    def replace(self, **changes) -> "LocalGP":
        """A copy with some settings changed; names are normalized as on construction."""
        return dataclasses.replace(self, **changes)

    def fit(
        self,
        X: Float[Array, "n m"],
        y: Float[Array, "n"],
        prior: Optional[Prior] = None,
        key: Optional[Array] = None,
    ) -> Emulator:
        """Bind training data and resolve the prior: the one host-side step.

        Without a prior, `Prior.from_data(X, y)` is used, which needs `key` when X has more
        than 1000 rows (it subsamples them). Hyperparameters are fitted per location, when
        the emulator is called -- not here.
        """
        X = _floating(X)
        y = jnp.asarray(y, X.dtype)
        n = X.shape[0]
        if y.shape != (n,):
            raise ValueError("y must be a vector with one entry per row of X")
        if not (6 <= self.init < self.neighbors):
            raise ValueError("need 6 <= init < neighbors")
        if n <= self.neighbors:
            raise ValueError("need more training points than neighbors")

        candidates = (
            _default_candidates(self.neighbors, n, self.search)
            if self.candidates is None
            else int(self.candidates)
        )
        if not (self.neighbors < candidates <= n):
            raise ValueError("need neighbors < candidates <= number of training points")

        if prior is None:
            prior = Prior.from_data(np.asarray(X), np.asarray(y), key=key)
        return Emulator(self.replace(candidates=candidates), X, y, prior)
