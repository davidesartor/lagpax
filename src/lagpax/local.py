"""The local approximate GP scheme, port of laGP's laGPsep and aGPsep.

`Local` is the scheme: settings plus a criterion and a search. `Local.condition` binds data
and resolves the prior -- the one host-side step -- and returns a `Conditioned` emulator.

Both are pytrees. In `Local`, the integers and the trend are static (they shape the
program) and the criterion and search are children; a `Conditioned` adds the data and the
prior as leaves. So an emulator can be passed through `jax.jit`, differentiated with
respect to its data, or batched with `vmap`, like any other JAX value. `Conditioned.local`
is the per-location function; `predict` and `refine` are conveniences built on it.
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
from .covar import covar, distance
from .criteria import NN, Criterion, as_criterion
from .gp import GP, Prediction, Trend, _floating
from .mle import fit
from .prior import Prior
from .search import Continuous, Ray, Search, as_search


def _default_close(end: int, n: int, search: Search) -> int:
    """laGP's default: a wider candidate set when the search does not scan all of it."""
    wide = isinstance(search, (Ray, Continuous))
    return int(min((1000 + end) * (10 if wide else 1), n))


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
    scheme: "Local",
    Xloc: Float[Array, "close m"],
    Zloc: Float[Array, "close"],
    Xref: Float[Array, "nref m"],
    d0: Float[Array, "m"],
    g0: Scalar,
) -> Int[Array, "end"]:
    """Grow the local design one point at a time, returning positions into Xloc."""
    start, end = scheme.start, scheme.end
    criterion, search = scheme.criterion, scheme.search

    # nearest neighbour needs no search: the candidates are already distance-ordered
    if isinstance(criterion, NN):
        return jnp.arange(end)

    # laGP takes the box from the candidates only, after the starting design is removed
    rect = jnp.stack([jnp.min(Xloc[start:], axis=0), jnp.max(Xloc[start:], axis=0)])
    pos = jnp.concatenate([jnp.arange(start), jnp.zeros(end - start, int)])
    mask = jnp.arange(end) < start
    taken = jnp.arange(scheme.close) < start

    # column j of `kxc` is covar(d0, Xloc, Xloc[pos[j]]), so a step appends one column
    # instead of rebuilding the whole candidate/design cross-covariance
    cached = criterion.cached and search.cached
    kxc = (
        jnp.zeros((scheme.close, end), Xloc.dtype)
        .at[:, :start]
        .set(covar(d0, Xloc, Xloc[:start]))
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
                trend=scheme.trend,
            )
            kw = {} if kxc is None else {"kx": kxc[:, :capacity]}
            w = search(criterion, gp, Xloc, Xref, taken, t - start, rect, **kw)
            if kxc is not None:
                kxc = kxc.at[:, t].set(covar(d0, Xloc, Xloc[w][None])[:, 0])
            return pos.at[t].set(w), mask.at[t].set(True), taken.at[w].set(True), kxc

        return body

    state = (pos, mask, taken, kxc)
    for lo, hi, capacity in _buckets(start, end):
        state = jax.lax.fori_loop(lo, hi, greedy(capacity), state)
    return state[0]


@partial(jax.jit, static_argnames=("lite",))
def _core(
    emu: "Conditioned",
    Xref: Float[Array, "nref m"],
    d0: Float[Array, "m"],
    *,
    lite: bool,
) -> Prediction:
    """One local approximate GP: choose a design, fit it, predict at Xref."""
    scheme, X, Z = emu.scheme, emu.X, emu.Z
    g0 = jnp.asarray(emu.prior.g.start, X.dtype)

    # only the `close` nearest candidates are ever used, and top_k breaks ties on the
    # lower index the way a stable sort does
    dst = jnp.min(distance(Xref, X), axis=0)
    sel = jax.lax.top_k(-dst, scheme.close)[1]
    Xloc, Zloc = X[sel], Z[sel]

    pos = _design(scheme, Xloc, Zloc, Xref, d0, g0)
    gp = GP(Xloc[pos], Zloc[pos], d0, g0, trend=scheme.trend)
    gp, fitted = fit(gp, emu.prior, maxit=scheme.maxit)

    pred = gp.predict(Xref, lite=lite)
    return pred._replace(Xi=sel[pos], fit=fitted)


class Conditioned(NamedTuple):
    """A `Local` scheme bound to data and a resolved prior: an emulator, and a pytree.

    Everything here is JAX. `local` is the per-location primitive -- pure, so
    `jax.vmap(lambda x: emu.local(x[None]))(XX)` works as well as `predict` does -- and
    `emu._replace(Z=Znew)` gives an emulator on new responses without re-resolving the prior.
    """

    scheme: "Local"
    X: Float[Array, "n m"]
    Z: Float[Array, "n"]
    prior: Prior

    def _dstart(self, dstart, shape) -> Float[Array, "..."]:
        """The starting lengthscale: the prior's, or a caller's clipped into its box."""
        dtype = self.X.dtype
        if dstart is None:
            return jnp.broadcast_to(jnp.asarray(self.prior.d.start, dtype), shape)
        d = jnp.broadcast_to(jnp.asarray(dstart, dtype), shape)
        lo = jnp.asarray(self.prior.d.min, dtype)
        hi = jnp.asarray(self.prior.d.max, dtype)
        return jnp.clip(d, lo, hi)

    def local(
        self,
        Xref: Float[Array, "nref m"],
        dstart: Optional[Float[Array, "m"]] = None,
        lite: bool = True,
    ) -> Prediction:
        """One local GP at one reference set, matching laGPsep. `s2` is the raw scale.

        `dstart` overrides the starting lengthscale (clipped into the prior's box), which
        is what a warm-started second pass needs.
        """
        Xref = jnp.atleast_2d(jnp.asarray(Xref, self.X.dtype))
        if Xref.shape[-1] != self.X.shape[-1]:
            raise ValueError("mismatch Xref and X cols")
        d0 = self._dstart(dstart, (self.X.shape[-1],))
        return _core(self, Xref, d0, lite=lite)

    def predict(
        self,
        XX: Float[Array, "nn m"],
        batch: Optional[int] = 256,
        dstart: Optional[Float[Array, "nn m"]] = None,
    ) -> Prediction:
        """One local GP per predictive location, matching aGPsep.

        `local` mapped over the rows of XX with `jax.lax.map`. `batch` bounds how many
        locations are in flight at once; it trades peak memory for throughput and does
        not change results. For other batching -- `vmap`, sharding -- map `local` yourself.
        """
        XX = jnp.atleast_2d(jnp.asarray(XX, self.X.dtype))
        if XX.shape[-1] != self.X.shape[-1]:
            raise ValueError("mismatch XX and X cols")
        D0 = self._dstart(dstart, XX.shape)

        def one(loc):
            x, d = loc
            return self.local(x[None, :], dstart=d)

        out = jax.lax.map(one, (XX, D0), batch_size=batch)
        return out._replace(mean=out.mean[:, 0], s2=out.s2[:, 0])

    def refine(
        self,
        XX: Float[Array, "nn m"],
        passes: Sequence[Any] = ("nn", "alc"),
        batch: Optional[int] = 256,
    ) -> Prediction:
        """Repeat `predict`, warm-starting each pass from the lengthscales the last one fit.

        Design selection scores candidates at the *starting* lengthscale, so a pass that
        inherits per-location lengthscales picks a neighbourhood respecting which dimensions
        matter. Port of laGP's aGP.seq. The first pass only exists to supply lengthscales,
        so `("nn", "alc")` beats `("alc", "alc")` at a fraction of the cost.

        The loop stays Python rather than `lax.scan` because passes may name different
        criteria, which compile to different functions; it unrolls at trace time, so the
        whole thing still jits and differentiates as one graph.
        """
        if not passes:
            raise ValueError("passes must name at least one pass")

        XX = jnp.atleast_2d(jnp.asarray(XX, self.X.dtype))
        dstart = self._dstart(None, XX.shape)
        for criterion in passes:
            emu = self._replace(scheme=self.scheme.replace(criterion=criterion))
            out = emu.predict(XX, batch=batch, dstart=dstart)
            dstart = jnp.where(jnp.isfinite(out.fit.d), out.fit.d, dstart)
        return out


@pytree_config
class Local:
    """The local scheme's settings; bind data with `condition`.

    `criterion` and `search` are a grid, not a list of methods: any `Criterion` works with
    any `Search`. Names are accepted for both (`"alc"`, `"ray"`) and plain scoring
    functions for the criterion; all are normalized to objects on construction. `close`
    bounds the candidate set; its default is ten times wider for ray and continuous,
    matching laGP, since those only touch the candidates their starting points reach.

    A pytree: the integers and `trend` are static, `criterion` and `search` are children.
    """

    start: int = static(default=6)
    end: int = static(default=50)
    criterion: Any = "alc"
    search: Any = "exhaustive"
    trend: Trend = static(default="zero")
    close: Optional[int] = static(default=None)
    maxit: int = static(default=100)

    def __post_init__(self):
        object.__setattr__(self, "criterion", as_criterion(self.criterion))
        object.__setattr__(self, "search", as_search(self.search))

    def replace(self, **changes) -> "Local":
        """A copy with some settings changed; names are normalized as on construction."""
        return dataclasses.replace(self, **changes)

    def condition(
        self,
        X: Float[Array, "n m"],
        Z: Float[Array, "n"],
        prior: Optional[Prior] = None,
        key: Optional[Array] = None,
    ) -> Conditioned:
        """Bind data and resolve the prior, the one host-side step in the pipeline.

        `key` is only needed when no prior is given and X has over 1000 rows: laGP's darg
        then subsamples the design to set the lengthscale range.
        """
        X = _floating(X)
        Z = jnp.asarray(Z, X.dtype)
        n = X.shape[0]
        if len(Z) != n:
            raise ValueError("length(Z) != nrow(X)")
        if not (6 <= self.start < self.end):
            raise ValueError("must have 6 <= start < end")
        if n <= self.end:
            raise ValueError("nrow(X) <= end, so nothing to do")

        close = (
            _default_close(self.end, n, self.search)
            if self.close is None
            else int(self.close)
        )
        if not (self.end < close <= n):
            raise ValueError("must have end < close <= n")

        if prior is None:
            prior = Prior.from_design(np.asarray(X), np.asarray(Z), key=key)
        return Conditioned(self.replace(close=close), X, Z, prior)


def agp(
    X: Float[Array, "n m"],
    Z: Float[Array, "n"],
    XX: Float[Array, "nn m"],
    prior: Optional[Prior] = None,
    key: Optional[Array] = None,
    batch: Optional[int] = 256,
    passes: Optional[Sequence[Any]] = None,
    **settings: Any,
) -> Prediction:
    """Local approximate GP at every predictive location, matching aGPsep.

    `passes` runs the iterated refinement of `aGP.seq` instead of a single pass.
    """
    emu = Local(**settings).condition(X, Z, prior=prior, key=key)
    if passes is None:
        return emu.predict(XX, batch=batch)
    return emu.refine(XX, passes=passes, batch=batch)


def lagp(
    X: Float[Array, "n m"],
    Z: Float[Array, "n"],
    Xref: Float[Array, "nref m"],
    prior: Optional[Prior] = None,
    key: Optional[Array] = None,
    lite: bool = True,
    **settings: Any,
) -> Prediction:
    """Local approximate GP at one reference set, matching laGPsep."""
    emu = Local(**settings).condition(X, Z, prior=prior, key=key)
    return emu.local(Xref, lite=lite)
