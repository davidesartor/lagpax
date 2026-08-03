"""The local approximate GP scheme, port of laGP's laGPsep and aGPsep.

Configuration is separate from data (`Local` vs `Local.condition`), and the criterion is
separate from the search that uses it. `condition` resolves the prior once, on the host,
ahead of every bit of JAX work — that seam is what keeps the per-location path pure.
"""

from functools import partial
from collections.abc import Sequence
from typing import Any, NamedTuple, Optional
from jaxtyping import Array, Bool, Float, Int, Scalar

import jax
import jax.numpy as jnp
import numpy as np

from .covar import distance
from .criteria import Criterion, _named as _named_criterion, nn
from .gp import GP, Prediction, Trend
from .mle import Fit, fit
from .prior import Prior, Range
from .search import _named as _named_search, exhaustive


class _Cfg(NamedTuple):
    """Everything the compiled core branches on; hashable so it can be a static argument."""

    start: int
    end: int
    close: int
    criterion: Any
    search: Any
    trend: Trend
    lite: bool
    d_estimate: bool
    g_estimate: bool
    maxit: int


def _default_close(end: int, n: int, search: Any) -> int:
    """laGP's default: a wider candidate set when the search does not scan all of it."""
    wide = search in ("ray", "continuous")
    return int(min((1000 + end) * (10 if wide else 1), n))


def _ranges(prior: Prior, m: int):
    """Traced prior arrays; the estimate flags stay on the host as static config."""
    d0 = jnp.broadcast_to(jnp.asarray(prior.d.start, jnp.float64), (m,))
    return (
        d0,
        jnp.asarray(prior.g.start, jnp.float64),
        (
            jnp.asarray(prior.d.min),
            jnp.asarray(prior.d.max),
            jnp.asarray(prior.d.ab[0]),
            jnp.asarray(prior.d.ab[1]),
        ),
        (
            jnp.asarray(prior.g.min),
            jnp.asarray(prior.g.max),
            jnp.asarray(prior.g.ab[0]),
            jnp.asarray(prior.g.ab[1]),
        ),
    )


def _prior_of(dr, gr, cfg: _Cfg) -> Prior:
    """Rebuild a Prior from traced bounds, so `mle.fit` has one code path."""
    return Prior(
        d=Range(min=dr[0], max=dr[1], ab=(dr[2], dr[3]), estimate=cfg.d_estimate),
        g=Range(min=gr[0], max=gr[1], ab=(gr[2], gr[3]), estimate=cfg.g_estimate),
    )


def _design(
    criterion: Criterion,
    search,
    Xloc: Float[Array, "close m"],
    Zloc: Float[Array, "close"],
    Xref: Float[Array, "nref m"],
    d0: Float[Array, "m"],
    g0: Scalar,
    cfg: _Cfg,
) -> Int[Array, "end"]:
    """Grow the local design one point at a time, returning positions into Xloc."""
    start, end = cfg.start, cfg.end

    # nearest neighbour needs no search: the candidates are already distance-ordered
    if criterion is nn:
        return jnp.arange(end)

    rect = jnp.stack([jnp.min(Xloc, axis=0), jnp.max(Xloc, axis=0)])
    pos = jnp.concatenate([jnp.arange(start), jnp.zeros(end - start, int)])
    mask = jnp.arange(end) < start
    taken = jnp.arange(cfg.close) < start

    def greedy(t: Scalar, state):
        pos, mask, taken = state
        gp = GP(Xloc[pos], Zloc[pos], d0, g0, mask, trend=cfg.trend)
        w = search(criterion, gp, Xloc, Xref, taken, t, rect)
        return pos.at[t].set(w), mask.at[t].set(True), taken.at[w].set(True)

    pos, _, _ = jax.lax.fori_loop(start, end, greedy, (pos, mask, taken))
    return pos


@partial(jax.jit, static_argnames=("cfg",))
def _core(
    Xref: Float[Array, "nref m"],
    X: Float[Array, "n m"],
    Z: Float[Array, "n"],
    d0: Float[Array, "m"],
    g0: Scalar,
    dr,
    gr,
    *,
    cfg: _Cfg,
) -> Prediction:
    """One local approximate GP: choose a design, fit it, predict at Xref."""
    criterion = _named_criterion(cfg.criterion)
    search = _named_search(cfg.search)

    # only the `close` nearest candidates are ever used, and top_k breaks ties on the
    # lower index the way a stable sort does
    dst = jnp.min(distance(Xref, X), axis=0)
    sel = jax.lax.top_k(-dst, cfg.close)[1]
    Xloc, Zloc = X[sel], Z[sel]

    pos = _design(criterion, search, Xloc, Zloc, Xref, d0, g0, cfg)
    gp = GP(Xloc[pos], Zloc[pos], d0, g0, trend=cfg.trend)
    gp, fitted = fit(gp, _prior_of(dr, gr, cfg), maxit=cfg.maxit)

    pred = gp.predict(Xref, lite=cfg.lite)
    return pred._replace(Xi=sel[pos], fit=fitted)


class Conditioned(NamedTuple):
    """A `Local` scheme bound to data and a resolved prior; the JAX-only half of the API."""

    scheme: "Local"
    X: Float[Array, "n m"]
    Z: Float[Array, "n"]
    prior: Prior

    def _cfg(self, criterion=None, lite: bool = True) -> _Cfg:
        s = self.scheme
        return _Cfg(
            start=s.start,
            end=s.end,
            close=s.close,
            criterion=s.criterion if criterion is None else criterion,
            search=s.search,
            trend=s.trend,
            lite=lite,
            d_estimate=bool(self.prior.d.estimate),
            g_estimate=bool(self.prior.g.estimate),
            maxit=s.maxit,
        )

    def local(
        self,
        Xref: Float[Array, "nref m"],
        lite: bool = True,
    ) -> Prediction:
        """One local GP at one reference set, matching laGPsep. `s2` is the raw scale."""
        Xref = jnp.atleast_2d(jnp.asarray(Xref, jnp.float64))
        d0, g0, dr, gr = _ranges(self.prior, self.X.shape[-1])
        return _core(Xref, self.X, self.Z, d0, g0, dr, gr, cfg=self._cfg(lite=lite))

    def predict(
        self,
        XX: Float[Array, "nn m"],
        batch: Optional[int] = 64,
        criterion=None,
        dstart: Optional[Float[Array, "nn m"]] = None,
    ) -> Prediction:
        """One local GP per predictive location, matching aGPsep.

        `batch` bounds how many locations are in flight at once; it trades peak memory for
        throughput and does not change results. `dstart` overrides the starting lengthscale
        per location, which is what makes a warm-started second pass worthwhile.
        """
        XX = jnp.atleast_2d(jnp.asarray(XX, jnp.float64))
        m = self.X.shape[-1]
        if XX.shape[-1] != m:
            raise ValueError("mismatch XX and X cols")

        d0, g0, dr, gr = _ranges(self.prior, m)
        D0 = jnp.broadcast_to(d0, (XX.shape[0], m))
        if dstart is not None:
            D0 = jnp.broadcast_to(jnp.asarray(dstart, jnp.float64), (XX.shape[0], m))
            D0 = jnp.clip(D0, dr[0], dr[1])

        cfg = self._cfg(criterion=criterion, lite=True)

        def one(loc):
            xref, d = loc
            return _core(xref[None, :], self.X, self.Z, d, g0, dr, gr, cfg=cfg)

        out = jax.lax.map(one, (XX, D0), batch_size=batch)
        return out._replace(mean=out.mean[:, 0], s2=out.s2[:, 0])

    def refine(
        self,
        XX: Float[Array, "nn m"],
        passes: Sequence[Any] = ("nn", "alc"),
        batch: Optional[int] = 64,
    ) -> Prediction:
        """Repeat `predict`, warm-starting each pass from the lengthscales the last one fit.

        Design selection scores candidates at the *starting* lengthscale, so a pass that
        inherits per-location lengthscales picks a neighbourhood respecting which dimensions
        matter. Port of laGP's aGP.seq. The first pass only exists to supply lengthscales,
        so `("nn", "alc")` beats `("alc", "alc")` at a fraction of the cost.
        """
        if not passes:
            raise ValueError("passes must name at least one pass")

        XX = jnp.atleast_2d(jnp.asarray(XX, jnp.float64))
        dstart = jnp.broadcast_to(
            jnp.asarray(self.prior.d.start, jnp.float64), XX.shape
        )
        for criterion in passes:
            out = self.predict(XX, batch=batch, criterion=criterion, dstart=dstart)
            dstart = jnp.where(jnp.isfinite(out.fit.d), out.fit.d, dstart)
        return out


class Local(NamedTuple):
    """Settings for the local scheme; bind data with `condition`.

    `criterion` and `search` are a grid, not a list of methods: any callable with the
    criterion signature works, and a search needing configuration is a `functools.partial`
    of the one in `lagpax.search`. `close` is meaningful only for `search="exhaustive"` —
    ray and continuous search look at the candidate set through their own starting points.
    """

    start: int = 6
    end: int = 50
    criterion: Any = "alc"
    search: Any = "exhaustive"
    trend: Trend = "zero"
    close: Optional[int] = None
    maxit: int = 100

    def condition(
        self,
        X: Float[Array, "n m"],
        Z: Float[Array, "n"],
        prior: Optional[Prior] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> Conditioned:
        """Bind data and resolve the prior, the one host-side step in the pipeline."""
        X = jnp.asarray(X, jnp.float64)
        Z = jnp.asarray(Z, jnp.float64)
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
            prior = Prior.from_design(np.asarray(X), np.asarray(Z), rng=rng)
        return Conditioned(self._replace(close=close), X, Z, prior)


def agp(
    X: Float[Array, "n m"],
    Z: Float[Array, "n"],
    XX: Float[Array, "nn m"],
    prior: Optional[Prior] = None,
    rng: Optional[np.random.Generator] = None,
    batch: Optional[int] = 64,
    passes: Optional[Sequence[Any]] = None,
    **settings: Any,
) -> Prediction:
    """Local approximate GP at every predictive location, matching aGPsep.

    `passes` runs the iterated refinement of `aGP.seq` instead of a single pass.
    """
    emu = Local(**settings).condition(X, Z, prior=prior, rng=rng)
    if passes is None:
        return emu.predict(XX, batch=batch)
    return emu.refine(XX, passes=passes, batch=batch)


def lagp(
    X: Float[Array, "n m"],
    Z: Float[Array, "n"],
    Xref: Float[Array, "nref m"],
    prior: Optional[Prior] = None,
    rng: Optional[np.random.Generator] = None,
    lite: bool = True,
    **settings: Any,
) -> Prediction:
    """Local approximate GP at one reference set, matching laGPsep."""
    emu = Local(**settings).condition(X, Z, prior=prior, rng=rng)
    return emu.local(Xref, lite=lite)
