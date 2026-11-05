"""Hyperparameter estimation, port of laGP's mleGPsep / mleGPsep_nug / jmleGPsep."""

from typing import NamedTuple
from jaxtyping import Array, Float, Int, Scalar

import jax
import jax.numpy as jnp
from vlse.optim import minimise

from .gp import GP, llik_d, llik_dg, llik_g
from .optim import bracketed_newton, implicit_argmin, r_ftol
from .prior import Prior

sg = jax.lax.stop_gradient


class Fit(NamedTuple):
    """Fitted hyperparameters plus the lengthscale optimizer's diagnostics.

    `conv` is laGP's: 0 converged, 1 hit maxit, 102 started on the lower boundary and was
    reset instead of optimized. `its` counts lengthscale iterations; the nugget's
    safeguarded Newton either converges or clamps, so its own count says little.
    """

    d: Float[Array, "m"]
    g: Scalar
    its: Int
    conv: Int


class MleD(NamedTuple):
    gp: GP
    d: Float[Array, "m"]
    its: Int
    conv: Int


class MleG(NamedTuple):
    gp: GP
    g: Scalar
    its: Int


def _bounds(v: Scalar, m: int, dtype) -> Float[Array, "m"]:
    return jnp.broadcast_to(jnp.atleast_1d(jnp.asarray(v, dtype)), (m,))


def _solve_d(gp: GP, d0, g, dmin, dmax, ab, maxit: int):
    """The forward lengthscale solve on a GP already cut off from the gradient tape.

    Hyperparameters come in as arguments and go out as arguments: `jmle` alternates over
    them, and rebuilding the GP between half-steps would factorize a K nothing reads.
    """
    res = minimise(
        lambda d: -llik_dg(gp, d, g, dab=ab),
        d0,
        bounds=(dmin, dmax),
        tol=0.0,
        ftol=r_ftol(d0.dtype),
        max_iterations=maxit,
    )
    # R's convergence code: 1 only when maxit ran out before any stopping rule fired
    stopped = (res.error <= 0.0) | res.converged_ftol | res.failed_linesearch
    conv = jnp.where(stopped | (res.iteration < maxit), 0, 1)

    # laGP resets rather than optimizing when started on the lower boundary
    reset = jnp.any(d0 <= dmin)
    d = jnp.where(reset, 0.9 * dmin + 0.1 * dmax, res.x)
    return d, jnp.where(reset, 0, res.iteration), jnp.where(reset, 102, conv)


def _solve_g(gp: GP, d, g0, gmin, gmax, ab, maxit: int):
    """The forward nugget solve at a given lengthscale, likewise without rebuilding.

    Newton needs both derivatives at the same `g`, and one jvp of the gradient produces
    them from a single factorization -- `grad` and `grad(grad(...))` as separate calls
    Cholesky the same K twice per iteration. Sharing the pass reassociates the arithmetic,
    so `g` moves in its last ulps; see CLAUDE.md on why that is judged on parity, not on
    bit-identity.
    """
    dobj = jax.grad(lambda g: llik_dg(gp, d, g, gab=ab))
    res = bracketed_newton(
        lambda g: jax.jvp(dobj, (g,), (jnp.ones_like(g),)),
        g0,
        gmin,
        gmax,
        maxiter=maxit,
    )
    return res.x, res.its


def mle_d(
    gp: GP,
    dmin: Scalar,
    dmax: Scalar,
    ab: tuple[Scalar, Scalar] = (0.0, 0.0),
    maxit: int = 100,
) -> MleD:
    """MAP lengthscale by box-constrained L-BFGS, matching mleGPsep(param="d")."""
    dmin = _bounds(dmin, gp.m, gp.Z.dtype)
    dmax = _bounds(dmax, gp.m, gp.Z.dtype)
    solved, its, conv = _solve_d(
        sg(gp), sg(gp.d), sg(gp.g), sg(dmin), sg(dmax), sg(ab), maxit
    )

    # conv=102 is a reset to a constant, not a solve, so it carries no gradient either way
    d = implicit_argmin(lambda d: -llik_d(gp, d, dab=ab), solved, dmin, dmax)
    d = jnp.where(conv == 102, solved, d)
    return MleD(gp=gp.at(d=d), d=d, its=its, conv=conv)


def mle_g(
    gp: GP,
    gmin: Scalar,
    gmax: Scalar,
    ab: tuple[Scalar, Scalar] = (0.0, 0.0),
    maxit: int = 100,
) -> MleG:
    """MAP nugget by safeguarded Newton, matching mleGPsep(param="g")."""
    solved, its = _solve_g(
        sg(gp), sg(gp.d), sg(gp.g), sg(gmin), sg(gmax), sg(ab), maxit
    )

    g = implicit_argmin(
        lambda v: -llik_g(gp, v[0], gab=ab),
        jnp.atleast_1d(solved),
        _bounds(gmin, 1, gp.Z.dtype),
        _bounds(gmax, 1, gp.Z.dtype),
    )[0]
    return MleG(gp=gp.at(g=g), g=g, its=its)


def jmle(
    gp: GP,
    drange: tuple[Scalar, Scalar],
    grange: tuple[Scalar, Scalar],
    dab: tuple[Scalar, Scalar] = (0.0, 0.0),
    gab: tuple[Scalar, Scalar] = (0.0, 0.0),
    N: int = 100,
    maxit: int = 100,
) -> tuple[GP, Fit]:
    """Alternate lengthscale and nugget estimation, matching jmleGPsep."""
    m, dtype = gp.m, gp.Z.dtype
    dmin, dmax = _bounds(drange[0], m, dtype), _bounds(drange[1], m, dtype)
    gmin, gmax = _bounds(grange[0], 1, dtype), _bounds(grange[1], 1, dtype)

    def cond(state):
        _, _, dits, gits, dconv, i = state
        settled = (gits <= 2) & (dits <= m + 1) & (dconv == 0)
        return (i < N) & ~(settled | (dconv > 1))

    # the loop carries the hyperparameters, not a GP: nothing in either half-step reads a
    # factorization, so rebuilding one per iteration would be two dead Choleskys
    def body(state):
        d0, g0, _, _, _, i = state
        d, dits, dconv = _solve_d(frozen, d0, g0, dmin, dmax, dab, maxit)
        g, gits = _solve_g(frozen, d, g0, gmin[0], gmax[0], gab, maxit)
        return (d, g, dits, gits, dconv, i + 1)

    frozen = sg(gp)
    init = sg(
        (gp.d, gp.g, jnp.asarray(m + 2), jnp.asarray(3), jnp.asarray(0), jnp.asarray(0))
    )
    dhat, ghat, dits, _, dconv, _ = jax.lax.while_loop(cond, body, init)

    # both subproblems have converged, so their answer is a joint stationary point of the
    # one MAP objective and the implicit correction can be taken over (d, g) together
    theta = implicit_argmin(
        lambda t: -llik_dg(gp, t[:m], t[m], dab=dab, gab=gab),
        jnp.concatenate([dhat, jnp.atleast_1d(ghat)]),
        jnp.concatenate([dmin, gmin]),
        jnp.concatenate([dmax, gmax]),
        free=jnp.concatenate([jnp.broadcast_to(dconv != 102, (m,)), jnp.ones(1, bool)]),
    )
    out = gp.at(d=theta[:m], g=theta[m])
    return out, Fit(d=out.d, g=out.g, its=dits, conv=dconv)


def fit(gp: GP, prior: Prior, maxit: int = 100) -> tuple[GP, Fit]:
    """Estimate whichever hyperparameters the prior marks estimable, per mleGPsep.switch."""
    drange = (jnp.asarray(prior.d.min), jnp.asarray(prior.d.max))
    grange = (jnp.asarray(prior.g.min), jnp.asarray(prior.g.max))
    dab = tuple(jnp.asarray(v) for v in prior.d.ab)
    gab = tuple(jnp.asarray(v) for v in prior.g.ab)
    zero = jnp.asarray(0)

    if prior.d.estimate and prior.g.estimate:
        return jmle(gp, drange, grange, dab, gab, maxit=maxit)
    if prior.d.estimate:
        r = mle_d(gp, *drange, ab=dab, maxit=maxit)
        return r.gp, Fit(d=r.d, g=r.gp.g, its=r.its, conv=r.conv)
    if prior.g.estimate:
        r = mle_g(gp, *grange, ab=gab, maxit=maxit)
        return r.gp, Fit(d=r.gp.d, g=r.g, its=r.its, conv=zero)
    return gp, Fit(d=gp.d, g=gp.g, its=zero, conv=zero)
