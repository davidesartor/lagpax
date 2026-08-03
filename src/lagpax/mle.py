"""Hyperparameter estimation, port of laGP's mleGPsep / mleGPsep_nug / jmleGPsep."""

from typing import NamedTuple
from jaxtyping import Array, Float, Int, Scalar

import jax
import jax.numpy as jnp

from .gp import GP, d2llik_g, dllik_g, llik_d
from .optim import bracketed_newton, lbfgsb
from .prior import Prior


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


def mle_d(
    gp: GP,
    dmin: Scalar,
    dmax: Scalar,
    ab: tuple[Scalar, Scalar] = (0.0, 0.0),
    maxit: int = 100,
) -> MleD:
    """MAP lengthscale by box-constrained L-BFGS, matching mleGPsep(param="d")."""
    m = gp.m
    dmin = jnp.broadcast_to(jnp.atleast_1d(jnp.asarray(dmin, gp.Z.dtype)), (m,))
    dmax = jnp.broadcast_to(jnp.atleast_1d(jnp.asarray(dmax, gp.Z.dtype)), (m,))

    res = lbfgsb(lambda d: -llik_d(gp, d, dab=ab), gp.d, dmin, dmax, maxiter=maxit)

    # laGP resets rather than optimizing when started on the lower boundary
    reset = jnp.any(gp.d <= dmin)
    d = jnp.where(reset, 0.9 * dmin + 0.1 * dmax, res.x)
    its = jnp.where(reset, 0, res.its)
    conv = jnp.where(reset, 102, res.conv)
    return MleD(gp=gp.at(d=d), d=d, its=its, conv=conv)


def mle_g(
    gp: GP,
    gmin: Scalar,
    gmax: Scalar,
    ab: tuple[Scalar, Scalar] = (0.0, 0.0),
    maxit: int = 100,
) -> MleG:
    """MAP nugget by safeguarded Newton, matching mleGPsep(param="g")."""
    res = bracketed_newton(
        lambda g: dllik_g(gp, g, gab=ab),
        lambda g: d2llik_g(gp, g, gab=ab),
        gp.g,
        gmin,
        gmax,
        maxiter=maxit,
    )
    return MleG(gp=gp.at(g=res.x), g=res.x, its=res.its)


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
    m = gp.m

    def cond(state):
        gp, dits, gits, dconv, i = state
        settled = (gits <= 2) & (dits <= m + 1) & (dconv == 0)
        return (i < N) & ~(settled | (dconv > 1))

    def body(state):
        gp, _, _, _, i = state
        d = mle_d(gp, drange[0], drange[1], ab=dab, maxit=maxit)
        g = mle_g(d.gp, grange[0], grange[1], ab=gab, maxit=maxit)
        return (g.gp, d.its, g.its, d.conv, i + 1)

    init = (gp, jnp.asarray(m + 2), jnp.asarray(3), jnp.asarray(0), jnp.asarray(0))
    gp, dits, _, dconv, _ = jax.lax.while_loop(cond, body, init)
    return gp, Fit(d=gp.d, g=gp.g, its=dits, conv=dconv)


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
