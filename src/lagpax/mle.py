"""Hyperparameter estimation, port of laGP's mleGPsep / mleGPsep_nug / jmleGPsep."""

from typing import NamedTuple
from jaxtyping import Array, Float, Int, Scalar

import jax
import jax.numpy as jnp

from .gp import GPsep, d2llik_gpsep_g, dllik_gpsep_g, llik_d, llik_g, newparams_gpsep
from .optim import bracketed_newton, lbfgsb


class MleD(NamedTuple):
    """Result of a lengthscale MLE/MAP: `conv` is 0 ok, 1 maxit, 102 boundary reset."""

    gp: GPsep
    d: Float[Array, "m"]
    its: Int
    conv: Int


class MleG(NamedTuple):
    """Result of a nugget MLE/MAP."""

    gp: GPsep
    g: Scalar
    its: Int


class Jmle(NamedTuple):
    """Result of alternating lengthscale/nugget estimation."""

    gp: GPsep
    d: Float[Array, "m"]
    g: Scalar
    dits: Int
    gits: Int
    dconv: Int


def mle_gpsep_d(
    gp: GPsep,
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
    return MleD(gp=newparams_gpsep(gp, d=d), d=d, its=its, conv=conv)


def mle_gpsep_nug(
    gp: GPsep,
    gmin: Scalar,
    gmax: Scalar,
    ab: tuple[Scalar, Scalar] = (0.0, 0.0),
    maxit: int = 100,
) -> MleG:
    """MAP nugget by safeguarded Newton, matching mleGPsep(param="g")."""
    res = bracketed_newton(
        lambda g: dllik_gpsep_g(gp, g, gab=ab),
        lambda g: d2llik_gpsep_g(gp, g, gab=ab),
        gp.g,
        gmin,
        gmax,
        maxiter=maxit,
    )
    return MleG(gp=newparams_gpsep(gp, g=res.x), g=res.x, its=res.its)


def jmle_gpsep(
    gp: GPsep,
    drange: tuple[Scalar, Scalar],
    grange: tuple[Scalar, Scalar],
    dab: tuple[Scalar, Scalar] = (0.0, 0.0),
    gab: tuple[Scalar, Scalar] = (0.0, 0.0),
    N: int = 100,
    maxit: int = 100,
) -> Jmle:
    """Alternate lengthscale and nugget estimation, matching jmleGPsep."""
    m = gp.m

    def cond(state):
        gp, dits, gits, dconv, i = state
        settled = (gits <= 2) & (dits <= m + 1) & (dconv == 0)
        return (i < N) & ~(settled | (dconv > 1))

    def body(state):
        gp, _, _, _, i = state
        d = mle_gpsep_d(gp, drange[0], drange[1], ab=dab, maxit=maxit)
        g = mle_gpsep_nug(d.gp, grange[0], grange[1], ab=gab, maxit=maxit)
        return (g.gp, d.its, g.its, d.conv, i + 1)

    init = (gp, jnp.asarray(m + 2), jnp.asarray(3), jnp.asarray(0), jnp.asarray(0))
    gp, dits, gits, dconv, _ = jax.lax.while_loop(cond, body, init)
    return Jmle(gp=gp, d=gp.d, g=gp.g, dits=dits, gits=gits, dconv=dconv)


def mle_switch(
    gp: GPsep,
    d_mle: bool,
    g_mle: bool,
    drange: tuple[Scalar, Scalar],
    grange: tuple[Scalar, Scalar],
    dab: tuple[Scalar, Scalar],
    gab: tuple[Scalar, Scalar],
    maxit: int = 100,
) -> tuple[GPsep, dict]:
    """Dispatch on which hyperparameters are estimated, matching mleGPsep.switch."""
    if d_mle and g_mle:
        r = jmle_gpsep(gp, drange, grange, dab, gab, maxit=maxit)
        return r.gp, dict(d=r.d, g=r.g, dits=r.dits, gits=r.gits, dconv=r.dconv)
    if d_mle:
        r = mle_gpsep_d(gp, drange[0], drange[1], ab=dab, maxit=maxit)
        return r.gp, dict(d=r.d, dits=r.its, dconv=r.conv)
    if g_mle:
        r = mle_gpsep_nug(gp, grange[0], grange[1], ab=gab, maxit=maxit)
        return r.gp, dict(g=r.g, gits=r.its)
    return gp, {}
