"""Separable Gaussian process, port of laGP's GPsep (gp_sep.c)."""

from typing import NamedTuple
from jaxtyping import Array, Bool, Float, Scalar

import jax
import jax.numpy as jnp
import jax.scipy as jsp

from .covar import covar_sep, covar_sep_symm

SDEPS = float(jnp.sqrt(jnp.finfo(jnp.float64).eps))


class GPsep(NamedTuple):
    """laGP GPsep state: data, hyperparameters, and the cached Cholesky-derived quantities.

    `mask` marks rows of (X, Z) that are part of the model; masked-out rows get an
    identity block in K so that padded (fixed-shape, vmappable) designs behave exactly
    like a smaller design.
    """

    X: Float[Array, "n m"]
    Z: Float[Array, "n"]
    d: Float[Array, "m"]
    g: Scalar
    mask: Bool[Array, "n"]
    K: Float[Array, "n n"]
    Ki: Float[Array, "n n"]
    ldetK: Scalar
    KiZ: Float[Array, "n"]
    phi: Scalar

    @property
    def n(self) -> Scalar:
        return jnp.sum(self.mask.astype(self.Z.dtype))

    @property
    def m(self) -> int:
        return self.X.shape[-1]


def new_gpsep(
    X: Float[Array, "n m"],
    Z: Float[Array, "n"],
    d: Float[Array, "m"] | Scalar,
    g: Scalar,
    mask: Bool[Array, "n"] | None = None,
) -> GPsep:
    """Build a separable GP, matching laGP's newGPsep/buildGPsep."""
    n, m = X.shape
    d = jnp.broadcast_to(jnp.atleast_1d(jnp.asarray(d, X.dtype)), (m,))
    g = jnp.asarray(g, X.dtype)
    mask = jnp.ones(n, bool) if mask is None else mask

    # masked-out rows/cols become identity so they contribute nothing to ldetK or phi
    K = covar_sep_symm(d, X, g)
    K = jnp.where(mask[:, None] & mask[None, :], K, jnp.eye(n, dtype=X.dtype))
    Z = jnp.where(mask, Z, 0.0)

    K_sqrt, is_lower = jsp.linalg.cho_factor(K)
    ldetK = 2.0 * jnp.sum(jnp.log(jnp.diag(K_sqrt)))
    Ki = jsp.linalg.cho_solve((K_sqrt, is_lower), jnp.eye(n, dtype=X.dtype))

    KiZ = Ki @ Z
    phi = jnp.dot(Z, KiZ)
    return GPsep(
        X=X, Z=Z, d=d, g=g, mask=mask, K=K, Ki=Ki, ldetK=ldetK, KiZ=KiZ, phi=phi
    )


def newparams_gpsep(gp: GPsep, d=None, g=None) -> GPsep:
    """Rebuild a GP with new hyperparameters, matching laGP's newparamsGPsep."""
    return new_gpsep(
        gp.X,
        gp.Z,
        gp.d if d is None else d,
        gp.g if g is None else g,
        gp.mask,
    )


def _dgamma_log(x: Float[Array, "..."], a: Scalar, b: Scalar) -> Float[Array, "..."]:
    """log density of Gamma(shape=a, rate=b), i.e. R's dgamma(x, a, scale=1/b, log=TRUE)."""
    return (a - 1.0) * jnp.log(x) - b * x + a * jnp.log(b) - jsp.special.gammaln(a)


def llik_gpsep(
    gp: GPsep,
    dab: tuple[Scalar, Scalar] = (0.0, 0.0),
    gab: tuple[Scalar, Scalar] = (0.0, 0.0),
) -> Scalar:
    """Log marginal likelihood (plus optional Gamma log-priors), matching llikGPsep."""
    n = gp.n
    llik = -0.5 * (n * jnp.log(0.5 * gp.phi) + gp.ldetK)

    # priors are only applied when both hyperparameters of the Gamma are positive
    use_dab = (dab[0] > 0) & (dab[1] > 0)
    db = jnp.where(use_dab, dab[1], 1.0)
    prior_d = jnp.sum(jnp.where(gp.d > 0, _dgamma_log(gp.d, dab[0], db), 0.0))
    llik = llik + jnp.where(use_dab, prior_d, 0.0)

    use_gab = (gp.g > 0) & (gab[0] > 0) & (gab[1] > 0)
    gb = jnp.where(use_gab, gab[1], 1.0)
    gs = jnp.where(use_gab, gp.g, 1.0)
    llik = llik + jnp.where(use_gab, _dgamma_log(gs, gab[0], gb), 0.0)
    return llik


def llik_d(
    gp: GPsep, d: Float[Array, "m"], dab: tuple[Scalar, Scalar] = (0.0, 0.0)
) -> Scalar:
    """llik as a function of the lengthscale only, for gradients and optimization."""
    return llik_gpsep(newparams_gpsep(gp, d=d), dab=dab)


def llik_g(gp: GPsep, g: Scalar, gab: tuple[Scalar, Scalar] = (0.0, 0.0)) -> Scalar:
    """llik as a function of the nugget only, for gradients and optimization."""
    return llik_gpsep(newparams_gpsep(gp, g=g), gab=gab)


dllik_gpsep_d = jax.grad(llik_d, argnums=1)
dllik_gpsep_g = jax.grad(llik_g, argnums=1)
d2llik_gpsep_g = jax.grad(dllik_gpsep_g, argnums=1)


class Prediction(NamedTuple):
    """Student-t predictive parameters returned by predGPsep."""

    mean: Float[Array, "nn"]
    s2: Float[Array, "nn"] | Float[Array, "nn nn"]
    df: Scalar
    llik: Scalar


def pred_gpsep(
    gp: GPsep,
    XX: Float[Array, "nn m"],
    lite: bool = True,
    nonug: bool = False,
) -> Prediction:
    """Student-t predictive equations, matching predGPsep / predGPsep_lite."""
    g = SDEPS if nonug else gp.g
    df = gp.n
    phidf = gp.phi / df
    llik = -0.5 * (df * jnp.log(0.5 * gp.phi) + gp.ldetK)

    k = covar_sep(gp.d, gp.X, XX)
    k = jnp.where(gp.mask[:, None], k, 0.0)
    ktKi = k.T @ gp.Ki
    mean = ktKi @ gp.Z

    if lite:
        ktKik = jnp.sum(ktKi.T * k, axis=0)
        s2 = phidf * (1.0 + g - ktKik)
    else:
        s2 = phidf * (covar_sep_symm(gp.d, XX, g) - ktKi @ k)
    return Prediction(mean=mean, s2=s2, df=df, llik=llik)


def update_gpsep(
    gp: GPsep, Xnew: Float[Array, "k m"], Znew: Float[Array, "k"]
) -> GPsep:
    """Append observations and rebuild, matching updateGPsep up to the rank-1 shortcut."""
    return new_gpsep(
        jnp.concatenate([gp.X, Xnew]),
        jnp.concatenate([gp.Z, Znew]),
        gp.d,
        gp.g,
        jnp.concatenate([gp.mask, jnp.ones(len(Znew), bool)]),
    )
