"""Separable Gaussian process, port of laGP's GPsep (gp_sep.c), plus trend models."""

from typing import Any, Literal, NamedTuple, Optional
from jaxtyping import Array, Bool, Float, Int, Scalar

import jax
import jax.numpy as jnp
import jax.scipy as jsp

from .covar import covar, covar_symm

# laGP's constant, R's sqrt(.Machine$double.eps): a parity value, not the working precision
SDEPS = float(jnp.sqrt(jnp.finfo(jnp.float64).eps))

Trend = Literal["zero", "constant", "linear"]


def trend_basis(trend: Trend, X: Float[Array, "n m"]) -> Float[Array, "n p"]:
    """Regression basis F: empty for simple kriging, 1 for ordinary, [1, X] for universal."""
    n = X.shape[0]
    if trend == "zero":
        return jnp.zeros((n, 0), X.dtype)
    if trend == "constant":
        return jnp.ones((n, 1), X.dtype)
    if trend == "linear":
        return jnp.concatenate([jnp.ones((n, 1), X.dtype), X], axis=-1)
    raise ValueError(f"unknown trend {trend!r}")


def _trend_of(p: int) -> Trend:
    """Recover the trend name from the basis width, for rebuilds that change n."""
    return "zero" if p == 0 else ("constant" if p == 1 else "linear")


class Prediction(NamedTuple):
    """Student-t predictive parameters; `Xi` and `fit` are filled by the local schemes."""

    mean: Float[Array, "nn"]
    s2: Float[Array, "nn"] | Float[Array, "nn nn"]
    df: Scalar
    llik: Scalar
    Xi: Optional[Int[Array, "..."]] = None
    fit: Optional[Any] = None

    @property
    def var(self) -> Float[Array, "nn"]:
        """The Student-t variance rather than its scale, which is what laGP's aGPsep returns."""
        return self.s2 * self.df / (self.df - 2.0)


class GP(NamedTuple):
    """Separable GP state: data, hyperparameters, and the cached factorizations.

    `mask` marks rows of (X, Z) that are part of the model; masked-out rows get an
    identity block in K, a zeroed Z entry and a zeroed F row, so a padded (fixed-shape,
    vmappable) design behaves exactly like a smaller one.

    Construct as `GP(X, Z, d, g, mask=None, trend="zero")`. The fields after `F` are
    derived, and are only ever passed explicitly when JAX rebuilds the pytree.
    """

    X: Float[Array, "n m"]
    Z: Float[Array, "n"]
    d: Float[Array, "m"]
    g: Scalar
    mask: Bool[Array, "n"]
    F: Float[Array, "n p"]
    K: Float[Array, "n n"]
    Ki: Float[Array, "n n"]
    ldetK: Scalar
    KiZ: Float[Array, "n"]
    phi: Scalar
    b: Float[Array, "p"]
    Kfi: Float[Array, "p p"]
    ldetF: Scalar

    @property
    def n(self) -> Scalar:
        return jnp.sum(self.mask.astype(self.Z.dtype))

    @property
    def m(self) -> int:
        return self.X.shape[-1]

    @property
    def p(self) -> int:
        return self.F.shape[-1]

    @property
    def df(self) -> Scalar:
        """Student-t degrees of freedom: n, less one per profiled trend coefficient."""
        return self.n - self.p

    @property
    def resid(self) -> Float[Array, "n"]:
        """Z with the fitted trend removed; Z itself under simple kriging."""
        return self.Z if self.p == 0 else self.Z - self.F @ self.b

    def at(self, d=None, g=None) -> "GP":
        """Rebuild with new hyperparameters, matching laGP's newparamsGPsep."""
        return GP(
            self.X,
            self.Z,
            self.d if d is None else d,
            self.g if g is None else g,
            self.mask,
            self.F,
        )

    def llik(self, prior=None) -> Scalar:
        """Log marginal likelihood, or the MAP objective when a prior is supplied."""
        return _llik(self, *_ab(prior))

    def predict(
        self,
        XX: Float[Array, "nn m"],
        lite: bool = True,
        nugget: bool = True,
    ) -> Prediction:
        """Student-t predictive equations, matching predGPsep / predGPsep_lite."""
        return _predict(self, XX, lite=lite, nugget=nugget)

    def update(self, Xnew: Float[Array, "k m"], Znew: Float[Array, "k"]) -> "GP":
        """Append observations and rebuild, matching updateGPsep up to the rank-1 shortcut."""
        return GP(
            jnp.concatenate([self.X, Xnew]),
            jnp.concatenate([self.Z, Znew]),
            self.d,
            self.g,
            jnp.concatenate([self.mask, jnp.ones(len(Znew), bool)]),
            trend=_trend_of(self.p),
        )


def _new(
    cls,
    X: Float[Array, "n m"],
    Z: Float[Array, "n"],
    d: Float[Array, "m"] | Scalar,
    g: Scalar,
    mask: Optional[Bool[Array, "n"]] = None,
    F: Optional[Float[Array, "n p"]] = None,
    K: Optional[Float[Array, "n n"]] = None,
    Ki: Optional[Float[Array, "n n"]] = None,
    ldetK: Optional[Scalar] = None,
    KiZ: Optional[Float[Array, "n"]] = None,
    phi: Optional[Scalar] = None,
    b: Optional[Float[Array, "p"]] = None,
    Kfi: Optional[Float[Array, "p p"]] = None,
    ldetF: Optional[Scalar] = None,
    *,
    trend: Trend = "zero",
) -> GP:
    """Build from data, or pass through when JAX is rebuilding the pytree."""
    if K is not None:
        return tuple.__new__(
            cls, (X, Z, d, g, mask, F, K, Ki, ldetK, KiZ, phi, b, Kfi, ldetF)
        )
    return _build(X, Z, d, g, mask, F, trend)


# assigned rather than declared: typing.NamedTuple refuses a __new__ in the class body,
# and _make/_replace capture tuple.__new__ at class creation so they still bypass this
GP.__new__ = staticmethod(_new)


def _floating(x) -> Array:
    """`x` as an array in its own floating dtype; integer input takes JAX's default float.

    Nothing here forces float64: under `jax_enable_x64` numpy input stays float64, which
    is what laGP parity needs, and without it the whole pipeline runs in float32.
    """
    x = jnp.asarray(x)
    return x if jnp.issubdtype(x.dtype, jnp.floating) else x.astype(float)


def _build(X, Z, d, g, mask, F, trend) -> GP:
    """Factorize K, then profile out the trend coefficients and the scale."""
    X = _floating(X)
    Z = jnp.asarray(Z, X.dtype)
    n, m = X.shape
    d = jnp.broadcast_to(jnp.atleast_1d(jnp.asarray(d, X.dtype)), (m,))
    g = jnp.asarray(g, X.dtype)
    mask = jnp.ones(n, bool) if mask is None else mask
    F = trend_basis(trend, X) if F is None else F

    # masked-out rows/cols become identity so they contribute nothing to ldetK or phi
    K = covar_symm(d, X, g)
    K = jnp.where(mask[:, None] & mask[None, :], K, jnp.eye(n, dtype=X.dtype))
    Z = jnp.where(mask, Z, 0.0)
    F = jnp.where(mask[:, None], F, 0.0)

    K_sqrt, is_lower = jsp.linalg.cho_factor(K)
    ldetK = 2.0 * jnp.sum(jnp.log(jnp.diag(K_sqrt)))
    Ki = jsp.linalg.cho_solve((K_sqrt, is_lower), jnp.eye(n, dtype=X.dtype))

    p = F.shape[-1]
    if p == 0:
        b = jnp.zeros((0,), X.dtype)
        Kfi = jnp.zeros((0, 0), X.dtype)
        ldetF = jnp.zeros((), X.dtype)
        resid = Z
    else:
        KiF = Ki @ F
        F_sqrt, F_lower = jsp.linalg.cho_factor(F.T @ KiF)
        ldetF = 2.0 * jnp.sum(jnp.log(jnp.diag(F_sqrt)))
        Kfi = jsp.linalg.cho_solve((F_sqrt, F_lower), jnp.eye(p, dtype=X.dtype))
        b = jsp.linalg.cho_solve((F_sqrt, F_lower), KiF.T @ Z)
        resid = Z - F @ b

    KiZ = Ki @ resid
    phi = jnp.dot(resid, KiZ)
    return tuple.__new__(
        GP, (X, Z, d, g, mask, F, K, Ki, ldetK, KiZ, phi, b, Kfi, ldetF)
    )


def _ab(prior) -> tuple[tuple[Scalar, Scalar], tuple[Scalar, Scalar]]:
    """(dab, gab) from a Prior; all zeros without one, which switches both priors off."""
    if prior is None:
        return (0.0, 0.0), (0.0, 0.0)
    dab = (0.0, 0.0) if prior.d.ab is None else tuple(prior.d.ab)
    gab = (0.0, 0.0) if prior.g.ab is None else tuple(prior.g.ab)
    return dab, gab


def _dgamma_log(x: Float[Array, "..."], a: Scalar, b: Scalar) -> Float[Array, "..."]:
    """log density of Gamma(shape=a, rate=b), i.e. R's dgamma(x, a, scale=1/b, log=TRUE)."""
    return (a - 1.0) * jnp.log(x) - b * x + a * jnp.log(b) - jsp.special.gammaln(a)


def _llik_from(df, ldetK, ldetF, phi, d, g, dab, gab) -> Scalar:
    """Log marginal likelihood plus optional Gamma log-priors, matching llikGPsep."""
    llik = -0.5 * (df * jnp.log(0.5 * phi) + ldetK + ldetF)

    # priors only apply when both hyperparameters of the Gamma are positive
    use_dab = (dab[0] > 0) & (dab[1] > 0)
    db = jnp.where(use_dab, dab[1], 1.0)
    prior_d = jnp.sum(jnp.where(d > 0, _dgamma_log(d, dab[0], db), 0.0))
    llik = llik + jnp.where(use_dab, prior_d, 0.0)

    use_gab = (g > 0) & (gab[0] > 0) & (gab[1] > 0)
    gb = jnp.where(use_gab, gab[1], 1.0)
    gs = jnp.where(use_gab, g, 1.0)
    return llik + jnp.where(use_gab, _dgamma_log(gs, gab[0], gb), 0.0)


def _llik(gp: GP, dab, gab) -> Scalar:
    return _llik_from(gp.df, gp.ldetK, gp.ldetF, gp.phi, gp.d, gp.g, dab, gab)


def _profile(gp: GP, d, g) -> tuple[Scalar, Scalar, Scalar]:
    """(ldetK, ldetF, phi) at new hyperparameters, without materializing K's inverse.

    `_build` inverts K explicitly because ALC needs the inverse; a likelihood never does.
    Every MLE inner iteration is this function and its derivatives, so skipping the inverse
    takes the objective from two O(n^3) solves down to one Cholesky.
    """
    X, n = gp.X, gp.X.shape[0]
    d = jnp.broadcast_to(jnp.atleast_1d(jnp.asarray(d, X.dtype)), (gp.m,))
    K = covar_symm(d, X, g)
    K = jnp.where(gp.mask[:, None] & gp.mask[None, :], K, jnp.eye(n, dtype=X.dtype))

    chol = jsp.linalg.cho_factor(K)
    ldetK = 2.0 * jnp.sum(jnp.log(jnp.diag(chol[0])))

    # Z and F were masked when the GP was built, so they need no masking again here
    if gp.p == 0:
        resid, ldetF = gp.Z, jnp.zeros((), X.dtype)
    else:
        KiF = jsp.linalg.cho_solve(chol, gp.F)
        fchol = jsp.linalg.cho_factor(gp.F.T @ KiF)
        ldetF = 2.0 * jnp.sum(jnp.log(jnp.diag(fchol[0])))
        resid = gp.Z - gp.F @ jsp.linalg.cho_solve(fchol, KiF.T @ gp.Z)
    return ldetK, ldetF, jnp.dot(resid, jsp.linalg.cho_solve(chol, resid))


def llik_d(gp: GP, d: Float[Array, "m"], dab=(0.0, 0.0)) -> Scalar:
    """llik as a function of the lengthscale only, for gradients and optimization."""
    return _llik_from(gp.df, *_profile(gp, d, gp.g), d, gp.g, dab, (0.0, 0.0))


def llik_g(gp: GP, g: Scalar, gab=(0.0, 0.0)) -> Scalar:
    """llik as a function of the nugget only, for gradients and optimization."""
    return _llik_from(gp.df, *_profile(gp, gp.d, g), gp.d, g, (0.0, 0.0), gab)


def llik_dg(gp: GP, d: Float[Array, "m"], g: Scalar, dab=(0.0, 0.0), gab=(0.0, 0.0)):
    """llik as a function of both hyperparameters; the objective jmle alternates over."""
    return _llik_from(gp.df, *_profile(gp, d, g), d, g, dab, gab)


dllik_d = jax.grad(llik_d, argnums=1)
dllik_g = jax.grad(llik_g, argnums=1)
d2llik_g = jax.grad(dllik_g, argnums=1)


def _predict(gp: GP, XX, lite: bool, nugget: bool) -> Prediction:
    g = gp.g if nugget else SDEPS
    df = gp.df
    phidf = gp.phi / df
    llik = -0.5 * (df * jnp.log(0.5 * gp.phi) + gp.ldetK + gp.ldetF)

    k = jnp.where(gp.mask[:, None], covar(gp.d, gp.X, XX), 0.0)
    ktKi = k.T @ gp.Ki
    mean = ktKi @ gp.resid

    # trend contribution: the fitted mean, plus the uncertainty in having fitted it
    if gp.p:
        f = trend_basis(_trend_of(gp.p), XX)
        u = f - ktKi @ gp.F
        mean = mean + f @ gp.b

    if lite:
        s2 = 1.0 + g - jnp.sum(ktKi.T * k, axis=0)
        if gp.p:
            s2 = s2 + jnp.sum((u @ gp.Kfi) * u, axis=-1)
    else:
        s2 = covar_symm(gp.d, XX, g) - ktKi @ k
        if gp.p:
            s2 = s2 + u @ gp.Kfi @ u.T
    return Prediction(mean=mean, s2=phidf * s2, df=df, llik=llik)
