"""An exact Gaussian process on one (possibly padded) design.

The model is laGP's separable GP: correlation `exp(-sum (x - x')^2 / theta)` plus `nugget` on
the diagonal, an unknown scale profiled out of the likelihood, and a Student-t predictive with
`dof` degrees of freedom. A `trend` adds a regression mean whose coefficients are profiled out
by generalized least squares.
"""

from typing import Any, Literal, NamedTuple, Optional
from jaxtyping import Array, Bool, Float, Int, Scalar

import jax
import jax.numpy as jnp
import jax.scipy as jsp

from .kernel import correlation, correlation_matrix

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
    """A Student-t predictive distribution at a set of locations.

    `variance` is the predictive variance (not the Student-t scale); `covariance` is filled
    only for `full_cov=True`. `neighbors` and `fit` are filled by the local model: the rows of
    the training set each local GP used, and its fitted hyperparameters.
    """

    mean: Float[Array, "nn"]
    variance: Float[Array, "nn"]
    dof: Scalar
    log_likelihood: Scalar
    covariance: Optional[Float[Array, "nn nn"]] = None
    neighbors: Optional[Int[Array, "..."]] = None
    fit: Optional[Any] = None


class GP(NamedTuple):
    """GP state: data, hyperparameters, and the cached factorizations.

    Construct as `GP(X, y, theta, nugget, mask=None, trend="zero")`; the remaining fields are
    derived and only passed explicitly when JAX rebuilds the pytree.

    `mask` marks the rows that are part of the model. Masked-out rows get an identity block
    in K, a zeroed y entry and a zeroed trend row, so a padded (fixed-shape, vmappable)
    design behaves exactly like the smaller one.
    """

    X: Float[Array, "n m"]
    y: Float[Array, "n"]
    theta: Float[Array, "m"]
    nugget: Scalar
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
        """Number of active (unmasked) observations."""
        return jnp.sum(self.mask.astype(self.y.dtype))

    @property
    def m(self) -> int:
        return self.X.shape[-1]

    @property
    def p(self) -> int:
        return self.F.shape[-1]

    @property
    def dof(self) -> Scalar:
        """Student-t degrees of freedom: n, less one per profiled trend coefficient."""
        return self.n - self.p

    @property
    def resid(self) -> Float[Array, "n"]:
        """y with the fitted trend removed; y itself without a trend."""
        return self.y if self.p == 0 else self.y - self.F @ self.b

    def replace(self, theta=None, nugget=None) -> "GP":
        """The same data under new hyperparameters."""
        return GP(
            self.X,
            self.y,
            self.theta if theta is None else theta,
            self.nugget if nugget is None else nugget,
            self.mask,
            self.F,
        )

    def log_likelihood(self, prior=None) -> Scalar:
        """Log marginal likelihood with the scale profiled out; plus log-priors given a Prior.

        The Gamma prior applies in the prior's own space: on theta, as in laGP, or on the
        lengthscale for a `Prior(lengthscale=...)`.
        """
        if prior is None:
            return _llik(self, (0.0, 0.0), (0.0, 0.0))
        ll = _llik(self, prior.theta_gamma, prior.nugget_gamma)
        return ll if prior.space == "theta" else ll + prior.kernel_log_prior(self.theta)

    def predict(
        self,
        Xnew: Float[Array, "nn m"],
        full_cov: bool = False,
        nugget: bool = True,
    ) -> Prediction:
        """Student-t predictive at Xnew. `nugget=False` predicts the latent, noise-free surface."""
        return _predict(self, Xnew, full_cov=full_cov, nugget=nugget)

    def append(self, Xnew: Float[Array, "k m"], ynew: Float[Array, "k"]) -> "GP":
        """Add observations and refactorize."""
        return GP(
            jnp.concatenate([self.X, Xnew]),
            jnp.concatenate([self.y, ynew]),
            self.theta,
            self.nugget,
            jnp.concatenate([self.mask, jnp.ones(len(ynew), bool)]),
            trend=_trend_of(self.p),
        )


def _new(
    cls,
    X: Float[Array, "n m"],
    y: Float[Array, "n"],
    theta: Float[Array, "m"] | Scalar,
    nugget: Scalar,
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
            cls, (X, y, theta, nugget, mask, F, K, Ki, ldetK, KiZ, phi, b, Kfi, ldetF)
        )
    return _build(X, y, theta, nugget, mask, F, trend)


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
    # internal names: Z = y, d = theta, g = nugget
    """Factorize K, then profile out the trend coefficients and the scale."""
    X = _floating(X)
    Z = jnp.asarray(Z, X.dtype)
    n, m = X.shape
    d = jnp.broadcast_to(jnp.atleast_1d(jnp.asarray(d, X.dtype)), (m,))
    g = jnp.asarray(g, X.dtype)
    mask = jnp.ones(n, bool) if mask is None else mask
    F = trend_basis(trend, X) if F is None else F

    # masked-out rows/cols become identity so they contribute nothing to ldetK or phi
    K = correlation_matrix(X, d, g)
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
    return _llik_from(gp.dof, gp.ldetK, gp.ldetF, gp.phi, gp.theta, gp.nugget, dab, gab)


def _profile(gp: GP, d, g) -> tuple[Scalar, Scalar, Scalar]:
    """(ldetK, ldetF, phi) at new hyperparameters, without materializing K's inverse.

    `_build` inverts K explicitly because ALC needs the inverse; a likelihood never does.
    Every MLE inner iteration is this function and its derivatives, so skipping the inverse
    takes the objective from two O(n^3) solves down to one Cholesky.
    """
    X, n = gp.X, gp.X.shape[0]
    d = jnp.broadcast_to(jnp.atleast_1d(jnp.asarray(d, X.dtype)), (gp.m,))
    K = correlation_matrix(X, d, g)
    K = jnp.where(gp.mask[:, None] & gp.mask[None, :], K, jnp.eye(n, dtype=X.dtype))

    chol = jsp.linalg.cho_factor(K)
    ldetK = 2.0 * jnp.sum(jnp.log(jnp.diag(chol[0])))

    # Z and F were masked when the GP was built, so they need no masking again here
    if gp.p == 0:
        resid, ldetF = gp.y, jnp.zeros((), X.dtype)
    else:
        KiF = jsp.linalg.cho_solve(chol, gp.F)
        fchol = jsp.linalg.cho_factor(gp.F.T @ KiF)
        ldetF = 2.0 * jnp.sum(jnp.log(jnp.diag(fchol[0])))
        resid = gp.y - gp.F @ jsp.linalg.cho_solve(fchol, KiF.T @ gp.y)
    return ldetK, ldetF, jnp.dot(resid, jsp.linalg.cho_solve(chol, resid))


def llik_theta(gp: GP, theta: Float[Array, "m"], ab=(0.0, 0.0)) -> Scalar:
    """Log-likelihood as a function of theta alone, for gradients and optimization."""
    return _llik_from(
        gp.dof, *_profile(gp, theta, gp.nugget), theta, gp.nugget, ab, (0.0, 0.0)
    )


def llik_nugget(gp: GP, nugget: Scalar, ab=(0.0, 0.0)) -> Scalar:
    """Log-likelihood as a function of the nugget alone."""
    return _llik_from(
        gp.dof, *_profile(gp, gp.theta, nugget), gp.theta, nugget, (0.0, 0.0), ab
    )


def llik_both(gp: GP, theta, nugget, theta_ab=(0.0, 0.0), nugget_ab=(0.0, 0.0)):
    """Log-likelihood as a function of both hyperparameters."""
    return _llik_from(
        gp.dof, *_profile(gp, theta, nugget), theta, nugget, theta_ab, nugget_ab
    )


dllik_theta = jax.grad(llik_theta, argnums=1)
dllik_nugget = jax.grad(llik_nugget, argnums=1)


def _moments(gp: GP, XX, nugget: bool = True):
    """Predictive mean and Student-t *scale* (not variance) at XX; the criteria need it."""
    g = gp.nugget if nugget else SDEPS
    k = jnp.where(gp.mask[:, None], correlation(gp.X, XX, gp.theta), 0.0)
    ktKi = k.T @ gp.Ki
    mean = ktKi @ gp.resid
    s2 = 1.0 + g - jnp.sum(ktKi.T * k, axis=0)
    if gp.p:
        f = trend_basis(_trend_of(gp.p), XX)
        u = f - ktKi @ gp.F
        mean = mean + f @ gp.b
        s2 = s2 + jnp.sum((u @ gp.Kfi) * u, axis=-1)
    return mean, gp.phi / gp.dof * s2


def _predict(gp: GP, XX, full_cov: bool, nugget: bool) -> Prediction:
    df = gp.dof
    llik = -0.5 * (df * jnp.log(0.5 * gp.phi) + gp.ldetK + gp.ldetF)

    # the Student-t scale, then its variance: scale * dof / (dof - 2)
    if not full_cov:
        mean, scale = _moments(gp, XX, nugget)
        variance = scale * df / (df - 2.0)
        return Prediction(mean=mean, variance=variance, dof=df, log_likelihood=llik)

    g = gp.nugget if nugget else SDEPS
    k = jnp.where(gp.mask[:, None], correlation(gp.X, XX, gp.theta), 0.0)
    ktKi = k.T @ gp.Ki
    mean = ktKi @ gp.resid
    s2 = correlation_matrix(XX, gp.theta, g) - ktKi @ k
    if gp.p:
        f = trend_basis(_trend_of(gp.p), XX)
        u = f - ktKi @ gp.F
        mean = mean + f @ gp.b
        s2 = s2 + u @ gp.Kfi @ u.T
    cov = gp.phi / df * s2 * df / (df - 2.0)
    return Prediction(
        mean=mean,
        variance=jnp.diagonal(cov),
        dof=df,
        log_likelihood=llik,
        covariance=cov,
    )
