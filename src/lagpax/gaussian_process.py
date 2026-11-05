"""An exact Gaussian process on one design, with the scale profiled out of the likelihood.

The model: correlation `kernel.correlation` plus a nugget on the diagonal, an unknown scale
profiled out, an optional regression trend whose coefficients are profiled out by generalized
least squares, and a Student-t predictive distribution.

`active` marks the rows that belong to the model. Inactive rows get an identity block in the
correlation matrix, a zero output and a zero trend row, which makes a fixed-size padded design
behave exactly like the smaller one -- the property that lets a growing design live in XLA's
fixed shapes.
"""

from __future__ import annotations

from typing import Any, Literal, NamedTuple, Optional

import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Array, Bool, Float, Int, Scalar

from ._pytree import pytree_dataclass, static
from .kernel import correlation, correlation_with_nugget
from .priors import ROOT_MACHINE_EPSILON, Prior

Trend = Literal["zero", "constant", "linear"]


def trend_basis(trend: Trend, inputs: Float[Array, "n d"]) -> Float[Array, "n p"]:
    count = inputs.shape[0]
    if trend == "zero":
        return jnp.zeros((count, 0), inputs.dtype)
    if trend == "constant":
        return jnp.ones((count, 1), inputs.dtype)
    if trend == "linear":
        return jnp.concatenate([jnp.ones((count, 1), inputs.dtype), inputs], axis=-1)
    raise ValueError(f"unknown trend {trend!r}")


def as_floating(values: Any) -> Array:
    """`values` as an array in its own floating dtype, or JAX's default float for integers."""
    values = jnp.asarray(values)
    return values if jnp.issubdtype(values.dtype, jnp.floating) else values.astype(float)


@pytree_dataclass
class Hyperparameters:
    squared_lengthscales: Float[Array, "d"]
    nugget: Scalar


class Prediction(NamedTuple):
    """A Student-t predictive distribution.

    `variance` is the predictive variance. `covariance` is filled only when a full covariance
    was asked for. A local model fills `neighbors` (the training rows each local GP used) and
    `fit` (its fitted hyperparameters).
    """

    mean: Float[Array, "m"]
    variance: Float[Array, "m"]
    degrees_of_freedom: Scalar
    log_likelihood: Scalar
    covariance: Optional[Float[Array, "m m"]] = None
    neighbors: Optional[Int[Array, "..."]] = None
    fit: Optional[Any] = None


@pytree_dataclass
class GaussianProcess:
    inputs: Float[Array, "n d"]
    outputs: Float[Array, "n"]
    active: Bool[Array, "n"]
    hyperparameters: Hyperparameters
    trend_basis: Float[Array, "n p"]
    inverse_correlation: Float[Array, "n n"]
    log_det_correlation: Scalar
    trend_coefficients: Float[Array, "p"]
    inverse_trend_information: Float[Array, "p p"]
    log_det_trend_information: Scalar
    residual_sum_of_squares: Scalar
    trend: Trend = static(default="zero")

    @classmethod
    def from_data(
        cls,
        inputs: Float[Array, "n d"],
        outputs: Float[Array, "n"],
        hyperparameters: Hyperparameters,
        active: Optional[Bool[Array, "n"]] = None,
        trend: Trend = "zero",
    ) -> GaussianProcess:
        inputs = as_floating(inputs)
        count = inputs.shape[0]
        active = jnp.ones(count, bool) if active is None else active
        outputs = jnp.where(active, jnp.asarray(outputs, inputs.dtype), 0.0)
        basis = jnp.where(active[:, None], trend_basis(trend, inputs), 0.0)
        factors = _factorize(inputs, outputs, active, basis, hyperparameters)
        trend_identity = jnp.eye(basis.shape[-1], dtype=inputs.dtype)
        return cls(
            inputs=inputs,
            outputs=outputs,
            active=active,
            hyperparameters=factors.hyperparameters,
            trend_basis=basis,
            inverse_correlation=jsp.linalg.cho_solve(
                factors.cholesky, jnp.eye(count, dtype=inputs.dtype)
            ),
            log_det_correlation=factors.log_det_correlation,
            trend_coefficients=factors.trend_coefficients,
            inverse_trend_information=(
                trend_identity
                if factors.information_cholesky is None
                else jsp.linalg.cho_solve(factors.information_cholesky, trend_identity)
            ),
            log_det_trend_information=factors.log_det_trend_information,
            residual_sum_of_squares=factors.residual_sum_of_squares,
            trend=trend,
        )

    @property
    def active_count(self) -> Scalar:
        return jnp.sum(self.active.astype(self.outputs.dtype))

    @property
    def dimension(self) -> int:
        return self.inputs.shape[-1]

    @property
    def trend_size(self) -> int:
        return self.trend_basis.shape[-1]

    @property
    def degrees_of_freedom(self) -> Scalar:
        return self.active_count - self.trend_size

    @property
    def residuals(self) -> Float[Array, "n"]:
        if self.trend_size == 0:
            return self.outputs
        return self.outputs - self.trend_basis @ self.trend_coefficients

    def with_hyperparameters(self, hyperparameters: Hyperparameters) -> GaussianProcess:
        return GaussianProcess.from_data(
            self.inputs, self.outputs, hyperparameters, self.active, self.trend
        )

    def with_observations(
        self, new_inputs: Float[Array, "k d"], new_outputs: Float[Array, "k"]
    ) -> GaussianProcess:
        return GaussianProcess.from_data(
            jnp.concatenate([self.inputs, new_inputs]),
            jnp.concatenate([self.outputs, new_outputs]),
            self.hyperparameters,
            jnp.concatenate([self.active, jnp.ones(len(new_outputs), bool)]),
            self.trend,
        )

    def log_likelihood(self, prior: Optional[Prior] = None) -> Scalar:
        """Profiled log-likelihood, plus the log prior density when a prior is given."""
        return _with_log_prior(
            _profiled_log_likelihood(
                self.degrees_of_freedom,
                self.log_det_correlation,
                self.log_det_trend_information,
                self.residual_sum_of_squares,
            ),
            prior,
            self.hyperparameters,
        )

    def predict(
        self,
        new_inputs: Float[Array, "m d"],
        full_covariance: bool = False,
        include_nugget: bool = True,
    ) -> Prediction:
        """Student-t predictive at `new_inputs`; `include_nugget=False` predicts the latent surface."""
        dof = self.degrees_of_freedom
        mean, scale = predictive_mean_and_scale(self, new_inputs, include_nugget, full_covariance)
        variance = scale * dof / (dof - 2.0)
        if not full_covariance:
            return Prediction(mean, variance, dof, self.log_likelihood())
        return Prediction(mean, jnp.diagonal(variance), dof, self.log_likelihood(), variance)


def predictive_mean_and_scale(
    gp: GaussianProcess,
    new_inputs: Float[Array, "m d"],
    include_nugget: bool = True,
    full_covariance: bool = False,
) -> tuple[Float[Array, "m"], Float[Array, "m"] | Float[Array, "m m"]]:
    """Predictive mean and squared Student-t scale (variance * (dof - 2) / dof) at new_inputs,
    per point or as a full matrix."""
    nugget = gp.hyperparameters.nugget if include_nugget else ROOT_MACHINE_EPSILON
    squared_lengthscales = gp.hyperparameters.squared_lengthscales
    cross = jnp.where(
        gp.active[:, None],
        correlation(gp.inputs, new_inputs, squared_lengthscales),
        0.0,
    )
    weights = cross.T @ gp.inverse_correlation
    mean = weights @ gp.residuals
    if full_covariance:
        prior_correlation = correlation_with_nugget(new_inputs, squared_lengthscales, nugget)
        unscaled = prior_correlation - weights @ cross
    else:
        unscaled = 1.0 + nugget - jnp.sum(weights.T * cross, axis=0)
    if gp.trend_size:
        new_basis = trend_basis(gp.trend, new_inputs)
        trend_uncertainty = new_basis - weights @ gp.trend_basis
        mean = mean + new_basis @ gp.trend_coefficients
        if full_covariance:
            unscaled = (
                unscaled + trend_uncertainty @ gp.inverse_trend_information @ trend_uncertainty.T
            )
        else:
            unscaled = unscaled + jnp.sum(
                (trend_uncertainty @ gp.inverse_trend_information) * trend_uncertainty,
                axis=-1,
            )
    return mean, gp.residual_sum_of_squares / gp.degrees_of_freedom * unscaled


def log_likelihood_at(
    gp: GaussianProcess, hyperparameters: Hyperparameters, prior: Optional[Prior] = None
) -> Scalar:
    """`gp.with_hyperparameters(h).log_likelihood(prior)` with one Cholesky and no inverse.

    The objective every hyperparameter solve iterates on, so it skips the explicit inverse that
    `from_data` keeps for prediction and design criteria.
    """
    factors = _factorize(gp.inputs, gp.outputs, gp.active, gp.trend_basis, hyperparameters)
    log_likelihood = _profiled_log_likelihood(
        gp.degrees_of_freedom,
        factors.log_det_correlation,
        factors.log_det_trend_information,
        factors.residual_sum_of_squares,
    )
    return _with_log_prior(log_likelihood, prior, factors.hyperparameters)


class _Factors(NamedTuple):
    hyperparameters: Hyperparameters
    cholesky: tuple[Float[Array, "n n"], bool]
    log_det_correlation: Scalar
    information_cholesky: Optional[tuple[Float[Array, "p p"], bool]]
    log_det_trend_information: Scalar
    trend_coefficients: Float[Array, "p"]
    residual_sum_of_squares: Scalar


def _factorize(
    inputs: Float[Array, "n d"],
    outputs: Float[Array, "n"],
    active: Bool[Array, "n"],
    basis: Float[Array, "n p"],
    hyperparameters: Hyperparameters,
) -> _Factors:
    """Cholesky of the masked correlation, the trend's GLS fit, and the residual sum of squares."""
    count, dimension = inputs.shape
    squared_lengthscales = jnp.broadcast_to(
        jnp.atleast_1d(jnp.asarray(hyperparameters.squared_lengthscales, inputs.dtype)),
        (dimension,),
    )
    hyperparameters = Hyperparameters(
        squared_lengthscales, jnp.asarray(hyperparameters.nugget, inputs.dtype)
    )
    correlation_matrix = jnp.where(
        active[:, None] & active[None, :],
        correlation_with_nugget(inputs, squared_lengthscales, hyperparameters.nugget),
        jnp.eye(count, dtype=inputs.dtype),
    )
    cholesky = jsp.linalg.cho_factor(correlation_matrix)
    log_det_correlation = 2.0 * jnp.sum(jnp.log(jnp.diag(cholesky[0])))

    if basis.shape[-1] == 0:
        information_cholesky = None
        log_det_trend_information = jnp.zeros((), inputs.dtype)
        trend_coefficients = jnp.zeros((0,), inputs.dtype)
        residuals = outputs
    else:
        whitened_basis = jsp.linalg.cho_solve(cholesky, basis)
        information_cholesky = jsp.linalg.cho_factor(basis.T @ whitened_basis)
        log_det_trend_information = 2.0 * jnp.sum(jnp.log(jnp.diag(information_cholesky[0])))
        trend_coefficients = jsp.linalg.cho_solve(information_cholesky, whitened_basis.T @ outputs)
        residuals = outputs - basis @ trend_coefficients

    return _Factors(
        hyperparameters,
        cholesky,
        log_det_correlation,
        information_cholesky,
        log_det_trend_information,
        trend_coefficients,
        jnp.dot(residuals, jsp.linalg.cho_solve(cholesky, residuals)),
    )


def _profiled_log_likelihood(
    degrees_of_freedom: Scalar,
    log_det_correlation: Scalar,
    log_det_trend_information: Scalar,
    residual_sum_of_squares: Scalar,
) -> Scalar:
    return -0.5 * (
        degrees_of_freedom * jnp.log(0.5 * residual_sum_of_squares)
        + log_det_correlation
        + log_det_trend_information
    )


def _with_log_prior(
    log_likelihood: Scalar, prior: Optional[Prior], hyperparameters: Hyperparameters
) -> Scalar:
    if prior is None:
        return log_likelihood
    if prior.kernel.prior is not None:
        kernel_values = prior.from_squared_lengthscales(hyperparameters.squared_lengthscales)
        densities = prior.kernel.prior.log_density(kernel_values)
        log_likelihood = log_likelihood + jnp.sum(jnp.where(kernel_values > 0, densities, 0.0))
    return log_likelihood + prior.nugget_log_density(hyperparameters.nugget)
