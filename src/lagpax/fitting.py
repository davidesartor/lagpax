"""Hyperparameter estimation: maximum likelihood, or MAP under a `Prior`.

The kernel parameter is fitted by L-BFGS-B in the prior's own kernel space, the nugget by a
safeguarded Newton iteration, and when both are learned they alternate until neither moves (laGP's
jmleGPsep). Every solver runs without a gradient tape; `with_implicit_gradient` attaches the
derivative of the optimum afterwards, so `jax.grad` through a fit is exact at the optimum.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, Scalar
from vlse.optim import minimise

from .gaussian_process import GaussianProcess, Hyperparameters, log_likelihood_at
from .optimize import (
    R_OPTIM_RELATIVE_TOLERANCE,
    newton_maximize_on_interval,
    with_implicit_gradient,
)
from .priors import Prior

frozen = jax.lax.stop_gradient


class FitResult(NamedTuple):
    """Fitted hyperparameters, and the kernel solver's iteration count and convergence."""

    squared_lengthscales: Float[Array, "d"]
    nugget: Scalar
    iterations: Int[Array, ""]
    converged: Bool[Array, ""]

    @property
    def lengthscales(self) -> Float[Array, "d"]:
        return jnp.sqrt(0.5 * self.squared_lengthscales)


def fit_hyperparameters(
    gp: GaussianProcess, prior: Prior, max_iterations: int = 100
) -> tuple[GaussianProcess, FitResult]:
    """Fit whatever the prior marks `learn`, returning the refit process and the result."""
    learn_kernel, learn_nugget = bool(prior.kernel.learn), bool(prior.nugget.learn)
    if learn_kernel and learn_nugget:
        return fit_jointly(gp, prior, max_iterations)
    if learn_kernel:
        return fit_squared_lengthscales(gp, prior, max_iterations)
    if learn_nugget:
        return fit_nugget(gp, prior, max_iterations)
    hyperparameters = gp.hyperparameters
    unchanged = FitResult(
        hyperparameters.squared_lengthscales,
        hyperparameters.nugget,
        jnp.asarray(0),
        jnp.asarray(True),
    )
    return gp, unchanged


def fit_squared_lengthscales(
    gp: GaussianProcess, prior: Prior, max_iterations: int = 100
) -> tuple[GaussianProcess, FitResult]:
    """MAP kernel parameter at the process's nugget."""
    lower, upper = _kernel_bounds(prior, gp)
    nugget = gp.hyperparameters.nugget
    start = _kernel_start(prior, gp, lower, upper)
    solution, iterations, converged = _solve_kernel(
        frozen(gp),
        frozen(start),
        frozen(nugget),
        frozen(lower),
        frozen(upper),
        frozen(prior),
        max_iterations,
    )
    solution = with_implicit_gradient(
        lambda values: -_kernel_log_posterior(gp, values, nugget, prior),
        solution,
        lower,
        upper,
    )
    squared = prior.to_squared_lengthscales(solution)
    refit = gp.with_hyperparameters(Hyperparameters(squared, nugget))
    return refit, FitResult(squared, nugget, iterations, converged)


def fit_nugget(
    gp: GaussianProcess, prior: Prior, max_iterations: int = 100
) -> tuple[GaussianProcess, FitResult]:
    """MAP nugget at the process's squared lengthscales."""
    dtype = gp.inputs.dtype
    squared = gp.hyperparameters.squared_lengthscales
    nugget, iterations = _solve_nugget(
        frozen(gp),
        frozen(squared),
        frozen(gp.hyperparameters.nugget),
        frozen(prior.nugget.lower),
        frozen(prior.nugget.upper),
        frozen(prior),
        max_iterations,
    )
    nugget = with_implicit_gradient(
        lambda values: -_nugget_log_posterior(gp, squared, values[0], prior),
        jnp.atleast_1d(nugget),
        _filled(prior.nugget.lower, 1, dtype),
        _filled(prior.nugget.upper, 1, dtype),
    )[0]
    refit = gp.with_hyperparameters(Hyperparameters(squared, nugget))
    return refit, FitResult(squared, nugget, iterations, jnp.asarray(True))


def fit_jointly(
    gp: GaussianProcess, prior: Prior, max_iterations: int = 100, max_rounds: int = 100
) -> tuple[GaussianProcess, FitResult]:
    """Alternate kernel and nugget solves until both settle."""
    dimension, dtype = gp.dimension, gp.inputs.dtype
    kernel_lower, kernel_upper = _kernel_bounds(prior, gp)
    nugget_lower = _filled(prior.nugget.lower, 1, dtype)
    nugget_upper = _filled(prior.nugget.upper, 1, dtype)
    fixed_gp, fixed_prior = frozen(gp), frozen(prior)

    def unsettled(state):
        _, _, kernel_iterations, nugget_iterations, converged, round_index = state
        settled = (
            (nugget_iterations <= 2) & (kernel_iterations <= dimension + 1) & converged
        )
        return (round_index < max_rounds) & ~settled

    def alternate(state):
        kernel_values, nugget, _, _, _, round_index = state
        kernel_values, kernel_iterations, converged = _solve_kernel(
            fixed_gp,
            kernel_values,
            nugget,
            kernel_lower,
            kernel_upper,
            fixed_prior,
            max_iterations,
        )
        squared = fixed_prior.to_squared_lengthscales(kernel_values)
        nugget, nugget_iterations = _solve_nugget(
            fixed_gp,
            squared,
            nugget,
            nugget_lower[0],
            nugget_upper[0],
            fixed_prior,
            max_iterations,
        )
        return (
            kernel_values,
            nugget,
            kernel_iterations,
            nugget_iterations,
            converged,
            round_index + 1,
        )

    start = _kernel_start(prior, gp, kernel_lower, kernel_upper)
    initial_state = frozen(
        (
            start,
            gp.hyperparameters.nugget,
            jnp.asarray(dimension + 2),
            jnp.asarray(3),
            jnp.asarray(True),
            0,
        )
    )
    kernel_values, nugget, kernel_iterations, _, converged, _ = jax.lax.while_loop(
        unsettled, alternate, initial_state
    )

    joint = with_implicit_gradient(
        lambda values: (
            -_kernel_log_posterior(
                gp,
                values[:dimension],
                values[dimension],
                prior,
                include_nugget_prior=True,
            )
        ),
        jnp.concatenate([kernel_values, jnp.atleast_1d(nugget)]),
        jnp.concatenate([kernel_lower, nugget_lower]),
        jnp.concatenate([kernel_upper, nugget_upper]),
    )
    squared = prior.to_squared_lengthscales(joint[:dimension])
    nugget = joint[dimension]
    refit = gp.with_hyperparameters(Hyperparameters(squared, nugget))
    return refit, FitResult(squared, nugget, kernel_iterations, converged)


def _kernel_log_posterior(
    gp: GaussianProcess,
    kernel_values: Float[Array, "d"],
    nugget: Scalar,
    prior: Prior,
    include_nugget_prior: bool = False,
) -> Scalar:
    squared = prior.to_squared_lengthscales(kernel_values)
    total = log_likelihood_at(gp, Hyperparameters(squared, nugget))
    if prior.kernel.prior is not None:
        densities = prior.kernel.prior.log_density(kernel_values)
        total = total + jnp.sum(jnp.where(kernel_values > 0, densities, 0.0))
    if include_nugget_prior:
        total = total + prior.nugget_log_density(nugget)
    return total


def _nugget_log_posterior(
    gp: GaussianProcess,
    squared_lengthscales: Float[Array, "d"],
    nugget: Scalar,
    prior: Prior,
) -> Scalar:
    likelihood = log_likelihood_at(gp, Hyperparameters(squared_lengthscales, nugget))
    return likelihood + prior.nugget_log_density(nugget)


def _solve_kernel(
    gp: GaussianProcess,
    start: Float[Array, "d"],
    nugget: Scalar,
    lower: Float[Array, "d"],
    upper: Float[Array, "d"],
    prior: Prior,
    max_iterations: int,
) -> tuple[Float[Array, "d"], Int[Array, ""], Bool[Array, ""]]:
    state = minimise(
        lambda values: -_kernel_log_posterior(gp, values, nugget, prior),
        start,
        bounds=(lower, upper),
        tol=0.0,
        ftol=R_OPTIM_RELATIVE_TOLERANCE,
        max_iterations=max_iterations,
    )
    stopped_by_a_rule = (
        (state.error <= 0.0) | state.converged_ftol | state.failed_linesearch
    )
    converged = stopped_by_a_rule | (state.iteration < max_iterations)
    return state.x, state.iteration, converged


def _solve_nugget(
    gp: GaussianProcess,
    squared_lengthscales: Float[Array, "d"],
    start: Scalar,
    lower: Scalar,
    upper: Scalar,
    prior: Prior,
    max_iterations: int,
) -> tuple[Scalar, Int[Array, ""]]:
    slope = jax.grad(
        lambda nugget: _nugget_log_posterior(gp, squared_lengthscales, nugget, prior)
    )
    return newton_maximize_on_interval(
        lambda nugget: jax.jvp(slope, (nugget,), (jnp.ones_like(nugget),)),
        start,
        lower,
        upper,
        max_iterations=max_iterations,
    )


def _filled(value, size: int, dtype) -> Float[Array, "size"]:
    return jnp.broadcast_to(jnp.atleast_1d(jnp.asarray(value, dtype)), (size,))


def _kernel_bounds(
    prior: Prior, gp: GaussianProcess
) -> tuple[Float[Array, "d"], Float[Array, "d"]]:
    dtype = gp.inputs.dtype
    return (
        _filled(prior.kernel.lower, gp.dimension, dtype),
        _filled(prior.kernel.upper, gp.dimension, dtype),
    )


def _kernel_start(
    prior: Prior,
    gp: GaussianProcess,
    lower: Float[Array, "d"],
    upper: Float[Array, "d"],
) -> Float[Array, "d"]:
    current = prior.from_squared_lengthscales(gp.hyperparameters.squared_lengthscales)
    return jnp.clip(_filled(current, gp.dimension, gp.inputs.dtype), lower, upper)
