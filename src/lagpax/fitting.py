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
    gp: GaussianProcess, prior: Prior, max_iterations: int = 100, max_rounds: int = 100
) -> tuple[GaussianProcess, FitResult]:
    """Fit whatever the prior marks `learn`, returning the refit process and the result.

    `iterations` and `converged` describe the kernel solve, or the nugget's when only the nugget
    is learned. Hyperparameters that are not learned keep their values and their gradients.
    """
    learn_kernel, learn_nugget = bool(prior.kernel.learn), bool(prior.nugget.learn)
    current = gp.hyperparameters
    if not (learn_kernel or learn_nugget):
        unchanged = FitResult(
            current.squared_lengthscales, current.nugget, jnp.asarray(0), jnp.asarray(True)
        )
        return gp, unchanged

    dimension, dtype = gp.dimension, gp.inputs.dtype
    live_kernel = _filled(
        prior.from_squared_lengthscales(current.squared_lengthscales), dimension, dtype
    )
    live_nugget = _filled(current.nugget, 1, dtype)
    live = jnp.concatenate([live_kernel, live_nugget])
    lower, upper = (
        jnp.concatenate(
            [
                _filled(getattr(prior.kernel, side), dimension, dtype)
                if learn_kernel
                else live_kernel,
                _filled(getattr(prior.nugget, side), 1, dtype) if learn_nugget else live_nugget,
            ]
        )
        for side in ("lower", "upper")
    )
    fixed_gp, fixed_prior = frozen(gp), frozen(prior)

    def solve_kernel(kernel_values, nugget):
        state = minimise(
            lambda values: -_kernel_log_posterior(fixed_gp, values, nugget, fixed_prior),
            kernel_values,
            bounds=(lower[:dimension], upper[:dimension]),
            tol=0.0,
            ftol=R_OPTIM_RELATIVE_TOLERANCE,
            max_iterations=max_iterations,
        )
        stopped_by_a_rule = (state.error <= 0.0) | state.converged_ftol | state.failed_linesearch
        return state.x, state.iteration, stopped_by_a_rule | (state.iteration < max_iterations)

    def solve_nugget(squared_lengthscales, nugget):
        slope = jax.grad(
            lambda g: _nugget_log_posterior(fixed_gp, squared_lengthscales, g, fixed_prior)
        )
        return newton_maximize_on_interval(
            lambda g: jax.jvp(slope, (g,), (jnp.ones_like(g),)),
            nugget,
            lower[dimension],
            upper[dimension],
            max_iterations=max_iterations,
        )

    kernel_values = frozen(jnp.clip(live[:dimension], lower[:dimension], upper[:dimension]))
    nugget = frozen(current.nugget)
    if learn_kernel and learn_nugget:

        def unsettled(state):
            _, _, kernel_iterations, nugget_iterations, converged, round_index = state
            settled = (nugget_iterations <= 2) & (kernel_iterations <= dimension + 1) & converged
            return (round_index < max_rounds) & ~settled

        def alternate(state):
            kernel_values, nugget, _, _, _, round_index = state
            kernel_values, kernel_iterations, converged = solve_kernel(kernel_values, nugget)
            squared = fixed_prior.to_squared_lengthscales(kernel_values)
            nugget, nugget_iterations = solve_nugget(squared, nugget)
            return (
                kernel_values,
                nugget,
                kernel_iterations,
                nugget_iterations,
                converged,
                round_index + 1,
            )

        initial_state = (
            kernel_values,
            nugget,
            jnp.asarray(dimension + 2),
            jnp.asarray(3),
            jnp.asarray(True),
            0,
        )
        kernel_values, nugget, iterations, _, converged, _ = jax.lax.while_loop(
            unsettled, alternate, initial_state
        )
    elif learn_kernel:
        kernel_values, iterations, converged = solve_kernel(kernel_values, nugget)
    else:
        nugget, iterations = solve_nugget(frozen(current.squared_lengthscales), nugget)
        converged = jnp.asarray(True)

    learned = jnp.concatenate([jnp.full(dimension, learn_kernel), jnp.array([learn_nugget])])
    solution = with_implicit_gradient(
        lambda values: (
            -_kernel_log_posterior(
                gp,
                jnp.where(learned, values, live)[:dimension],
                jnp.where(learned, values, live)[dimension],
                prior,
                include_nugget_prior=True,
            )
        ),
        jnp.concatenate([kernel_values, jnp.atleast_1d(nugget)]),
        lower,
        upper,
        free=learned,
    )
    squared = (
        prior.to_squared_lengthscales(solution[:dimension])
        if learn_kernel
        else current.squared_lengthscales
    )
    nugget = solution[dimension] if learn_nugget else current.nugget
    refit = gp.with_hyperparameters(Hyperparameters(squared, nugget))
    return refit, FitResult(squared, nugget, iterations, converged)


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


def _filled(value, size: int, dtype) -> Float[Array, "size"]:
    return jnp.broadcast_to(jnp.atleast_1d(jnp.asarray(value, dtype)), (size,))
