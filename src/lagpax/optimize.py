"""Solver pieces shared by the hyperparameter fits and the continuous design search.

L-BFGS-B itself is `vlse.optim.minimise`, called with R's `optim` stopping rule.
"""

import math
from typing import Callable, Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, Scalar

R_OPTIM_RELATIVE_TOLERANCE = 1e7 * float(jnp.finfo(jnp.float64).eps)
"""R's default `factr` (1e7) as scipy's `ftol`. Defined against double precision in every dtype: scaled by
float32's epsilon it would be ~1.2, and every fit would stop after one step."""


def with_implicit_gradient(
    objective: Callable[[Float[Array, "k"]], Scalar],
    solution: Float[Array, "k"],
    lower: Float[Array, "k"],
    upper: Float[Array, "k"],
    free: Optional[Bool[Array, "k"]] = None,
) -> Float[Array, "k"]:
    """`solution`, unchanged in value, carrying the derivative of the argmin of `objective`.

    The solvers are `lax.while_loop`s, which reverse mode cannot differentiate. At an interior
    optimum the implicit function theorem gives the argmin's derivative, and one Newton step from
    the frozen solution has exactly that derivative; subtracting the step's own value leaves the
    solution bit-identical. Coordinates on a bound, or not marked `free`, get zero gradient.
    """
    solution = jax.lax.stop_gradient(solution)
    interior = (solution > lower) & (solution < upper)
    free = interior if free is None else interior & free

    hessian = jax.lax.stop_gradient(jax.hessian(objective)(solution))
    both_free = free[:, None] & free[None, :]
    hessian = jnp.where(
        both_free, hessian, jnp.eye(solution.shape[0], dtype=solution.dtype)
    )
    newton_step = jnp.linalg.solve(
        hessian, jnp.where(free, jax.grad(objective)(solution), 0.0)
    )
    newton_step = jnp.where(free & jnp.isfinite(newton_step), newton_step, 0.0)
    return solution - (newton_step - jax.lax.stop_gradient(newton_step))


def newton_maximize_on_interval(
    first_and_second_derivative: Callable[[Scalar], tuple[Scalar, Scalar]],
    start: Scalar,
    lower: Scalar,
    upper: Scalar,
    max_iterations: int = 100,
    tolerance: Optional[float] = None,
) -> tuple[Scalar, Int[Array, ""]]:
    """Maximize a unimodal scalar function by Newton on its derivative, bisecting on bad steps.

    Returns the maximizer and the iteration count. A derivative that keeps its sign over the
    whole interval puts the maximum on that end, with zero iterations. `tolerance` defaults to
    the square root of the working precision, R's default for `optimize`.
    """
    dtype = jnp.result_type(start, lower, upper, float)
    lower, upper = jnp.asarray(lower, dtype), jnp.asarray(upper, dtype)
    tolerance = (
        math.sqrt(float(jnp.finfo(dtype).eps)) if tolerance is None else tolerance
    )
    slope_at_lower = first_and_second_derivative(lower)[0]
    slope_at_upper = first_and_second_derivative(upper)[0]

    def unfinished(state):
        _, _, _, iteration, done = state
        return (iteration < max_iterations) & ~done

    def step(state):
        point, bracket_low, bracket_high, iteration, _ = state
        slope, curvature = first_and_second_derivative(point)
        bracket_low = jnp.where(slope > 0, point, bracket_low)
        bracket_high = jnp.where(slope > 0, bracket_high, point)

        newton = point - slope / jnp.where(curvature == 0, -1.0, curvature)
        newton_is_safe = (
            (newton > bracket_low) & (newton < bracket_high) & (curvature < 0)
        )
        following = jnp.where(
            newton_is_safe, newton, 0.5 * (bracket_low + bracket_high)
        )
        done = (
            (jnp.abs(slope) < tolerance)
            | (jnp.abs(following - point) < tolerance)
            | (bracket_high - bracket_low < tolerance)
        )
        return following, bracket_low, bracket_high, iteration + 1, done

    initial_state = (
        jnp.clip(start, lower, upper),
        lower,
        upper,
        jnp.asarray(0),
        jnp.asarray(False),
    )
    maximizer, _, _, iterations, _ = jax.lax.while_loop(unfinished, step, initial_state)

    on_boundary = (slope_at_lower <= 0) | (slope_at_upper >= 0)
    boundary = jnp.where(slope_at_lower <= 0, lower, upper)
    return (
        jnp.where(on_boundary, boundary, maximizer),
        jnp.where(on_boundary, 0, iterations),
    )
