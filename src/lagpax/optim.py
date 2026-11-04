"""Safeguarded Newton and the implicit-argmin gradient; box-constrained solves use vlse.

Lengthscale and continuous-search solves call `vlse.optim.minimise` (L-BFGS-B) directly,
with R's optim stopping rule: `factr = 1e7` scaled by machine epsilon is scipy's `ftol`,
and `pgtol = 0` disables the projected-gradient test. `R_FTOL` is that product.
"""

from typing import Callable, NamedTuple, Optional
from jaxtyping import Array, Bool, Float, Int, Scalar

import jax
import jax.numpy as jnp

EPSMCH = float(jnp.finfo(jnp.float64).eps)
SDEPS = float(jnp.sqrt(EPSMCH))
R_FTOL = 1e7 * EPSMCH  # optim(method="L-BFGS-B")'s default factr, as scipy's ftol


def implicit_argmin(
    fun: Callable[[Float[Array, "n"]], Scalar],
    x: Float[Array, "n"],
    lower: Float[Array, "n"],
    upper: Float[Array, "n"],
    free: Optional[Bool[Array, "n"]] = None,
) -> Float[Array, "n"]:
    """Give a solver's answer the gradient the argmin has. The value is returned unchanged.

    Both solvers (vlse's L-BFGS-B and `bracketed_newton`) are `lax.while_loop`s, which
    reverse-mode cannot differentiate, and unrolling them would differentiate the iterates rather than the answer. Instead: at an
    interior optimum `grad fun(x) = 0`, so the implicit function theorem moves the argmin
    with the upstream data as `-H^-1 d(grad fun)/d(upstream)`. One Newton step taken from a
    frozen `x` has exactly that derivative, and subtracting its own `stop_gradient` leaves
    the returned value bit-identical to the solver's.

    A coordinate pinned to a bound is not a stationary point, so it stays frozen and its
    gradient is zero -- the honest answer for an active constraint, not a smooth one. Pass
    `free` to freeze more than the bounds do; the remaining coordinates then get the
    derivative of the argmin over them with the frozen ones held fixed, which is what the
    caller means when the lengthscale came from laGP's boundary reset rather than a solve.
    """
    x = jax.lax.stop_gradient(x)
    interior = (x > lower) & (x < upper)
    free = interior if free is None else (interior & free)

    H = jax.lax.stop_gradient(jax.hessian(fun)(x))
    H = jnp.where(free[:, None] & free[None, :], H, jnp.eye(x.shape[0], dtype=x.dtype))
    step = jnp.linalg.solve(H, jnp.where(free, jax.grad(fun)(x), 0.0))

    # a degenerate Hessian means the optimum is not locally identified; give up on the
    # gradient rather than propagating a NaN through an otherwise valid prediction
    step = jnp.where(free & jnp.isfinite(step), step, 0.0)
    return x - (step - jax.lax.stop_gradient(step))


class NewtonResult(NamedTuple):
    """Scalar root-find outcome for the nugget MLE."""

    x: Scalar
    its: Int


def bracketed_newton(
    dfun: Callable[[Scalar], tuple[Scalar, Scalar]],
    x0: Scalar,
    lower: Scalar,
    upper: Scalar,
    maxiter: int = 100,
    tol: float = SDEPS,
) -> NewtonResult:
    """Maximize a unimodal scalar objective by Newton on its derivative, bisecting on slips.

    Replaces laGP's Newton-with-Brent-fallback (mleGPsep_nug): same interior stationary
    point, and the same clamping to a boundary when the derivative never changes sign.

    `dfun` returns the first and second derivative as a pair rather than as two callables:
    for the nugget MLE both come out of one forward-over-reverse pass sharing a single
    factorization of K, where two calls would Cholesky it twice per iteration.
    """
    lower = jnp.asarray(lower, jnp.float64)
    upper = jnp.asarray(upper, jnp.float64)
    dlo, dhi = dfun(lower)[0], dfun(upper)[0]

    def cond(state):
        x, lo, hi, it, done = state
        return (it < maxiter) & ~done

    def body(state):
        x, lo, hi, it, _ = state
        dx, d2x = dfun(x)
        lo = jnp.where(dx > 0, x, lo)
        hi = jnp.where(dx > 0, hi, x)

        newton = x - dx / jnp.where(d2x == 0, -1.0, d2x)
        inside = (newton > lo) & (newton < hi) & (d2x < 0)
        xn = jnp.where(inside, newton, 0.5 * (lo + hi))
        done = (jnp.abs(dx) < tol) | (jnp.abs(xn - x) < tol) | (hi - lo < tol)
        return (xn, lo, hi, it + 1, done)

    # a derivative that never changes sign puts the optimum on a boundary
    on_boundary = (dlo <= 0) | (dhi >= 0)
    boundary = jnp.where(dlo <= 0, lower, upper)

    x0 = jnp.clip(x0, lower, upper)
    x, _, _, its, _ = jax.lax.while_loop(
        cond, body, (x0, lower, upper, jnp.asarray(0), jnp.asarray(False))
    )
    return NewtonResult(
        x=jnp.where(on_boundary, boundary, x), its=jnp.where(on_boundary, 0, its)
    )
