"""Box-constrained L-BFGS and safeguarded Newton, standing in for R's optim/optimize."""

from typing import Callable, NamedTuple, Optional
from jaxtyping import Array, Bool, Float, Int, Scalar

import jax
import jax.numpy as jnp

EPSMCH = float(jnp.finfo(jnp.float64).eps)
SDEPS = float(jnp.sqrt(EPSMCH))


def implicit_argmin(
    fun: Callable[[Float[Array, "n"]], Scalar],
    x: Float[Array, "n"],
    lower: Float[Array, "n"],
    upper: Float[Array, "n"],
    free: Optional[Bool[Array, "n"]] = None,
) -> Float[Array, "n"]:
    """Give a solver's answer the gradient the argmin has. The value is returned unchanged.

    Both solvers below are `lax.while_loop`s, which reverse-mode cannot differentiate, and
    unrolling them would differentiate the iterates rather than the answer. Instead: at an
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


class LbfgsbResult(NamedTuple):
    """Optimizer outcome; `conv` is 0 on convergence and 1 when maxiter was hit."""

    x: Float[Array, "n"]
    f: Scalar
    its: Int
    conv: Int


def _two_loop(
    q: Float[Array, "n"],
    S: Float[Array, "h n"],
    Y: Float[Array, "h n"],
    rho: Float[Array, "h"],
    valid: Float[Array, "h"],
) -> Float[Array, "n"]:
    """L-BFGS two-loop recursion over a ring buffer ordered oldest -> newest."""

    def backward(q, sy):
        s, y, r, v = sy
        alpha = v * r * jnp.dot(s, q)
        return q - alpha * y, alpha

    q, alphas = jax.lax.scan(backward, q, (S, Y, rho, valid), reverse=True)

    # initial Hessian scaling from the newest stored pair
    yy = jnp.dot(Y[-1], Y[-1])
    gamma = jnp.where(
        valid[-1] * (yy > 0), jnp.dot(S[-1], Y[-1]) / jnp.where(yy > 0, yy, 1.0), 1.0
    )
    q = q * gamma

    def forward(q, sy):
        s, y, r, v, alpha = sy
        beta = v * r * jnp.dot(y, q)
        return q + (alpha - beta) * s, None

    q, _ = jax.lax.scan(forward, q, (S, Y, rho, valid, alphas))
    return q


def lbfgsb(
    fun: Callable[[Float[Array, "n"]], Scalar],
    x0: Float[Array, "n"],
    lower: Float[Array, "n"],
    upper: Float[Array, "n"],
    maxiter: int = 100,
    history: int = 10,
    factr: float = 1e7,
    pgtol: float = 0.0,
    max_linesearch: int = 20,
) -> LbfgsbResult:
    """Minimize `fun` over a box, mirroring the stopping rules of R's L-BFGS-B (factr/pgtol)."""
    value_and_grad = jax.value_and_grad(fun)
    n = x0.shape[0]
    lower = jnp.broadcast_to(jnp.asarray(lower, x0.dtype), (n,))
    upper = jnp.broadcast_to(jnp.asarray(upper, x0.dtype), (n,))
    ftol = factr * EPSMCH

    def project_grad(x, grad):
        at_lower = (x <= lower) & (grad > 0)
        at_upper = (x >= upper) & (grad < 0)
        return jnp.where(at_lower | at_upper, 0.0, grad)

    def line_search(x, f, grad, p):
        """Backtrack on the projected arc, evaluating every trial step at once.

        Picking the first Armijo-feasible alpha is what sequential halving returns, but
        under vmap a while_loop costs the batch maximum, so the probes go in parallel.
        """
        alphas = 0.5 ** jnp.arange(max_linesearch + 1, dtype=x.dtype)
        xs = jnp.clip(x + alphas[:, None] * p, lower, upper)
        fs = jax.vmap(fun)(xs)

        feasible = (fs <= f + 1e-4 * ((xs - x) @ grad)) & jnp.isfinite(fs)
        # no feasible step leaves the smallest one, as the sequential loop did on maxit
        i = jnp.where(jnp.any(feasible), jnp.argmax(feasible), max_linesearch)
        return xs[i], fs[i]

    def cond(state):
        _, _, _, _, _, _, it, conv = state
        return (it < maxiter) & (conv < 0)

    def body(state):
        x, f, grad, S, Y, rho, it, _ = state
        pg = project_grad(x, grad)
        valid = (rho != 0.0).astype(x.dtype)

        direction = -_two_loop(pg, S, Y, rho, valid)
        direction = jnp.where(pg == 0.0, 0.0, direction)
        # fall back on steepest descent when the quasi-Newton direction is not descending
        direction = jnp.where(jnp.dot(direction, pg) < 0.0, direction, -pg)

        xn, fn = line_search(x, f, grad, direction)
        fn, gn = value_and_grad(xn)

        s, y = xn - x, gn - grad
        sy, yy = jnp.dot(s, y), jnp.dot(y, y)
        keep = sy > EPSMCH * yy
        S = jnp.where(keep, jnp.roll(S, -1, axis=0).at[-1].set(s), S)
        Y = jnp.where(keep, jnp.roll(Y, -1, axis=0).at[-1].set(y), Y)
        rho = jnp.where(
            keep, jnp.roll(rho, -1).at[-1].set(1.0 / jnp.where(sy > 0, sy, 1.0)), rho
        )

        scale = jnp.maximum(jnp.maximum(jnp.abs(f), jnp.abs(fn)), 1.0)
        stalled = (f - fn) <= ftol * scale
        flat = jnp.max(jnp.abs(project_grad(xn, gn))) <= pgtol
        conv = jnp.where(stalled | flat, 0, -1)
        return (xn, fn, gn, S, Y, rho, it + 1, conv)

    f0, g0 = value_and_grad(x0)
    init = (
        x0,
        f0,
        g0,
        jnp.zeros((history, n), x0.dtype),
        jnp.zeros((history, n), x0.dtype),
        jnp.zeros(history, x0.dtype),
        jnp.asarray(0),
        jnp.asarray(-1),
    )
    x, f, _, _, _, _, its, conv = jax.lax.while_loop(cond, body, init)
    return LbfgsbResult(x=x, f=f, its=its, conv=jnp.where(conv < 0, 1, 0))


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
