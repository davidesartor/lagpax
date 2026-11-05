"""Hyperparameter estimation: maximum likelihood, or MAP under a `Prior`.

The kernel parameter is fitted by L-BFGS-B (`vlse.optim.minimise`) in the prior's own space
-- theta, or the lengthscale -- and the nugget by a safeguarded Newton iteration. When both
are learned they alternate until neither moves, as laGP's jmleGPsep does.

Every solver is a `lax.while_loop`, so its answer is computed without a gradient tape and
`optim.implicit_argmin` attaches the implicit-function-theorem derivative afterwards: the
returned values are the solvers', and `jax.grad` through `fit` is exact at the optimum.
"""

from typing import NamedTuple
from jaxtyping import Array, Bool, Float, Int, Scalar

import jax
import jax.numpy as jnp
from vlse.optim import minimise

from .gp import GP, llik_both, llik_nugget
from .optim import bracketed_newton, implicit_argmin, r_ftol
from .prior import Prior, _gamma_log_prior

sg = jax.lax.stop_gradient


class FitInfo(NamedTuple):
    """Fitted hyperparameters and the kernel solver's diagnostics.

    `theta` is the kernel's squared lengthscale whatever space the prior was in;
    `lengthscale` is the same thing as `sqrt(theta / 2)`. `iterations` counts the kernel
    solver's steps; `converged` is False only when it ran out of iterations.
    """

    theta: Float[Array, "m"]
    nugget: Scalar
    iterations: Int[Array, ""]
    converged: Bool[Array, ""]

    @property
    def lengthscale(self) -> Float[Array, "m"]:
        return jnp.sqrt(0.5 * self.theta)


def _box(v, m: int, dtype) -> Float[Array, "m"]:
    return jnp.broadcast_to(jnp.atleast_1d(jnp.asarray(v, dtype)), (m,))


def _kernel_llik(gp: GP, v, nugget, prior: Prior, nugget_ab=(0.0, 0.0)) -> Scalar:
    """The objective in the prior's kernel space: likelihood plus both log-priors."""
    if prior.space == "theta":
        return llik_both(gp, v, nugget, prior.theta_gamma, nugget_ab)
    a, b = prior.lengthscale.gamma or (0.0, 0.0)
    return llik_both(gp, prior.to_theta(v), nugget, (0.0, 0.0), nugget_ab) + (
        _gamma_log_prior(v, a, b)
    )


def _solve_kernel(gp: GP, v0, nugget, lower, upper, prior: Prior, maxiter: int):
    """L-BFGS-B on the kernel parameter, with R's optim stopping rule."""
    res = minimise(
        lambda v: -_kernel_llik(gp, v, nugget, prior),
        v0,
        bounds=(lower, upper),
        tol=0.0,
        ftol=r_ftol(v0.dtype),
        max_iterations=maxiter,
    )
    stopped = (res.error <= 0.0) | res.converged_ftol | res.failed_linesearch
    return res.x, res.iteration, stopped | (res.iteration < maxiter)


def _solve_nugget(gp: GP, theta, g0, lower, upper, ab, maxiter: int):
    """Newton on the nugget's score, both derivatives from one forward-over-reverse pass."""
    score = jax.grad(lambda g: llik_both(gp, theta, g, (0.0, 0.0), ab))
    res = bracketed_newton(
        lambda g: jax.jvp(score, (g,), (jnp.ones_like(g),)),
        g0,
        lower,
        upper,
        maxiter=maxiter,
    )
    return res.x, res.its


def fit_kernel(gp: GP, prior: Prior, maxiter: int = 100) -> tuple[GP, FitInfo]:
    """MAP kernel parameter at the GP's nugget."""
    p, dtype = prior.kernel, gp.y.dtype
    lower, upper = _box(p.lower, gp.m, dtype), _box(p.upper, gp.m, dtype)
    v0 = jnp.clip(_box(prior.from_theta(gp.theta), gp.m, dtype), lower, upper)
    frozen = sg(gp)
    v, its, ok = _solve_kernel(
        frozen, sg(v0), sg(gp.nugget), sg(lower), sg(upper), sg(prior), maxiter
    )
    v = implicit_argmin(lambda v: -_kernel_llik(gp, v, gp.nugget, prior), v, lower, upper)
    theta = prior.to_theta(v)
    return gp.replace(theta=theta), FitInfo(theta, gp.nugget, its, ok)


def fit_nugget(gp: GP, prior: Prior, maxiter: int = 100) -> tuple[GP, FitInfo]:
    """MAP nugget at the GP's kernel parameter."""
    p, dtype = prior.nugget, gp.y.dtype
    ab = prior.nugget_gamma
    g, its = _solve_nugget(
        sg(gp), sg(gp.theta), sg(gp.nugget), sg(p.lower), sg(p.upper), sg(ab), maxiter
    )
    g = implicit_argmin(
        lambda v: -llik_nugget(gp, v[0], ab=ab),
        jnp.atleast_1d(g),
        _box(p.lower, 1, dtype),
        _box(p.upper, 1, dtype),
    )[0]
    return gp.replace(nugget=g), FitInfo(gp.theta, g, its, jnp.asarray(True))


def fit_both(
    gp: GP, prior: Prior, maxiter: int = 100, rounds: int = 100
) -> tuple[GP, FitInfo]:
    """Alternate kernel and nugget solves until both settle, as laGP's jmleGPsep does."""
    m, dtype = gp.m, gp.y.dtype
    kp, gp_ = prior.kernel, prior.nugget
    klo, khi = _box(kp.lower, m, dtype), _box(kp.upper, m, dtype)
    glo, ghi = _box(gp_.lower, 1, dtype), _box(gp_.upper, 1, dtype)
    nab = prior.nugget_gamma
    frozen, fprior = sg(gp), sg(prior)

    def cond(state):
        _, _, kits, gits, ok, i = state
        settled = (gits <= 2) & (kits <= m + 1) & ok
        return (i < rounds) & ~settled

    # the loop carries the hyperparameters, not a GP: neither half-step reads a
    # factorization, so rebuilding one per round would be two dead Choleskys
    def body(state):
        v0, g0, _, _, _, i = state
        v, kits, ok = _solve_kernel(frozen, v0, g0, klo, khi, fprior, maxiter)
        theta = fprior.to_theta(v)
        g, gits = _solve_nugget(frozen, theta, g0, glo[0], ghi[0], sg(nab), maxiter)
        return v, g, kits, gits, ok, i + 1

    v0 = jnp.clip(_box(prior.from_theta(gp.theta), m, dtype), klo, khi)
    init = sg((v0, gp.nugget, jnp.asarray(m + 2), jnp.asarray(3), jnp.asarray(True), 0))
    v, g, kits, _, ok, _ = jax.lax.while_loop(cond, body, init)

    # both halves converged, so the answer is a joint stationary point of one objective
    # and the implicit correction can be taken over (kernel, nugget) together
    joint = implicit_argmin(
        lambda t: -_kernel_llik(gp, t[:m], t[m], prior, nab),
        jnp.concatenate([v, jnp.atleast_1d(g)]),
        jnp.concatenate([klo, glo]),
        jnp.concatenate([khi, ghi]),
    )
    theta, g = prior.to_theta(joint[:m]), joint[m]
    return gp.replace(theta=theta, nugget=g), FitInfo(theta, g, kits, ok)


def fit(gp: GP, prior: Prior, maxiter: int = 100) -> tuple[GP, FitInfo]:
    """Fit whichever hyperparameters the prior marks `learn`, returning the refit GP."""
    learn_k, learn_g = bool(prior.kernel.learn), bool(prior.nugget.learn)
    if learn_k and learn_g:
        return fit_both(gp, prior, maxiter=maxiter)
    if learn_k:
        return fit_kernel(gp, prior, maxiter=maxiter)
    if learn_g:
        return fit_nugget(gp, prior, maxiter=maxiter)
    done = jnp.asarray(True)
    return gp, FitInfo(gp.theta, gp.nugget, jnp.asarray(0), done)
