"""Differentiating through the fit, and running the whole scheme under jit/vmap.

The optimizers are `lax.while_loop`s, so the gradient does not come from their iterates --
it comes from the implicit function theorem applied at the answer. These tests check the
two halves separately: the machinery against an argmin with a closed form, and the real
fit against an argmin solved to machine precision.
"""

import numpy as np
import pytest

import jax
import jax.numpy as jnp

import lagpax as lagp
from lagpax.gp import GP, llik_d, llik_g
from lagpax.local import Local
from lagpax.mle import fit, mle_d, mle_g
from lagpax.optim import implicit_argmin
from lagpax.prior import Prior, Range


def _quadratic(p, A, M):
    """argmin of 0.5 x'Ax - (Mp)'x, whose exact derivative is A^-1 M."""
    b = M @ p
    return lambda x: 0.5 * x @ A @ x - b @ x


@pytest.fixture(scope="module")
def quad():
    rng = np.random.default_rng(3)
    B = rng.random((4, 4))
    A = jnp.asarray(B @ B.T + 4.0 * np.eye(4))
    M = jnp.asarray(rng.random((4, 3)))
    return A, M, jnp.asarray(rng.random(3))


def test_implicit_argmin_reproduces_a_closed_form_jacobian(quad):
    """A quadratic's argmin is A^-1 M p, so the implicit gradient has an exact target."""
    A, M, p = quad
    lo, hi = jnp.full(4, -1e3), jnp.full(4, 1e3)

    def solved(p):
        fun = _quadratic(p, A, M)
        return implicit_argmin(fun, jnp.linalg.solve(A, M @ p), lo, hi)

    np.testing.assert_allclose(
        np.asarray(jax.jacobian(solved)(p)),
        np.asarray(jnp.linalg.solve(A, M)),
        rtol=1e-11,
    )
    np.testing.assert_array_equal(
        np.asarray(solved(p)), np.asarray(jnp.linalg.solve(A, M @ p))
    )


def test_implicit_argmin_leaves_the_value_bit_identical(quad):
    """The solver's answer must survive untouched: the parity suite compares values."""
    A, M, p = quad
    lo, hi = jnp.full(4, -1e3), jnp.full(4, 1e3)
    x = jnp.asarray([0.11, -0.37, 2.5, 0.04])
    got = implicit_argmin(_quadratic(p, A, M), x, lo, hi)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(x))


def test_a_coordinate_on_a_bound_has_exactly_zero_gradient(quad):
    """An active constraint is not smooth, and is reported as flat rather than guessed at."""
    A, M, p = quad
    lo = jnp.array([-1e3, -1e3, -1e3, -1e3])
    hi = jnp.array([1e3, 1e-3, 1e3, 1e3])

    def solved(p):
        x = jnp.clip(jnp.linalg.solve(A, M @ p), lo, hi)
        return implicit_argmin(_quadratic(p, A, M), x, lo, hi)

    jac = np.asarray(jax.jacobian(solved)(p))
    assert np.all(jac[1] == 0.0)
    assert np.any(jac[0] != 0.0)


def _exact_argmin_d(X, Z, d0, g, warm):
    """d* solved from the stationarity condition itself, good to machine precision.

    Finite differences of `lbfgsb`'s own output cannot check this gradient: it stops on R's
    `factr`, so it returns roughly eight significant digits, and a central difference of
    that carries |d| * 1e-8 / 2h of noise -- around 7e-5 at the largest step truncation
    allows. Root-finding `grad llik = 0` removes that floor.
    """
    from scipy.optimize import root

    gp = GP(jnp.asarray(X), jnp.asarray(Z), d0, g)
    grad = jax.jit(jax.grad(lambda d: -llik_d(gp, d)))
    sol = root(lambda d: np.asarray(grad(jnp.asarray(d))), warm, tol=1e-13)

    # hybr reports failure once it stops improving, which at this tolerance means it has
    # already landed -- the residual is the thing that matters, not its stopping message
    assert np.max(np.abs(grad(jnp.asarray(sol.x)))) < 1e-9
    return sol.x


def test_lengthscale_gradient_matches_an_exactly_solved_argmin(noisy):
    """d(d*)/dZ through `mle_d`, against a differenced machine-precision argmin."""
    X, Z = noisy
    X, Z = X[:60], Z[:60]
    d0, g = jnp.full(3, 0.5), 1e-2
    lo, hi = 1e-3, 10.0

    fitted = lambda Zv: mle_d(GP(jnp.asarray(X), Zv, d0, g), lo, hi, maxit=200).d
    got = np.asarray(jax.jacobian(fitted)(jnp.asarray(Z)))

    warm = np.asarray(fitted(jnp.asarray(Z)))
    step = 1e-4
    for k in (0, 7, 23):
        hi_z, lo_z = Z.copy(), Z.copy()
        hi_z[k] += step
        lo_z[k] -= step
        want = (
            _exact_argmin_d(X, hi_z, d0, g, warm)
            - _exact_argmin_d(X, lo_z, d0, g, warm)
        ) / (2 * step)
        np.testing.assert_allclose(got[:, k], want, rtol=1e-5, atol=1e-9)


def test_nugget_gradient_matches_finite_differences(noisy):
    """The nugget's optimum is interior on `noisy`, which is what makes it checkable."""
    from scipy.optimize import brentq

    X, Z = noisy
    X, Z = X[:60], Z[:60]
    d0, g0 = jnp.full(3, 0.5), 1e-2
    lo, hi = 1e-6, 1.0

    fitted = lambda Zv: mle_g(GP(jnp.asarray(X), Zv, d0, g0), lo, hi, maxit=200).g
    got = np.asarray(jax.grad(fitted)(jnp.asarray(Z)))

    def exact(Znp):
        gp = GP(jnp.asarray(X), jnp.asarray(Znp), d0, g0)
        dg = jax.jit(jax.grad(lambda g: -llik_g(gp, g)))
        return brentq(lambda g: float(dg(g)), lo, hi, xtol=1e-15, rtol=1e-15)

    step = 1e-4
    for k in (1, 12, 40):
        hi_z, lo_z = Z.copy(), Z.copy()
        hi_z[k] += step
        lo_z[k] -= step
        want = (exact(hi_z) - exact(lo_z)) / (2 * step)
        assert got[k] == pytest.approx(want, rel=1e-5, abs=1e-12)


def test_gradient_is_zero_when_the_nugget_is_pinned(small):
    """The noiseless case pins g at its lower bound, and a bound carries no gradient."""
    X, Z = small
    gp = GP(jnp.asarray(X), jnp.asarray(Z), jnp.full(3, 0.5), 1e-2)
    lo, hi = 1e-8, 1.0

    fitted = lambda Zv: mle_g(GP(jnp.asarray(X), Zv, jnp.full(3, 0.5), 1e-2), lo, hi).g
    assert float(fitted(jnp.asarray(Z))) == pytest.approx(lo, rel=1e-12)
    np.testing.assert_array_equal(np.asarray(jax.grad(fitted)(jnp.asarray(Z))), 0.0)


def test_the_boundary_reset_carries_no_gradient(small):
    """conv=102 replaces the solve with 0.9*dmin + 0.1*dmax, a constant in the data."""
    X, Z = small
    lo, hi = 1e-3, 10.0

    fitted = lambda Zv: mle_d(GP(jnp.asarray(X), Zv, jnp.full(3, lo), 1e-2), lo, hi)
    res = fitted(jnp.asarray(Z))
    assert int(res.conv) == 102

    jac = jax.jacobian(lambda Zv: fitted(Zv).d)(jnp.asarray(Z))
    np.testing.assert_array_equal(np.asarray(jac), 0.0)


def _emu(X, Z, **settings):
    prior = Prior.from_design(np.asarray(X), np.asarray(Z), g={"mle": True})
    return Local(**settings).condition(X, Z, prior=prior)


def test_the_whole_scheme_runs_under_jit(medium):
    """Acceptance: everything after `condition` is one JAX function, host callbacks and all."""
    X, Z = medium
    XX = np.random.default_rng(11).random((5, X.shape[1]))
    prior = Prior.from_design(X, Z, g={"mle": True})
    scheme = Local(start=6, end=20)

    def run(X, Z, XX):
        return scheme.condition(X, Z, prior=prior).predict(XX, batch=None).mean

    plain = np.asarray(run(jnp.asarray(X), jnp.asarray(Z), jnp.asarray(XX)))
    compiled = np.asarray(jax.jit(run)(jnp.asarray(X), jnp.asarray(Z), jnp.asarray(XX)))

    # reassociation under jit costs a few ulps, the same reason the batch-invariance
    # checks elsewhere in the suite are 1e-6..1e-9 rather than exact
    np.testing.assert_allclose(compiled, plain, rtol=1e-9)


def test_the_scheme_vmaps_over_datasets(medium):
    """Not just over predictive locations: X and Z can carry a leading batch axis too."""
    X, Z = medium
    rng = np.random.default_rng(12)
    XX = rng.random((3, X.shape[1]))
    prior = Prior.from_design(X, Z)
    scheme = Local(start=6, end=20)

    Zs = jnp.stack([jnp.asarray(Z), jnp.asarray(Z + 0.5 * rng.standard_normal(len(Z)))])
    run = lambda Zv: scheme.condition(X, Zv, prior=prior).predict(XX, batch=None)
    batched = jax.vmap(run)(Zs)

    # `vmap` rewrites the arithmetic and a batched `while_loop` shares its trip count, so
    # one ulp of difference in the likelihood flips `lbfgsb`'s relative-change stopping test
    # and the two runs stop at different iterates -- separated by the solver's own
    # resolution, sqrt(factr * eps) ~ 4.7e-5 relative in `d`. The likelihood is stationary
    # there, so the gap it induces is *second* order in that: ~2e-6 relative, which is what
    # the tolerance below is. It is not `factr * eps`; that bounds the improvement of the
    # last accepted step, not the distance from either stopping point to the optimum.
    for i in range(2):
        one = run(Zs[i])
        np.testing.assert_allclose(
            np.asarray(batched.mean[i]), np.asarray(one.mean), rtol=1e-4, atol=1e-8
        )
        np.testing.assert_allclose(
            np.asarray(batched.llik[i]), np.asarray(one.llik), rtol=1e-5
        )


def test_gradients_reach_the_data_through_the_whole_scheme(noisy):
    """The end-to-end claim: `jax.grad` of a prediction w.r.t. the training responses.

    The comparison is a finite difference of the pipeline, so it inherits `lbfgsb`'s
    stopping tolerance -- see `_exact_argmin_d` for why that floors the agreement near
    1e-4. The exactly-solved comparisons above are what pin the gradient itself.
    """
    X, Z = noisy
    XX = np.random.default_rng(13).random((2, X.shape[1]))
    emu = _emu(X, Z, start=6, end=20)

    f = lambda Zv: jnp.sum(emu._replace(Z=Zv).predict(XX, batch=None).mean)
    Zj = jnp.asarray(Z)
    got = np.asarray(jax.grad(f)(Zj))
    assert np.count_nonzero(got) > 10

    step = 1e-4
    for k in np.argsort(-np.abs(got))[:4]:
        want = float(f(Zj.at[k].add(step)) - f(Zj.at[k].add(-step))) / (2 * step)
        assert got[k] == pytest.approx(want, rel=2e-3)


def test_gradients_reach_the_design_locations(noisy):
    """d(prediction)/dX as well as dZ: the design enters through the covariance."""
    X, Z = noisy
    XX = np.random.default_rng(14).random((2, X.shape[1]))
    emu = _emu(X, Z, start=6, end=20)

    f = lambda Xv: jnp.sum(emu._replace(X=Xv).predict(XX, batch=None).mean)
    Xj = jnp.asarray(X)
    got = np.asarray(jax.grad(f)(Xj))
    assert np.count_nonzero(got) > 10

    step = 1e-4
    flat = np.argsort(-np.abs(got.ravel()))[:4]
    for idx in flat:
        i, j = np.unravel_index(idx, got.shape)
        want = float(f(Xj.at[i, j].add(step)) - f(Xj.at[i, j].add(-step))) / (2 * step)
        assert got[i, j] == pytest.approx(want, rel=2e-3)


def test_refine_is_differentiable(medium):
    """`refine` unrolls its passes at trace time, so the gradient survives them."""
    X, Z = medium
    XX = np.random.default_rng(15).random((2, X.shape[1]))
    emu = _emu(X, Z, start=6, end=20)

    f = lambda Zv: jnp.sum(emu._replace(Z=Zv).refine(XX, batch=None).mean)
    got = np.asarray(jax.grad(f)(jnp.asarray(Z)))
    assert np.all(np.isfinite(got))
    assert np.count_nonzero(got) > 10
