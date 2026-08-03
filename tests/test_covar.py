"""covar.py against laGP's distance/covar_sep."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import lagpax as lagp
from lagpax.covar import covar_symm, dcovar


def test_distance_matches_r(R, small):
    X, _ = small
    Y = X[:7] + 0.01
    got = np.asarray(lagp.distance(jnp.array(X), jnp.array(Y)))
    want = np.asarray(R("distance(X, Y)", X=X, Y=Y))
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)


def test_distance_symmetric_matches_r(R, small):
    X, _ = small
    got = np.asarray(lagp.distance(jnp.array(X)))
    want = np.asarray(R("distance(X)", X=X))
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)


def test_covar_matches_r(R, small):
    X, _ = small
    Y = X[:7] + 0.01
    d = np.array([0.3, 0.7, 1.1])
    got = np.asarray(lagp.covar(jnp.array(d), jnp.array(X), jnp.array(Y)))
    want = np.asarray(
        R(
            "exp(-distance(X %*% diag(1/sqrt(d)), Y %*% diag(1/sqrt(d))))",
            X=X,
            Y=Y,
            d=d,
        )
    )
    np.testing.assert_allclose(got, want, rtol=1e-14, atol=0)


def test_covar_symm_diagonal_carries_nugget(small):
    X, _ = small
    d, g = jnp.array([0.3, 0.7, 1.1]), 1e-3
    K = covar_symm(d, jnp.array(X), g)
    np.testing.assert_allclose(np.diag(np.asarray(K)), 1.0 + g, rtol=0, atol=0)
    np.testing.assert_allclose(np.asarray(K), np.asarray(K).T, rtol=0, atol=0)


def test_covar_gradient_finite_on_the_diagonal(small):
    """laGP's kernel is smooth at zero separation; the JAX gradient must not be NaN."""
    X, _ = small
    d = jnp.array([0.3, 0.7, 1.1])
    grad = jax.grad(lambda d: covar_symm(d, jnp.array(X), 1e-3).sum())(d)
    assert np.all(np.isfinite(np.asarray(grad)))


def test_dcovar_matches_autodiff(small):
    X, _ = small
    Y = X[:7] + 0.01
    d = jnp.array([0.3, 0.7, 1.1])
    K = lagp.covar(d, jnp.array(X), jnp.array(Y))
    analytic = np.asarray(dcovar(d, jnp.array(X), jnp.array(Y), K))
    auto = np.asarray(
        jax.jacobian(lambda d: lagp.covar(d, jnp.array(X), jnp.array(Y)))(d)
    )
    np.testing.assert_allclose(
        analytic, np.moveaxis(auto, -1, 0), rtol=1e-12, atol=1e-14
    )


@pytest.mark.parametrize("n", [1, 5])
def test_covar_shapes(small, n):
    X, _ = small
    d = jnp.array([0.3, 0.7, 1.1])
    assert lagp.covar(d, jnp.array(X[:n]), jnp.array(X)).shape == (n, len(X))
