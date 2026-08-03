"""JAX-side properties: float64, jit/vmap/grad, and device-independent results."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import lagpax as lagp
from lagpax.mle import mle_d

D = np.array([0.3, 0.7, 1.1])
G = 1e-3


def _gp(small):
    X, Z = small
    return lagp.GP(jnp.array(X), jnp.array(Z), jnp.array(D), G)


def test_x64_is_enabled_by_importing_the_package():
    assert jax.config.jax_enable_x64
    assert jnp.zeros(1).dtype == jnp.float64


def test_gp_is_a_pytree(small):
    gp = _gp(small)
    leaves, treedef = jax.tree.flatten(gp)
    assert all(isinstance(v, jax.Array) for v in leaves)
    assert jax.tree.unflatten(treedef, leaves).phi == gp.phi


def test_llik_jits_and_differentiates(small):
    X, Z = small
    d = jnp.array(D)

    def f(d):
        return lagp.GP(jnp.array(X), jnp.array(Z), d, G).llik()

    assert float(jax.jit(f)(d)) == pytest.approx(float(f(d)), rel=1e-14)
    grad = np.asarray(jax.grad(f)(d))
    assert np.all(np.isfinite(grad))

    eps = 1e-6
    for k in range(len(D)):
        step = jnp.zeros(len(D)).at[k].set(eps)
        fd = (float(f(d + step)) - float(f(d - step))) / (2 * eps)
        assert grad[k] == pytest.approx(fd, rel=1e-5, abs=1e-7)


def test_pred_vmaps_over_locations(small):
    X, _ = small
    gp = _gp(small)
    XX = jnp.array(X[:6] + 0.05)
    batched = jax.vmap(lambda x: gp.predict(x[None, :]).mean[0])(XX)
    direct = gp.predict(XX).mean
    np.testing.assert_allclose(np.asarray(batched), np.asarray(direct), rtol=1e-13)


def test_mle_d_jits(small):
    gp = _gp(small)
    f = jax.jit(lambda gp: mle_d(gp, 1e-3, 10.0, ab=(1.5, 3.9)).d)
    np.testing.assert_allclose(
        np.asarray(f(gp)),
        np.asarray(mle_d(gp, 1e-3, 10.0, ab=(1.5, 3.9)).d),
        rtol=1e-12,
    )


def test_no_nans_anywhere_in_a_local_fit(medium):
    X, Z = medium
    out = lagp.lagp(X, Z, X[:1] + 0.01, start=6, end=30)
    for name, value in [("mean", out.mean), ("s2", out.s2), ("llik", out.llik)]:
        assert np.all(np.isfinite(np.asarray(value))), name
    assert np.all(np.isfinite(np.asarray(out.fit.d)))


def test_local_design_indices_are_unique(medium):
    X, Z = medium
    for criterion in ("nn", "alc"):
        Xi = np.asarray(
            lagp.lagp(X, Z, X[:1] + 0.01, start=6, end=30, criterion=criterion).Xi
        )
        assert len(np.unique(Xi)) == len(Xi)


def test_masked_gp_has_no_nan_gradient(small):
    """The identity padding must not leak NaNs into reverse-mode AD."""
    X, Z = small
    mask = jnp.arange(len(X)) < 20

    def f(d):
        return lagp.GP(jnp.array(X), jnp.array(Z), d, G, mask).llik()

    assert np.all(np.isfinite(np.asarray(jax.grad(f)(jnp.array(D)))))


def test_predictions_are_deterministic(medium):
    X, Z = medium
    a = lagp.lagp(X, Z, X[:1] + 0.01, start=6, end=30)
    b = lagp.lagp(X, Z, X[:1] + 0.01, start=6, end=30)
    assert float(a.mean[0]) == float(b.mean[0])
    np.testing.assert_array_equal(np.asarray(a.Xi), np.asarray(b.Xi))
