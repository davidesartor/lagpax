"""JAX-side properties: float64, jit/vmap/grad, and device-independent results."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import lagpax as lagp

D = np.array([0.3, 0.7, 1.1])
G = 1e-3


def _gp(small):
    X, Z = small
    return lagp.new_gpsep(jnp.array(X), jnp.array(Z), jnp.array(D), G)


def test_x64_is_enabled_by_importing_the_package():
    assert jax.config.jax_enable_x64
    assert jnp.zeros(1).dtype == jnp.float64


def test_gpsep_is_a_pytree(small):
    gp = _gp(small)
    leaves, treedef = jax.tree.flatten(gp)
    assert all(isinstance(v, jax.Array) for v in leaves)
    assert jax.tree.unflatten(treedef, leaves).phi == gp.phi


def test_llik_jits_and_differentiates(small):
    X, Z = small
    d = jnp.array(D)

    def f(d):
        return lagp.llik_gpsep(lagp.new_gpsep(jnp.array(X), jnp.array(Z), d, G))

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
    batched = jax.vmap(lambda x: lagp.pred_gpsep(gp, x[None, :]).mean[0])(XX)
    direct = lagp.pred_gpsep(gp, XX).mean
    np.testing.assert_allclose(np.asarray(batched), np.asarray(direct), rtol=1e-13)


def test_mle_d_jits(small):
    gp = _gp(small)
    f = jax.jit(lambda gp: lagp.mle_gpsep_d(gp, 1e-3, 10.0, ab=(1.5, 3.9)).d)
    np.testing.assert_allclose(
        np.asarray(f(gp)),
        np.asarray(lagp.mle_gpsep_d(gp, 1e-3, 10.0, ab=(1.5, 3.9)).d),
        rtol=1e-12,
    )


def test_no_nans_anywhere_in_a_local_fit(medium):
    X, Z = medium
    out = lagp.lagpsep(X[:1] + 0.01, X, Z, start=6, end=30, method="alc")
    for name, value in [("mean", out.mean), ("s2", out.s2), ("llik", out.llik)]:
        assert np.all(np.isfinite(np.asarray(value))), name
    assert np.all(np.isfinite(np.asarray(out.mle["d"])))


def test_local_design_indices_are_unique(medium):
    X, Z = medium
    for method in ("nn", "alc"):
        Xi = np.asarray(lagp.lagpsep(X[:1] + 0.01, X, Z, 6, 30, method=method).Xi)
        assert len(np.unique(Xi)) == len(Xi)


def test_masked_gp_has_no_nan_gradient(small):
    """The identity padding must not leak NaNs into reverse-mode AD."""
    X, Z = small
    mask = jnp.arange(len(X)) < 20

    def f(d):
        gp = lagp.new_gpsep(jnp.array(X), jnp.array(Z), d, G, mask)
        return lagp.llik_gpsep(gp)

    assert np.all(np.isfinite(np.asarray(jax.grad(f)(jnp.array(D)))))


def test_predictions_are_deterministic(medium):
    X, Z = medium
    a = lagp.lagpsep(X[:1] + 0.01, X, Z, 6, 30, method="alc")
    b = lagp.lagpsep(X[:1] + 0.01, X, Z, 6, 30, method="alc")
    assert float(a.mean[0]) == float(b.mean[0])
    np.testing.assert_array_equal(np.asarray(a.Xi), np.asarray(b.Xi))
