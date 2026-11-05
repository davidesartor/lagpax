"""JAX-side properties: dtype handling, jit/vmap/grad, and device-independent results."""

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


_FLOAT32_RUN = """
import jax, numpy as np
import lagpax
assert not jax.config.jax_enable_x64, "importing lagpax must not flip a global switch"
rng = np.random.default_rng(0)
X = rng.random((200, 2))
Z = np.sin(4 * X[:, 0]) + X[:, 1]
XX = rng.random((5, 2))
out = lagpax.agp(X, Z, XX, start=6, end=20)
assert out.mean.dtype == np.float32, out.mean.dtype
assert np.allclose(np.asarray(out.mean), np.sin(4 * XX[:, 0]) + XX[:, 1], atol=1e-2)
assert np.all(np.asarray(out.fit.its) > 1), "float32 fits must not stop after one step"
# s2 = 1 + g - k' Ki k is a cancellation; at g=1e-4 float32 cannot promise its sign
print("ok")
"""


def test_importing_the_package_leaves_x64_alone_and_float32_runs():
    """No global side effect at import; without x64 the pipeline runs in float32.

    Needs a fresh interpreter: this session turned x64 on in conftest, for parity.
    """
    import subprocess
    import sys

    res = subprocess.run(
        [sys.executable, "-c", _FLOAT32_RUN], capture_output=True, text=True
    )
    assert res.returncode == 0, res.stderr[-2000:]


def test_the_suite_runs_in_float64():
    """conftest enables x64 before any data exists; every parity tolerance assumes it."""
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


def _emu(medium, **settings):
    X, Z = medium
    return lagp.Local(start=6, end=16, **settings).condition(X, Z)


def test_the_emulator_is_a_pytree_that_jit_takes_as_an_argument(medium):
    """A Conditioned emulator crosses a jit boundary like any other JAX value."""
    emu = _emu(medium)
    XX = medium[0][:3] + 0.01
    leaves = jax.tree.leaves(emu)
    assert any(v is emu.X for v in leaves) and any(v is emu.Z for v in leaves)

    eager = emu.predict(XX, batch=None).mean
    jitted = jax.jit(lambda e, x: e.predict(x, batch=None).mean)(emu, XX)
    np.testing.assert_allclose(np.asarray(jitted), np.asarray(eager), rtol=1e-9)


def test_vmap_over_the_per_location_function_matches_predict(medium):
    """`local` is the primitive; `predict` is lax.map over it, and vmap must agree.

    Not bit-identical: a batched while_loop shares its trip count across the batch, so the
    fits stop at slightly different iterates (ROADMAP: "jit preserves the trajectory, vmap
    does not") -- hence the batch-invariance tolerance, not 1e-12.
    """
    emu = _emu(medium)
    XX = medium[0][:4] + 0.01
    mapped = jax.vmap(lambda x: emu.local(x[None]).mean[0])(jnp.asarray(XX))
    np.testing.assert_allclose(
        np.asarray(mapped), np.asarray(emu.predict(XX).mean), rtol=1e-4, atol=1e-8
    )


def test_ieci_incumbent_is_data_and_its_flags_are_program(medium):
    """Moving `fmin` -- every step of a BO loop -- must not recompile; `nugget` must."""
    from lagpax.local import _core

    # `local` dispatches `_core` at top level, where its compile cache is what counts;
    # inside `predict` it is traced into lax.map's body instead
    x = medium[0][:1] + 0.01
    _emu(medium, criterion=lagp.IECI(fmin=0.0)).local(x)
    compiled = _core._cache_size()

    _emu(medium, criterion=lagp.IECI(fmin=0.5)).local(x)
    assert _core._cache_size() == compiled

    _emu(medium, criterion=lagp.IECI(fmin=0.5, nugget=False)).local(x)
    assert _core._cache_size() == compiled + 1


def test_names_and_objects_are_the_same_scheme():
    assert lagp.Local(criterion="mspe", search="ray") == lagp.Local(
        criterion=lagp.MSPE(), search=lagp.Ray()
    )
    with pytest.raises(ValueError, match="unknown search"):
        lagp.Local(search="teleport")
