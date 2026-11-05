"""ALC criterion against laGP's alcGPsep."""

import jax
import jax.numpy as jnp
import numpy as np

import lagpax as lagp
from helpers import make_gp

D = np.array([0.3, 0.7, 1.1])
G = 1e-3

ALC = (
    "gpi <- newGPsep(X, Z, d=d, g=g[1], dK=TRUE); "
    "out <- alcGPsep(gpi, Xcand, Xref); deleteGPsep(gpi); out"
)


def _split(small, ncand=12, nref=4):
    X, Z = small
    n = len(X) - ncand - nref
    return X[:n], Z[:n], X[n : n + ncand] + 0.01, X[n + ncand :] + 0.02


def test_alc_matches_r_single_reference(R, small):
    Xd, Zd, Xcand, Xref = _split(small, nref=1)
    gp = make_gp(jnp.array(Xd), jnp.array(Zd), jnp.array(D), G)
    got = np.asarray(lagp.ActiveLearningCohn()(gp, jnp.array(Xcand), jnp.array(Xref)))
    want = np.asarray(
        R(ALC, X=Xd, Z=Zd, d=D, g=np.array([G]), Xcand=Xcand, Xref=Xref)
    ).ravel()
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-14)


def test_alc_matches_r_multiple_references(R, small):
    Xd, Zd, Xcand, Xref = _split(small)
    gp = make_gp(jnp.array(Xd), jnp.array(Zd), jnp.array(D), G)
    got = np.asarray(lagp.ActiveLearningCohn()(gp, jnp.array(Xcand), jnp.array(Xref)))
    want = np.asarray(
        R(ALC, X=Xd, Z=Zd, d=D, g=np.array([G]), Xcand=Xcand, Xref=Xref)
    ).ravel()
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-14)


def test_alc_matches_r_on_a_wider_design(R, wide):
    X, Z = wide
    d = np.linspace(0.4, 1.6, X.shape[1])
    Xd, Zd, Xcand, Xref = X[:150], Z[:150], X[150:180], X[180:] + 0.01
    gp = make_gp(jnp.array(Xd), jnp.array(Zd), jnp.array(d), G)
    got = np.asarray(lagp.ActiveLearningCohn()(gp, jnp.array(Xcand), jnp.array(Xref)))
    want = np.asarray(
        R(ALC, X=Xd, Z=Zd, d=d, g=np.array([G]), Xcand=Xcand, Xref=Xref)
    ).ravel()
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-14)


def test_alc_is_nonnegative_and_finite(small):
    Xd, Zd, Xcand, Xref = _split(small)
    gp = make_gp(jnp.array(Xd), jnp.array(Zd), jnp.array(D), G)
    alc = np.asarray(lagp.ActiveLearningCohn()(gp, jnp.array(Xcand), jnp.array(Xref)))
    assert np.all(np.isfinite(alc)) and np.all(alc >= 0)


def test_alc_prefers_candidates_near_the_reference(small):
    Xd, Zd, _, Xref = _split(small, nref=1)
    gp = make_gp(jnp.array(Xd), jnp.array(Zd), jnp.array(D), G)
    Xcand = np.vstack([Xref[0] + 0.01, Xref[0] + 5.0])
    near, far = np.asarray(
        lagp.ActiveLearningCohn()(gp, jnp.array(Xcand), jnp.array(Xref))
    )
    assert near > far


def test_alc_ignores_masked_rows(small):
    Xd, Zd, Xcand, Xref = _split(small)
    k = len(Xd) - 5
    mask = np.arange(len(Xd)) < k
    padded = make_gp(jnp.array(Xd), jnp.array(Zd), jnp.array(D), G, jnp.array(mask))
    exact = make_gp(jnp.array(Xd[:k]), jnp.array(Zd[:k]), jnp.array(D), G)
    a = lagp.ActiveLearningCohn()(padded, jnp.array(Xcand), jnp.array(Xref))
    b = lagp.ActiveLearningCohn()(exact, jnp.array(Xcand), jnp.array(Xref))
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-10, atol=1e-15)


def test_alc_is_jittable_and_vmappable(small):
    Xd, Zd, Xcand, Xref = _split(small)
    gp = make_gp(jnp.array(Xd), jnp.array(Zd), jnp.array(D), G)
    alc = lagp.ActiveLearningCohn()
    single = jax.jit(alc)(gp, jnp.array(Xcand), jnp.array(Xref))
    batched = jax.vmap(lambda x: alc(gp, x[None, :], jnp.array(Xref)))(jnp.array(Xcand))
    np.testing.assert_allclose(
        np.asarray(single), np.asarray(batched).ravel(), rtol=1e-9
    )
