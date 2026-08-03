"""GPsep likelihood, gradients and predictive equations against laGP."""

import jax.numpy as jnp
import numpy as np
import pytest

import lagpax as lagp

D = np.array([0.3, 0.7, 1.1])
G = 1e-3

NEWGP = "gpi <- newGPsep(X, Z, d=d, g=g[1], dK=TRUE)"


def _gp(small):
    X, Z = small
    return lagp.new_gpsep(jnp.array(X), jnp.array(Z), jnp.array(D), G)


def test_llik_matches_r(R, small):
    X, Z = small
    got = float(lagp.llik_gpsep(_gp(small)))
    want = R.scalar(
        f"{NEWGP}; out <- llikGPsep(gpi); deleteGPsep(gpi); out",
        X=X,
        Z=Z,
        d=D,
        g=np.array([G]),
    )
    assert got == pytest.approx(want, rel=1e-12, abs=1e-12)


def test_llik_with_priors_matches_r(R, small):
    X, Z = small
    dab, gab = np.array([1.5, 3.9]), np.array([2.0, 5.0])
    got = float(lagp.llik_gpsep(_gp(small), dab=tuple(dab), gab=tuple(gab)))
    want = R.scalar(
        f"{NEWGP}; out <- llikGPsep(gpi, dab=dab, gab=gab); deleteGPsep(gpi); out",
        X=X,
        Z=Z,
        d=D,
        g=np.array([G]),
        dab=dab,
        gab=gab,
    )
    assert got == pytest.approx(want, rel=1e-12, abs=1e-12)


def test_llik_ignores_priors_when_ab_is_zero(R, small):
    X, Z = small
    plain = float(lagp.llik_gpsep(_gp(small)))
    zeroed = float(lagp.llik_gpsep(_gp(small), dab=(0.0, 0.0), gab=(0.0, 0.0)))
    assert plain == zeroed


def test_dllik_d_matches_r(R, small):
    X, Z = small
    gp = _gp(small)
    got = np.asarray(lagp.dllik_gpsep_d(gp, gp.d, (0.0, 0.0)))
    want = np.asarray(
        R(
            f"{NEWGP}; out <- laGP:::dllikGPsep(gpi, param='d'); deleteGPsep(gpi); out",
            X=X,
            Z=Z,
            d=D,
            g=np.array([G]),
        )
    ).ravel()
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-10)


def test_dllik_d_with_prior_matches_r(R, small):
    X, Z = small
    gp, ab = _gp(small), np.array([1.5, 3.9])
    got = np.asarray(lagp.dllik_gpsep_d(gp, gp.d, tuple(ab)))
    want = np.asarray(
        R(
            f"{NEWGP}; out <- laGP:::dllikGPsep(gpi, param='d', ab=ab); deleteGPsep(gpi); out",
            X=X,
            Z=Z,
            d=D,
            g=np.array([G]),
            ab=ab,
        )
    ).ravel()
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-10)


def test_dllik_g_matches_r(R, small):
    X, Z = small
    gp = _gp(small)
    got = float(lagp.dllik_gpsep_g(gp, gp.g, (0.0, 0.0)))
    want = R.scalar(
        f"{NEWGP}; out <- laGP:::dllikGPsep(gpi, param='g'); deleteGPsep(gpi); out",
        X=X,
        Z=Z,
        d=D,
        g=np.array([G]),
    )
    assert got == pytest.approx(want, rel=1e-8, abs=1e-10)


def test_pred_lite_matches_r(R, small):
    X, Z = small
    XX = np.asarray(X[:6] + 0.05)
    p = lagp.pred_gpsep(_gp(small), jnp.array(XX), lite=True)
    want = R(
        f"{NEWGP}; out <- predGPsep(gpi, XX, lite=TRUE); deleteGPsep(gpi); "
        "c(out$mean, out$s2, out$df, out$llik)",
        X=X,
        Z=Z,
        d=D,
        g=np.array([G]),
        XX=XX,
    )
    want = np.asarray(want).ravel()
    nn = len(XX)
    np.testing.assert_allclose(np.asarray(p.mean), want[:nn], rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(
        np.asarray(p.s2), want[nn : 2 * nn], rtol=1e-8, atol=1e-11
    )
    assert float(p.df) == pytest.approx(want[2 * nn])
    assert float(p.llik) == pytest.approx(want[2 * nn + 1], rel=1e-12)


def test_pred_full_matches_r(R, small):
    X, Z = small
    XX = np.asarray(X[:6] + 0.05)
    p = lagp.pred_gpsep(_gp(small), jnp.array(XX), lite=False)
    mean = np.asarray(
        R(
            f"{NEWGP}; out <- predGPsep(gpi, XX, lite=FALSE); deleteGPsep(gpi); out$mean",
            X=X,
            Z=Z,
            d=D,
            g=np.array([G]),
            XX=XX,
        )
    ).ravel()
    Sigma = np.asarray(
        R(
            f"{NEWGP}; out <- predGPsep(gpi, XX, lite=FALSE); deleteGPsep(gpi); out$Sigma",
            X=X,
            Z=Z,
            d=D,
            g=np.array([G]),
            XX=XX,
        )
    )
    np.testing.assert_allclose(np.asarray(p.mean), mean, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(np.asarray(p.s2), Sigma, rtol=1e-8, atol=1e-11)


def test_pred_nonug_uses_sdeps(small):
    X, Z = small
    gp = _gp(small)
    XX = jnp.array(X[:3] + 0.05)
    with_nug = lagp.pred_gpsep(gp, XX, lite=True).s2
    without = lagp.pred_gpsep(gp, XX, lite=True, nonug=True).s2
    assert np.all(np.asarray(without) < np.asarray(with_nug))


def test_mask_matches_a_smaller_design(small):
    """A padded design with masked rows must behave exactly like the unpadded one."""
    X, Z = small
    k = 25
    mask = np.arange(len(X)) < k
    padded = lagp.new_gpsep(
        jnp.array(X), jnp.array(Z), jnp.array(D), G, jnp.array(mask)
    )
    exact = lagp.new_gpsep(jnp.array(X[:k]), jnp.array(Z[:k]), jnp.array(D), G)

    assert float(padded.n) == k
    assert float(padded.phi) == pytest.approx(float(exact.phi), rel=1e-12)
    assert float(padded.ldetK) == pytest.approx(float(exact.ldetK), rel=1e-12)
    assert float(lagp.llik_gpsep(padded)) == pytest.approx(
        float(lagp.llik_gpsep(exact)), rel=1e-12
    )

    XX = jnp.array(X[:5] + 0.05)
    a, b = lagp.pred_gpsep(padded, XX), lagp.pred_gpsep(exact, XX)
    np.testing.assert_allclose(
        np.asarray(a.mean), np.asarray(b.mean), rtol=1e-11, atol=1e-13
    )
    np.testing.assert_allclose(
        np.asarray(a.s2), np.asarray(b.s2), rtol=1e-8, atol=1e-12
    )


def test_update_matches_rebuild(small):
    X, Z = small
    gp = lagp.new_gpsep(jnp.array(X[:30]), jnp.array(Z[:30]), jnp.array(D), G)
    updated = lagp.update_gpsep(gp, jnp.array(X[30:]), jnp.array(Z[30:]))
    rebuilt = _gp(small)
    assert float(lagp.llik_gpsep(updated)) == pytest.approx(
        float(lagp.llik_gpsep(rebuilt)), rel=1e-12
    )


def test_newparams_rebuilds_cached_quantities(small):
    X, Z = small
    gp = _gp(small)
    moved = lagp.newparams_gpsep(gp, d=jnp.array([0.5, 0.5, 0.5]), g=1e-2)
    direct = lagp.new_gpsep(
        jnp.array(X), jnp.array(Z), jnp.array([0.5, 0.5, 0.5]), 1e-2
    )
    assert float(moved.phi) == pytest.approx(float(direct.phi), rel=1e-14)
    assert float(moved.ldetK) == pytest.approx(float(direct.ldetK), rel=1e-14)
