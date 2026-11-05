"""GP likelihood, gradients and predictive equations against laGP."""

import jax.numpy as jnp
import numpy as np
import pytest

import lagpax as lagp
from lagpax.gp import dllik_nugget, dllik_theta
from lagpax.prior import Param, Prior

D = np.array([0.3, 0.7, 1.1])
G = 1e-3

NEWGP = "gpi <- newGPsep(X, Z, d=d, g=g[1], dK=TRUE)"


def _prior(dab=(0.0, 0.0), gab=(0.0, 0.0)) -> Prior:
    """A Prior carrying only the Gamma (shape, rate) pairs, which is all the likelihood reads."""
    return Prior(theta=Param(gamma=dab), nugget=Param(gamma=gab))


def _gp(small):
    X, Z = small
    return lagp.GP(jnp.array(X), jnp.array(Z), jnp.array(D), G)


def test_llik_matches_r(R, small):
    X, Z = small
    got = float(_gp(small).log_likelihood())
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
    got = float(_gp(small).log_likelihood(_prior(tuple(dab), tuple(gab))))
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
    plain = float(_gp(small).log_likelihood())
    zeroed = float(_gp(small).log_likelihood(_prior()))
    assert plain == zeroed


def test_dllik_theta_matches_r(R, small):
    X, Z = small
    gp = _gp(small)
    got = np.asarray(dllik_theta(gp, gp.theta))
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


def test_dllik_theta_with_prior_matches_r(R, small):
    X, Z = small
    gp, ab = _gp(small), np.array([1.5, 3.9])
    got = np.asarray(dllik_theta(gp, gp.theta, tuple(ab)))
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


def test_dllik_nugget_matches_r(R, small):
    X, Z = small
    gp = _gp(small)
    got = float(dllik_nugget(gp, gp.nugget))
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
    p = _gp(small).predict(jnp.array(XX))
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
    # laGP returns the Student-t scale; lagpax returns the variance, scale * df / (df - 2)
    df = want[2 * nn]
    np.testing.assert_allclose(
        np.asarray(p.variance), want[nn : 2 * nn] * df / (df - 2), rtol=1e-8, atol=1e-11
    )
    assert float(p.dof) == pytest.approx(df)
    assert float(p.log_likelihood) == pytest.approx(want[2 * nn + 1], rel=1e-12)


def test_pred_full_matches_r(R, small):
    X, Z = small
    XX = np.asarray(X[:6] + 0.05)
    p = _gp(small).predict(jnp.array(XX), full_cov=True)
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
    df = float(p.dof)
    np.testing.assert_allclose(
        np.asarray(p.covariance), Sigma * df / (df - 2), rtol=1e-8, atol=1e-11
    )
    np.testing.assert_array_equal(
        np.asarray(p.variance), np.diag(np.asarray(p.covariance))
    )


def test_pred_without_nugget_uses_sdeps(small):
    X, Z = small
    gp = _gp(small)
    XX = jnp.array(X[:3] + 0.05)
    with_nug = gp.predict(XX).variance
    without = gp.predict(XX, nugget=False).variance
    assert np.all(np.asarray(without) < np.asarray(with_nug))


def test_mask_matches_a_smaller_design(small):
    """A padded design with masked rows must behave exactly like the unpadded one."""
    X, Z = small
    k = 25
    mask = np.arange(len(X)) < k
    padded = lagp.GP(jnp.array(X), jnp.array(Z), jnp.array(D), G, jnp.array(mask))
    exact = lagp.GP(jnp.array(X[:k]), jnp.array(Z[:k]), jnp.array(D), G)

    assert float(padded.n) == k
    assert float(padded.phi) == pytest.approx(float(exact.phi), rel=1e-12)
    assert float(padded.ldetK) == pytest.approx(float(exact.ldetK), rel=1e-12)
    assert float(padded.log_likelihood()) == pytest.approx(
        float(exact.log_likelihood()), rel=1e-12
    )

    XX = jnp.array(X[:5] + 0.05)
    a, b = padded.predict(XX), exact.predict(XX)
    np.testing.assert_allclose(
        np.asarray(a.mean), np.asarray(b.mean), rtol=1e-11, atol=1e-13
    )
    np.testing.assert_allclose(
        np.asarray(a.variance), np.asarray(b.variance), rtol=1e-8, atol=1e-12
    )


def test_append_matches_rebuild(small):
    X, Z = small
    gp = lagp.GP(jnp.array(X[:30]), jnp.array(Z[:30]), jnp.array(D), G)
    appended = gp.append(jnp.array(X[30:]), jnp.array(Z[30:]))
    rebuilt = _gp(small)
    assert float(appended.log_likelihood()) == pytest.approx(
        float(rebuilt.log_likelihood()), rel=1e-12
    )


def test_replace_rebuilds_cached_quantities(small):
    X, Z = small
    gp = _gp(small)
    moved = gp.replace(theta=jnp.array([0.5, 0.5, 0.5]), nugget=1e-2)
    direct = lagp.GP(jnp.array(X), jnp.array(Z), jnp.array([0.5, 0.5, 0.5]), 1e-2)
    assert float(moved.phi) == pytest.approx(float(direct.phi), rel=1e-14)
    assert float(moved.ldetK) == pytest.approx(float(direct.ldetK), rel=1e-14)
