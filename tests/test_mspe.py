"""MSPE and Fisher criteria against laGP's mspeGP, fishGP and dmus2GP.

Both live in laGP's *isotropic* family, where the lengthscale is one number. A separable GP
with every `d_k` equal is that GP exactly, so parity is testable; the port's derivative is
the directional one along the all-ones direction, which is the same thing here.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import lagpax as lagp
from lagpax.criteria import _dpred, fisher_info

D = 0.6
G = 1e-3

ISO = "gpi <- newGP(X, Z, d=d[1], g=g[1], dK=TRUE); "


def _split(design, ncand=12, nref=4):
    X, Z = design
    n = len(X) - ncand - nref
    return X[:n], Z[:n], X[n : n + ncand] + 0.01, X[n + ncand :] + 0.02


def _gp(Xd, Zd, d=D):
    return lagp.GP(jnp.array(Xd), jnp.array(Zd), jnp.full(Xd.shape[1], d), G)


def _r(R, script, Xd, Zd, d=D, **data):
    return R(ISO + script, X=Xd, Z=Zd, d=np.array([d]), g=np.array([G]), **data)


def test_dmus2_matches_r(R, small):
    Xd, Zd, Xcand, _ = _split(small)
    mu, dmu, s2, ds2 = _dpred(_gp(Xd, Zd), jnp.array(Xcand))

    # as.matrix, because a data.frame comes back from rpy2 without usable column names
    want = np.asarray(
        _r(
            R,
            "out <- as.matrix(laGP:::dmus2GP(gpi, XX)); deleteGP(gpi); out",
            Xd,
            Zd,
            XX=Xcand,
        )
    )
    for got, col, rtol in (
        (mu, 0, 1e-10),
        (dmu, 1, 1e-9),
        (s2, 3, 1e-10),
        (ds2, 4, 1e-9),
    ):
        np.testing.assert_allclose(np.asarray(got), want[:, col], rtol=rtol)


def test_fisher_info_matches_the_analytic_second_derivative(R, small):
    """laGP's fishinfoGP is -d2llik from its hand-written dllikGP; ours is autodiff."""
    Xd, Zd, Xcand, _ = _split(small)
    gp = _gp(Xd, Zd)
    efi = np.asarray(
        _r(
            R,
            "out <- laGP:::fishGP(gpi, Xcand); deleteGP(gpi); out",
            Xd,
            Zd,
            Xcand=Xcand,
        )
    ).ravel()

    # fishGP adds the per-candidate terms on top of gp->F, so back them out to isolate it
    _, dmu, s2, ds2 = _dpred(gp, jnp.array(Xcand))
    got = np.asarray(fisher_info(gp))
    want = efi - np.asarray(0.5 * jnp.square(ds2 / s2) + jnp.square(dmu) / s2)
    np.testing.assert_allclose(np.full_like(want, got), want, rtol=1e-8)


def test_fisher_matches_r(R, small):
    Xd, Zd, Xcand, _ = _split(small)
    got = np.asarray(lagp.fisher(_gp(Xd, Zd), jnp.array(Xcand)))
    want = np.asarray(
        _r(
            R,
            "out <- laGP:::fishGP(gpi, Xcand); deleteGP(gpi); out",
            Xd,
            Zd,
            Xcand=Xcand,
        )
    ).ravel()
    np.testing.assert_allclose(got, want, rtol=1e-8)


def test_fisher_matches_r_on_a_wider_design(R, wide):
    X, Z = wide
    Xd, Zd, Xcand = X[:150], Z[:150], X[150:180]
    got = np.asarray(lagp.fisher(_gp(Xd, Zd), jnp.array(Xcand)))
    want = np.asarray(
        _r(
            R,
            "out <- laGP:::fishGP(gpi, Xcand); deleteGP(gpi); out",
            Xd,
            Zd,
            Xcand=Xcand,
        )
    ).ravel()
    np.testing.assert_allclose(got, want, rtol=1e-8)


@pytest.mark.parametrize("fi", [True, False])
def test_mspe_matches_r(R, small, fi):
    Xd, Zd, Xcand, Xref = _split(small)
    got = np.asarray(lagp.MSPE(fi=fi)(_gp(Xd, Zd), jnp.array(Xcand), jnp.array(Xref)))
    want = np.asarray(
        _r(
            R,
            f"out <- laGP:::mspeGP(gpi, Xcand, Xref, fi={'TRUE' if fi else 'FALSE'}); "
            "deleteGP(gpi); out",
            Xd,
            Zd,
            Xcand=Xcand,
            Xref=Xref,
        )
    ).ravel()
    np.testing.assert_allclose(got, want, rtol=1e-8)


def test_mspe_matches_r_with_one_reference(R, small):
    Xd, Zd, Xcand, Xref = _split(small, nref=1)
    got = np.asarray(lagp.MSPE()(_gp(Xd, Zd), jnp.array(Xcand), jnp.array(Xref)))
    want = np.asarray(
        _r(
            R,
            "out <- laGP:::mspeGP(gpi, Xcand, Xref, fi=TRUE); deleteGP(gpi); out",
            Xd,
            Zd,
            Xcand=Xcand,
            Xref=Xref,
        )
    ).ravel()
    np.testing.assert_allclose(got, want, rtol=1e-8)


def test_mspe_matches_r_on_a_wider_design(R, wide):
    X, Z = wide
    Xd, Zd, Xcand, Xref = X[:150], Z[:150], X[150:180], X[180:] + 0.01
    got = np.asarray(lagp.MSPE()(_gp(Xd, Zd), jnp.array(Xcand), jnp.array(Xref)))
    want = np.asarray(
        _r(
            R,
            "out <- laGP:::mspeGP(gpi, Xcand, Xref, fi=TRUE); deleteGP(gpi); out",
            Xd,
            Zd,
            Xcand=Xcand,
            Xref=Xref,
        )
    ).ravel()
    np.testing.assert_allclose(got, want, rtol=1e-8)


def test_mspe_is_minimized_and_fisher_is_maximized():
    assert lagp.MSPE().minimize is True
    assert getattr(lagp.fisher, "minimize", False) is False


def test_mspe_tracks_alc_when_the_fisher_term_is_off(small):
    """Without `fi`, mspe is an affine, decreasing function of alc -- so it reranks nothing."""
    Xd, Zd, Xcand, Xref = _split(small)
    gp = _gp(Xd, Zd)
    alc = np.asarray(lagp.alc(gp, jnp.array(Xcand), jnp.array(Xref)))
    got = np.asarray(lagp.MSPE(fi=False)(gp, jnp.array(Xcand), jnp.array(Xref)))
    assert np.argmin(got) == np.argmax(alc)


def test_the_criteria_jit_and_differentiate(small):
    Xd, Zd, Xcand, Xref = _split(small)
    gp = _gp(Xd, Zd)
    Xcand, Xref = jnp.array(Xcand), jnp.array(Xref)

    for criterion in (lagp.MSPE(), lagp.fisher):
        got = criterion(gp, Xcand, Xref)
        # 1e-9, not exact: these run a Hessian through a Cholesky, and jit reassociates it
        np.testing.assert_allclose(
            np.asarray(jax.jit(criterion)(gp, Xcand, Xref)), np.asarray(got), rtol=1e-9
        )
        grad = jax.grad(lambda X: jnp.sum(criterion(gp, X, Xref)))(Xcand)
        assert np.all(np.isfinite(np.asarray(grad)))


def test_the_criteria_drive_the_local_scheme(small):
    X, Z = small
    XX = X[:5] + 0.01
    for criterion in ("mspe", "fisher"):
        out = lagp.agp(X, Z, XX, criterion=criterion, start=6, end=12, close=25)
        assert np.all(np.isfinite(np.asarray(out.mean)))
        assert out.Xi.shape == (len(XX), 12)


def test_mspe_named_lookup_uses_lagps_default():
    from lagpax.criteria import as_criterion

    Xd = np.random.default_rng(0).random((20, 2))
    Zd = Xd.sum(-1)
    gp = _gp(Xd, Zd)
    Xcand, Xref = jnp.array(Xd[:5] + 0.01), jnp.array(Xd[5:8] + 0.02)
    np.testing.assert_allclose(
        np.asarray(as_criterion("mspe")(gp, Xcand, Xref)),
        np.asarray(lagp.MSPE(fi=True)(gp, Xcand, Xref)),
        rtol=1e-12,
    )
