"""Hyperparameter estimation against laGP's mleGPsep / jmleGPsep.

Our L-BFGS is not R's Fortran L-BFGS-B, so the iterate trajectory differs. The
contract tested here is the one that matters: the same optimum, verified by both
the parameter values and by our objective being no worse than laGP's.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import lagpax as lagp
from lagpax.gp import dllik_g
from lagpax.mle import jmle, mle_d, mle_g
from lagpax.optim import bracketed_newton, lbfgsb
from lagpax.prior import Prior, Range

G = 1e-4
DRANGE = (1e-3, 10.0)
GRANGE = (float(np.sqrt(np.finfo(float).eps)), 1.0)

NEWGP = "gpi <- newGPsep(X, Z, d=d, g=g[1], dK=TRUE)"


def _gp(design, d0, g=G):
    X, Z = design
    return lagp.GP(jnp.array(X), jnp.array(Z), jnp.array(d0), g)


def _llik_at(design, d, g, dab=(0.0, 0.0), gab=(0.0, 0.0)):
    X, Z = design
    gp = lagp.GP(
        jnp.array(X), jnp.array(Z), jnp.array(np.atleast_1d(d)), float(g)
    )
    return float(gp.llik(Prior(d=Range(ab=dab), g=Range(ab=gab))))


@pytest.mark.parametrize("design", ["small", "medium"])
def test_mle_d_finds_the_same_optimum_as_r(R, request, design):
    data = request.getfixturevalue(design)
    X, Z = data
    d0 = np.full(X.shape[1], 0.5)
    dab = (1.5, 3.9)

    res = mle_d(_gp(data, d0), DRANGE[0], DRANGE[1], ab=dab)
    want = np.asarray(
        R(
            f"{NEWGP}; out <- mleGPsep(gpi, param='d', tmin=tmin, tmax=tmax, ab=ab); "
            "deleteGPsep(gpi); out$d",
            X=X,
            Z=Z,
            d=d0,
            g=np.array([G]),
            tmin=np.array([DRANGE[0]]),
            tmax=np.array([DRANGE[1]]),
            ab=np.array(dab),
        )
    ).ravel()

    got = np.asarray(res.d)
    np.testing.assert_allclose(got, want, rtol=2e-3, atol=1e-6)
    assert _llik_at(data, got, G, dab=dab) >= _llik_at(data, want, G, dab=dab) - 1e-6


def test_mle_d_respects_the_box(R, wide):
    X, Z = wide
    d0 = np.full(X.shape[1], 0.4)
    lo, hi = 0.3, 0.6
    res = mle_d(_gp((X, Z), d0), lo, hi)
    d = np.asarray(res.d)
    assert np.all(d >= lo - 1e-12) and np.all(d <= hi + 1e-12)


def test_mle_d_on_the_lower_boundary_resets_like_lagp(small):
    """laGP refuses to optimize from the boundary and jumps to 0.9*min + 0.1*max."""
    X, Z = small
    lo, hi = 0.2, 5.0
    res = mle_d(_gp(small, np.full(X.shape[1], lo)), lo, hi)
    np.testing.assert_allclose(np.asarray(res.d), 0.9 * lo + 0.1 * hi, rtol=1e-12)
    assert int(res.conv) == 102
    assert int(res.its) == 0


def _r_mle_nug(R, X, Z, d0, gab):
    return R.scalar(
        f"{NEWGP}; out <- mleGPsep(gpi, param='g', tmin=tmin, tmax=tmax, ab=ab); "
        "deleteGPsep(gpi); out$g",
        X=X,
        Z=Z,
        d=d0,
        g=np.array([1e-2]),
        tmin=np.array([GRANGE[0]]),
        tmax=np.array([GRANGE[1]]),
        ab=np.array(gab),
    )


@pytest.mark.parametrize("design", ["small", "noisy"])
def test_mle_nug_finds_the_same_optimum_as_r(R, request, design):
    data = request.getfixturevalue(design)
    X, Z = data
    d0 = np.full(X.shape[1], 0.5)
    gab = (1.5, 4.0)

    got = float(mle_g(_gp(data, d0, g=1e-2), *GRANGE, ab=gab).g)
    want = _r_mle_nug(R, X, Z, d0, gab)

    assert got == pytest.approx(want, rel=1e-4, abs=1e-10)
    assert _llik_at(data, d0, got, gab=gab) >= _llik_at(data, d0, want, gab=gab) - 1e-8


def test_mle_nug_is_no_worse_than_r_on_a_noiseless_design(R, medium):
    """With no noise the optimum is pinned at gmin, where laGP stops early."""
    X, Z = medium
    d0 = np.full(X.shape[1], 0.5)
    gab = (1.5, 4.0)

    got = float(mle_g(_gp(medium, d0, g=1e-2), *GRANGE, ab=gab).g)
    want = _r_mle_nug(R, X, Z, d0, gab)

    assert GRANGE[0] <= got <= GRANGE[1]
    assert _llik_at(medium, d0, got, gab=gab) >= _llik_at(medium, d0, want, gab=gab)


def test_mle_nug_gradient_vanishes_at_the_optimum(noisy):
    X, Z = noisy
    d0 = np.full(X.shape[1], 0.5)
    gab = (1.5, 4.0)
    res = mle_g(_gp(noisy, d0, g=1e-2), *GRANGE, ab=gab)
    assert GRANGE[0] < float(res.g) < GRANGE[1]
    assert abs(float(dllik_g(res.gp, res.g, gab=gab))) < 1e-4


def _r_jmle(R, X, Z, d0, dab, gab):
    return np.asarray(
        R(
            f"{NEWGP}; out <- jmleGPsep(gpi, drange=drange, grange=grange, "
            "dab=dab, gab=gab); deleteGPsep(gpi); as.numeric(out[1,])",
            X=X,
            Z=Z,
            d=d0,
            g=np.array([G]),
            drange=np.array(DRANGE),
            grange=np.array(GRANGE),
            dab=np.array(dab),
            gab=np.array(gab),
        )
    ).ravel()


@pytest.mark.parametrize("design", ["small", "noisy"])
def test_jmle_matches_r(R, request, design):
    data = request.getfixturevalue(design)
    X, Z = data
    d0 = np.full(X.shape[1], 0.5)
    dab, gab = (1.5, 3.9), (1.5, 4.0)
    m = X.shape[1]

    _, res = jmle(_gp(data, d0), DRANGE, GRANGE, dab, gab)
    want = _r_jmle(R, X, Z, d0, dab, gab)
    d_want, g_want = want[:m], want[m]

    np.testing.assert_allclose(np.asarray(res.d), d_want, rtol=1e-3, atol=1e-6)
    assert float(res.g) == pytest.approx(g_want, rel=5e-3, abs=1e-8)

    ours = _llik_at(data, np.asarray(res.d), float(res.g), dab=dab, gab=gab)
    theirs = _llik_at(data, d_want, g_want, dab=dab, gab=gab)
    assert ours >= theirs - 1e-5 * max(1.0, abs(theirs))


def test_jmle_is_no_worse_than_r_on_a_noiseless_design(R, medium):
    X, Z = medium
    d0 = np.full(X.shape[1], 0.5)
    dab, gab = (1.5, 3.9), (1.5, 4.0)
    m = X.shape[1]

    _, res = jmle(_gp(medium, d0), DRANGE, GRANGE, dab, gab)
    want = _r_jmle(R, X, Z, d0, dab, gab)

    ours = _llik_at(medium, np.asarray(res.d), float(res.g), dab=dab, gab=gab)
    theirs = _llik_at(medium, want[:m], want[m], dab=dab, gab=gab)
    assert ours >= theirs - 1e-5 * max(1.0, abs(theirs))


@pytest.mark.parametrize("d_est,g_est", [(False, False), (True, False), (False, True), (True, True)])
def test_fit_estimates_only_what_the_prior_marks(small, d_est, g_est):
    """Fit has the same four fields whichever hyperparameters move, so it stays a pytree."""
    X, Z = small
    d0 = np.full(X.shape[1], 0.5)
    gp = _gp(small, d0)
    prior = Prior(
        d=Range(min=DRANGE[0], max=DRANGE[1], ab=(1.5, 3.9), estimate=d_est),
        g=Range(min=GRANGE[0], max=GRANGE[1], ab=(1.5, 4.0), estimate=g_est),
    )

    moved, out = lagp.fit(gp, prior)
    assert isinstance(out, lagp.Fit)
    assert np.allclose(np.asarray(out.d), np.asarray(moved.d))
    assert float(out.g) == float(moved.g)
    assert (not np.allclose(np.asarray(out.d), d0)) == d_est
    assert (float(out.g) != G) == g_est


def test_lbfgsb_on_a_bounded_quadratic():
    """The optimum sits outside the box, so every coordinate must clamp."""
    lo = jnp.array([0.0, 0.0])
    hi = jnp.array([1.0, 1.0])
    target = jnp.array([2.0, -1.0])
    res = lbfgsb(
        lambda x: jnp.sum((x - target) ** 2), jnp.array([0.5, 0.5]), lo, hi
    )
    np.testing.assert_allclose(np.asarray(res.x), [1.0, 0.0], atol=1e-8)


def test_bracketed_newton_on_a_scalar_root():
    root = bracketed_newton(
        lambda x: jnp.cos(x), lambda x: -jnp.sin(x), jnp.array(1.0), 0.1, 3.0
    )
    assert float(root.x) == pytest.approx(np.pi / 2, abs=1e-9)
