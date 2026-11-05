"""Hyperparameter fitting against laGP's mleGPsep / jmleGPsep.

The solvers are not laGP's, so the iterate trajectory differs. The contract tested here is
the one that matters: the same optimum, verified both by the parameter values and by our
objective being no worse than laGP's.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import lagpax as lagp
from lagpax.fit import _kernel_llik, fit_both, fit_kernel, fit_nugget
from lagpax.gp import dllik_nugget
from lagpax.optim import bracketed_newton
from lagpax.prior import Param, Prior

G = 1e-4
DRANGE = (1e-3, 10.0)
GRANGE = (float(np.sqrt(np.finfo(float).eps)), 1.0)

NEWGP = "gpi <- newGPsep(X, Z, d=d, g=g[1], dK=TRUE)"


def _gp(design, d0, g=G):
    X, Z = design
    return lagp.GP(jnp.array(X), jnp.array(Z), jnp.array(d0), g)


def _llik_at(design, d, g, dab=(0.0, 0.0), gab=(0.0, 0.0)):
    X, Z = design
    gp = lagp.GP(jnp.array(X), jnp.array(Z), jnp.array(np.atleast_1d(d)), float(g))
    return float(
        gp.log_likelihood(Prior(theta=Param(gamma=dab), nugget=Param(gamma=gab)))
    )


def _prior(
    dab=(0.0, 0.0), gab=(0.0, 0.0), drange=DRANGE, grange=GRANGE, learn=(True, True)
):
    """A theta-space Prior with laGP's (range, ab) arguments spelled as Params."""
    return Prior(
        theta=Param(lower=drange[0], upper=drange[1], gamma=dab, learn=learn[0]),
        nugget=Param(lower=grange[0], upper=grange[1], gamma=gab, learn=learn[1]),
    )


@pytest.mark.parametrize("design", ["small", "medium"])
def test_theta_fit_finds_the_same_optimum_as_r(R, request, design):
    data = request.getfixturevalue(design)
    X, Z = data
    d0 = np.full(X.shape[1], 0.5)
    dab = (1.5, 3.9)

    _, res = fit_kernel(_gp(data, d0), _prior(dab=dab))
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

    got = np.asarray(res.theta)
    np.testing.assert_allclose(got, want, rtol=2e-3, atol=1e-6)
    assert _llik_at(data, got, G, dab=dab) >= _llik_at(data, want, G, dab=dab) - 1e-6


def test_theta_fit_respects_the_box(R, wide):
    X, Z = wide
    d0 = np.full(X.shape[1], 0.4)
    lo, hi = 0.3, 0.6
    _, res = fit_kernel(_gp((X, Z), d0), _prior(drange=(lo, hi)))
    d = np.asarray(res.theta)
    assert np.all(d >= lo - 1e-12) and np.all(d <= hi + 1e-12)


def test_a_start_on_the_bound_is_simply_optimized(small):
    """laGP refuses to optimize from the lower bound and jumps into the box; lagpax just
    optimizes, so it reaches the optimum an interior start reaches."""
    X, Z = small
    lo, hi = 0.2, 5.0
    prior = _prior(drange=(lo, hi))
    _, edge = fit_kernel(_gp(small, np.full(X.shape[1], lo)), prior)
    _, inner = fit_kernel(_gp(small, np.full(X.shape[1], 1.0)), prior)
    assert bool(edge.converged) and int(edge.iterations) > 0
    assert _llik_at(small, edge.theta, G) == pytest.approx(
        _llik_at(small, inner.theta, G), abs=1e-6
    )


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
def test_nugget_fit_finds_the_same_optimum_as_r(R, request, design):
    data = request.getfixturevalue(design)
    X, Z = data
    d0 = np.full(X.shape[1], 0.5)
    gab = (1.5, 4.0)

    got = float(fit_nugget(_gp(data, d0, g=1e-2), _prior(gab=gab))[1].nugget)
    want = _r_mle_nug(R, X, Z, d0, gab)

    assert got == pytest.approx(want, rel=1e-4, abs=1e-10)
    assert _llik_at(data, d0, got, gab=gab) >= _llik_at(data, d0, want, gab=gab) - 1e-8


def test_nugget_fit_is_no_worse_than_r_on_a_noiseless_design(R, medium):
    """With no noise the optimum is pinned at gmin, where laGP stops early."""
    X, Z = medium
    d0 = np.full(X.shape[1], 0.5)
    gab = (1.5, 4.0)

    got = float(fit_nugget(_gp(medium, d0, g=1e-2), _prior(gab=gab))[1].nugget)
    want = _r_mle_nug(R, X, Z, d0, gab)

    assert GRANGE[0] <= got <= GRANGE[1]
    assert _llik_at(medium, d0, got, gab=gab) >= _llik_at(medium, d0, want, gab=gab)


def test_nugget_fit_gradient_vanishes_at_the_optimum(noisy):
    X, Z = noisy
    d0 = np.full(X.shape[1], 0.5)
    gab = (1.5, 4.0)
    gp, res = fit_nugget(_gp(noisy, d0, g=1e-2), _prior(gab=gab))
    assert GRANGE[0] < float(res.nugget) < GRANGE[1]
    assert abs(float(dllik_nugget(gp, res.nugget, ab=gab))) < 1e-4


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
def test_joint_fit_matches_r(R, request, design):
    data = request.getfixturevalue(design)
    X, Z = data
    d0 = np.full(X.shape[1], 0.5)
    dab, gab = (1.5, 3.9), (1.5, 4.0)
    m = X.shape[1]

    _, res = fit_both(_gp(data, d0), _prior(dab, gab))
    want = _r_jmle(R, X, Z, d0, dab, gab)
    d_want, g_want = want[:m], want[m]

    np.testing.assert_allclose(np.asarray(res.theta), d_want, rtol=1e-3, atol=1e-6)
    assert float(res.nugget) == pytest.approx(g_want, rel=5e-3, abs=1e-8)

    ours = _llik_at(data, np.asarray(res.theta), float(res.nugget), dab=dab, gab=gab)
    theirs = _llik_at(data, d_want, g_want, dab=dab, gab=gab)
    assert ours >= theirs - 1e-5 * max(1.0, abs(theirs))


def test_joint_fit_is_no_worse_than_r_on_a_noiseless_design(R, medium):
    X, Z = medium
    d0 = np.full(X.shape[1], 0.5)
    dab, gab = (1.5, 3.9), (1.5, 4.0)
    m = X.shape[1]

    _, res = fit_both(_gp(medium, d0), _prior(dab, gab))
    want = _r_jmle(R, X, Z, d0, dab, gab)

    ours = _llik_at(medium, np.asarray(res.theta), float(res.nugget), dab=dab, gab=gab)
    theirs = _llik_at(medium, want[:m], want[m], dab=dab, gab=gab)
    assert ours >= theirs - 1e-5 * max(1.0, abs(theirs))


@pytest.mark.parametrize(
    "d_est,g_est", [(False, False), (True, False), (False, True), (True, True)]
)
def test_fit_estimates_only_what_the_prior_marks(small, d_est, g_est):
    """FitInfo has the same fields whichever hyperparameters move, so it stays a pytree."""
    X, Z = small
    d0 = np.full(X.shape[1], 0.5)
    gp = _gp(small, d0)
    prior = _prior((1.5, 3.9), (1.5, 4.0), learn=(d_est, g_est))

    moved, out = lagp.fit(gp, prior)
    assert isinstance(out, lagp.FitInfo)
    assert np.allclose(np.asarray(out.theta), np.asarray(moved.theta))
    assert float(out.nugget) == float(moved.nugget)
    assert (not np.allclose(np.asarray(out.theta), d0)) == d_est
    assert (float(out.nugget) != G) == g_est


def test_bracketed_newton_on_a_scalar_root():
    root = bracketed_newton(
        lambda x: (jnp.cos(x), -jnp.sin(x)), jnp.array(1.0), 0.1, 3.0
    )
    assert float(root.x) == pytest.approx(np.pi / 2, abs=1e-9)


def _ell_prior(gamma=(0.0, 0.0), learn=(True, False)):
    """The same box as `_prior()`, expressed on l = sqrt(theta / 2)."""
    lo, hi = (float(np.sqrt(v / 2)) for v in DRANGE)
    return Prior(
        lengthscale=Param(lower=lo, upper=hi, gamma=gamma, learn=learn[0]),
        nugget=Param(lower=GRANGE[0], upper=GRANGE[1], learn=learn[1]),
    )


@pytest.mark.parametrize("design", ["small", "noisy"])
def test_without_a_prior_the_space_does_not_move_the_optimum(request, design):
    """The likelihood is invariant to reparameterization, so the MLE is the same point."""
    data = request.getfixturevalue(design)
    d0 = np.full(data[0].shape[1], 0.5)
    _, in_theta = fit_kernel(_gp(data, d0), _prior())
    _, in_ell = fit_kernel(_gp(data, d0), _ell_prior())
    np.testing.assert_allclose(
        np.asarray(in_ell.theta), np.asarray(in_theta.theta), rtol=2e-3
    )
    assert _llik_at(data, in_ell.theta, G) == pytest.approx(
        _llik_at(data, in_theta.theta, G), abs=1e-6
    )


def test_a_lengthscale_prior_is_a_map_in_lengthscale(noisy):
    """With a Gamma prior on l, the solution is stationary for llik + log p(l) in l."""
    import jax

    d0 = np.full(noisy[0].shape[1], 0.5)
    prior = _ell_prior(gamma=(1.5, 3.0))
    gp, res = fit_kernel(_gp(noisy, d0), prior)
    ell = res.lengthscale
    assert np.all(np.asarray(ell) > prior.lengthscale.lower)
    grad = jax.grad(lambda v: _kernel_llik(gp, v, gp.nugget, prior))(ell)
    assert np.max(np.abs(np.asarray(grad))) < 1e-3

    # and it is a different point from the theta-space MAP under a theta prior
    _, theta_map = fit_kernel(_gp(noisy, d0), _prior(dab=(1.5, 3.0)))
    assert not np.allclose(
        np.asarray(res.theta), np.asarray(theta_map.theta), rtol=1e-3
    )
