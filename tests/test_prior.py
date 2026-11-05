"""Default priors (laGP's darg / garg / Igamma.inv) against laGP's prior.R."""

import jax
import numpy as np
import pytest
from scipy.special import gamma as sp_gamma

from lagpax.prior import Param, Prior, darg, garg, get_ds, igamma_inv

FIELDS = "c(out$start, out$min, out$max, as.numeric(out$mle), out$ab)"


def _r_arg(R, call: str, **data) -> np.ndarray:
    return np.asarray(R(f"out <- {call}; {FIELDS}", **data)).ravel()


def _assert_darg(got: Param, want: np.ndarray, nstart: int = 1):
    start, rest = want[:nstart], want[nstart:]
    np.testing.assert_allclose(np.atleast_1d(np.asarray(got.init)), start, rtol=1e-13)
    assert got.lower == pytest.approx(rest[0], rel=1e-13)
    assert got.upper == pytest.approx(rest[1], rel=1e-13)
    assert float(got.learn) == rest[2]
    np.testing.assert_allclose(np.asarray(got.gamma), rest[3:5], rtol=1e-10)


@pytest.mark.parametrize("a", [0.7, 1.5, 2.0, 5.0])
@pytest.mark.parametrize("frac", [0.05, 0.5, 0.95])
@pytest.mark.parametrize("lower", [0, 1])
def test_igamma_inv_matches_r(R, a, frac, lower):
    """laGP always calls this as Igamma.inv(a, frac*gamma(a), lower, 0)."""
    got = igamma_inv(a, frac * sp_gamma(a), lower=bool(lower))
    want = R.scalar(f"laGP:::Igamma.inv({a}, {frac}*gamma({a}), {lower}, 0)")
    assert got == pytest.approx(want, rel=1e-9)


def test_get_ds_matches_r(R, medium):
    X, _ = medium
    start, dmin, dmax = get_ds(X)
    want = np.asarray(
        R("Ds <- laGP:::getDs(X); c(Ds$start, Ds$min, Ds$max)", X=X)
    ).ravel()
    np.testing.assert_allclose([start, dmin, dmax], want, rtol=1e-13)


@pytest.mark.parametrize("design", ["small", "medium", "wide"])
def test_darg_default_matches_r(R, request, design):
    X, _ = request.getfixturevalue(design)
    _assert_darg(darg(None, X), _r_arg(R, "darg(NULL, X)", X=X))


def test_darg_scalar_start_matches_r(R, medium):
    X, _ = medium
    _assert_darg(darg(0.5, X), _r_arg(R, "darg(0.5, X)", X=X))


def test_darg_no_mle_matches_r(R, medium):
    X, _ = medium
    _assert_darg(
        darg(Param(init=0.5, learn=False), X),
        _r_arg(R, "darg(list(start=0.5, mle=FALSE), X)", X=X),
    )


def test_darg_explicit_box_matches_r(R, medium):
    X, _ = medium
    _assert_darg(
        darg(Param(lower=1e-4, upper=3.0), X),
        _r_arg(R, "darg(list(min=1e-4, max=3), X)", X=X),
    )


def test_darg_vector_start(R, wide):
    X, _ = wide
    start = np.linspace(0.2, 0.9, X.shape[1])
    got = darg(Param(init=start, upper=4.0, lower=1e-3), X)
    want = _r_arg(R, "darg(list(start=start, max=4, min=1e-3), X)", X=X, start=start)
    _assert_darg(got, want, nstart=len(start))


@pytest.mark.parametrize("design", ["small", "medium"])
def test_garg_default_matches_r(R, request, design):
    _, Z = request.getfixturevalue(design)
    _assert_darg(garg(None, Z), _r_arg(R, "garg(NULL, Z)", Z=Z))


def test_garg_scalar_matches_r(R, medium):
    _, Z = medium
    _assert_darg(garg(1 / 10000, Z), _r_arg(R, "garg(1/10000, Z)", Z=Z))


def test_garg_mle_matches_r(R, medium):
    _, Z = medium
    _assert_darg(garg(Param(learn=True), Z), _r_arg(R, "garg(list(mle=TRUE), Z)", Z=Z))


def test_darg_rejects_start_outside_box(medium):
    X, _ = medium
    with pytest.raises(ValueError):
        darg(Param(init=10.0, lower=0.1, upper=1.0), X)


def test_get_ds_subsamples_deterministically_with_a_key(small):
    X, _ = small
    a = get_ds(X, sample=20, key=jax.random.key(0))
    b = get_ds(X, sample=20, key=jax.random.key(0))
    assert a == b


def test_subsampling_without_a_key_is_an_error_not_a_hidden_draw(small):
    """laGP draws from R's global RNG; lagpax has none, so the key must be explicit."""
    X, _ = small
    get_ds(X, sample=len(X))  # no subsample, no key needed
    with pytest.raises(ValueError, match="key=jax.random.key"):
        get_ds(X, sample=20)


def test_from_data_is_darg_and_garg(medium):
    X, Z = medium
    prior = Prior.from_data(X, Z)
    assert prior.space == "theta"
    np.testing.assert_array_equal(
        np.asarray(prior.theta.init), np.asarray(darg(None, X).init)
    )
    assert prior.nugget.init == 1e-4 and prior.nugget.learn is False


def test_lengthscale_space_maps_the_theta_defaults(medium):
    """l = sqrt(theta / 2) for the start and box; the Gamma prior is recentered on l."""
    X, Z = medium
    t = Prior.from_data(X, Z).theta
    ell = Prior.from_data(X, Z, space="lengthscale").lengthscale
    np.testing.assert_allclose(
        np.asarray(ell.init), np.sqrt(np.asarray(t.init) / 2), rtol=1e-15
    )
    assert ell.lower == pytest.approx(np.sqrt(t.lower / 2), rel=1e-15)
    assert ell.upper == pytest.approx(np.sqrt(t.upper / 2), rel=1e-15)
    # same rule as darg's -- Gamma(1.5, b), 95% of the mass below the upper bound -- in l
    from scipy.stats import gamma

    a, b = ell.gamma
    assert a == 1.5
    assert gamma.cdf(ell.upper, a, scale=1 / b) == pytest.approx(0.95, rel=1e-9)


def test_a_prior_lives_in_exactly_one_space():
    with pytest.raises(ValueError, match="exactly one"):
        Prior(nugget=Param(), theta=Param(), lengthscale=Param())
    with pytest.raises(ValueError, match="exactly one"):
        Prior(nugget=Param())
    p = Prior(nugget=Param(), lengthscale=Param(init=0.3))
    with pytest.raises(ValueError, match="other space"):
        p.replace(theta=0.5)
    assert p.replace(lengthscale=0.4).lengthscale.init == 0.4
