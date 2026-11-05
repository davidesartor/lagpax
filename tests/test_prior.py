"""Default priors (laGP's darg / garg) and the Gamma prior rule, against laGP's prior.R."""

import jax
import numpy as np
import pytest

from lagpax.priors import (
    GammaPrior,
    Param,
    Prior,
    default_nugget,
    default_squared_lengthscale,
    pairwise_distance_summary,
)

FIELDS = "c(out$start, out$min, out$max, as.numeric(out$mle), out$ab)"


def _r_arg(R, call: str, **data) -> np.ndarray:
    return np.asarray(R(f"out <- {call}; {FIELDS}", **data)).ravel()


def _assert_darg(got: Param, want: np.ndarray, nstart: int = 1):
    start, rest = want[:nstart], want[nstart:]
    np.testing.assert_allclose(
        np.atleast_1d(np.asarray(got.initial)), start, rtol=1e-13
    )
    assert got.lower == pytest.approx(rest[0], rel=1e-13)
    assert got.upper == pytest.approx(rest[1], rel=1e-13)
    assert float(got.learn) == rest[2]
    shape_rate = (0.0, 0.0) if got.prior is None else (got.prior.shape, got.prior.rate)
    np.testing.assert_allclose(shape_rate, rest[3:5], rtol=1e-10)


@pytest.mark.parametrize("shape", [0.7, 1.5, 2.0, 5.0])
@pytest.mark.parametrize("mass", [0.05, 0.5, 0.95])
def test_gamma_prior_with_mass_below_matches_lagps_rule(R, shape, mass):
    """laGP sets rate = Igamma.inv(a, mass * gamma(a), lower=1, log=0) / upper."""
    upper = 2.7
    got = GammaPrior.with_mass_below(upper, shape=shape, mass=mass)
    want = R.scalar(
        f"laGP:::Igamma.inv({shape}, {mass}*gamma({shape}), 1, 0) / {upper}"
    )
    assert got.shape == shape
    assert got.rate == pytest.approx(want, rel=1e-9)


def test_distance_summary_matches_r(R, medium):
    X, _ = medium
    start, dmin, dmax = pairwise_distance_summary(X)
    want = np.asarray(
        R("Ds <- laGP:::getDs(X); c(Ds$start, Ds$min, Ds$max)", X=X)
    ).ravel()
    np.testing.assert_allclose([start, dmin, dmax], want, rtol=1e-13)


@pytest.mark.parametrize("design", ["small", "medium", "wide"])
def test_default_squared_lengthscale_default_matches_r(R, request, design):
    X, _ = request.getfixturevalue(design)
    _assert_darg(default_squared_lengthscale(None, X), _r_arg(R, "darg(NULL, X)", X=X))


def test_default_squared_lengthscale_scalar_start_matches_r(R, medium):
    X, _ = medium
    _assert_darg(default_squared_lengthscale(0.5, X), _r_arg(R, "darg(0.5, X)", X=X))


def test_default_squared_lengthscale_no_mle_matches_r(R, medium):
    X, _ = medium
    _assert_darg(
        default_squared_lengthscale(Param(initial=0.5, learn=False), X),
        _r_arg(R, "darg(list(start=0.5, mle=FALSE), X)", X=X),
    )


def test_default_squared_lengthscale_explicit_box_matches_r(R, medium):
    X, _ = medium
    _assert_darg(
        default_squared_lengthscale(Param(lower=1e-4, upper=3.0), X),
        _r_arg(R, "darg(list(min=1e-4, max=3), X)", X=X),
    )


def test_default_squared_lengthscale_vector_start(R, wide):
    X, _ = wide
    start = np.linspace(0.2, 0.9, X.shape[1])
    got = default_squared_lengthscale(Param(initial=start, upper=4.0, lower=1e-3), X)
    want = _r_arg(R, "darg(list(start=start, max=4, min=1e-3), X)", X=X, start=start)
    _assert_darg(got, want, nstart=len(start))


@pytest.mark.parametrize("design", ["small", "medium"])
def test_default_nugget_default_matches_r(R, request, design):
    _, Z = request.getfixturevalue(design)
    _assert_darg(default_nugget(None, Z), _r_arg(R, "garg(NULL, Z)", Z=Z))


def test_default_nugget_scalar_matches_r(R, medium):
    _, Z = medium
    _assert_darg(default_nugget(1 / 10000, Z), _r_arg(R, "garg(1/10000, Z)", Z=Z))


def test_default_nugget_mle_matches_r(R, medium):
    _, Z = medium
    _assert_darg(
        default_nugget(Param(learn=True), Z), _r_arg(R, "garg(list(mle=TRUE), Z)", Z=Z)
    )


def test_default_squared_lengthscale_rejects_start_outside_box(medium):
    X, _ = medium
    with pytest.raises(ValueError):
        default_squared_lengthscale(Param(initial=10.0, lower=0.1, upper=1.0), X)


def test_distance_summary_subsamples_deterministically_with_a_key(small):
    X, _ = small
    a = pairwise_distance_summary(X, sample_size=20, key=jax.random.key(0))
    b = pairwise_distance_summary(X, sample_size=20, key=jax.random.key(0))
    assert a == b


def test_subsampling_without_a_key_is_an_error_not_a_hidden_draw(small):
    """laGP draws from R's global RNG; lagpax has none, so the key must be explicit."""
    X, _ = small
    pairwise_distance_summary(X, sample_size=len(X))  # no subsample, no key needed
    with pytest.raises(ValueError, match="key=jax.random.key"):
        pairwise_distance_summary(X, sample_size=20)


def test_from_data_is_lagps_defaults(medium):
    X, Z = medium
    prior = Prior.from_data(X, Z)
    assert prior.kernel_space == "squared_lengthscale"
    np.testing.assert_array_equal(
        np.asarray(prior.squared_lengthscale.initial),
        np.asarray(default_squared_lengthscale(None, X).initial),
    )
    assert prior.nugget.initial == 1e-4 and prior.nugget.learn is False


def test_lengthscale_space_maps_the_theta_defaults(medium):
    """l = sqrt(theta / 2) for the start and box; the Gamma prior is recentered on l."""
    X, Z = medium
    t = Prior.from_data(X, Z).squared_lengthscale
    ell = Prior.from_data(X, Z, kernel_space="lengthscale").lengthscale
    np.testing.assert_allclose(
        np.asarray(ell.initial), np.sqrt(np.asarray(t.initial) / 2), rtol=1e-15
    )
    assert ell.lower == pytest.approx(np.sqrt(t.lower / 2), rel=1e-15)
    assert ell.upper == pytest.approx(np.sqrt(t.upper / 2), rel=1e-15)
    # same rule as darg's -- Gamma(1.5, b), 95% of the mass below the upper bound -- in l
    from scipy.stats import gamma

    a, b = ell.prior.shape, ell.prior.rate
    assert a == 1.5
    assert gamma.cdf(ell.upper, a, scale=1 / b) == pytest.approx(0.95, rel=1e-9)


def test_a_prior_lives_in_exactly_one_space():
    with pytest.raises(ValueError, match="exactly one of squared_lengthscale= and lengthscale="):
        Prior(nugget=Param(), squared_lengthscale=Param(), lengthscale=Param())
    with pytest.raises(ValueError, match="exactly one of squared_lengthscale= and lengthscale="):
        Prior(nugget=Param())
    p = Prior(nugget=Param(), lengthscale=Param(initial=0.3))
    with pytest.raises(ValueError, match="other kernel space"):
        p.replace(squared_lengthscale=0.5)
    assert p.replace(lengthscale=0.4).lengthscale.initial == 0.4
