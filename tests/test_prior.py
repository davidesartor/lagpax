"""darg / garg / Igamma.inv against laGP's prior.R."""

import numpy as np
import pytest
from scipy.special import gamma as sp_gamma

import lagpax as lagp

FIELDS = "c(out$start, out$min, out$max, as.numeric(out$mle), out$ab)"


def _r_arg(R, call: str, **data) -> np.ndarray:
    return np.asarray(R(f"out <- {call}; {FIELDS}", **data)).ravel()


def _assert_darg(got: lagp.Darg | lagp.Garg, want: np.ndarray, nstart: int = 1):
    start, rest = want[:nstart], want[nstart:]
    np.testing.assert_allclose(np.atleast_1d(np.asarray(got.start)), start, rtol=1e-13)
    assert got.min == pytest.approx(rest[0], rel=1e-13)
    assert got.max == pytest.approx(rest[1], rel=1e-13)
    assert float(got.mle) == rest[2]
    np.testing.assert_allclose(np.asarray(got.ab), rest[3:5], rtol=1e-10)


@pytest.mark.parametrize("a", [0.7, 1.5, 2.0, 5.0])
@pytest.mark.parametrize("frac", [0.05, 0.5, 0.95])
@pytest.mark.parametrize("lower", [0, 1])
def test_igamma_inv_matches_r(R, a, frac, lower):
    """laGP always calls this as Igamma.inv(a, frac*gamma(a), lower, 0)."""
    got = lagp.igamma_inv(a, frac * sp_gamma(a), lower=bool(lower))
    want = R.scalar(f"laGP:::Igamma.inv({a}, {frac}*gamma({a}), {lower}, 0)")
    assert got == pytest.approx(want, rel=1e-9)


def test_get_ds_matches_r(R, medium):
    X, _ = medium
    start, dmin, dmax = lagp.get_ds(X)
    want = np.asarray(
        R("Ds <- laGP:::getDs(X); c(Ds$start, Ds$min, Ds$max)", X=X)
    ).ravel()
    np.testing.assert_allclose([start, dmin, dmax], want, rtol=1e-13)


@pytest.mark.parametrize("design", ["small", "medium", "wide"])
def test_darg_default_matches_r(R, request, design):
    X, _ = request.getfixturevalue(design)
    _assert_darg(lagp.darg(None, X), _r_arg(R, "darg(NULL, X)", X=X))


def test_darg_scalar_start_matches_r(R, medium):
    X, _ = medium
    _assert_darg(lagp.darg(0.5, X), _r_arg(R, "darg(0.5, X)", X=X))


def test_darg_no_mle_matches_r(R, medium):
    X, _ = medium
    _assert_darg(
        lagp.darg({"start": 0.5, "mle": False}, X),
        _r_arg(R, "darg(list(start=0.5, mle=FALSE), X)", X=X),
    )


def test_darg_explicit_box_matches_r(R, medium):
    X, _ = medium
    _assert_darg(
        lagp.darg({"min": 1e-4, "max": 3.0}, X),
        _r_arg(R, "darg(list(min=1e-4, max=3), X)", X=X),
    )


def test_darg_vector_start(R, wide):
    X, _ = wide
    start = np.linspace(0.2, 0.9, X.shape[1])
    got = lagp.darg({"start": start, "max": 4.0, "min": 1e-3}, X)
    want = _r_arg(R, "darg(list(start=start, max=4, min=1e-3), X)", X=X, start=start)
    _assert_darg(got, want, nstart=len(start))


@pytest.mark.parametrize("design", ["small", "medium"])
def test_garg_default_matches_r(R, request, design):
    _, Z = request.getfixturevalue(design)
    _assert_darg(lagp.garg(None, Z), _r_arg(R, "garg(NULL, Z)", Z=Z))


def test_garg_scalar_matches_r(R, medium):
    _, Z = medium
    _assert_darg(lagp.garg(1 / 10000, Z), _r_arg(R, "garg(1/10000, Z)", Z=Z))


def test_garg_mle_matches_r(R, medium):
    _, Z = medium
    _assert_darg(lagp.garg({"mle": True}, Z), _r_arg(R, "garg(list(mle=TRUE), Z)", Z=Z))


def test_darg_rejects_start_outside_box(medium):
    X, _ = medium
    with pytest.raises(ValueError):
        lagp.darg({"start": 10.0, "min": 0.1, "max": 1.0}, X)


def test_get_ds_subsamples_deterministically_with_a_seed(small):
    X, _ = small
    a = lagp.get_ds(X, samp_size=20, rng=np.random.default_rng(0))
    b = lagp.get_ds(X, samp_size=20, rng=np.random.default_rng(0))
    assert a == b
