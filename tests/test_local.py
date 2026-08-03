"""laGPsep / aGPsep against the R package."""

import numpy as np
import pytest

import lagpax as lagp

START, END = 6, 30


def _laGPsep(R, Xref, X, Z, method, lite=False, **kw):
    extra = "".join(f", {k}={v}" for k, v in kw.items())
    lite_arg = "TRUE" if lite else "FALSE"
    script = (
        f"out <- laGPsep(Xref, {START}, {END}, X, Z, method='{method}', "
        f"lite={lite_arg}{extra})"
    )
    R(script, Xref=Xref, X=X, Z=Z)
    return R


def _fetch(R, expr):
    return np.asarray(R(expr)).ravel()


@pytest.mark.parametrize("method", ["nn", "alc"])
def test_lagpsep_selects_the_same_local_design(R, medium, method):
    X, Z = medium
    Xref = X[:1] + 0.01
    got = lagp.lagpsep(Xref, X, Z, start=START, end=END, method=method)
    _laGPsep(R, Xref, X, Z, method, lite=True)
    want = _fetch(R, "out$Xi").astype(int) - 1
    np.testing.assert_array_equal(np.sort(np.asarray(got.Xi)), np.sort(want))


@pytest.mark.parametrize("method", ["nn", "alc"])
def test_lagpsep_prediction_matches_r(R, medium, method):
    X, Z = medium
    Xref = X[:1] + 0.01
    got = lagp.lagpsep(Xref, X, Z, start=START, end=END, method=method)
    _laGPsep(R, Xref, X, Z, method, lite=True)
    want = _fetch(R, "c(out$mean, out$s2, out$df, out$llik)")

    assert float(got.mean[0]) == pytest.approx(want[0], rel=1e-6, abs=1e-8)
    assert float(got.s2[0]) == pytest.approx(want[1], rel=1e-4, abs=1e-12)
    assert float(got.df) == pytest.approx(want[2])
    assert float(got.llik) == pytest.approx(want[3], rel=1e-6)


@pytest.mark.parametrize("method", ["nn", "alc"])
def test_lagpsep_mle_matches_r(R, medium, method):
    X, Z = medium
    Xref = X[:1] + 0.01
    got = lagp.lagpsep(Xref, X, Z, start=START, end=END, method=method)
    _laGPsep(R, Xref, X, Z, method, lite=True)
    want = _fetch(R, "as.numeric(out$mle[1, grep('^d\\\\.', names(out$mle))])")
    np.testing.assert_allclose(np.asarray(got.mle["d"]), want, rtol=5e-3, atol=1e-6)


def test_lagpsep_multiple_references_full_covariance(R, medium):
    X, Z = medium
    Xref = X[:4] + 0.01
    got = lagp.lagpsep(Xref, X, Z, start=START, end=END, method="alc", lite=False)
    _laGPsep(R, Xref, X, Z, "alc", lite=False)
    mean = _fetch(R, "out$mean")
    Sigma = np.asarray(R("out$Sigma"))

    np.testing.assert_allclose(np.asarray(got.mean), mean, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(np.asarray(got.s2), Sigma, rtol=1e-3, atol=1e-12)


def test_lagpsep_close_limits_the_candidate_pool(R, medium):
    X, Z = medium
    Xref = X[:1] + 0.01
    got = lagp.lagpsep(Xref, X, Z, start=START, end=END, method="alc", close=120)
    _laGPsep(R, Xref, X, Z, "alc", lite=True, close=120)
    want = _fetch(R, "out$Xi").astype(int) - 1
    np.testing.assert_array_equal(np.sort(np.asarray(got.Xi)), np.sort(want))


@pytest.mark.parametrize("method", ["nn", "alc"])
def test_agpsep_matches_r(R, medium, method):
    X, Z = medium
    XX = X[:8] + 0.015
    got = lagp.agpsep(X, Z, XX, start=START, end=END, method=method, batch=4)
    R(
        f"out <- aGPsep(X, Z, XX, start={START}, end={END}, method='{method}', verb=0)",
        X=X,
        Z=Z,
        XX=XX,
    )
    mean = _fetch(R, "out$mean")
    var = _fetch(R, "out$var")

    np.testing.assert_allclose(np.asarray(got.mean), mean, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(np.asarray(got.var), var, rtol=1e-3, atol=1e-12)


def test_agpsep_is_batch_invariant(medium):
    X, Z = medium
    XX = X[:8] + 0.015
    a = lagp.agpsep(X, Z, XX, start=START, end=END, method="alc", batch=2)
    b = lagp.agpsep(X, Z, XX, start=START, end=END, method="alc", batch=None)
    np.testing.assert_allclose(np.asarray(a.mean), np.asarray(b.mean), rtol=1e-9)
    np.testing.assert_allclose(np.asarray(a.var), np.asarray(b.var), rtol=1e-6, atol=1e-11)


def test_agpsep_agrees_with_looping_lagpsep(medium):
    X, Z = medium
    XX = X[:4] + 0.015
    batched = lagp.agpsep(X, Z, XX, start=START, end=END, method="alc")
    for i, xx in enumerate(XX):
        one = lagp.lagpsep(xx[None, :], X, Z, start=START, end=END, method="alc")
        df = float(one.df)
        assert float(batched.mean[i]) == pytest.approx(float(one.mean[0]), rel=1e-9)
        assert float(batched.var[i]) == pytest.approx(
            float(one.s2[0]) * df / (df - 2.0), rel=1e-6, abs=1e-11
        )


def test_lagpsep_rejects_bad_sizes(small):
    X, Z = small
    with pytest.raises(ValueError):
        lagp.lagpsep(X[:1], X, Z, start=4, end=10)
    with pytest.raises(ValueError):
        lagp.lagpsep(X[:1], X, Z, start=6, end=len(X))
    with pytest.raises(ValueError):
        lagp.lagpsep(X[:1], X, Z, start=6, end=10, close=10)


def test_lagpsep_nn_takes_the_nearest_points(medium):
    X, Z = medium
    Xref = X[:1] + 0.01
    got = lagp.lagpsep(Xref, X, Z, start=START, end=END, method="nn")
    dist = np.sum(np.square(X - Xref), axis=-1)
    np.testing.assert_array_equal(
        np.sort(np.asarray(got.Xi)), np.sort(np.argsort(dist)[:END])
    )
