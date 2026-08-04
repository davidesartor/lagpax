"""The local scheme (lagp / agp) against laGP's laGPsep / aGPsep."""

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


@pytest.mark.parametrize("criterion", ["nn", "alc"])
def test_lagp_selects_the_same_local_design(R, medium, criterion):
    X, Z = medium
    Xref = X[:1] + 0.01
    got = lagp.lagp(X, Z, Xref, start=START, end=END, criterion=criterion)
    _laGPsep(R, Xref, X, Z, criterion, lite=True)
    want = _fetch(R, "out$Xi").astype(int) - 1
    np.testing.assert_array_equal(np.sort(np.asarray(got.Xi)), np.sort(want))


@pytest.mark.parametrize("criterion", ["nn", "alc"])
def test_lagp_prediction_matches_r(R, medium, criterion):
    X, Z = medium
    Xref = X[:1] + 0.01
    got = lagp.lagp(X, Z, Xref, start=START, end=END, criterion=criterion)
    _laGPsep(R, Xref, X, Z, criterion, lite=True)
    want = _fetch(R, "c(out$mean, out$s2, out$df, out$llik)")

    assert float(got.mean[0]) == pytest.approx(want[0], rel=1e-6, abs=1e-8)
    assert float(got.s2[0]) == pytest.approx(want[1], rel=1e-4, abs=1e-12)
    assert float(got.df) == pytest.approx(want[2])
    assert float(got.llik) == pytest.approx(want[3], rel=1e-6)


@pytest.mark.parametrize("criterion", ["nn", "alc"])
def test_lagp_fit_matches_r(R, medium, criterion):
    X, Z = medium
    Xref = X[:1] + 0.01
    got = lagp.lagp(X, Z, Xref, start=START, end=END, criterion=criterion)
    _laGPsep(R, Xref, X, Z, criterion, lite=True)
    want = _fetch(R, "as.numeric(out$mle[1, grep('^d\\\\.', names(out$mle))])")
    np.testing.assert_allclose(np.asarray(got.fit.d), want, rtol=5e-3, atol=1e-6)


def test_lagp_multiple_references_full_covariance(R, medium):
    X, Z = medium
    Xref = X[:4] + 0.01
    got = lagp.lagp(X, Z, Xref, start=START, end=END, lite=False)
    _laGPsep(R, Xref, X, Z, "alc", lite=False)
    mean = _fetch(R, "out$mean")
    Sigma = np.asarray(R("out$Sigma"))

    np.testing.assert_allclose(np.asarray(got.mean), mean, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(np.asarray(got.s2), Sigma, rtol=1e-3, atol=1e-12)


def test_lagp_close_limits_the_candidate_pool(R, medium):
    X, Z = medium
    Xref = X[:1] + 0.01
    got = lagp.lagp(X, Z, Xref, start=START, end=END, close=120)
    _laGPsep(R, Xref, X, Z, "alc", lite=True, close=120)
    want = _fetch(R, "out$Xi").astype(int) - 1
    np.testing.assert_array_equal(np.sort(np.asarray(got.Xi)), np.sort(want))


@pytest.mark.parametrize("criterion", ["nn", "alc"])
def test_agp_matches_r(R, medium, criterion):
    X, Z = medium
    XX = X[:8] + 0.015
    got = lagp.agp(X, Z, XX, start=START, end=END, criterion=criterion, batch=4)
    R(
        f"out <- aGPsep(X, Z, XX, start={START}, end={END}, method='{criterion}', verb=0)",
        X=X,
        Z=Z,
        XX=XX,
    )
    mean = _fetch(R, "out$mean")
    var = _fetch(R, "out$var")

    np.testing.assert_allclose(np.asarray(got.mean), mean, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(np.asarray(got.var), var, rtol=1e-3, atol=1e-12)


def test_agp_is_batch_invariant(medium):
    X, Z = medium
    XX = X[:8] + 0.015
    a = lagp.agp(X, Z, XX, start=START, end=END, batch=2)
    b = lagp.agp(X, Z, XX, start=START, end=END, batch=None)
    np.testing.assert_allclose(np.asarray(a.mean), np.asarray(b.mean), rtol=1e-9)
    np.testing.assert_allclose(
        np.asarray(a.var), np.asarray(b.var), rtol=1e-6, atol=1e-11
    )


def test_agp_agrees_with_looping_lagp(medium):
    X, Z = medium
    XX = X[:4] + 0.015
    batched = lagp.agp(X, Z, XX, start=START, end=END)
    for i, xx in enumerate(XX):
        one = lagp.lagp(X, Z, xx[None, :], start=START, end=END)
        assert float(batched.mean[i]) == pytest.approx(float(one.mean[0]), rel=1e-9)
        assert float(batched.var[i]) == pytest.approx(
            float(one.var[0]), rel=1e-6, abs=1e-11
        )


def test_conditioning_once_matches_the_wrappers(medium):
    """`condition` resolves the prior; reusing it must not change a single number."""
    X, Z = medium
    XX = X[:4] + 0.015
    emu = lagp.Local(start=START, end=END).condition(X, Z)
    a = emu.predict(XX)
    b = lagp.agp(X, Z, XX, start=START, end=END)
    np.testing.assert_array_equal(np.asarray(a.mean), np.asarray(b.mean))
    np.testing.assert_array_equal(np.asarray(a.Xi), np.asarray(b.Xi))


def test_lagp_rejects_bad_sizes(small):
    X, Z = small
    with pytest.raises(ValueError):
        lagp.lagp(X, Z, X[:1], start=4, end=10)
    with pytest.raises(ValueError):
        lagp.lagp(X, Z, X[:1], start=6, end=len(X))
    with pytest.raises(ValueError):
        lagp.lagp(X, Z, X[:1], start=6, end=10, close=10)


@pytest.mark.parametrize("end", [7, 8, 32, 50, 200])
def test_design_buckets_cover_every_step_with_room_for_it(end):
    """Every greedy step lands in exactly one bucket whose capacity holds its live rows."""
    start = 6
    buckets = lagp.local._buckets(start, end)

    covered = [t for lo, hi, _ in buckets for t in range(lo, hi)]
    assert covered == list(range(start, end))
    for lo, hi, capacity in buckets:
        assert hi - 1 <= capacity <= end
    # logarithmic in the design size, not one trace per step
    assert len(buckets) <= max(1, (end - 1).bit_length())


def test_lagp_nn_takes_the_nearest_points(medium):
    X, Z = medium
    Xref = X[:1] + 0.01
    got = lagp.lagp(X, Z, Xref, start=START, end=END, criterion="nn")
    dist = np.sum(np.square(X - Xref), axis=-1)
    np.testing.assert_array_equal(
        np.sort(np.asarray(got.Xi)), np.sort(np.argsort(dist)[:END])
    )


def test_refine_is_no_worse_than_a_single_pass(medium):
    """The second pass inherits per-location lengthscales, so it should not be worse."""
    X, Z = medium
    XX = X[:16] + 0.015
    truth = np.sin(4 * XX[:, 0]) + np.cos(3 * XX[:, 1:3].sum(-1)) + 0.1 * XX.sum(-1)

    emu = lagp.Local(start=START, end=END).condition(X, Z)
    one = np.asarray(emu.predict(XX).mean)
    two = np.asarray(emu.refine(XX, passes=("nn", "alc")).mean)

    rmse = lambda a: np.sqrt(np.mean((a - truth) ** 2))
    assert rmse(two) <= 1.05 * rmse(one)
