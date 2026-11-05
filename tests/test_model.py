"""The local model (LocalGP / Emulator) against laGP's laGPsep / aGPsep."""

import numpy as np
import pytest

import lagpax as lagp
import lagpax.local as lagpax_local

START, END = 6, 30


def _model(**settings):
    return lagp.LocalGP(initial_neighbors=START, neighbors=END, **settings)


def _local(X, Z, Xref, full_covariance=False, **settings):
    return _model(**settings).fit(X, Z).predict_local(Xref, full_covariance=full_covariance)


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
def test_local_selects_the_same_local_design(R, medium, criterion):
    X, Z = medium
    Xref = X[:1] + 0.01
    got = _local(X, Z, Xref, criterion=criterion)
    _laGPsep(R, Xref, X, Z, criterion, lite=True)
    want = _fetch(R, "out$Xi").astype(int) - 1
    np.testing.assert_array_equal(np.sort(np.asarray(got.neighbors)), np.sort(want))


@pytest.mark.parametrize("criterion", ["nn", "alc"])
def test_local_prediction_matches_r(R, medium, criterion):
    X, Z = medium
    Xref = X[:1] + 0.01
    got = _local(X, Z, Xref, criterion=criterion)
    _laGPsep(R, Xref, X, Z, criterion, lite=True)
    want = _fetch(R, "c(out$mean, out$s2, out$df, out$llik)")

    df = want[2]
    assert float(got.mean[0]) == pytest.approx(want[0], rel=1e-6, abs=1e-8)
    # laGPsep returns the Student-t scale; lagpax the variance, scale * df / (df - 2)
    assert float(got.variance[0]) == pytest.approx(
        want[1] * df / (df - 2), rel=1e-4, abs=1e-12
    )
    assert float(got.degrees_of_freedom) == pytest.approx(df)
    assert float(got.log_likelihood) == pytest.approx(want[3], rel=1e-6)


@pytest.mark.parametrize("criterion", ["nn", "alc"])
def test_local_fit_matches_r(R, medium, criterion):
    X, Z = medium
    Xref = X[:1] + 0.01
    got = _local(X, Z, Xref, criterion=criterion)
    _laGPsep(R, Xref, X, Z, criterion, lite=True)
    want = _fetch(R, "as.numeric(out$mle[1, grep('^d\\\\.', names(out$mle))])")
    np.testing.assert_allclose(np.asarray(got.fit.squared_lengthscales), want, rtol=5e-3, atol=1e-6)


def test_local_multiple_references_full_covariance(R, medium):
    X, Z = medium
    Xref = X[:4] + 0.01
    got = _local(X, Z, Xref, full_covariance=True)
    _laGPsep(R, Xref, X, Z, "alc", lite=False)
    mean = _fetch(R, "out$mean")
    Sigma = np.asarray(R("out$Sigma"))
    df = float(_fetch(R, "out$df")[0])

    np.testing.assert_allclose(np.asarray(got.mean), mean, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(
        np.asarray(got.covariance), Sigma * df / (df - 2), rtol=1e-3, atol=1e-12
    )


def test_local_close_limits_the_candidate_pool(R, medium):
    X, Z = medium
    Xref = X[:1] + 0.01
    got = _local(X, Z, Xref, candidates=120)
    _laGPsep(R, Xref, X, Z, "alc", lite=True, close=120)
    want = _fetch(R, "out$Xi").astype(int) - 1
    np.testing.assert_array_equal(np.sort(np.asarray(got.neighbors)), np.sort(want))


@pytest.mark.parametrize("criterion", ["nn", "alc"])
def test_emulator_matches_r(R, medium, criterion):
    X, Z = medium
    XX = X[:8] + 0.015
    got = _model(criterion=criterion).fit(X, Z)(XX, batch_size=4)
    R(
        f"out <- aGPsep(X, Z, XX, start={START}, end={END}, method='{criterion}', verb=0)",
        X=X,
        Z=Z,
        XX=XX,
    )
    mean = _fetch(R, "out$mean")
    var = _fetch(R, "out$var")

    np.testing.assert_allclose(np.asarray(got.mean), mean, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(np.asarray(got.variance), var, rtol=1e-3, atol=1e-12)


def test_emulator_is_batch_invariant(medium):
    X, Z = medium
    XX = X[:8] + 0.015
    emu = _model().fit(X, Z)
    a, b = emu(XX, batch_size=2), emu(XX, batch_size=None)
    np.testing.assert_allclose(np.asarray(a.mean), np.asarray(b.mean), rtol=1e-9)
    np.testing.assert_allclose(
        np.asarray(a.variance), np.asarray(b.variance), rtol=1e-6, atol=1e-11
    )


def test_emulator_agrees_with_looping_lagp(medium):
    X, Z = medium
    XX = X[:4] + 0.015
    emu = _model().fit(X, Z)
    batched = emu(XX)
    for i, xx in enumerate(XX):
        one = emu.predict_local(xx[None, :])
        assert float(batched.mean[i]) == pytest.approx(float(one.mean[0]), rel=1e-9)
        assert float(batched.variance[i]) == pytest.approx(
            float(one.variance[0]), rel=1e-6, abs=1e-11
        )


def test_an_explicit_default_prior_changes_nothing(medium):
    """`fit` resolves `Prior.from_data` when no prior is given; passing it must agree."""
    X, Z = medium
    XX = X[:4] + 0.015
    a = _model().fit(X, Z)(XX)
    b = _model().fit(X, Z, prior=lagp.Prior.from_data(X, Z))(XX)
    np.testing.assert_array_equal(np.asarray(a.mean), np.asarray(b.mean))
    np.testing.assert_array_equal(np.asarray(a.neighbors), np.asarray(b.neighbors))


def test_fit_rejects_bad_sizes(small):
    X, Z = small
    with pytest.raises(ValueError):
        lagp.LocalGP(initial_neighbors=4, neighbors=10).fit(X, Z)
    with pytest.raises(ValueError):
        lagp.LocalGP(initial_neighbors=6, neighbors=len(X)).fit(X, Z)
    with pytest.raises(ValueError):
        lagp.LocalGP(initial_neighbors=6, neighbors=10, candidates=10).fit(X, Z)


@pytest.mark.parametrize("end", [7, 8, 32, 50, 200])
def test_design_buckets_cover_every_step_with_room_for_it(end):
    """Every greedy step lands in exactly one bucket whose capacity holds its live rows."""
    start = 6
    buckets = lagpax_local.design_buckets(start, end)

    covered = [t for lo, hi, _ in buckets for t in range(lo, hi)]
    assert covered == list(range(start, end))
    for lo, hi, capacity in buckets:
        assert hi - 1 <= capacity <= end
    # logarithmic in the design size, not one trace per step
    assert len(buckets) <= max(1, (end - 1).bit_length())


def test_local_nn_takes_the_nearest_points(medium):
    X, Z = medium
    Xref = X[:1] + 0.01
    got = _local(X, Z, Xref, criterion="nn")
    dist = np.sum(np.square(X - Xref), axis=-1)
    np.testing.assert_array_equal(
        np.sort(np.asarray(got.neighbors)), np.sort(np.argsort(dist)[:END])
    )


def test_refine_is_no_worse_than_a_single_pass(medium):
    """The second pass inherits per-location lengthscales, so it should not be worse."""
    X, Z = medium
    XX = X[:16] + 0.015
    truth = np.sin(4 * XX[:, 0]) + np.cos(3 * XX[:, 1:3].sum(-1)) + 0.1 * XX.sum(-1)

    emu = _model().fit(X, Z)
    one = np.asarray(emu(XX).mean)
    two = np.asarray(emu.refine(XX, passes=("nn", "alc")).mean)

    rmse = lambda a: np.sqrt(np.mean((a - truth) ** 2))
    assert rmse(two) <= 1.05 * rmse(one)
