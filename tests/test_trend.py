"""Trend models, which laGP has no equivalent for -- so property tests, not parity.

The claim being tested is calibration: profiling out a trend widens the intervals by the
uncertainty in the fitted mean, which simple kriging drops. rmse is not the claim.
"""

import numpy as np
import pytest

import jax
import jax.numpy as jnp

import lagpax as lagp
from helpers import make_gp
from lagpax.gaussian_process import trend_basis

D = np.array([0.3, 0.7, 1.1])
G = 1e-6


def _gp(X, Z, trend, d=D, mask=None):
    return make_gp(jnp.asarray(X), jnp.asarray(Z), jnp.asarray(d), G, mask, trend=trend)


def test_zero_trend_is_the_untrended_model(small):
    """The parity default has to stay the code path the whole suite validates."""
    X, Z = small
    plain, zero = _gp(X, Z, "zero"), make_gp(jnp.asarray(X), jnp.asarray(Z), D, G)
    assert plain.trend_size == 0
    assert float(plain.degrees_of_freedom) == float(zero.degrees_of_freedom) == len(X)

    XX = X[:5] + 0.05
    np.testing.assert_array_equal(
        np.asarray(plain.predict(XX).mean), np.asarray(zero.predict(XX).mean)
    )
    np.testing.assert_array_equal(
        np.asarray(plain.predict(XX).variance), np.asarray(zero.predict(XX).variance)
    )


@pytest.mark.parametrize("trend,p", [("constant", 1), ("linear", 4)])
def test_degrees_of_freedom_drop_one_per_coefficient(small, trend, p):
    X, Z = small
    gp = _gp(X, Z, trend)
    assert gp.trend_size == p
    assert float(gp.degrees_of_freedom) == len(X) - p
    assert float(gp.predict(X[:3] + 0.05).degrees_of_freedom) == len(X) - p


def test_constant_trend_coefficient_is_the_gls_mean(small):
    """`b` is the generalized least squares estimate, and the prediction reverts to it.

    Simple kriging reverts to zero far from the design; with a constant trend it has to
    revert to the GLS mean of Z instead, which is the whole reason for profiling it out.
    """
    X, Z = small
    gp = _gp(X, Z, "constant")

    Ki = np.asarray(gp.inverse_correlation)
    ones = np.ones(len(Z))
    gls = float((ones @ Ki @ Z) / (ones @ Ki @ ones))
    assert float(gp.trend_coefficients[0]) == pytest.approx(gls, rel=1e-10)

    far = np.full((3, X.shape[1]), 30.0)
    np.testing.assert_allclose(np.asarray(gp.predict(far).mean), gls, rtol=1e-10)
    np.testing.assert_allclose(
        np.asarray(_gp(X, Z, "zero").predict(far).mean), 0.0, atol=1e-12
    )


def test_trend_recovers_a_linear_response_outside_the_design():
    """Simple kriging reverts to zero away from the data; a linear trend extrapolates."""
    X = np.linspace(0.0, 1.0, 20)[:, None]
    Z = 3.0 + 2.0 * X[:, 0]
    XX = np.array([[3.0]])

    zero = float(_gp(X, Z, "zero", d=np.array([0.1])).predict(XX).mean[0])
    linear = float(_gp(X, Z, "linear", d=np.array([0.1])).predict(XX).mean[0])
    assert abs(zero) < 1e-6
    assert linear == pytest.approx(9.0, rel=1e-6)


def test_mask_matches_a_smaller_design_with_a_trend(small):
    """The fourth mask invariant: masked rows must drop out of F as well as K and Z."""
    X, Z = small
    k = 25
    mask = jnp.asarray(np.arange(len(X)) < k)
    for trend in ("constant", "linear"):
        padded = _gp(X, Z, trend, mask=mask)
        exact = _gp(X[:k], Z[:k], trend)

        assert float(padded.degrees_of_freedom) == float(exact.degrees_of_freedom)
        np.testing.assert_allclose(
            np.asarray(padded.trend_coefficients), np.asarray(exact.trend_coefficients), rtol=1e-10
        )
        assert float(padded.residual_sum_of_squares) == pytest.approx(float(exact.residual_sum_of_squares), rel=1e-10)
        assert float(padded.log_likelihood()) == pytest.approx(
            float(exact.log_likelihood()), rel=1e-10
        )

        XX = X[:5] + 0.05
        a, b = padded.predict(XX), exact.predict(XX)
        np.testing.assert_allclose(
            np.asarray(a.mean), np.asarray(b.mean), rtol=1e-10, atol=1e-12
        )
        np.testing.assert_allclose(
            np.asarray(a.variance), np.asarray(b.variance), rtol=1e-8, atol=1e-12
        )


def _coverage(trend, reps=200, seed=2):
    """Empirical coverage of 90% intervals just past the end of a one-sided design.

    Extrapolating off the end is where the uncertainty in the fitted mean matters, and
    where simple kriging is overconfident: its mean reverts to zero while its interval
    only ever accounts for the correlation.
    """
    from scipy.stats import t as student_t

    rng = np.random.default_rng(seed)
    f = lambda u: 5.0 * u[:, 0] + np.sin(6 * u[:, 0])
    hits = 0
    for _ in range(reps):
        x = np.sort(rng.random(24))[:, None]
        Z = f(x) + 0.05 * rng.standard_normal(len(x))

        xx = np.array([[x[-1, 0] + 0.4]])
        gp = make_gp(x, Z, np.array([0.05]), 1e-4, trend=trend)
        pred = gp.predict(xx)
        df = float(pred.degrees_of_freedom)
        scale = float(pred.variance[0]) * (df - 2) / df  # the Student-t scale, squared
        half = student_t.ppf(0.95, df) * np.sqrt(scale)
        hits += abs(float(pred.mean[0]) - float(f(xx)[0])) <= half
    return hits / reps


def test_trend_improves_interval_coverage():
    """The actual claim: better-calibrated intervals, not a better mean.

    Extrapolating 0.4 past the design with a lengthscale of 0.05, simple kriging covers
    ~0.30 where it promises 0.90. Both trends fix that; `linear` then over-covers, because
    profiling two coefficients out of 24 points is not free.
    """
    zero, constant, linear = (_coverage(t) for t in ("zero", "constant", "linear"))
    assert zero < 0.5
    assert abs(constant - 0.90) < abs(zero - 0.90)
    assert abs(linear - 0.90) < abs(zero - 0.90)
    assert constant >= 0.90 and linear >= 0.90


def test_gradients_flow_through_the_trend_predictive(small):
    """Differentiable prediction has to survive the extra solve."""
    X, Z = small
    gp = _gp(X, Z, "linear")
    x0 = jnp.asarray(X[0] + 0.05)

    # the variance is a cancellation, so its finite difference is the noisy side of the
    # comparison and sets the tolerance; the step is large for the same reason
    step = 1e-4
    for field, rtol in (("mean", 1e-6), ("variance", 1e-3)):
        f = lambda x: getattr(gp.predict(x[None]), field)[0]
        got = np.asarray(jax.grad(f)(x0))
        want = np.array(
            [
                float(f(x0.at[k].add(step)) - f(x0.at[k].add(-step))) / (2 * step)
                for k in range(len(x0))
            ]
        )
        np.testing.assert_allclose(got, want, rtol=rtol, atol=1e-9)


def test_the_local_model_accepts_a_trend(medium):
    """`trend` is static config, so it has to reach the compiled core intact."""
    X, Z = medium
    XX = np.random.default_rng(8).random((5, X.shape[1]))
    model = lagp.LocalGP(initial_neighbors=6, neighbors=20)
    zero = model.fit(X, Z)(XX, batch_size=None)
    for trend, p in (("constant", 1), ("linear", X.shape[1] + 1)):
        out = model.replace(trend=trend).fit(X, Z)(XX, batch_size=None)
        np.testing.assert_array_equal(np.asarray(out.degrees_of_freedom), 20 - p)
        assert np.all(np.isfinite(np.asarray(out.mean)))
    np.testing.assert_array_equal(np.asarray(zero.degrees_of_freedom), 20)


def test_unknown_trend_is_rejected(small):
    X, Z = small
    with pytest.raises(ValueError):
        trend_basis("quadratic", jnp.asarray(X))
