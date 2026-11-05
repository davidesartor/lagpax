"""IECI against laGP:::ieciGPsep, and the direction that makes it a BO criterion."""

import numpy as np
import pytest

import jax
import jax.numpy as jnp

import lagpax as lagp


def _split(X, Z, ndesign=25, ncand=10):
    d = np.array([0.4, 0.6, 0.5][: X.shape[1]] + [0.5] * (X.shape[1] - 3))
    gp = lagp.GP(X[:ndesign], Z[:ndesign], d, 1e-6)
    Xcand = jnp.asarray(X[ndesign : ndesign + ncand])
    Xref = jnp.asarray(X[ndesign + ncand : ndesign + ncand + 5])
    return gp, Xcand, Xref, float(np.min(Z[:ndesign]))


def _new_gpsep(R, gp):
    R(
        "gpi <- newGPsep(Xd, Zd, d=dd, g=gg, dK=TRUE)",
        Xd=np.asarray(gp.X),
        Zd=np.asarray(gp.y),
        dd=np.asarray(gp.theta),
        gg=np.array([float(gp.nugget)]),
    )


# EI subtracts two O(1) terms to land on an O(1e-4) answer on `noisy`, the same
# cancellation that makes `s2` looser than `mean` elsewhere in the suite
@pytest.mark.parametrize("fixture,rtol", [("small", 1e-10), ("noisy", 1e-9)])
@pytest.mark.parametrize("nugget", [True, False])
def test_ieci_matches_iecigpsep(R, request, fixture, rtol, nugget):
    X, Z = request.getfixturevalue(fixture)
    gp, Xcand, Xref, fmin = _split(X, Z)
    _new_gpsep(R, gp)

    want = np.asarray(
        R(
            f"laGP:::ieciGPsep(gpi, Xcand, {fmin!r}, Xref,"
            f" nonug={'FALSE' if nugget else 'TRUE'})",
            Xcand=np.asarray(Xcand),
            Xref=np.asarray(Xref),
        )
    ).ravel()
    got = np.asarray(lagp.IECI(fmin=fmin, nugget=nugget)(gp, Xcand, Xref))
    np.testing.assert_allclose(got, want, rtol=rtol)


def test_ieci_weights_match_lagp(R, small):
    """laGP scales the average by `w` instead of renormalizing, which changes magnitude."""
    X, Z = small
    gp, Xcand, Xref, fmin = _split(X, Z)
    _new_gpsep(R, gp)
    w = np.random.default_rng(0).random(len(Xref))

    want = np.asarray(
        R(
            f"laGP:::ieciGPsep(gpi, Xcand, {fmin!r}, Xref, w=w)",
            Xcand=np.asarray(Xcand),
            Xref=np.asarray(Xref),
            w=w,
        )
    ).ravel()
    got = np.asarray(lagp.IECI(fmin=fmin, w=jnp.asarray(w))(gp, Xcand, Xref))
    np.testing.assert_allclose(got, want, rtol=1e-10)


def test_ieci_is_minimized_and_alc_is_not():
    """The direction test: on a 1-d bowl, IECI reaches for the incumbent minimum.

    ALC scores variance reduction, so it goes to the design edges. If the two criteria
    ranked candidates the same way, one of them would be wrong.
    """
    X = np.linspace(0.0, 1.0, 11)[:, None]
    Z = (X[:, 0] - 0.2) ** 2
    gp = lagp.GP(X, Z, np.array([0.1]), 1e-6)
    Xcand = jnp.asarray(np.linspace(0.02, 0.98, 49)[:, None])

    scores = lagp.IECI(fmin=float(Z.min()))(gp, Xcand, Xcand)
    best = float(Xcand[jnp.argmin(scores), 0])
    assert abs(best - 0.2) < 0.05

    edge = float(Xcand[jnp.argmax(lagp.alc(gp, Xcand, Xcand)), 0])
    assert min(edge, 1.0 - edge) < 0.05


def test_ieci_declares_its_direction():
    """The greedy loop reads `minimize` off the criterion instead of hardcoding argmax."""
    assert lagp.IECI(fmin=0.0).minimize is True
    assert getattr(lagp.alc, "minimize", False) is False


def test_ieci_gradient_matches_finite_differences(small):
    """Differentiability is the reason a criterion needs no hand-written derivative."""
    X, Z = small
    gp, Xcand, Xref, fmin = _split(X, Z)
    criterion = lagp.IECI(fmin=fmin)
    scalar = lambda x: criterion(gp, x[None], Xref)[0]

    x0 = Xcand[0]
    got = np.asarray(jax.grad(scalar)(x0))
    step = 1e-6
    want = np.array(
        [
            float(scalar(x0.at[k].add(step)) - scalar(x0.at[k].add(-step))) / (2 * step)
            for k in range(len(x0))
        ]
    )
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-12)


def test_the_local_model_runs_with_ieci(medium):
    """Acceptance: a configurable criterion drops into the local scheme unchanged."""
    X, Z = medium
    XX = np.random.default_rng(4).random((6, X.shape[1]))
    fmin = float(Z.min())
    model = lagp.LocalGP(init=6, neighbors=20)
    out = model.replace(criterion=lagp.IECI(fmin=fmin)).fit(X, Z)(XX, batch=3)
    assert np.all(np.isfinite(np.asarray(out.mean)))
    assert np.asarray(out.neighbors).shape == (6, 20)

    # one reference location, so ECI is monotone in the variance reduction there and the
    # minimize criterion has to reproduce ALC's design exactly -- proof the direction is
    # handled generically rather than by luck
    alc_design = model.fit(X, Z)(XX, batch=3).neighbors
    np.testing.assert_array_equal(np.asarray(out.neighbors), np.asarray(alc_design))

    # with several reference locations the monotone tie breaks and the designs diverge
    Xref = XX[:3]
    ours = model.replace(criterion=lagp.IECI(fmin=fmin)).fit(X, Z).local(Xref).neighbors
    theirs = model.fit(X, Z).local(Xref).neighbors
    assert not np.array_equal(np.asarray(ours), np.asarray(theirs))


def test_ieci_by_name_explains_it_needs_fmin(medium):
    X, Z = medium
    with pytest.raises(ValueError, match="fmin"):
        lagp.LocalGP(init=6, neighbors=20, criterion="ieci")
