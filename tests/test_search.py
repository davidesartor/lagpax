"""Ray and continuous search, laGP's alcray/alcopt without laGP's RNG.

Value parity is not available here: laGP draws the ray starting points with `sample()`
and its 1-D search along a ray is C-only. What *is* testable is that the pieces which do
have a reference agree (the continuous optimum against `alcoptGPsep`), that the search
degenerates to exhaustive ALC when it is allowed to look everywhere, and that the designs
it produces are as good as laGP's by the measure the search is trying to optimize.
"""

import numpy as np
import pytest

import jax
import jax.numpy as jnp

import lagpax as lagp
from lagpax import search as searches
from lagpax.criteria import alc
from lagpax.optim import r_ftol
from vlse.optim import minimise

START, END = 6, 30


def _local(X, Z, Xref, **settings):
    """One local GP at Xref (laGPsep's job), with the shared design sizes."""
    return lagp.LocalGP(init=START, neighbors=END, **settings).fit(X, Z).local(Xref)


def _predict(X, Z, XX, batch=256, **settings):
    """One local GP per row of XX (aGPsep's job)."""
    return lagp.LocalGP(init=START, neighbors=END, **settings).fit(X, Z)(
        XX, batch=batch
    )


def _state(X, Z, Xref, ndesign=11):
    """A greedy loop frozen mid-flight: a small GP, and the rest as candidates."""
    prior = lagp.Prior.from_data(X, Z)
    d0 = np.broadcast_to(np.asarray(prior.theta.init, float), (X.shape[1],))
    g0 = float(prior.nugget.init)
    order = np.argsort(np.sum(np.square(X - Xref), axis=-1))
    gp = lagp.GP(X[order[:ndesign]], Z[order[:ndesign]], d0, g0)
    Xcand = jnp.asarray(X[order[ndesign:]])
    rect = jnp.stack([Xcand.min(0), Xcand.max(0)])
    return gp, Xcand, rect, d0, g0


def _small_pool(X, Z, Xref, ncand=40):
    gp, Xcand, _, _, _ = _state(X, Z, Xref)
    Xcand = Xcand[:ncand]
    rect = jnp.stack([Xcand.min(0), Xcand.max(0)])
    return gp, Xcand, rect


def test_ray_family_covers_every_candidate(medium):
    """Candidate r sits at s=0 on ray r, so the ray continuum cannot lose to exhaustive.

    This is the property the search rests on; only the snap back to a candidate can give
    ground, and `test_ray_degenerates_to_exhaustive` bounds how much.
    """
    X, Z = medium
    for xref in np.random.default_rng(0).random((5, X.shape[1])):
        Xref = jnp.asarray(xref[None, :])
        gp, Xcand, rect = _small_pool(X, Z, np.asarray(Xref))
        Xend = searches._ray_ends(Xref, Xcand, rect)

        zeros = jnp.zeros(len(Xcand))
        _, val = searches._golden_max(
            lambda s: alc(gp, Xcand + s[:, None] * (Xend - Xcand), Xref),
            zeros,
            zeros + 1.0,
            40,
        )
        best = float(jnp.max(alc(gp, Xcand, Xref)))
        assert float(jnp.max(val)) >= best * (1.0 - 1e-6)


def test_ray_degenerates_to_exhaustive(medium):
    """With a ray per candidate, ray search must recover the exhaustive ALC optimum.

    Index equality is the usual outcome but not guaranteed: the maximum along the rays is
    off-candidate, and snapping can land one candidate over. The criterion value is what
    the search is for, so that is what is asserted.
    """
    X, Z = medium
    for xref in np.random.default_rng(0).random((5, X.shape[1])):
        Xref = jnp.asarray(xref[None, :])
        gp, Xcand, rect = _small_pool(X, Z, np.asarray(Xref))
        taken = jnp.zeros(len(Xcand), bool)
        score = alc(gp, Xcand, Xref)

        want = lagp.Exhaustive()(alc, gp, Xcand, Xref, taken, 0, rect)
        for t in range(4):
            got = lagp.Ray(numrays=len(Xcand), iters=40)(
                alc, gp, Xcand, Xref, taken, jnp.asarray(t), rect
            )
            assert float(score[got]) >= 0.99 * float(score[want])


def _projected_grad(f, x, rect):
    """Gradient with the components pushing into a bound zeroed."""
    grad = jax.grad(f)(x)
    free = ((x > rect[0] + 1e-12) | (grad < 0)) & ((x < rect[1] - 1e-12) | (grad > 0))
    return jnp.where(free, grad, 0.0)


def test_alcoptgpsep_optima_are_stationary_for_our_gradient(R, medium):
    """laGP's ALC optimum has to be an optimum for us too, or `jax.grad` disagrees with
    laGP's hand-written `dalcGPsep`.

    Started from `alcoptGPsep`'s answer, our projected L-BFGS must stay in that basin and
    must not come out worse. Starting both optimizers from the same *candidate* proves much
    less: ALC over the box is multimodal, and the two land in different basins about a
    quarter of the time, each winning some. laGP stops on its own `tol`, so it does not sit
    exactly on the stationary point -- we polish it by <1% of the box and gain value.
    """
    X, Z = medium
    for xref in np.random.default_rng(11).random((4, X.shape[1])):
        Xref = xref[None, :]
        gp, Xcand, rect, d0, g0 = _state(X, Z, Xref)

        R(
            "gpi <- newGPsep(Xd, Zd, d=dd, g=gg, dK=TRUE);"
            " opt <- alcoptGPsep(gpi, Xref, Xstart, rect[1,], rect[2,])",
            Xd=np.asarray(gp.X),
            Zd=np.asarray(gp.y),
            dd=d0,
            gg=np.array([g0]),
            Xref=Xref,
            Xstart=np.asarray(Xcand[0]),
            rect=np.asarray(rect),
        )
        want = np.asarray(R("opt$par")).ravel()
        want_alc = R.scalar("alcGPsep(gpi, matrix(opt$par, nrow=1), Xref)")

        objective = lambda x: -alc(gp, x[None], jnp.asarray(Xref))[0]
        res = minimise(
            objective,
            jnp.asarray(want),
            (rect[0], rect[1]),
            tol=0.0,
            ftol=r_ftol(jnp.float64),
        )
        got_alc = float(alc(gp, res.x[None], jnp.asarray(Xref))[0])

        assert got_alc >= want_alc * (1.0 - 1e-8)
        width = np.asarray(rect[1] - rect[0])
        assert float(np.max(np.abs(np.asarray(res.x) - want) / width)) < 0.01

        # and we sit closer to stationary than laGP's own stopping point does
        projected = lambda x: jnp.max(jnp.abs(_projected_grad(objective, x, rect)))
        assert float(projected(res.x)) < 0.1 * float(projected(jnp.asarray(want)))


def _var_of(X, Z, idx, Xref, d0, g0):
    return float(lagp.GP(X[idx], Z[idx], d0, g0).predict(Xref).variance[0])


@pytest.mark.parametrize("search,method", [("ray", "alcray"), ("continuous", "alcopt")])
def test_search_designs_are_competitive_with_lagp(R, medium, search, method):
    """laGP's own criterion, applied to both designs: ours must not be much worse.

    Step-by-step agreement is unreachable — laGP's ray starts are drawn with R's RNG, so
    the two greedy paths diverge on the first step and stay diverged. What has to hold is
    that the design we end up with is as good as theirs at the thing ALC measures.
    """
    X, Z = medium
    prior = lagp.Prior.from_data(X, Z)
    d0 = np.broadcast_to(np.asarray(prior.theta.init, float), (X.shape[1],))
    g0 = float(prior.nugget.init)

    ratios = []
    for xref in np.random.default_rng(7).random((6, X.shape[1])):
        Xref = xref[None, :]
        Xi = (
            np.asarray(
                R(
                    f"set.seed(1); out <- laGPsep(Xref, {START}, {END}, X, Z,"
                    f" method='{method}', lite=TRUE); out$Xi",
                    Xref=Xref,
                    X=X,
                    Z=Z,
                )
            )
            .ravel()
            .astype(int)
            - 1
        )
        ours = np.asarray(_local(X, Z, Xref, search=search).neighbors)
        ratios.append(
            _var_of(X, Z, ours, Xref, d0, g0) / _var_of(X, Z, Xi, Xref, d0, g0)
        )

    assert float(np.median(ratios)) <= 1.5


def test_search_accuracy_tracks_exhaustive(medium):
    """The point of ray/continuous is cost, not accuracy: they must not cost accuracy."""
    X, Z = medium
    XX = np.random.default_rng(99).random((40, X.shape[1]))
    truth = np.sin(4 * XX[:, 0]) + np.cos(3 * XX[:, 1:3].sum(-1)) + 0.1 * XX.sum(-1)
    rmse = lambda p: float(np.sqrt(np.mean((np.asarray(p.mean) - truth) ** 2)))

    nearest = rmse(_predict(X, Z, XX, batch=8, criterion="nn"))
    full = rmse(_predict(X, Z, XX, batch=8))
    for search in ("ray", "continuous"):
        got = rmse(_predict(X, Z, XX, batch=8, search=search))
        assert got <= 1.5 * full
        assert got <= nearest


def test_exhaustive_stays_the_default(medium):
    """Every divergent option must leave the laGP-equivalent configuration untouched."""
    X, Z = medium
    Xref = X[:1] + 0.01
    a = _local(X, Z, Xref)
    b = _local(X, Z, Xref, search="exhaustive")
    np.testing.assert_array_equal(np.asarray(a.neighbors), np.asarray(b.neighbors))
    np.testing.assert_array_equal(np.asarray(a.mean), np.asarray(b.mean))


@pytest.mark.parametrize("search", ["ray", "continuous"])
def test_search_is_batch_invariant(medium, search):
    """Both searches have to survive `lax.map`, which is where fixed shapes get tested."""
    X, Z = medium
    XX = X[:8] + 0.015
    a = _predict(X, Z, XX, batch=2, search=search)
    b = _predict(X, Z, XX, batch=None, search=search)
    np.testing.assert_array_equal(np.asarray(a.neighbors), np.asarray(b.neighbors))
    np.testing.assert_allclose(np.asarray(a.mean), np.asarray(b.mean), rtol=1e-9)


def test_ray_end_stays_inside_the_box():
    """ray.end casts far and pulls back; the pulled-back point must be on the box."""
    rng = np.random.default_rng(3)
    rect = jnp.asarray(np.stack([np.zeros(4), np.ones(4)]))
    Xref = jnp.asarray(rng.random((1, 4)))
    Xstart = jnp.asarray(rng.random((7, 4)))
    Xend = searches._ray_ends(Xref, Xstart, rect)

    assert bool(jnp.all(Xend >= rect[0] - 1e-12))
    assert bool(jnp.all(Xend <= rect[1] + 1e-12))
    # each ray keeps its direction and reaches a face
    step = Xend - Xstart
    assert bool(jnp.all(jnp.sum(step * (Xstart - Xref), axis=-1) >= -1e-12))
    on_face = jnp.min(
        jnp.minimum(jnp.abs(Xend - rect[0]), jnp.abs(Xend - rect[1])), axis=-1
    )
    assert bool(jnp.all(on_face < 1e-9))


def test_unknown_search_is_rejected():
    with pytest.raises(ValueError):
        lagp.LocalGP(search="teleport")


def test_a_search_is_configured_by_its_settings(medium):
    """`criterion x search` is a grid: configuring a search must not need a new name."""
    X, Z = medium
    Xref = X[:1] + 0.01
    out = _local(X, Z, Xref, search=lagp.Ray(numrays=8, iters=20))
    assert np.asarray(out.neighbors).shape == (END,)
    assert np.isfinite(float(out.mean[0]))
