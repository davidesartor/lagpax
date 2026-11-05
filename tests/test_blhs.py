"""Block Latin hypercube subsampling against laGP's blhs / blhs.loop."""

import jax
import numpy as np
import pytest

import lagpax as lagp
from lagpax.blhs import blocks, bootstrap, latin, subsample
from lagpax.gp import llik_d
from lagpax.mle import mle_d

G = 1e-3

# blhs' only randomness is one `sample` per input dimension, so replaying the same seed
# reproduces exactly the Latin hypercube the R call used
LABELS = (
    "set.seed(seed[1]); L <- matrix(NA, m[1], ncol(X)); "
    "for(h in 1:ncol(X)) L[,h] <- sample(1:m[1], m[1], replace=FALSE); L"
)
SUB = "set.seed(seed[1]); sub <- laGP::blhs(y=Z, X=X, m=m[1]); "


@pytest.mark.parametrize("seed", [1, 7, 42])
def test_blocks_matches_r(R, small, seed):
    X, Z = small
    m = np.array([3])
    args = dict(X=X, Z=Z, m=m, seed=np.array([seed]))

    labels = np.asarray(R(LABELS, **args)).astype(int)
    got = blocks(X, int(m[0]), labels)

    want_X = np.atleast_2d(np.asarray(R(SUB + "as.matrix(sub$xs)", **args)))
    want_Z = np.asarray(R(SUB + "as.numeric(sub$ys)", **args)).ravel()
    np.testing.assert_array_equal(X[got], want_X)
    np.testing.assert_array_equal(Z[got], want_Z)


def test_blocks_matches_r_on_a_wider_design(R, medium):
    X, Z = medium
    m = np.array([3])
    args = dict(X=X, Z=Z, m=m, seed=np.array([11]))

    got = blocks(X, int(m[0]), np.asarray(R(LABELS, **args)).astype(int))
    want = np.atleast_2d(np.asarray(R(SUB + "as.matrix(sub$xs)", **args)))
    np.testing.assert_array_equal(X[got], want)


def test_latin_is_a_permutation_per_dimension():
    L = latin(jax.random.key(0), 5, 3)
    assert L.shape == (5, 3)
    for col in L.T:
        np.testing.assert_array_equal(np.sort(col), np.arange(1, 6))


def test_subsample_keeps_rows_intact(small):
    X, Z = small
    Xs, Zs = subsample(X, Z, m=3, key=jax.random.key(3))
    assert 0 < len(Xs) < len(X) and Xs.shape[1] == X.shape[1]
    for x, z in zip(Xs, Zs):
        i = np.flatnonzero((X == x).all(-1))[0]
        assert Z[i] == z


def test_subsample_rejects_shapes_blhs_rejects(small):
    X, Z = small
    with pytest.raises(ValueError, match="at least 2 input"):
        subsample(X[:, :1], Z, m=3, key=jax.random.key(0))
    with pytest.raises(ValueError, match="m must be positive"):
        subsample(X, Z, m=0, key=jax.random.key(0))
    with pytest.raises(ValueError, match=r"m <= n"):
        subsample(X, Z, m=20, key=jax.random.key(0))


def test_bootstrap_median_matches_r(R, medium):
    """laGP fits each subsample with mleGPsep on a tenfold-widened upper bound.

    Compared as an objective, not an argmin: neither optimizer reproduces the other's
    iterate trajectory, and `factr=1e7` leaves the two answers `sqrt(factr*eps) ~ 5e-5`
    apart while the MAP objective they reach agrees far more tightly. The median of `K`
    inherits exactly that gap, so it is checked loosely and the likelihood strictly.
    """
    X, Z = medium
    m, K = 3, 5
    prior = lagp.Prior.from_design(X, Z)
    ab = np.asarray(prior.d.ab)

    # bootstrap's own key schedule: one split for a prior it may need, then one key per draw
    key, _ = jax.random.split(jax.random.key(9))
    draws = [subsample(X, Z, m, key=k) for k in jax.random.split(key, K)]
    theirs = [
        np.asarray(
            R(
                "gpsepi <- newGPsep(Xs, Zs, d=start, g=g[1], dK=TRUE); "
                "out <- mleGPsep(gpsepi, tmin=tmin[1], tmax=tmax[1], ab=ab, maxit=100)$d; "
                "deleteGPsep(gpsepi); out",
                Xs=Xs,
                Zs=Zs,
                start=np.broadcast_to(prior.d.start, (X.shape[1],)),
                g=np.array([G]),
                tmin=np.array([prior.d.min]),
                tmax=np.array([10.0 * prior.d.max]),
                ab=ab,
            )
        ).ravel()
        for Xs, Zs in draws
    ]
    ours = [
        np.asarray(
            mle_d(
                lagp.GP(Xs, Zs, prior.d.start, G),
                prior.d.min,
                10.0 * prior.d.max,
                ab=tuple(ab),
                maxit=100,
            ).d
        )
        for Xs, Zs in draws
    ]

    # `- 1e-6` is the same slack tests/test_mle.py allows: both optimizers stop on R's
    # factr, so the objectives agree to that and neither side is reliably the better one
    for (Xs, Zs), mine, theirs_d in zip(draws, ours, theirs):
        gp = lagp.GP(Xs, Zs, prior.d.start, G)
        assert (
            llik_d(gp, mine, dab=tuple(ab))
            >= llik_d(gp, theirs_d, dab=tuple(ab)) - 1e-6
        )

    got = bootstrap(X, Z, m, K=K, prior=prior, g=G, key=jax.random.key(9))
    np.testing.assert_array_equal(np.asarray(got.d.start), np.median(ours, axis=0))
    np.testing.assert_allclose(
        np.asarray(got.d.start), np.median(theirs, axis=0), rtol=1e-3
    )


def test_bootstrap_returns_a_usable_prior(medium):
    X, Z = medium
    prior = lagp.Prior.from_design(X, Z)
    boot = lagp.bootstrap(X, Z, m=3, K=3, prior=prior, key=jax.random.key(1))

    start = np.asarray(boot.d.start)
    assert start.shape == (X.shape[1],)
    assert np.all(start >= boot.d.min) and np.all(start <= 10.0 * boot.d.max)

    # the only field that changes: everything else has to survive for `condition` to work
    assert (boot.d.min, boot.d.max, boot.d.ab) == (prior.d.min, prior.d.max, prior.d.ab)
    out = lagp.agp(X, Z, X[:4] + 0.01, prior=boot, start=6, end=12)
    assert np.all(np.isfinite(np.asarray(out.mean)))


def test_bootstrap_is_deterministic_given_a_key(medium):
    X, Z = medium
    prior = lagp.Prior.from_design(X, Z)
    kw = dict(m=3, K=3, prior=prior)
    a = lagp.bootstrap(X, Z, key=jax.random.key(4), **kw)
    b = lagp.bootstrap(X, Z, key=jax.random.key(4), **kw)
    np.testing.assert_array_equal(np.asarray(a.d.start), np.asarray(b.d.start))
