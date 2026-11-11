"""Parity on real response surfaces: the SFU test functions from `vlse`, against laGP.

The other parity modules use random-feature designs that are smooth and well conditioned by
construction. These are the published optimization benchmarks instead -- narrow valleys,
huge dynamic range, many local minima -- which is where a lengthscale fit goes wrong.

What is asserted is what holds on such surfaces, measured with both the pre-vlse solver and
the current one:

- The full-GP joint MLE is never worse than laGP's. On surfaces whose likelihood has a
  single basin the two optima also coincide; elsewhere (Branin, Michalewicz, Rosenbrock,
  Goldstein-Price) ours is often *higher*, by up to ~14 log-likelihood units.
- aGP predictions agree with laGP to ~1e-9..1e-6 at the median on every surface. On the
  multimodal ones (Michalewicz, Ackley) a few locations' local fits land in a different
  basin from laGP's and move the prediction by O(1); there only the median and the rmse
  against the true function are held, not the max.
"""

import jax.numpy as jnp
import numpy as np
import pytest
import vlse
from helpers import fit_both, lagp_prior, make_gp

import lagpax as lagp
from lagpax.priors import Prior, default_nugget, default_squared_lengthscale

SURFACES = {
    "Branin": lambda: vlse.Branin(normalized=True),
    "Hartmann3": lambda: vlse.Hartmann3(normalized=True),
    "Hartmann6": lambda: vlse.Hartmann6(normalized=True),
    "Michalewicz4": lambda: vlse.Michalewicz(d=4, normalized=True),
    "Ackley4": lambda: vlse.Ackley(d=4, normalized=True),
    "Rosenbrock4": lambda: vlse.Rosenbrock(d=4, normalized=True),
    "GoldsteinPrice": lambda: vlse.GoldsteinPrice(normalized=True),
    "Shekel": lambda: vlse.Shekel(normalized=True),
}
# likelihood with one basin: the joint MLE must coincide, not just be no worse
UNIMODAL_FIT = {"Hartmann3", "Hartmann6", "Ackley4", "Shekel"}
# local likelihoods with several basins: per-location predictions may legitimately split
MULTIMODAL = {"Michalewicz4", "Ackley4"}

DRANGE = (1e-3, 10.0)
GRANGE = (float(np.sqrt(np.finfo(float).eps)), 1.0)
DAB, GAB = (1.5, 3.9), (1.5, 4.0)
START, END = 6, 30


def _sample(f, n: int, seed: int):
    """Uniform design on the unit cube, response standardized: laGP is simple kriging."""
    X = np.random.default_rng(seed).random((n, f.d))
    y = np.asarray(f(jnp.asarray(X)))
    mu, sd = y.mean(), y.std()
    return X, (y - mu) / sd, lambda XX: (np.asarray(f(jnp.asarray(XX))) - mu) / sd


def _llik(X, Z, d, g):
    gp = make_gp(jnp.asarray(X), jnp.asarray(Z), jnp.asarray(d), float(g))
    return float(gp.log_likelihood(lagp_prior(DAB, GAB)))


@pytest.mark.parametrize("name", list(SURFACES))
def test_joint_fit_on_test_functions(R, name):
    X, Z, _ = _sample(SURFACES[name](), n=150, seed=1)
    m = X.shape[1]
    d0, g0 = np.full(m, 0.5), 1e-3

    gp = make_gp(jnp.asarray(X), jnp.asarray(Z), jnp.asarray(d0), g0)
    prior = lagp_prior(DAB, GAB, DRANGE, GRANGE)
    _, ours = fit_both(gp, prior)
    want = np.asarray(
        R(
            "gpi <- newGPsep(X, Z, d=d, g=g, dK=TRUE); "
            "out <- jmleGPsep(gpi, drange=dr, grange=gr, dab=dab, gab=gab); "
            "deleteGPsep(gpi); as.numeric(out[1,])",
            X=X,
            Z=Z,
            d=d0,
            g=np.array([g0]),
            dr=np.array(DRANGE),
            gr=np.array(GRANGE),
            dab=np.array(DAB),
            gab=np.array(GAB),
        )
    ).ravel()

    theirs = _llik(X, Z, want[:m], want[m])
    assert _llik(X, Z, ours.squared_lengthscales, ours.nugget) >= theirs - 1e-5 * max(
        1.0, abs(theirs)
    )
    if name in UNIMODAL_FIT:
        np.testing.assert_allclose(np.asarray(ours.squared_lengthscales), want[:m], rtol=1e-3)
        assert float(ours.nugget) == pytest.approx(want[m], rel=1e-3)


@pytest.mark.parametrize("criterion", ["nn", "alc"])
@pytest.mark.parametrize("name", list(SURFACES))
def test_local_model_on_test_functions(R, name, criterion):
    X, Z, truth = _sample(SURFACES[name](), n=800, seed=2)
    XX = np.random.default_rng(3).random((40, X.shape[1]))

    # both sides get the same prior; under n=1000 darg does not subsample, but sharing it
    # keeps the comparison about the port and not about the prior
    d = default_squared_lengthscale(None, X)
    prior = Prior(squared_lengthscale=d, nugget=default_nugget(1 / 10000, Z))
    model = lagp.LocalGP(initial_neighbors=START, neighbors=END, criterion=criterion)
    got = model.fit(X, Z, prior=prior)(XX)
    R(
        f"out <- aGPsep(X, Z, XX, start={START}, end={END}, d={R.darg_list(d)}, "
        f"method='{criterion}', verb=0)",
        X=X,
        Z=Z,
        XX=XX,
    )
    mean = np.asarray(R("out$mean")).ravel()
    gap = np.abs(np.asarray(got.mean) - mean)

    assert np.median(gap) < 1e-4
    if name not in MULTIMODAL:
        assert gap.max() < 1e-2

    rmse = lambda p: float(np.sqrt(np.mean((np.asarray(p) - truth(XX)) ** 2)))
    assert rmse(got.mean) <= 1.1 * rmse(mean)
