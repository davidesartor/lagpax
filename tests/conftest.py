"""Shared fixtures: the R laGP reference and the designs the comparisons run on."""

import os

# laGP is float64 throughout; lagpax follows the input dtype, so parity needs x64 on first
import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest

R_LIB = os.environ.get("LAGPAX_R_LIB", "~/R/library")

_skipped: set[str] = set()


def pytest_runtest_logreport(report):
    if report.skipped and not hasattr(report, "wasxfail"):
        _skipped.add(report.nodeid)


def pytest_sessionfinish(session, exitstatus):
    """Under LAGPAX_STRICT a skip fails the run: it means R never loaded and nothing was compared."""
    if _skipped and os.environ.get("LAGPAX_STRICT"):
        session.exitstatus = 1
        print(
            f"\nLAGPAX_STRICT: {len(_skipped)} test(s) skipped, so the suite proved nothing:"
        )
        for nodeid in sorted(_skipped):
            print(f"  {nodeid}")


class RlaGP:
    """Evaluate R snippets with numpy inputs pushed into the R global environment."""

    def __init__(self, ro, numpy2ri):
        self.ro = ro
        self.numpy2ri = numpy2ri

    def __call__(self, script: str, **data):
        from rpy2.robjects.conversion import localconverter

        with localconverter(self.ro.default_converter + self.numpy2ri.converter) as cv:
            for name, value in data.items():
                self.ro.globalenv[name] = cv.py2rpy(np.asarray(value))
            return cv.rpy2py(self.ro.r(script))

    def scalar(self, script: str, **data) -> float:
        return float(np.asarray(self(script, **data)).ravel()[0])

    def darg_list(self, d) -> str:
        """R expression for a darg/garg-shaped list, so both sides get identical settings."""
        start = ",".join(repr(float(v)) for v in np.atleast_1d(d.start))
        return (
            f"list(start=c({start}), min={d.min!r}, max={d.max!r}, "
            f"mle={'TRUE' if d.estimate else 'FALSE'}, ab=c({d.ab[0]!r},{d.ab[1]!r}))"
        )


@pytest.fixture(scope="session")
def R() -> RlaGP:
    """laGP through rpy2; skips when rpy2 or R is not usable in this environment."""
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import numpy2ri
        from rpy2.robjects.packages import importr
    except Exception as e:  # rpy2 missing, or libR not on the loader path
        pytest.skip(f"rpy2 unavailable ({e}); run under `module load r/4.4.0`")

    # CI installs into R's own default library, so an empty LAGPAX_R_LIB means "leave it alone"
    if R_LIB:
        ro.r(f'.libPaths("{R_LIB}")')
    try:
        importr("laGP")
    except Exception as e:
        pytest.skip(f"laGP not importable ({e}); looked in {ro.r('.libPaths()')}")
    return RlaGP(ro, numpy2ri)


def _design(
    n: int, m: int, seed: int, noise: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.random((n, m))
    Z = np.sin(4 * X[:, 0]) + np.cos(3 * X[:, 1 : min(m, 3)].sum(-1)) + 0.1 * X.sum(-1)
    return X, Z + noise * rng.standard_normal(n)


@pytest.fixture(scope="session")
def small():
    """A design small enough that laGP's darg subsampling never triggers."""
    return _design(n=40, m=3, seed=0)


@pytest.fixture(scope="session")
def medium():
    """n < 1000, so darg is deterministic on both sides."""
    return _design(n=600, m=4, seed=1)


@pytest.fixture(scope="session")
def noisy():
    """Observation noise makes the nugget identifiable, so g has an interior optimum."""
    return _design(n=300, m=3, seed=5, noise=0.2)


@pytest.fixture(scope="session")
def wide():
    """Higher input dimension, where the lengthscale MLE is harder."""
    return _design(n=200, m=8, seed=2)
