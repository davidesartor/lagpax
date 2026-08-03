"""Shared pieces for the JAX-vs-laGP benchmarks: designs, the R bridge, and timing."""

import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")

# BLAS must be pinned before R loads, so laGP's omp.threads is the only knob that varies
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import time
from pathlib import Path
from typing import Callable

import numpy as np

BENCH_DIR = Path(__file__).resolve().parent
R_LIB = "~/R/library"


def ackley(X: np.ndarray) -> np.ndarray:
    """Ackley on [0, 1]^m rescaled to the usual [-32.768, 32.768]^m domain."""
    x = 65.536 * np.asarray(X) - 32.768
    m = x.shape[-1]
    return (
        -20 * np.exp(-0.2 * np.sqrt(np.sum(x**2, axis=-1) / m))
        - np.exp(np.sum(np.cos(2 * np.pi * x), axis=-1) / m)
        + 20
        + np.e
    )


def design(n: int, nn: int, m: int, seed: int) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    X, XX = rng.random((n, m)), rng.random((nn, m))
    return X, ackley(X), XX


def timeit(fn: Callable, repeat: int = 3) -> tuple[float, object]:
    """Best-of-`repeat` wall time, after one warm-up call that absorbs JIT compilation."""
    out = fn()
    best = np.inf
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return best, out


class RlaGP:
    """laGP through rpy2, with numpy arrays pushed into the R global environment."""

    def __init__(self, threads: int = 1):
        import rpy2.robjects as ro
        from rpy2.robjects import numpy2ri
        from rpy2.robjects.packages import importr

        self.ro, self.numpy2ri = ro, numpy2ri
        ro.r(f'.libPaths("{R_LIB}")')
        importr("laGP")
        self.threads = threads

    def __call__(self, script: str, **data):
        from rpy2.robjects.conversion import localconverter

        with localconverter(self.ro.default_converter + self.numpy2ri.converter) as cv:
            for name, value in data.items():
                self.ro.globalenv[name] = cv.py2rpy(np.asarray(value))
            return cv.rpy2py(self.ro.r(script))

    def agpsep(self, X, Z, XX, start: int, end: int, method: str, d=None):
        """Run aGPsep and return (mean, var), with laGP doing its own timing internally."""
        # darg subsamples with R's RNG when n > 1000, so parity runs must supply d explicitly
        arg = ""
        if d is not None:
            self(
                f"dprior <- list(start=dstart, mle={'TRUE' if d.mle else 'FALSE'}, "
                "min=dmin, max=dmax, ab=dab)",
                dstart=np.atleast_1d(d.start),
                dmin=np.array([d.min]),
                dmax=np.array([d.max]),
                dab=np.asarray(d.ab),
            )
            arg = ", d=dprior"
        self(
            f"out <- aGPsep(X, Z, XX, start={start}, end={end}, method='{method}'"
            f"{arg}, omp.threads={self.threads}, verb=0)",
            X=X,
            Z=Z,
            XX=XX,
        )
        mean = np.asarray(self("out$mean")).ravel()
        var = np.asarray(self("out$var")).ravel()
        return mean, var
