"""Wall-clock aGPsep: JAX (CPU or GPU) vs laGP (1 thread and all cores) -> bench/timing.csv"""

import argparse
import os

import numpy as np
import pandas as pd

from bench.common import BENCH_DIR, RlaGP, design, timeit

CASES = (
    [dict(n=2000, nn=nn, m=4, seeds=3) for nn in (100, 1000)]
    + [dict(n=n, nn=1000, m=4, seeds=3) for n in (8000, 32000)]
    # laGP on one core owns the budget here, so this regime runs a single seed
    + [dict(n=100_000, nn=10_000, m=4, seeds=1)]
)
START, END = 6, 50


def jax_backend() -> str:
    import jax

    return jax.devices()[0].platform


def bench_jax(case, seed, method, batch):
    import jax

    import lagpax as lagp

    X, Z, XX = design(seed=seed, **case)

    def run():
        out = lagp.agp(
            X, Z, XX, start=START, end=END, criterion=method, batch=batch
        )
        return jax.block_until_ready(out)

    return timeit(run)[0]


def bench_lagp(R, case, seed, method):
    X, Z, XX = design(seed=seed, **case)
    return timeit(lambda: R.agp(X, Z, XX, START, END, method), repeat=1)[0]


def run(methods, batch, skip_r, out="timing.csv") -> pd.DataFrame:
    backend = jax_backend()
    # cpu_count reports the machine; the cgroup/scheduler affinity is what we actually get
    cores = len(os.sched_getaffinity(0))
    engines = {
        f"jax-{backend}-{cores}core" if backend == "cpu" else f"jax-{backend}": None
    }
    if not skip_r:
        engines["lagp-1core"] = RlaGP(threads=1)
        if cores > 1:
            engines[f"lagp-{cores}core"] = RlaGP(threads=cores)

    rows = []
    for spec in CASES:
        case = {k: v for k, v in spec.items() if k != "seeds"}
        for method in methods:
            for seed in range(spec["seeds"]):
                for name, R in engines.items():
                    seconds = (
                        bench_jax(case, seed, method, batch)
                        if R is None
                        else bench_lagp(R, case, seed, method)
                    )
                    rows.append(
                        dict(
                            **case,
                            method=method,
                            seed=seed,
                            engine=name,
                            batch=batch if R is None else None,
                            seconds=seconds,
                        )
                    )
                    print(rows[-1], flush=True)
                    # rewritten every row so a walltime kill still leaves usable data
                    pd.DataFrame(rows).to_csv(BENCH_DIR / out, index=False)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--methods", nargs="+", default=["nn", "alc"])
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--skip-r", action="store_true")
    p.add_argument("--out", default="timing.csv")
    a = p.parse_args()
    print(run(a.methods, a.batch, a.skip_r, a.out).to_string(index=False))
