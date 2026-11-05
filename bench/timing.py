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

# laGP is linear in the number of predictive locations while we are nearly flat until the
# GPU saturates, so the location count -- not n -- is what the ratio actually turns on.
# The last point is jax-only territory: laGP needs about an hour of 16 cores to reach it.
LOCATIONS = [
    dict(n=8000, nn=nn, m=4, seeds=1)
    for nn in (250, 1000, 4000, 16_000, 64_000, 256_000)
]

SUITES = {"default": CASES, "locations": LOCATIONS}
START, END = 6, 50


def jax_backend() -> str:
    import jax

    return jax.devices()[0].platform


def bench_jax(case, seed, method, batch):
    import jax

    import lagpax as lagp

    X, Z, XX = design(seed=seed, **case)

    def run():
        model = lagp.LocalGP(initial_neighbors=START, neighbors=END, criterion=method)
        out = model.fit(X, Z, key=jax.random.key(seed))(XX, batch_size=batch)
        return jax.block_until_ready(out)

    # a repeat costs a full pass, and the spread shrinks as the pass gets longer
    return timeit(run, repeat=1 if case["nn"] >= 16_000 else 3)[0]


def bench_lagp(R, case, seed, method):
    X, Z, XX = design(seed=seed, **case)
    return timeit(lambda: R.agp(X, Z, XX, START, END, method), repeat=1)[0]


def run(
    methods,
    batch,
    skip_r,
    out="timing.csv",
    cases=CASES,
    seeds=None,
    r_threads=("1", "all"),
) -> pd.DataFrame:
    backend = jax_backend()
    # cpu_count reports the machine; the cgroup/scheduler affinity is what we actually get
    cores = len(os.sched_getaffinity(0))
    engines = {
        f"jax-{backend}-{cores}core" if backend == "cpu" else f"jax-{backend}": None
    }
    if not skip_r:
        # single-core laGP owns the budget at large nn, so it can be dropped on its own
        for spec in dict.fromkeys(cores if t == "all" else int(t) for t in r_threads):
            if spec == 1 or cores > 1:
                engines[f"lagp-{spec}core"] = RlaGP(threads=spec)

    rows = []
    for spec in cases:
        case = {k: v for k, v in spec.items() if k != "seeds"}
        for method in methods:
            for seed in range(seeds if seeds is not None else spec["seeds"]):
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
    p.add_argument("--suite", choices=sorted(SUITES), default="default")
    # a walltime-bounded run wants a subset: indices into the suite, and a seed override
    p.add_argument("--cases", nargs="+", type=int, default=None)
    p.add_argument("--seeds", type=int, default=None)
    p.add_argument("--r-threads", nargs="+", default=["1", "all"])
    a = p.parse_args()
    suite = SUITES[a.suite]
    cases = suite if a.cases is None else [suite[i] for i in a.cases]
    print(
        run(a.methods, a.batch, a.skip_r, a.out, cases, a.seeds, a.r_threads).to_string(
            index=False
        )
    )
