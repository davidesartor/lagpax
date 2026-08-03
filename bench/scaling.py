"""Cost per predictive location as the candidate set grows -> bench/scaling.csv

`close` is pinned to `n` so the candidate set really does grow: with laGP's default cap
every search looks flat and the plot says nothing. Exhaustive scores every candidate at
every greedy step, so it is linear in `n`; ray and continuous spend `iters` (or L-BFGS)
criterion evaluations on a fixed number of starting points, so they should be flat.
"""

import argparse

import numpy as np
import pandas as pd

from bench.common import BENCH_DIR, ackley, design, timeit

NS = (1000, 4000, 16000, 64000)
SEARCHES = ("exhaustive", "ray", "continuous")
START, END, M, NN = 6, 50, 4, 20


def run(ns, searches, out="scaling.csv") -> pd.DataFrame:
    import jax

    import lagpax as lagp

    rows = []
    for n in ns:
        X, Z, XX = design(n=n, nn=NN, m=M, seed=0)
        truth = ackley(XX)
        for search in searches:
            fn = lambda: jax.block_until_ready(
                lagp.agp(
                    X, Z, XX, start=START, end=END, search=search, close=n, batch=NN
                )
            )
            seconds, pred = timeit(fn)
            rows.append(
                dict(
                    n=n,
                    nn=NN,
                    m=M,
                    search=search,
                    seconds=seconds,
                    per_location=seconds / NN,
                    rmse=float(np.sqrt(np.mean((np.asarray(pred.mean) - truth) ** 2))),
                )
            )
            print(rows[-1], flush=True)
            pd.DataFrame(rows).to_csv(BENCH_DIR / out, index=False)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ns", nargs="+", type=int, default=list(NS))
    p.add_argument("--searches", nargs="+", default=list(SEARCHES))
    p.add_argument("--out", default="scaling.csv")
    a = p.parse_args()
    print(run(a.ns, a.searches, a.out).to_string(index=False))
