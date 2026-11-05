"""Do the JAX port and laGP predict the same thing? -> bench/accuracy.csv"""

import jax
import numpy as np
import pandas as pd

from bench.common import BENCH_DIR, RlaGP, design
import lagpax as lagp
from lagpax.prior import Prior, darg, garg

CASES = [
    dict(n=2000, nn=100, m=2),
    dict(n=2000, nn=100, m=4),
    dict(n=8000, nn=100, m=4),
    dict(n=8000, nn=100, m=8),
]
SEEDS = range(3)
START, END = 6, 50


def relative_error(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(a - b) / np.maximum(np.abs(b), 1e-12)


def run() -> pd.DataFrame:
    R = RlaGP()
    rows = []
    for case in CASES:
        for method in ("nn", "alc"):
            for seed in SEEDS:
                X, Z, XX = design(seed=seed, **case)
                # both sides get the same prior; darg's own subsample is RNG-dependent
                d = darg(None, X, key=jax.random.key(seed))
                prior = Prior(d=d, g=garg(1 / 10000, Z))
                ours = lagp.agp(
                    X, Z, XX, prior=prior, start=START, end=END, criterion=method
                )
                mean, var = R.agp(X, Z, XX, START, END, method, d=d)

                mean_err = relative_error(np.asarray(ours.mean), mean)
                var_err = relative_error(np.asarray(ours.var), var)
                spread = float(np.std(mean))
                rows.append(
                    dict(
                        **case,
                        method=method,
                        seed=seed,
                        mean_rel_err_max=float(mean_err.max()),
                        mean_rel_err_med=float(np.median(mean_err)),
                        mean_rmse_over_sd=float(
                            np.sqrt(np.mean((np.asarray(ours.mean) - mean) ** 2))
                            / spread
                        ),
                        var_rel_err_max=float(var_err.max()),
                        var_rel_err_med=float(np.median(var_err)),
                        # a location this far off means the two greedy searches diverged
                        frac_diverged=float(np.mean(mean_err > 1e-6)),
                    )
                )
                print(rows[-1])

    df = pd.DataFrame(rows)
    df.to_csv(BENCH_DIR / "accuracy.csv", index=False)
    return df


if __name__ == "__main__":
    print(run().to_string(index=False))
