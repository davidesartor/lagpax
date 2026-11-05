"""Plot the timing and accuracy CSVs -> bench/benchmark.png"""

import argparse

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bench.common import BENCH_DIR


def _speed_panel(ax, timing: pd.DataFrame, method: str):
    sub = timing[timing.method == method]
    grouped = sub.groupby(["engine", "n", "nn"]).seconds.median().reset_index()
    grouped["case"] = grouped.n.astype(str) + "x" + grouped.nn.astype(str)

    for engine, part in grouped.groupby("engine"):
        ax.plot(part.case, part.seconds, marker="o", label=engine)
    ax.set_yscale("log")
    ax.set_title(f"aGPsep, method={method}")
    ax.set_xlabel("n x nn")
    ax.set_ylabel("seconds (median of seeds)")
    ax.legend()


def _accuracy_panel(ax, accuracy: pd.DataFrame):
    grouped = accuracy.groupby(["method", "m"])[["mean_rel_err_med", "var_rel_err_med"]].max()
    grouped.plot.bar(ax=ax, logy=True)
    ax.set_title("JAX vs laGP predictive agreement")
    ax.set_ylabel("median relative error (worst case)")


def affine_fit(nn, seconds) -> tuple[float, float]:
    """Least-squares `fixed + per_location * nn`, the two numbers the sweep exists to get."""
    design = np.c_[np.ones(len(nn)), nn]
    fixed, per_location = np.linalg.lstsq(design, np.asarray(seconds), rcond=None)[0]
    return fixed, per_location


def _locations_figure(csvs: list[str], out: str):
    """Wall clock and speedup against the predictive-location count, at fixed n."""
    timing = pd.concat([pd.read_csv(BENCH_DIR / c) for c in csvs], ignore_index=True)
    methods = sorted(timing.method.unique())
    fig, axes = plt.subplots(2, len(methods), figsize=(6 * len(methods), 9), squeeze=False)

    for col, method in enumerate(methods):
        sub = timing[timing.method == method]
        grouped = sub.groupby(["engine", "nn"]).seconds.median().reset_index()

        top = axes[0][col]
        for engine, part in grouped.groupby("engine"):
            fixed, per_location = affine_fit(part.nn.values, part.seconds.values)
            label = f"{engine}: {fixed:.1f}s + {per_location * 1e3:.3f}ms/loc"
            top.plot(part.nn, part.seconds, marker="o", label=label)
        top.set(xscale="log", yscale="log", xlabel="predictive locations")
        top.set_ylabel("seconds")
        top.set_title(f"aGPsep at n={int(sub.n.iloc[0])}, method={method}")
        top.legend()

        # laGP is linear in nn and we are not, so the ratio is the story
        wide = grouped.pivot(index="nn", columns="engine", values="seconds")
        ours = next(c for c in wide.columns if c.startswith("jax"))
        bottom = axes[1][col]
        for engine in (c for c in wide.columns if c != ours):
            bottom.plot(wide.index, wide[engine] / wide[ours], marker="o", label=engine)
        bottom.axhline(1.0, color="grey", lw=0.8)
        bottom.set(xscale="log", xlabel="predictive locations")
        bottom.set_ylabel(f"speedup over {ours}")
        bottom.legend()

    fig.tight_layout()
    fig.savefig(BENCH_DIR / out, dpi=150)
    print(BENCH_DIR / out)


def main():
    timing = pd.read_csv(BENCH_DIR / "timing.csv")
    accuracy = pd.read_csv(BENCH_DIR / "accuracy.csv")

    methods = sorted(timing.method.unique())
    fig, axes = plt.subplots(1, len(methods) + 1, figsize=(6 * (len(methods) + 1), 5))
    for ax, method in zip(axes, methods):
        _speed_panel(ax, timing, method)
    _accuracy_panel(axes[-1], accuracy)

    fig.tight_layout()
    fig.savefig(BENCH_DIR / "benchmark.png", dpi=150)
    print(BENCH_DIR / "benchmark.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # the location sweep is its own figure: one n, many nn, and a speedup curve
    p.add_argument("--locations", nargs="+", default=None)
    p.add_argument("--out", default="locations.png")
    a = p.parse_args()
    if a.locations:
        _locations_figure(a.locations, a.out)
    else:
        main()
