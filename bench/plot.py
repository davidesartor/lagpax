"""Plot the timing and accuracy CSVs -> bench/benchmark.png"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
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
    grouped = accuracy.groupby(["method", "m"])[
        ["mean_rel_err_med", "var_rel_err_med"]
    ].max()
    grouped.plot.bar(ax=ax, logy=True)
    ax.set_title("JAX vs laGP predictive agreement")
    ax.set_ylabel("median relative error (worst case)")


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
    main()
