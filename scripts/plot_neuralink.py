"""Render the report figures from the published measurements, without recordings or weights."""
import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPORT = Path(__file__).resolve().parents[1] / "reports" / "neuralink"
COLORS = ["#0072B2", "#D55E00", "#007F64"]


def rows(name):
    with (REPORT / name).open() as f:
        return list(csv.DictReader(f))


def ecdf(ax, values, **style):
    ax.ecdf(values, **style)
    ax.set_ylabel("Fraction of recordings")
    ax.set_ylim(0, 1.04)


def save(fig, name):
    fig.savefig(REPORT / f"{name}.png", dpi=200)
    plt.close(fig)


def main():
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "axes.titlelocation": "left", "axes.labelsize": 10})
    fig, ax = plt.subplots(figsize=(6.7, 3.8), layout="constrained")
    test = rows("c128-h512-d3-bench.csv")
    # These mono PCM16 WAVs have a 44-byte header and no trailing chunks.
    samples = (np.array([int(r["wav"]) for r in test]) - 44) // 2
    for key, label, color, line in zip(("satzip", "lzma", "wavpack"), ("satzip", "LZMA", "WavPack"),
                                       COLORS, ("-", "--", "-.")):
        ecdf(ax, 8 * np.array([int(r[key]) for r in test]) / samples,
             label=label, color=color, ls=line)
    ax.set(xlabel="Complete stream size (bits per PCM sample)", xlim=(0, None),
           title=f"Lossless compression on {len(test)} held-out recordings")
    ax.legend(frameon=False)
    save(fig, "held-out-rates")

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10, 3.9), layout="constrained")
    sweep = sorted(rows("validation.csv"), key=lambda r: int(r["parameters"]))
    y = np.arange(len(sweep))
    ax.scatter([float(r["validation_bits_per_sample"]) for r in sweep], y, color=COLORS[0])
    ax.set_yticks(y, [r["model"] for r in sweep])
    ax.set(xlabel="Cross-entropy (bits/sample)", title="A   Validation model sweep")
    ax.ticklabel_format(axis="x", useOffset=False)
    ax.grid(axis="x", alpha=.2)
    corrs = rows("correlation.csv")
    ecdf(bx, [float(r["abs_r"]) for r in corrs], color=COLORS[2])
    bx.set(xlabel="Largest absolute correlation per recording", xlim=(0, .55),
           title=f"B   Aligned prefixes (n = {len(corrs)})")
    save(fig, "validation-and-correlation")


if __name__ == "__main__":
    main()
