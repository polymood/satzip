"""Build the Sentinel report figures from the measured streams and source tiles."""
import argparse
import csv

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from compare_tiles import ARCHIVES, PRIOR, REPORT
from satzip import lab


def preview(img, channels, fill):
    pixels = img[channels].astype(float)
    valid = np.ones(pixels.shape[1:], bool) if fill is None else (pixels != fill).all(axis=0)
    for band in pixels:
        low, high = np.percentile(band[valid], [2, 98])
        band[:] = np.clip((band - low) / max(high - low, 1), 0, 1)
    pixels[:, ~valid] = 0
    return pixels.transpose(1, 2, 0) if len(channels) == 3 else pixels[0]


def measure_and_preview():
    names = {f: s for s, f in lab.SAMPLES.items() if f}
    names.update(zip(lab.PRIORS[PRIOR], ("S2 Paris", "S2 Lyon", "S2 Nantes"), strict=True))
    with (REPORT / "measurements.csv").open() as f:
        measurements = list(csv.DictReader(f))
    atlas, axes = plt.subplots(2, 4, figsize=(11.2, 6.4), layout="constrained")
    rows = []
    for ax, (file, name) in zip(axes.flat, sorted(names.items(), key=lambda item: item[1]), strict=True):
        path = lab.DATA / file
        with np.load(path) as tile:
            img = tile["img"]
            fill = tile["fill"].item() if tile["fill"].size else None
        if name.startswith("S1"):
            channels, view = [0], "VV"
        elif name.startswith("S3"):
            channels, view = [5, 3, 1], "Oa08 / 06 / 04"
        else:
            channels, view = [2, 1, 0], "B04 / 03 / 02"
        ax.imshow(preview(img, channels, fill), cmap="gray", interpolation="antialiased")
        ax.set_title(name, loc="left", weight="bold")
        ax.set_xlabel(f"{view}   ·   {img.shape[2]} × {img.shape[1]}", fontsize=9)
        ax.set(xticks=[], yticks=[])
        for spine in ax.spines.values():
            spine.set_visible(False)
        records = [r for r in measurements if r["tile"] == file]
        assert {int(r["raw_bytes"]) for r in records} == {img.nbytes}
        sizes = {r["codec"]: int(r["bytes"]) for r in records}
        standard = min(b for k, b in sizes.items() if not k.startswith(("CPU", "GPU")))
        with np.load(ARCHIVES / f"{path.stem}.two-pass.npz") as archive:
            fitted = archive["stream"].nbytes
        online = (ARCHIVES / f"{path.stem}.gpu.sz").stat().st_size
        assert sizes["GPU online"] == online
        shared = sizes.get("GPU shared prior")
        if shared is not None:
            assert (ARCHIVES / f"{path.stem}.shared.sz").stat().st_size == shared
        rows.append(dict(tile=name, samples=img.size, raw_bytes=img.nbytes, best_standard_bytes=standard,
                         online_bytes=online, tile_fitted_stream_bytes=fitted, shared_stream_bytes=shared))
    atlas.savefig(REPORT / "sentinel-previews.jpg", dpi=170)
    plt.close(atlas)
    with (REPORT / "stream-sizes.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot_rates():
    with (REPORT / "stream-sizes.csv").open() as f:
        rows = []
        for record in csv.DictReader(f):
            name = record.pop("tile")
            sizes = {key: int(value) if value else None for key, value in record.items()}
            rows.append({"tile": name, **sizes})
    fig, (rates, savings) = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained",
                                         gridspec_kw={"width_ratios": [1.65, 1]})
    series = [("best_standard_bytes", "Best standard", "#777777", "s"),
              ("online_bytes", "Online", "#0072b2", "o"),
              ("tile_fitted_stream_bytes", "Tile-fitted", "#d55e00", "^"),
              ("shared_stream_bytes", "Shared prior", "#008568", "D")]
    for j, (key, label, color, marker) in enumerate(series):
        points = [(8 * r[key] / r["samples"], i + (j - 1.5) * .17) for i, r in enumerate(rows) if r[key]]
        rates.scatter(*zip(*points), marker=marker, color=color, label=label, s=24)
    rates.set_yticks(range(len(rows)), [r["tile"] for r in rows])
    rates.invert_yaxis()
    rates.set(xlabel="Compressed bits per band sample", title="A    Size by scene and method")
    rates.legend(loc="upper center", bbox_to_anchor=(.5, -.15), ncol=2, fontsize=9, frameon=False)
    held = [r for r in rows if r["shared_stream_bytes"]]
    gains = [100 * (1 - r["shared_stream_bytes"] / r["online_bytes"]) for r in held]
    bars = savings.barh([r["tile"].replace("S2 ", "") for r in held], gains, color="#008568", height=.46)
    savings.bar_label(bars, fmt="%.3f%%", padding=4, fontsize=9)
    savings.invert_yaxis()
    savings.set(xlabel="Stream reduction relative to online (%)", xlim=(0, max(gains) * 1.28),
                title="B    Benefit of shared weights")
    for ax in (rates, savings):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", alpha=.2)
        ax.set_axisbelow(True)
    fig.savefig(REPORT / "sentinel-stream-rates.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rates-only", action="store_true", help="use the published CSV without the source tiles")
    if not parser.parse_args().rates_only:
        measure_and_preview()
    plot_rates()
