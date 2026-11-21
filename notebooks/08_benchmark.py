import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")

with app.setup:
    import csv
    import io
    import subprocess
    import sys
    from collections import defaultdict

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    from satzip import lab

    SCRIPT = lab.ROOT / "scripts" / "compare_tiles.py"
    RUNS = lab.DATA / "compressed" / "measurements"
    BUNDLE = lab.DATA / "compressed" / "sentinel-tiles.zip"
    NAMES = {f: s for s, f in lab.SAMPLES.items() if f} | dict(
        zip(lab.PRIORS["S2 Paris+Lyon+Nantes"], ("S2 Paris", "S2 Lyon", "S2 Nantes")))
    FAMILIES = {"Standard codecs": "#eb6834", "satzip CPU": "#2a78d6", "satzip GPU": "#1baf7a"}


@app.function
def family(codec):
    return "satzip GPU" if codec.startswith("GPU") else "satzip CPU" if codec.startswith("CPU") else "Standard codecs"


@app.cell
def _():
    remeasure = mo.ui.run_button(label="Re-measure every tile (slow)")
    mo.vstack([mo.md("""
    # Benchmark every compression algorithm

    Compares every lossless codec from `scripts/compare_tiles.py` on each downloaded Sentinel tile:
    general-purpose compressors, PNG, JPEG 2000, CCSDS-121, JPEG XL (several layouts) and the satzip
    CPU, two-pass and GPU predictors. Each stream is decoded and checked against the input.

    The notebook opens with the published measurements. A fresh run takes tens of minutes because
    of the 2048 × 2048 tile. GPU runs need a ROCm device. Compression ratio = raw integer bytes /
    compressed bytes. Shared-prior sizes exclude the reusable decoder weights. CPU two-pass
    archive sizes include tile-specific weights.
    """), remeasure])
    return (remeasure,)


@app.cell
def _(remeasure):
    measurement_files = [lab.ROOT / "reports" / "tile_comparison" / "measurements.csv"]
    if remeasure.value:
        tiles = sorted(lab.DATA.glob("sentinel-*.npz"), key=lambda p: p.stat().st_size)
        mo.stop(not tiles, mo.md("Download the Sentinel tiles with `python scripts/download_data.py --no-lst`."))
        _modes = ["standard", "cpu", "two-pass"] + (["gpu"] if torch.version.hip and torch.cuda.is_available() else [])
        _jobs = [(t, m) for t in tiles for m in _modes]
        measurement_files = []
        for _tile, _mode in mo.status.progress_bar(_jobs, title="Measuring tiles", show_eta=True):
            subprocess.run([sys.executable, str(SCRIPT), _mode, str(_tile)], check=True, cwd=lab.ROOT)
            measurement_files.append(RUNS / f"{_tile.stem}.{_mode}.csv")
    return (measurement_files,)


@app.cell
def _(measurement_files):
    rows = []
    for _file in measurement_files:
        with _file.open() as _f:
            for _r in csv.DictReader(_f):
                _raw, _bytes = int(_r["raw_bytes"]), int(_r["bytes"])
                rows.append({"Tile": NAMES[_r["tile"]], "Codec": _r["codec"], "Family": family(_r["codec"]),
                             "Raw bytes": _raw, "Bytes": _bytes, "Shared weights": int(_r["shared_weight_bytes"]),
                             "Ratio": round(_raw / _bytes, 4), "Saving %": round(100 * (1 - _bytes / _raw), 2),
                             "Bits per sample": float(_r["bits_per_sample"])})
    mo.stop(not rows, mo.md("No measurements yet."))
    labels = list(dict.fromkeys(r["Tile"] for r in rows))
    raw = {r["Tile"]: r["Raw bytes"] for r in rows}
    best = {t: min((r for r in rows if r["Tile"] == t), key=lambda r: r["Bytes"]) for t in labels}
    for _r in rows:
        _r["% above tile best"] = round(100 * (_r["Bytes"] / best[_r["Tile"]]["Bytes"] - 1), 2)
    return best, labels, raw, rows


@app.cell
def _(labels, raw, rows):
    # Totals over every tile: codecs measured on all tiles, plus the best choice per tile in each family.
    _sizes = defaultdict(dict)
    for _r in rows:
        _sizes[_r["Codec"]][_r["Tile"]] = _r["Bytes"]
    _entries = [(c, family(c), sum(s.values())) for c, s in _sizes.items() if len(s) == len(labels)]
    for _fam in [*FAMILIES, None]:
        _picks = [min((r["Bytes"] for r in rows if r["Tile"] == t and _fam in (None, r["Family"])), default=None)
                  for t in labels]
        if None not in _picks:
            _entries.append((f"★ Best {_fam or 'of all'} per tile", _fam or "Best overall", sum(_picks)))
    _total_raw = sum(raw.values())
    leaderboard = sorted(({"Method": c, "Family": f, "Bytes": b, "Ratio": round(_total_raw / b, 3),
                           "Saving %": round(100 * (1 - b / _total_raw), 2)} for c, f, b in _entries),
                         key=lambda e: e["Bytes"])
    mo.vstack([mo.md(f"## Leaderboard: all {len(labels)} tiles ({_total_raw:,} raw bytes), each encoded independently"),
               mo.ui.table(leaderboard, selection=None, page_size=40)])
    return (leaderboard,)


@app.cell
def _(leaderboard, raw):
    _board = leaderboard[::-1]
    leaderboard_figure, _ax = plt.subplots(figsize=(10, 0.36 * len(_board) + 1.2), layout="constrained")
    _colors = [FAMILIES.get(e["Family"], "#0b0b0b") for e in _board]
    _bars = _ax.barh([e["Method"] for e in _board], [e["Ratio"] for e in _board], color=_colors, height=0.7)
    _ax.bar_label(_bars, [f"{e['Ratio']:.2f}×  ({e['Saving %']:.1f}% smaller)" for e in _board],
                  padding=4, fontsize=8, color="#52514e")
    for _tick, _e in zip(_ax.get_yticklabels(), _board):
        _tick.set_fontweight("bold" if _e["Method"].startswith("★") else "normal")
    if BUNDLE.exists():
        _ratio = sum(raw.values()) / BUNDLE.stat().st_size
        _ax.axvline(_ratio, color="#52514e", ls="--", lw=1)
        _ax.annotate(f"bundle with overlap dedup {_ratio:.2f}×", (_ratio, len(_board) - 0.5),
                     fontsize=8, color="#52514e", ha="right", va="bottom", xytext=(-3, 0), textcoords="offset points")
    _ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in [*FAMILIES.values(), "#0b0b0b"]],
               labels=[*FAMILIES, "Best overall"], loc="lower right", fontsize=8, frameon=False)
    _ax.set_xlabel("Compression ratio (higher is better)")
    _ax.set_xlim(1, max(e["Ratio"] for e in _board) * 1.22)
    _ax.spines[["top", "right"]].set_visible(False)
    _ax.grid(axis="x", color="0.9")
    _ax.set_axisbelow(True)
    leaderboard_figure
    return (leaderboard_figure,)


@app.cell
def _(best, labels, rows):
    _order = sorted(labels, key=lambda t: best[t]["Ratio"])
    tiles_figure, _ax = plt.subplots(figsize=(10, 0.75 * len(_order) + 1.2), layout="constrained")
    _y = np.arange(len(_order))
    _h = 0.8 / len(FAMILIES)
    for _i, (_fam, _color) in enumerate(FAMILIES.items()):
        _vals = [max((r["Ratio"] for r in rows if r["Tile"] == t and r["Family"] == _fam), default=0) for t in _order]
        _ax.barh(_y + (_i - 1) * _h, _vals, height=_h * 0.9, color=_color, label=f"Best {_fam}")
    for _yy, _t in zip(_y, _order):
        _ax.annotate(f"{best[_t]['Ratio']:.3f}×  {best[_t]['Codec']}", (best[_t]["Ratio"], _yy),
                     xytext=(4, 0), textcoords="offset points", va="center", fontsize=8, color="#52514e")
    _ax.set_yticks(_y, _order)
    _ax.set_xlim(1, max(best[t]["Ratio"] for t in labels) * 1.45)
    _ax.set_xlabel("Compression ratio of the best codec in each family (winner annotated)")
    _ax.legend(loc="lower right", fontsize=8, frameon=False)
    _ax.spines[["top", "right"]].set_visible(False)
    _ax.grid(axis="x", color="0.9")
    _ax.set_axisbelow(True)
    mo.vstack([mo.md("## Best result per tile"), tiles_figure])
    return (tiles_figure,)


@app.cell
def _(rows):
    hist_figure, (_ratio_ax, _gap_ax) = plt.subplots(1, 2, figsize=(12, 4.2), layout="constrained")
    _groups = {f: [r for r in rows if r["Family"] == f] for f in FAMILIES}
    _ratios = [r["Ratio"] for r in rows]
    _ratio_ax.hist([[r["Ratio"] for r in g] for g in _groups.values()], stacked=True,
                   bins=np.arange(np.floor(min(_ratios) * 10) / 10, max(_ratios) + 0.1, 0.1),
                   color=list(FAMILIES.values()), label=list(FAMILIES), edgecolor="white", linewidth=1)
    _ratio_ax.set_xlabel("Compression ratio (one count per tile × codec)")
    _ratio_ax.set_title("Distribution of compression ratios", fontsize=10)
    _gaps = [r["% above tile best"] for r in rows]
    _gap_ax.hist([[r["% above tile best"] for r in g] for g in _groups.values()], stacked=True,
                 bins=np.arange(0, max(_gaps) + 1, 1),
                 color=list(FAMILIES.values()), label=list(FAMILIES), edgecolor="white", linewidth=1)
    _gap_ax.set_xlabel("% larger than the best stream for the same tile (0 = winner)")
    _gap_ax.set_title("Distance from the per-tile winner", fontsize=10)
    for _ax in (_ratio_ax, _gap_ax):
        _ax.set_ylabel("Measurements")
        _ax.legend(fontsize=8, frameon=False)
        _ax.spines[["top", "right"]].set_visible(False)
    mo.vstack([mo.md("## Histograms"), hist_figure])
    return (hist_figure,)


@app.cell
def _(best, labels, rows):
    _std = {t: min((r for r in rows if r["Tile"] == t and r["Family"] == "Standard codecs"),
                   key=lambda r: r["Bytes"], default=None) for t in labels}
    _winners = [{"Tile": t, "Winner": best[t]["Codec"], "Bytes": best[t]["Bytes"], "Ratio": best[t]["Ratio"],
                 "Best standard": _std[t]["Codec"] if _std[t] else "",
                 "Saving vs best standard %": round(100 * (1 - best[t]["Bytes"] / _std[t]["Bytes"]), 2) if _std[t] else None}
                for t in labels]
    mo.vstack([mo.md("## Winners per tile"), mo.ui.table(_winners, selection=None),
               mo.md("## All measurements"), mo.ui.table(rows, selection=None, page_size=20)])
    return


@app.cell
def _(hist_figure, leaderboard_figure, rows, tiles_figure):
    _csv = io.StringIO()
    _writer = csv.DictWriter(_csv, fieldnames=rows[0])
    _writer.writeheader()
    _writer.writerows(rows)

    def _png(fig):
        _buf = io.BytesIO()
        fig.savefig(_buf, format="png", dpi=170, bbox_inches="tight")
        return _buf.getvalue()

    mo.hstack([mo.download(_csv.getvalue().encode(), filename="benchmark.csv", label="Measurements CSV"),
               mo.download(_png(leaderboard_figure), filename="leaderboard.png", label="Leaderboard PNG"),
               mo.download(_png(tiles_figure), filename="per-tile.png", label="Per-tile PNG"),
               mo.download(_png(hist_figure), filename="histograms.png", label="Histograms PNG")], justify="start")
    return


if __name__ == "__main__":
    app.run()
