import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Satellite compression benchmark results

    The Sentinel comparison is shown below. The LST figures update as verified chunks arrive.
    A positive saving means SZ2 uses fewer bytes than the selected reference codec.
    """)
    return


@app.cell(hide_code=True)
def _(Path, csv, mo):
    _root = mo.notebook_dir().parent if mo.notebook_dir() else Path.cwd()
    _dir = _root / "reports" / "sentinel"
    mo.stop(not (_dir / "streams.csv").is_file(), mo.md("The saved Sentinel comparison is unavailable."))
    with (_dir / "streams.csv").open() as _f:
        _rows = list(csv.DictReader(_f))
    _summary = []
    for _name in dict.fromkeys(r["Input"] for r in _rows):
        _sample = [r for r in _rows if r["Input"] == _name]
        _sz1, _sz2 = (int(next(r["Bytes"] for r in _sample if r["Codec"] == c)) for c in ("SZ1", "SZ2"))
        _best = min(int(r["Bytes"]) for r in _sample if r["Codec"].startswith("JPEG XL (")
                    and "baseline" not in r["Codec"])
        _summary.append({"Tile": _name.split("_512_")[0], "SZ1 bytes": _sz1, "SZ2 bytes": _sz2,
                         "JPEG XL bytes": _best, "SZ2 saving (%)": round(100 * (1 - _sz2 / _best), 2)})
    mo.vstack([mo.md(f"## Sentinel · {len(_summary)} measured tiles"),
               mo.image(str(_dir / "size-differences.png"), alt="Per-tile sizes and SZ2 savings against JPEG XL",
                        width=950),
               mo.accordion({"Sentinel measurements": mo.ui.table(_summary, selection=None)})])
    return


@app.cell(hide_code=True)
def _(cache_path, mo):
    refresh = mo.ui.refresh(options=["30s", "1m", "5m"], default_interval=None if cache_path.exists() else "1m",
                            label="Refresh LST measurements")
    refresh
    return (refresh,)


@app.cell(hide_code=True)
def _(cache_path, h5py, json, mo, names, path, refresh, run, satzip, workers):
    refresh.value
    names_list = [n.strip() for n in names.value.split(",")]
    with h5py.File(path.value) as _file:
        _h, _w = _file[satzip._nc_split(names_list[0])[0]].shape[-2:]
    grid = ((_h + 999) // 1000, (_w + 999) // 1000)
    expected_chunks = grid[0] * grid[1]
    if cache_path.exists():
        res = json.loads(cache_path.read_text())
    elif run.value:
        with mo.status.spinner("Running full benchmark") as _status:
            res = satzip.nc_benchmark(path.value, names_list, workers=workers.value, full=True,
                                      progress=lambda n, total: _status.update(title=f"{n} / {total} chunks verified"))
    else:
        _parts = cache_path.with_name(cache_path.name + ".chunks")
        res = [json.loads(p.read_text()) for p in sorted(_parts.glob("*.json"))]
    mo.stop(not res, mo.md("No LST measurements yet. Open **Benchmark controls** to start the run."))
    return expected_chunks, grid, names_list, res


@app.cell
def _(expected_chunks, mo, res):
    CODECS = ["NetCDF file", "shuffle+zlib-9", "shuffle+zstd-19", "lzma-9e", "CCSDS-121 (libaec)",
              "PNG", "JPEG 2000", "JPEG XL"]
    _layout_keys = {"JPEG XL effort 9": "JPEG XL layouts"}
    if "JPEG XL transformed layouts" in res[0]:
        _layout_keys["JPEG XL effort 9 + uncertainty"] = "JPEG XL transformed layouts"
    fixed_layouts = {}
    layout_rows = []
    for _label, _key in _layout_keys.items():
        _layout_totals = {layout: sum(r[_key][layout] for r in res) for layout in res[0][_key]}
        fixed_layouts[_label] = min(_layout_totals, key=_layout_totals.get)
        for _layout, _bytes in _layout_totals.items():
            layout_rows.append({"Variant": _label, "Layout": _layout, "Bytes": _bytes})
        layout_rows.append({"Variant": _label, "Layout": "Per-chunk minimum (before layout signalling)",
                            "Bytes": sum(min(r[_key].values()) for r in res)})
    sz_label = "SZ2 + uncertainty" if "JPEG XL transformed layouts" in res[0] else "SZ2"
    chunk_rows = []
    for _r in res:
        _row = {"row": _r["i"], "column": _r["j"], "valid samples": sum(_r["valid"]),
                "raw bytes": sum(_r["raw"])}
        _row.update({("JPEG XL effort 7" if c == "JPEG XL" else c): sum(_r[c]) for c in CODECS})
        _row["SZ1"] = _r["SZ1"]
        if sz_label != "SZ2":
            _row["SZ2 plain"] = _r["SZ2 plain"]
        _row[sz_label] = _r["satzip"]
        for _label, _key in _layout_keys.items():
            _row[_label] = _r[_key][fixed_layouts[_label]]
        chunk_rows.append(_row)
    size_columns = [c for c in chunk_rows[0] if c not in ("row", "column", "valid samples", "raw bytes")]
    totals = {c: sum(row[c] for row in chunk_rows) for c in size_columns}
    _raw = sum(row["raw bytes"] for row in chunk_rows)
    _populated = sum(row["valid samples"] > 0 for row in chunk_rows)
    total_rows = [{"Codec": c, "Bytes": totals[c], "MB": round(totals[c] / 1e6, 3),
                   "Ratio to raw": round(_raw / totals[c], 2) if totals[c] else None,
                   "Saving vs stored (%)": round(100 * (1 - totals[c] / totals["NetCDF file"]), 2)
                   if totals["NetCDF file"] else None}
                  for c in sorted(totals, key=totals.get)]
    _details = " · ".join(f"{label} uses {layout}" for label, layout in fixed_layouts.items())
    _enc, _dec = (sum(r[k] for r in res) / 60 for k in ("t_enc", "t_dec"))
    _coverage = "Complete product" if len(res) == expected_chunks else "Partial results"
    mo.vstack([mo.md(f"## LST · {_coverage} · {len(res)} / {expected_chunks} verified chunks\n\n"
                     f"The figures and totals cover these chunks only, including {_populated} with valid data. "
                     f"The integer grids occupy {_raw / 1e9:.3f} GB before compression. Positive savings mean smaller streams."),
               mo.ui.table(total_rows, selection=None, pagination=False), mo.md(_details),
               mo.accordion({"JPEG XL layout totals": mo.ui.table(layout_rows, selection=None, pagination=False)}),
               mo.md(f"{sz_label} summed worker time across chunks: {_enc:.1f} minutes encoding and "
                     f"{_dec:.1f} minutes decoding. Parallel workers overlap in wall time.")])
    return CODECS, chunk_rows, fixed_layouts, size_columns, sz_label, total_rows, totals


@app.cell
def _(BLUE, ORANGE, plt, sz_label, totals):
    size_figure, _axes = plt.subplots(1, 2, figsize=(13, 5.6), layout="constrained")
    for _ax, _labels, _title in zip(_axes, [list(totals), [c for c in totals if c.startswith(("SZ", "JPEG XL"))]],
                                    ["All codecs", "Neural and JPEG XL streams"]):
        _labels.sort(key=totals.get, reverse=True)
        _mb = [totals[c] / 1e6 for c in _labels]
        _bars = _ax.barh(_labels, _mb, color=[ORANGE if c == sz_label else BLUE for c in _labels])
        _ax.bar_label(_bars, fmt="%.1f", padding=3, fontsize=8)
        _ax.set_xlim(0, max(_mb) * 1.18)
        _ax.set_xlabel("Compressed size (decimal MB)")
        _ax.set_title(_title, loc="left")
        _ax.grid(axis="y", visible=False)
    size_figure
    return (size_figure,)


@app.cell
def _(mo, size_columns, sz_label):
    _choices = [c for c in size_columns if c != sz_label]
    reference = mo.ui.dropdown(_choices, value=next(c for c in reversed(_choices) if c.startswith("JPEG XL")),
                               label=f"Compare {sz_label} against")
    scope = mo.ui.dropdown(["Populated chunks", "All chunks", "Fill-only chunks"],
                           value="Populated chunks", label="Histogram population")
    mo.vstack([mo.md("## Where does the size difference come from?"), mo.hstack([reference, scope])])
    return reference, scope


@app.cell
def _(chunk_rows, mo, np, plt, reference, scope, sz_label):
    _ref = reference.value
    _rows = [r for r in chunk_rows if scope.value == "All chunks" or
             (r["valid samples"] > 0) == (scope.value == "Populated chunks")]
    mo.stop(not _rows, mo.md("This selection has no chunks."))
    _baseline = np.array([r[_ref] for r in _rows])
    _delta = _baseline - np.array([r[sz_label] for r in _rows])
    _defined = _baseline > 0
    _pct = 100 * _delta[_defined] / _baseline[_defined]
    hist_figure, _axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    for _ax, _values, _unit in zip(_axes, (_delta / 1024, _pct), ("KiB", "% of reference size")):
        if _values.size:
            _edges = np.histogram_bin_edges(_values, bins=min(40, max(1, int(np.sqrt(len(_values))))))
            if _edges[0] < 0 < _edges[-1]:
                _edges = np.unique(np.append(_edges, 0))
            _, _, _bars = _ax.hist(_values, bins=_edges, color="#276e81", edgecolor="white")
            for _bar, _right in zip(_bars, _edges[1:]):
                if _right <= 0:
                    _bar.set_facecolor("#a44d3d")
            _ax.axvline(np.median(_values), color="#cb631a", label="Median", linewidth=1.7)
            _ax.legend()
        _ax.axvline(0, color="0.25", linestyle="--", lw=1)
        _ax.set(xlabel=f"Saving per chunk ({_unit})", ylabel="Number of chunks")
    hist_figure.suptitle(f"{scope.value} · {sz_label} vs {_ref}\nPositive values mean {sz_label} uses fewer bytes")
    _wins, _ties = int(np.sum(_delta > 0)), int(np.sum(_delta == 0))
    _weighted = 100 * _delta.sum() / _baseline.sum() if _baseline.sum() else float("nan")
    mo.vstack([hist_figure, mo.md(f"**{len(_rows)} chunks** · {_wins} wins · {_ties} ties · "
                                  f"{len(_rows)-_wins-_ties} losses · {_weighted:.2f}% saving by total bytes. "
                                  f"The percentage histogram includes {len(_pct)} chunks with a positive reference size.")])
    return (hist_figure,)


@app.cell
def _(chunk_rows, grid, np, plt, reference, sz_label):
    from matplotlib.colors import ListedColormap

    # Status layer: white = not measured yet, grey = measured but fill only.
    _status = np.zeros(grid)
    _gain = np.full(grid, np.nan)
    for _r in chunk_rows:
        _status[_r["row"], _r["column"]] = 1
        if _r["valid samples"] and _r[reference.value] > 0:
            _gain[_r["row"], _r["column"]] = 100 * (1 - _r[sz_label] / _r[reference.value])
    _finite = np.abs(_gain[np.isfinite(_gain)])
    _limit = max(1, _finite.max()) if _finite.size else 1
    map_figure, _ax = plt.subplots(figsize=(11, 5.5), layout="constrained")
    _ax.imshow(_status, cmap=ListedColormap(["#ffffff", "#d9d9d9"]), vmin=0, vmax=1,
               origin="lower", interpolation="nearest")
    _im = _ax.imshow(_gain, cmap=plt.get_cmap("RdBu").with_extremes(bad=(0, 0, 0, 0)),
                     vmin=-_limit, vmax=_limit, origin="lower", interpolation="nearest")
    _ax.set_xticks(np.arange(-0.5, grid[1], 6), [f"{-180 + 10 * k:.0f}°" for k in range(0, grid[1] + 1, 6)])
    _ax.set_yticks(np.arange(-0.5, grid[0], 3), [f"{-90 + 10 * k:.0f}°" for k in range(0, grid[0] + 1, 3)])
    _ax.set(xlabel="Longitude (chunk edges)", ylabel="Latitude (chunk edges)",
            title=f"{sz_label} saving against {reference.value} · {int(_status.sum())} of {_status.size} chunks measured")
    _ax.grid(False)
    for _side in _ax.spines.values():
        _side.set_visible(True)
    map_figure.colorbar(_im, ax=_ax, shrink=.83, label="Saving (%) · grey = fill only · white = not measured")
    map_figure
    return (map_figure,)


@app.cell
def _():
    import csv
    import json
    import os
    from pathlib import Path

    import h5py
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    import satzip

    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.alpha": 0.25})
    BLUE, ORANGE = "#2a78d6", "#eb6834"
    return BLUE, ORANGE, Path, csv, h5py, json, mo, np, os, plt, satzip


@app.cell(hide_code=True)
def _(Path, mo):
    _root = mo.notebook_dir().parent if mo.notebook_dir() else Path.cwd()
    path = mo.ui.text(
        value=str(_root / "data/ESACCI-LST-L3C-LST-SLSTRB-0.01deg_1DAILY_DAY-20230101000000-fv4.00.nc"),
        label="NetCDF file", full_width=True)
    # Band order: put related variables next to each other. Each band sees the 2 bands before it.
    names = mo.ui.text(
        value="satze,sataz,solze,solaz,lst,t2m,ndvi,emis_0,emis_1,qual_flag,lst_uncertainty,lst_unc_ran,"
              "lst_unc_loc_atm,lst_unc_loc_sfc,lst_unc_sys,lcc,n,ncld",
        label="Variables in coding order (var_k = index k on the 2nd axis)", full_width=True)
    mo.accordion({"Input file and band order": mo.vstack([path, names])})
    return names, path


@app.cell
def _(h5py, mo, path):
    with h5py.File(path.value) as _f:
        _rows = [{"variable": k, "dtype": str(d.dtype), "shape": str(d.shape), "chunks": str(d.chunks),
                  "filters": ", ".join(f for f in ("gzip", "shuffle", "szip") if f in str(d._filters)),
                  "stored MB": round(d.id.get_storage_size() / 1e6, 1)}
                 for k, d in _f.items() if isinstance(d, h5py.Dataset)]
    mo.accordion({"Variables in the file": mo.ui.table(_rows, selection=None)})
    return


@app.cell
def _(mo):
    mo.md(r"""
    The variable `dtime` is float32 with fractions of 1/128 s. satzip and the image codecs read integers only,
    so the benchmark does not include `dtime`, `lat` and `lon` (14 MB of the file). The other 18 grids are int16.

    The full run processes chunks in parallel and writes the SZ2 results to a JSON cache next to the file.
    """)
    return


@app.cell
def _(mo):
    scan_uncertainty = mo.ui.run_button(label="Scan uncertainty grids")
    mo.vstack([
        mo.md("## Uncertainty prediction across the file\n\nCompare the stored total with the rounded "
              "root-sum-square of its four components in packed integer units. This reads the five ESA CCI "
              "LST grids chunk by chunk and can run independently of the compression benchmark."),
        scan_uncertainty,
    ])
    return (scan_uncertainty,)


@app.cell
def _(h5py, mo, np, path, scan_uncertainty):
    mo.stop(not scan_uncertainty.value)
    _components = ("lst_unc_ran", "lst_unc_loc_atm", "lst_unc_loc_sfc", "lst_unc_sys")
    _total = _shared = _exact = _within_one = _worst = 0
    with h5py.File(path.value) as _file:
        mo.stop(any(n not in _file for n in ("lst_uncertainty", *_components)),
                mo.md("Select an ESA CCI LST file containing total uncertainty and its four components."))
        _ds = _file["lst_uncertainty"]
        for _block in _ds.iter_chunks():
            _target = _ds[_block]
            _valid = _target != -32768
            _total += int(_valid.sum())
            if not _valid.any():
                continue
            _parts = np.stack([_file[n][_block] for n in _components])
            _valid &= (_parts != -32768).all(axis=0)
            if _valid.any():
                _prediction = np.rint(np.sqrt(np.square(_parts[:, _valid], dtype=np.int64).sum(axis=0)))
                _error = np.abs(_target[_valid] - _prediction)
                _shared += _error.size
                _exact += int((_error == 0).sum())
                _within_one += int((_error <= 1).sum())
                _worst = max(_worst, int(_error.max()))
    uncertainty_counts = {"Valid totals": _total, "With all components": _shared, "Exact": _exact,
                          "Within one count": _within_one, "Maximum error (counts)": _worst if _shared else None}
    mo.ui.table([uncertainty_counts], selection=None)
    return (uncertainty_counts,)


@app.cell
def _(mo, names, os, path, satzip):
    workers = mo.ui.number(start=1, stop=os.cpu_count() or 1, value=min(14, os.cpu_count() or 1), label="Parallel workers")
    run = mo.ui.run_button(label="Run or resume benchmark")
    cache_path = satzip.nc_cache_path(path.value, [n.strip() for n in names.value.split(",")], full=True)
    mo.accordion({"Benchmark controls": mo.vstack([workers, run, mo.md(f"Cache file: `{cache_path.name}`")])})
    return cache_path, run, workers


@app.cell
def _(mo):
    mo.md(r"""
    ## By variable

    satzip estimates the cost of each band from its adaptive token probabilities and low-bit counts.
    The final row reconciles the estimates with the measured stream sizes. It includes masks,
    headers, tables and differences between estimated symbol costs and range-coded bytes.
    """)
    return


@app.cell
def _(CODECS, mo, names_list, res, sz_label, totals):
    _rows = []
    for _i, _n in enumerate(names_list):
        _row = {"variable": _n}
        for _c in CODECS:
            _row[_c] = round(sum(r[_c][_i] for r in res) / 1e6, 2)
        _row["satzip"] = round(sum(r["satzip bits"][_i] for r in res) / 8e6, 2)
        _rows.append(_row)
    _est = sum(sum(r["satzip bits"]) for r in res) / 8
    _rows.append({"variable": "side information and coding difference", "satzip": round((totals[sz_label] - _est) / 1e6, 2)})
    mo.ui.table(_rows, selection=None, label="MB per variable")
    return


@app.cell
def _(mo):
    export_report = mo.ui.run_button(label="Save figures and measurement tables")
    mo.vstack([mo.md("## Export results\n\nSave the current figures as PNGs and a PDF, with exact per-chunk "
                     "sizes, totals and benchmark settings under `reports/`."), export_report])
    return (export_report,)


@app.cell
def _(Path, cache_path, chunk_rows, csv, export_report, fixed_layouts, hist_figure, json, map_figure, mo, names_list,
      path, reference, res, scope, size_figure, total_rows):
    mo.stop(not export_report.value)
    import hashlib
    from importlib.metadata import version
    from matplotlib.backends.backend_pdf import PdfPages

    _dir = Path("reports") / cache_path.name.removesuffix(".bench.json")
    _dir.mkdir(parents=True, exist_ok=True)
    with PdfPages(_dir / "benchmark.pdf") as _pdf:
        for _stem, _fig in [("total-sizes", size_figure), ("chunk-histograms", hist_figure), ("chunk-map", map_figure)]:
            _fig.savefig(_dir / f"{_stem}.png", dpi=170, bbox_inches="tight")
            _pdf.savefig(_fig, bbox_inches="tight")
    _layout_rows = [{"row": r["i"], "column": r["j"], "variant": key, "layout": layout, "bytes": size}
                    for r in res for key in ("JPEG XL layouts", "JPEG XL transformed layouts")
                    for layout, size in r.get(key, {}).items()]
    for _stem, _rows in [("totals", total_rows), ("chunks", chunk_rows), ("jpegxl-layouts", _layout_rows)]:
        with (_dir / f"{_stem}.csv").open("w", newline="") as _file:
            _writer = csv.DictWriter(_file, fieldnames=_rows[0].keys())
            _writer.writeheader()
            _writer.writerows(_rows)
    with Path(path.value).open("rb") as _file:
        _digest = hashlib.file_digest(_file, "sha256").hexdigest()
    _metadata = {"source": Path(path.value).name, "sha256": _digest, "variables_in_order": names_list,
                 "fill": -32768, "chunk_side": 1000, "chunks": len(res), "offline_prior": None,
                 "jpegxl_effort": 9, "fixed_layouts": fixed_layouts, "histogram_reference": reference.value,
                 "histogram_population": scope.value, "size_unit": "bytes",
                 "packages": {p: version(p) for p in ("torch", "numpy", "constriction", "imagecodecs", "h5py", "marimo")}}
    (_dir / "settings.json").write_text(json.dumps(_metadata, indent=2) + "\n")
    report_dir = _dir
    mo.md(f"Saved the figures and measurement tables to `{report_dir}`.")
    return (report_dir,)


if __name__ == "__main__":
    app.run()
