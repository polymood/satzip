import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")

with app.setup:
    import time
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    import satzip


@app.cell
def _():
    _intro = mo.md("""
    # Compare SZ1, SZ2 and lossless JPEG XL

    Select one or more cached Sentinel tiles or a local LST product. Each result includes a
    decode and equality check. For JPEG XL the table lists each channel layout and counts
    eight bytes per band for the integer offset. Bits per pixel is measured per band sample,
    including fill pixels. The time columns measure the satzip encoder and decoder.

    For the published LST comparisons use chunk (11, 19) or (1, 25) and the default band order.
    """)
    directory = mo.ui.text(value="data", label="Input directory")
    mo.vstack([_intro, directory])
    return (directory,)


@app.cell
def _(directory):
    _root = Path(directory.value).expanduser()
    _paths = sorted(p.name for p in _root.glob("*") if p.is_file() and p.suffix in (".npz", ".nc"))
    mo.stop(not _paths, mo.md("Select a directory containing cached Sentinel .npz tiles or an LST .nc file. "
                               "Notebook 01 can download the Sentinel samples."))
    _first = next((p for p in _paths if p.endswith(".npz")), _paths[0])
    settings = mo.md("""
    {files}

    {effort}

    {all_codecs}

    **LST chunk** (zero-based indices, 1000 × 1000 cells per chunk)

    {row} {col}

    {bands}
    """).batch(
        files=mo.ui.multiselect(_paths, value=[_first], label="Inputs"),
        effort=mo.ui.slider(1, 9, value=9, show_value=True, label="JPEG XL effort"),
        all_codecs=mo.ui.checkbox(value=True, label="Include all standard per-band codecs"),
        row=mo.ui.number(start=0, value=11, step=1, label="Chunk row"),
        col=mo.ui.number(start=0, value=19, step=1, label="Chunk column"),
        bands=mo.ui.text(value="satze,sataz,solze,solaz,lst,t2m,ndvi,emis_0,emis_1,qual_flag,"
                         "lst_uncertainty,lst_unc_ran,lst_unc_loc_atm,lst_unc_loc_sfc,lst_unc_sys,lcc,n,ncld",
                         full_width=True, label="LST variables in coding order"),
    ).form(submit_button_label="Run comparison")
    settings
    return (settings,)


@app.function
def compare_input(path, *, effort, row, col, names, all_codecs=True):
    recipe = None
    if path.suffix == ".nc":
        img, _ = satzip.nc_chunk(path, names, row, col)
        if not img.size:
            raise ValueError(f"Chunk ({row}, {col}) is outside {path.name}")
        fill = -32768
        recipe = satzip.uncertainty.lst_recipe(names)
        sample = f"{path.name} ({row}, {col})"
    else:
        with np.load(path) as tile:
            img = tile["img"]
            fill = int(tile["fill"].item()) if tile["fill"].size else None
        sample = path.name

    rows = []

    def record(codec, size, encode=None, decode=None):
        rows.append({"Input": sample, "Codec": codec, "Bytes": size, "Bits per pixel": round(8 * size / img.size, 4),
                     "Encode (s)": encode, "Decode (s)": decode})

    variants = [(1, None, "SZ1"), (2, None, "SZ2")]
    if recipe:
        variants.append((2, recipe, "SZ2 + uncertainty transform"))
    for version, transform, label in variants:
        start = time.perf_counter()
        stream, _ = satzip.compress(img, fill=fill, version=version, quadrature=transform)
        encoded = time.perf_counter()
        restored = satzip.decompress(stream)
        decoded = time.perf_counter()
        if restored.dtype != img.dtype or not np.array_equal(restored, img):
            raise ValueError(f"{sample} failed the {label} round trip")
        record(label, len(stream), round(encoded - start, 2), round(decoded - encoded, 2))

    for layout, size in satzip.jpegxl_layouts(img, effort=effort).items():
        record(f"JPEG XL ({layout}, effort {effort})", size)
    if all_codecs:
        for label, band_sizes in satzip.baselines(img).items():
            if label != "raw" and None not in band_sizes:
                record("JPEG XL (independent, effort 7 baseline)" if label == "JPEG XL" else label,
                       sum(band_sizes))

    if recipe:
        transformed = satzip.uncertainty.transform(img, recipe, fill)
        restored = satzip.uncertainty.transform(transformed, recipe, fill, inverse=True)
        if not np.array_equal(restored, img):
            raise ValueError("Uncertainty transform round trip failed")
        overhead = len(satzip.uncertainty.pack(recipe)) + 9  # recipe and fill
        sizes = satzip.jpegxl_layouts(transformed, effort=effort)
        record("JPEG XL + uncertainty transform (best layout)", min(sizes.values()) + overhead)
    elif fill is None:
        bands, table_bytes = satzip.palette.prepare(img, np.ones(img.shape, dtype=bool))
        tables, _ = satzip.palette.read(table_bytes, 0, img.dtype, len(img))
        restored = np.stack([band if table is None else table[band] for band, table in zip(bands, tables)])
        if not np.array_equal(restored, img):
            raise ValueError("Rank transform round trip failed")
        sizes = satzip.jpegxl_layouts(np.stack(bands), effort=effort)
        record("JPEG XL + rank transform (best layout)", min(sizes.values()) + len(table_bytes))
    return rows


@app.cell
def _(compare_input, directory, settings):
    mo.stop(settings.value is None, mo.md("Choose inputs and press **Run comparison**."))
    _s = settings.value
    mo.stop(not _s["files"], mo.md("Select at least one input to compare."))
    results = []
    for _file in mo.status.progress_bar(_s["files"], title="Comparing inputs"):
        results.extend(compare_input(Path(directory.value).expanduser() / _file, effort=_s["effort"],
                                     row=_s["row"], col=_s["col"],
                                     names=[n.strip() for n in _s["bands"].split(",") if n.strip()],
                                     all_codecs=_s["all_codecs"]))
    mo.ui.table(results, selection=None, label="Round-trip verified streams")
    return (results,)


@app.cell
def _(results):
    from matplotlib.ticker import MaxNLocator

    _samples = list(dict.fromkeys(r["Input"] for r in results))
    _y = np.arange(len(_samples))
    sentinel_figure, _axes = plt.subplots(1, 2, figsize=(11, max(3, 1.7 + .59 * len(_y))),
                                          sharey=True, layout="constrained")
    _ax, _gain = _axes
    _best_values, _sz2_values, _labels = [], [], []
    for _i, _sample in enumerate(_samples):
        _rows = [r for r in results if r["Input"] == _sample]
        _sz1, _sz2 = (next(r for r in _rows if r["Codec"] == c) for c in ("SZ1", "SZ2"))
        _jxl = min((r for r in _rows if r["Codec"].startswith("JPEG XL (") and "baseline" not in r["Codec"]),
                   key=lambda r: r["Bytes"])
        _values = [_sz1["Bits per pixel"], _sz2["Bits per pixel"], _jxl["Bits per pixel"]]
        _ax.plot([min(_values), max(_values)], [_i, _i], color="0.72")
        for _v, _name, _color, _shape in zip(_values, ("SZ1", "SZ2", "Best JPEG XL layout"),
                                              ("#848484", "#006b61", "#bb590c"), ("x", "o", "D")):
            _ax.scatter(_v, _i, label=_name if _i == 0 else None, color=_color, marker=_shape, zorder=3)
        _best_values.append(_jxl["Bytes"])
        _sz2_values.append(_sz2["Bytes"])
        _labels.append(_sample.split("_512_")[0].replace("sentinel-", "S").replace("olci-wfr-l2-netcdf", "OLCI")
                       if _sample.endswith(".npz") else _sample.split(".nc")[-1].strip() or _sample)
    _saving = 100 * (1 - np.array(_sz2_values) / _best_values)
    _bars = _gain.barh(_y, _saving, color=["#006b61" if s >= 0 else "#ae4630" for s in _saving])
    _gain.bar_label(_bars, fmt="%.1f%%", padding=4)
    _gain.margins(x=.22)
    _gain.axvline(0, color="0.4", lw=.8)
    _ax.set_yticks(_y, _labels)
    _ax.invert_yaxis()
    _ax.set_xlabel("Bits per band sample (including fill)")
    _gain.set_xlabel("SZ2 saving vs JPEG XL (%)")
    _ax.set_title("Complete streams")
    _gain.set_title("Paired size difference")
    _ax.legend(loc="lower left", bbox_to_anchor=(0, 1.12), ncol=3, fontsize=8)
    _gain.xaxis.set_major_locator(MaxNLocator(nbins=6))
    sentinel_figure
    return (sentinel_figure,)


@app.cell
def _(results, sentinel_figure):
    import csv
    import io

    _csv = io.StringIO()
    _writer = csv.DictWriter(_csv, fieldnames=results[0])
    _writer.writeheader()
    _writer.writerows(results)
    _png = io.BytesIO()
    sentinel_figure.savefig(_png, format="png", dpi=170, bbox_inches="tight")
    mo.hstack([mo.download(_csv.getvalue().encode(), filename="streams.csv", label="Download measurements"),
               mo.download(_png.getvalue(), filename="size-differences.png", label="Download figure")], justify="start")
    return


if __name__ == "__main__":
    app.run()
