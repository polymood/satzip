import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from satzip import lab

    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.alpha": 0.25})
    BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#8a8a85"
    CACHE = lab.ROOT / "reports" / "06_two_pass" / "results.json"


@app.cell
def _():
    mo.md(r"""
    # 06 · Two passes: train first, then send the weights

    SZ2 starts each image with an untrained network and learns while it codes.
    The first pixels are expensive because the model knows nothing yet.

    In a two-pass codec, the encoder first trains on the whole image for E epochs.
    Then it sends the weights (float16, about 14 KB for `hidden=64`) before the pixels. Two variants exist:

    - **two-pass + online**: the decoder starts from the sent weights and keeps learning (backpropagation) as in SZ2.
    - **two-pass, frozen** (`steps=0`): the decoder never trains. It runs only forward passes, so it decodes faster.

    The row **no learning** (`steps=0`, no weights) shows the cost of the bare MED predictor with the
    untrained network. The gap between it and SZ2 is the value of online learning.

    The sizes include the sent weights. Every measurement encodes, decodes and compares with the input.
    """)
    return


@app.cell
def _():
    VARIANTS = {
        "SZ2 (online only)": {},
        "no learning": {"steps": 0},
        **{f"two-pass E={e} + online": {"two_pass": e} for e in (1, 2, 4)},
        **{f"two-pass E={e}, frozen": {"two_pass": e, "steps": 0} for e in (1, 2, 4)},
    }
    configs = [{"sample": s, **v} for s in lab.SMALL for v in VARIANTS.values()]
    run = mo.ui.run_button(label=f"Compute missing results ({len(configs)} configurations, all cores)")
    mo.vstack([mo.md(f"Cache: `{CACHE.relative_to(lab.ROOT)}` "
                     f"({'found' if CACHE.exists() else 'not found, press the button'})"), run])
    return VARIANTS, configs, run


@app.cell
def _(VARIANTS, configs, run):
    mo.stop(not CACHE.exists() and not run.value, mo.md("No results yet."))
    with mo.status.progress_bar(total=len(configs), title="Measuring") as _bar:
        _rows = lab.grid(configs, CACHE, progress=lambda n, t: _bar.update())
    _names = [n for _ in lab.SMALL for n in VARIANTS]
    results = [{"variant": n, **r} for n, r in zip(_names, _rows)]
    ref = {r["sample"]: r for r in results if r["variant"] == "SZ2 (online only)"}
    table = [{"sample": r["sample"], "variant": r["variant"], "bits/pixel": round(r["bpp"], 4),
              "weights (bytes)": r["side bytes"],
              "size vs SZ2 (%)": round(100 * (r["bytes"] / ref[r["sample"]]["bytes"] - 1), 2),
              "pixel stream vs SZ2 (%)": round(100 * ((r["bytes"] - r["side bytes"]) / ref[r["sample"]]["bytes"] - 1), 2),
              "encode time vs SZ2 (×)": round(r["t_enc"] / ref[r["sample"]]["t_enc"], 2),
              "decode time vs SZ2 (×)": round(r["t_dec"] / ref[r["sample"]]["t_dec"], 2)} for r in results]
    mo.ui.table(table, selection=None)
    return ref, results, table


@app.cell
def _(VARIANTS, table):
    _samples = list(lab.SMALL)
    _variants = [v for v in VARIANTS if v != "SZ2 (online only)"]
    _y = np.arange(len(_variants))
    tp_fig, _axs = plt.subplots(1, len(_samples), figsize=(3.4 * len(_samples), 3.6), sharey=True,
                                layout="constrained")
    for _ax, _s in zip(_axs, _samples):
        _t = {t["variant"]: t for t in table if t["sample"] == _s}
        _total = [_t[v]["size vs SZ2 (%)"] for v in _variants]
        _pix = [_t[v]["pixel stream vs SZ2 (%)"] for v in _variants]
        _bars = _ax.barh(_y, _total, color=BLUE, height=0.6, label="total (with weights)")
        _ax.bar_label(_bars, fmt="%+.1f", padding=3, fontsize=7)
        _ax.scatter(_pix, _y, color=ORANGE, zorder=3, s=18, label="pixel stream only")
        _ax.axvline(0, color="0.3", lw=0.8)
        _ax.margins(x=0.2)
        _ax.set_title(_s, fontsize=10)
        _ax.set_xlabel("size vs SZ2 (%)")
        _ax.grid(axis="y", visible=False)
    _axs[0].set_yticks(_y, _variants, fontsize=8)
    _axs[0].invert_yaxis()
    _h, _l = _axs[0].get_legend_handles_labels()
    tp_fig.legend(_h, _l, loc="outside upper center", ncol=2, frameon=False, fontsize=8)
    tp_fig
    return


@app.cell
def _(ref, results):
    # The weights are a fixed cost; the pixel-stream saving grows with the image.
    # Break-even image size = weight bytes / (saving in bytes per pixel).
    _rows = []
    for _r in results:
        if not _r["two_pass"]:
            continue
        _saving = (ref[_r["sample"]]["bytes"] - (_r["bytes"] - _r["side bytes"])) / _r["pixels"]
        _rows.append({"sample": _r["sample"], "variant": _r["variant"],
                      "pixel-stream saving (bits/pixel)": round(8 * _saving, 4),
                      "break-even size (Mpixel samples)": round(_r["side bytes"] / _saving / 1e6, 2) if _saving > 0 else None})
    mo.vstack([mo.md("""
    ## When do the weights pay for themselves?

    The weights cost a fixed number of bytes. The saving on the pixel stream grows with the image.
    The table gives the image size (bands × height × width) at which the two-pass stream becomes smaller than SZ2,
    if the saving per pixel stays the same. An empty cell means that the pixel stream itself is not smaller.
    These samples hold 0.5 to 4.7 Mpixel samples. A full Sentinel-2 granule at 10 m holds about 480 Mpixel samples (4 bands).
    """), mo.ui.table(_rows, selection=None, pagination=False)])
    return


@app.cell
def _():
    mo.md(r"""
    ## Findings

    - **Online learning is worth 5 % to 28 %.** This is the gap between "no learning" and SZ2.
      The gain is largest on Sentinel-3 OLCI and on the LST stack.
    - **A frozen model is worse than online learning, even after 4 epochs on the same image.**
      On the LST stack it is worse than no learning at all. One fixed set of weights cannot fit 18 different grids.
      The online model keeps adapting to the current band and region. It works as a local, adaptive model,
      not as one global model of the image.
    - **Two passes followed by online learning** make the pixel stream 0.3 % to 5 % smaller.
      On these 512 × 512 samples the 14 KB of weights cancel that gain, except on Sentinel-3 OLCI (−3.9 % in total).
      On full scenes the weights are a negligible cost, so two passes are useful for large images.
    - **Frozen decoding is about 3× faster** (forward passes only). The size cost (+2 % to +30 %) is too high.

    Next ideas that follow from these results: tell the network which band it codes (band identity as an input),
    and make the start of each band cheaper.
    """)
    return


if __name__ == "__main__":
    app.run()
