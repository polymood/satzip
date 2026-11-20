import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell
def _():
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    import satzip

    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.alpha": 0.25, "lines.linewidth": 2})
    BLUE, ORANGE = "#2a78d6", "#eb6834"
    return BLUE, ORANGE, mo, np, plt, satzip, time


@app.cell
def _(mo):
    mo.md(r"""
    # satzip: neural lossless compression of Copernicus imagery

    [tufazip](https://github.com/Tufalabs/tufazip) compresses 1 GB of Wikipedia (enwik9) to 105 MB.
    It uses three parts:

    1. A transformer predicts a probability for each possible next token.
    2. An arithmetic coder writes the true token in about $-\log_2 p$ bits.
    3. The model trains online on each coded token. The decoder does the same training in the same order.
       Thus the decoder gets the same probabilities and can invert the coder.

    This notebook applies the same three parts to satellite images:

    | tufazip (text) | satzip (images) |
    |---|---|
    | 160M-parameter transformer, 2 × B200 GPU, 7 days | two 64-unit hidden layers, 1 CPU core |
    | next word token | next pixel value |
    | softmax over 4097 tokens | integer prediction error tokens conditioned on the neural scale estimate |
    | context: 192 previous tokens | context: causal neighbours W, N, NW, NE, WW, NN and a 3×3 window of 2 earlier bands |
    | arithmetic coder | range coder (`constriction`) |
    | offline "Cycle2" prior, then online training | optional offline prior, then one Adam step per wavefront |

    The pixel order is a wavefront $t = x + 2y$. All causal neighbours of a pixel are on earlier wavefronts.
    Thus the codec predicts and codes one full wavefront in one batch.
    The decoder calls the network about $W + 2H$ times for each band, not $W \times H$ times.

    SZ2 can first map the observed levels of a band to consecutive integers and store the inverse table.
    The network corrects the MED prediction using spatial neighbours and local regressions on earlier bands.
    The scale estimate selects a histogram of integer error tokens. Encoder and decoder update that histogram
    after each wavefront. The network still trains on the quantized Laplace loss.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Data

    All data comes from Microsoft Planetary Computer (anonymous access, no account).
    Each sample is a 512 × 512 window. The first call downloads the data to `data/`.
    """)
    return


@app.cell
def _(satzip):
    SUMMER = "2024-06-01T00:00:00Z/2024-09-30T23:59:59Z"
    CLEAR = {"eo:cloud_cover": {"lt": 5}}
    samples = {
        "Sentinel-2 L2A, Toulouse (B02 B03 B04 B08)":
            satzip.fetch("sentinel-2-l2a", 1.44, 43.60, ["B02", "B03", "B04", "B08"], 512, SUMMER, CLEAR),
        "Sentinel-2 L2A, Beauce farmland (B02 B03 B04 B08)":
            satzip.fetch("sentinel-2-l2a", 1.60, 48.25, ["B02", "B03", "B04", "B08"], 512, SUMMER, CLEAR),
        "Sentinel-1 GRD IW, Toulouse (VV VH)":
            satzip.fetch("sentinel-1-grd", 1.44, 43.60, ["vv", "vh"], 512),
        "Sentinel-3 OLCI WFR, Bay of Biscay (Oa03 to Oa08)":
            satzip.fetch("sentinel-3-olci-wfr-l2-netcdf", -5.0, 47.0,
                         [f"oa0{i}-reflectance" for i in range(3, 9)], 512,
                         "2024-07-01T00:00:00Z/2024-07-31T23:59:59Z"),
    }
    return CLEAR, SUMMER, samples


@app.cell
def _(np, plt, samples):
    def preview(img, fill):
        """First 3 bands (or band 0) stretched to the 2-98 percentile, for display only."""
        x = img[:3][::-1] if len(img) >= 3 else img[:1]
        x = x.astype(float)
        if fill is not None:
            x[x == fill] = np.nan
        lo, hi = np.nanpercentile(x, [2, 98])
        return np.clip((np.moveaxis(x, 0, -1) - lo) / (hi - lo), 0, 1).squeeze()

    _fig, _axs = plt.subplots(1, len(samples), figsize=(4 * len(samples), 4))
    for _ax, (_name, (_img, _fill)) in zip(_axs, samples.items()):
        _ax.imshow(preview(_img, _fill), cmap="gray")
        _ax.set_title(_name.split(",")[0] + "\n" + _name.split(",")[1].strip(), fontsize=9)
        _ax.axis("off")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Compression and lossless check

    The next cell compresses each sample, decodes it, and compares it to the original.
    It also runs the standard lossless codecs. Each codec decodes its output and the cell checks the result.
    """)
    return


@app.cell
def _(np, samples, satzip, time):
    results = {}
    for _name, (_img, _fill) in samples.items():
        _t = time.perf_counter()
        _data, _bits = satzip.compress(_img, fill=_fill)
        _te = time.perf_counter() - _t
        _t = time.perf_counter()
        assert np.array_equal(satzip.decompress(_data), _img), _name
        _td = time.perf_counter() - _t
        _sizes = {k: sum(v) for k, v in satzip.baselines(_img).items() if None not in v}
        _sizes["JPEG XL (best layout, effort 9)"] = min(satzip.jpegxl_layouts(_img).values())
        _sizes["satzip"] = len(_data)
        results[_name] = dict(sizes=_sizes, bits=_bits, t_enc=_te, t_dec=_td, n=_img.size)
    return (results,)


@app.cell
def _(mo, results):
    _codecs = list(next(iter(results.values()))["sizes"])
    _rows = []
    for _name, _r in results.items():
        _row = {"sample": _name}
        _best = min(v for k, v in _r["sizes"].items() if k not in ("raw", "satzip"))
        for _c in _codecs:
            _row[_c + " (bpp)"] = round(8 * _r["sizes"][_c] / _r["n"], 3)
        _row["satzip vs best other"] = f"{100 * (_r['sizes']['satzip'] / _best - 1):+.1f} %"
        _row["encode / decode (s)"] = f"{_r['t_enc']:.1f} / {_r['t_dec']:.1f}"
        _rows.append(_row)
    mo.vstack([
        mo.md("**Bits per pixel (lower is better). All results are lossless and checked.**"),
        mo.ui.table(_rows, selection=None),
    ])
    return


@app.cell
def _(BLUE, ORANGE, plt, results):
    _fig, _axs = plt.subplots(1, len(results), figsize=(4 * len(results), 3.2), sharex=False)
    for _ax, (_name, _r) in zip(_axs, results.items()):
        _s = {k: v for k, v in _r["sizes"].items() if k != "raw"}
        _names = sorted(_s, key=_s.get, reverse=True)
        _bpp = [8 * _s[k] / _r["n"] for k in _names]
        _ax.barh(_names, _bpp, color=[ORANGE if k == "satzip" else BLUE for k in _names], height=0.6)
        for _y, _v in enumerate(_bpp):
            _ax.text(_v, _y, f" {_v:.2f}", va="center", fontsize=8)
        _ax.set_title(_name.split(",")[0], fontsize=9)
        _ax.set_xlabel("bits per pixel")
        _ax.grid(axis="y", visible=False)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Where do the bits go?

    The map shows the estimated cost of the error token and its low bits for each pixel of band 1.
    Edges, texture and speckle cost more. Flat areas cost less.
    """)
    return


@app.cell
def _(plt, results):
    _fig, _axs = plt.subplots(1, len(results), figsize=(4 * len(results), 4))
    for _ax, (_name, _r) in zip(_axs, results.items()):
        _im = _ax.imshow(_r["bits"][0], cmap="Blues", vmin=0, vmax=16)
        _ax.set_title(_name.split(",")[0] + f"\nband 1: {_r['bits'][0].mean():.2f} bits/pixel", fontsize=9)
        _ax.axis("off")
    _fig.colorbar(_im, ax=_axs, shrink=0.7, label="bits per pixel")
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Online learning

    The model starts with no knowledge. It trains on each wavefront after the codec codes it.
    The curve shows the mean cost of all pixels coded so far, for each band of the Toulouse Sentinel-2 sample.
    The cost falls fast during the first few thousand pixels. After that, the model is stable.
    """)
    return


@app.cell
def _(BLUE, np, plt, results):
    def learning_curve(bits):
        """Mean cost of all pixels coded so far, in coding order (wavefront t = x + 2y)."""
        h, w = bits.shape
        y, x = np.mgrid[:h, :w]
        c = bits.ravel()[np.argsort((x + 2 * y).ravel(), kind="stable")]
        return np.cumsum(c) / np.arange(1, c.size + 1)

    _r = next(iter(results.values()))
    _fig, _ax = plt.subplots(figsize=(8, 3))
    _shades = ["#9ec5f4", "#5598e7", BLUE, "#104281"]
    for _b, _bits in enumerate(_r["bits"]):
        _ax.plot(learning_curve(_bits), color=_shades[_b % 4], label=f"band {_b + 1}")
    _ax.set_xscale("log")
    _ax.set_ylim(0, 20)
    _ax.legend(frameon=False, ncol=4)
    _ax.set_xlabel("pixels coded (log scale)")
    _ax.set_ylabel("mean bits per pixel so far")
    _fig.tight_layout()
    _fig
    return (learning_curve,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Offline prior (the tufazip "Cycle2 prior")

    tufazip starts from weights that it trained offline. The decoder must have the same weights,
    so tufazip counts them in the decompressor size (4.4 MB).

    The next cell trains a prior on Sentinel-2 windows from three other places (Paris, Lyon, Nantes).
    Then it compresses the Toulouse sample with and without this prior.
    The prior helps most at the start, when the online model has seen few pixels.
    For one small image, the prior size can be larger than the gain.
    For an archive of many images, you send the prior one time only.
    """)
    return


@app.cell
def _(CLEAR, SUMMER, np, samples, satzip):
    _train = [satzip.fetch("sentinel-2-l2a", lo, la, ["B02", "B03", "B04", "B08"], 512, SUMMER, CLEAR)[0]
              for lo, la in [(2.35, 48.85), (4.83, 45.76), (-1.55, 47.22)]]
    prior = satzip.pretrain(_train)
    _img, _ = next(iter(samples.values()))
    _cold, cold_bits = satzip.compress(_img)
    _warm, warm_bits = satzip.compress(_img, prior=prior)
    assert np.array_equal(satzip.decompress(_warm, prior=prior), _img)
    prior_table = [
        {"start": "no prior (online only)", "stream bytes": len(_cold), "bpp": round(8 * len(_cold) / _img.size, 3)},
        {"start": "offline prior + online", "stream bytes": len(_warm), "bpp": round(8 * len(_warm) / _img.size, 3)},
        {"start": "prior weights (sent one time)", "stream bytes": satzip.prior_nbytes(prior), "bpp": None},
    ]
    return cold_bits, prior_table, warm_bits


@app.cell
def _(BLUE, ORANGE, cold_bits, learning_curve, mo, plt, prior_table, warm_bits):
    _fig, _ax = plt.subplots(figsize=(8, 3))
    _ax.plot(learning_curve(cold_bits[0]), color=BLUE, label="no prior")
    _ax.plot(learning_curve(warm_bits[0]), color=ORANGE, label="offline prior")
    _ax.set_xscale("log")
    _ax.set_ylim(0, 20)
    _ax.legend(frameon=False)
    _ax.set_xlabel("pixels coded in band 1 (log scale)")
    _ax.set_ylabel("mean bits per pixel so far")
    _fig.tight_layout()
    mo.vstack([mo.ui.table(prior_table, selection=None), _fig])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Limits

    - These windows come from individual scenes. The table includes both the original per-band JPEG XL baseline
      and the smallest of the tested effort-9 layouts, including multichannel coding.

    - The encoder and the decoder must compute the same floating-point numbers.
      satzip uses one CPU thread and fixed seeds. Use the same machine type, PyTorch version and CPU for both.
      tufazip has the same limit: it decodes only on 2 × B200 GPUs with pinned versions.
    - The table above reports encode and decode time for each tile. Independent NetCDF chunks can run in parallel.
    - The codec reads integer bands only. For float data, find an exact integer scale first.
    """)
    return


if __name__ == "__main__":
    app.run()
