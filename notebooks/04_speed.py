import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")

with app.setup:
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from satzip import lab

    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.alpha": 0.25, "lines.linewidth": 2})
    SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # categorical, fixed order
    CACHE = lab.ROOT / "reports" / "04_speed" / "results.json"


@app.cell
def _():
    mo.md(r"""
    # 04 · Speed: train less often, use a smaller network

    The profile of SZ2 on one Sentinel-2 tile shows where the time goes:

    | part | share of encode time |
    |---|---|
    | backward pass, Adam step, loss | about 55 % |
    | features (numpy) | about 20 % |
    | forward pass and range coder | the remainder |

    SZ2 takes one Adam step after each wavefront. A wavefront holds only a few hundred pixels,
    so most of the training time is Python overhead, not arithmetic.

    This notebook tests two changes. Both change the stream, so the stream records them (format SZ3).

    1. **train_every = k**: predict k wavefronts with the frozen model, then take one Adam step on all their pixels.
    2. **hidden**: the width of the two hidden layers (SZ2 uses 64).

    Every measurement encodes, decodes and compares with the input.
    """)
    return


@app.cell
def _():
    TRAIN_EVERY = [1, 2, 4, 8, 16, 32]
    HIDDEN = [16, 32, 64]
    configs = [{"sample": s, "train_every": k, "hidden": h}
               for s in lab.SMALL for h in HIDDEN for k in TRAIN_EVERY]
    run = mo.ui.run_button(label=f"Compute missing results ({len(configs)} configurations, all cores)")
    mo.vstack([mo.md(f"Cache: `{CACHE.relative_to(lab.ROOT)}` "
                     f"({'found' if CACHE.exists() else 'not found, press the button'})"), run])
    return HIDDEN, TRAIN_EVERY, configs, run


@app.cell
def _(configs, run):
    mo.stop(not CACHE.exists() and not run.value, mo.md("No results yet."))
    with mo.status.progress_bar(total=len(configs), title="Measuring") as _bar:
        rows = lab.grid(configs, CACHE, progress=lambda n, t: _bar.update())
    return (rows,)


@app.cell
def _(rows):
    # Reference: the SZ2 default (train_every 1, hidden 64) of the same sample.
    ref = {r["sample"]: r for r in rows if r["train_every"] == 1 and r["hidden"] == 64}
    table = [{"sample": r["sample"], "hidden": r["hidden"], "train_every": r["train_every"],
              "bits/pixel": round(r["bpp"], 4),
              "size vs SZ2 (%)": round(100 * (r["bytes"] / ref[r["sample"]]["bytes"] - 1), 2),
              "encode (s)": round(r["t_enc"], 2), "decode (s)": round(r["t_dec"], 2),
              "speedup": round(ref[r["sample"]]["t_enc"] / r["t_enc"], 2),
              "Mpixel/s encode": round(r["pixels"] / r["t_enc"] / 1e6, 3)} for r in rows]
    mo.accordion({"All measurements": mo.ui.table(table, selection=None, pagination=True)})
    return ref, table


@app.cell
def _(HIDDEN, rows, ref):
    _samples = list(lab.SMALL)
    speed_fig, _axs = plt.subplots(1, len(_samples), figsize=(4 * len(_samples), 3.6), sharey=True,
                                   layout="constrained")
    for _ax, _s in zip(_axs, _samples):
        for _h, _c in zip(HIDDEN, SERIES):
            _r = sorted((r for r in rows if r["sample"] == _s and r["hidden"] == _h), key=lambda r: r["train_every"])
            _x = [ref[_s]["t_enc"] / r["t_enc"] for r in _r]
            _y = [100 * (r["bytes"] / ref[_s]["bytes"] - 1) for r in _r]
            _ax.plot(_x, _y, "-o", color=_c, markersize=4, label=f"hidden {_h}")
            for _xi, _yi, _ri in zip(_x, _y, _r):
                if _h == 64:
                    _ax.annotate(f"k={_ri['train_every']}", (_xi, _yi), fontsize=7, xytext=(3, 3),
                                 textcoords="offset points")
        _ax.axhline(0, color="0.4", lw=0.8)
        _ax.set_title(_s, fontsize=10)
        _ax.set_xlabel("encode speedup vs SZ2 (×)")
    _axs[0].set_ylabel("size vs SZ2 (%), lower is better")
    _axs[0].legend(frameon=False, fontsize=8)
    speed_fig
    return


@app.cell
def _():
    budget = mo.ui.slider(0.25, 5, step=0.25, value=1.0, show_value=True,
                          label="Accepted size cost on every sample (%)")
    budget
    return (budget,)


@app.cell
def _(budget, rows, ref):
    # Recommendation: the fastest setting that stays within the size budget on every sample.
    _settings = sorted({(r["train_every"], r["hidden"]) for r in rows})
    _summary = []
    for _k, _h in _settings:
        _r = [r for r in rows if r["train_every"] == _k and r["hidden"] == _h]
        _summary.append({"train_every": _k, "hidden": _h,
                         "worst size change (%)": round(max(100 * (r["bytes"] / ref[r["sample"]]["bytes"] - 1) for r in _r), 2),
                         "mean size change (%)": round(float(np.mean([100 * (r["bytes"] / ref[r["sample"]]["bytes"] - 1) for r in _r])), 2),
                         "mean encode speedup": round(float(np.mean([ref[r["sample"]]["t_enc"] / r["t_enc"] for r in _r])), 2),
                         "mean decode speedup": round(float(np.mean([ref[r["sample"]]["t_dec"] / r["t_dec"] for r in _r])), 2)})
    _ok = [s for s in _summary if s["worst size change (%)"] <= budget.value]
    best_speed = max(_ok, key=lambda s: s["mean encode speedup"])
    mo.vstack([
        mo.md(f"""
    ## Result

    - Fastest setting within +{budget.value} % size on every sample: **train_every = {best_speed['train_every']},
      hidden = {best_speed['hidden']}**. It encodes {best_speed['mean encode speedup']}× and decodes
      {best_speed['mean decode speedup']}× faster on average, for {best_speed['mean size change (%)']:+.2f} % size.
    - Less frequent training costs much more on the multi-band samples (Sentinel-3 OLCI, LST) than on
      Sentinel-1 and Sentinel-2. The model learns all the time, so each skipped update costs bits.
    - The speedup stops near 2.5×. Features and the range coder do not get faster with fewer updates.
    - The measurements ran 16 jobs in parallel on one machine. Absolute times include this contention.
      Compare the ratios, not the seconds.
    """),
        mo.ui.table(sorted(_summary, key=lambda s: -s["mean encode speedup"]), selection=None, pagination=False),
    ])
    return (best_speed,)


if __name__ == "__main__":
    app.run()
