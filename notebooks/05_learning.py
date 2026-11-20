import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from satzip import lab

    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.alpha": 0.25})
    BLUE, ORANGE, RED = "#2a78d6", "#eb6834", "#e34948"
    CACHE = lab.ROOT / "reports" / "05_learning" / "results.json"


@app.cell
def _():
    mo.md(r"""
    # 05 · Learning and context

    Notebook 04 shows that fewer updates cost many bits. The model learns all the time,
    so this notebook tests the opposite direction and the inputs of the model.
    It changes one setting at a time, starting from the SZ2 defaults:

    | setting | SZ2 | tested |
    |---|---|---|
    | `steps`: Adam steps on each wavefront | 1 | 2, 3, 4 |
    | `lr`: learning rate | 0.003 | 0.001, 0.006, 0.01 |
    | `hidden`: width of the hidden layers | 64 | 32, 128 |
    | `n_prev`: earlier bands used as context | 2 | 0 to 5 (at most bands − 1) |
    | band order | file order | greedy: each band follows the band with the most similar fine detail |

    The greedy order costs one byte per band in the size, because the decoder needs the permutation.
    Every measurement encodes, decodes and compares with the input.
    """)
    return


@app.function
def single_changes(sample, bands):
    yield {"sample": sample}
    for v in (2, 3, 4):
        yield {"sample": sample, "steps": v}
    for v in (1e-3, 6e-3, 1e-2):
        yield {"sample": sample, "lr": v}
    for v in (32, 128):
        yield {"sample": sample, "hidden": v}
    for v in range(0, min(5, bands - 1) + 1):
        if v != 2:
            yield {"sample": sample, "n_prev": v}
    if bands > 2:
        yield {"sample": sample, "order": "greedy"}


@app.cell
def _():
    _bands = {s: len(lab.load(s)[0]) for s in lab.SMALL}
    configs = [c for s in lab.SMALL for c in single_changes(s, _bands[s])]
    run = mo.ui.run_button(label=f"Compute missing results ({len(configs)} configurations, all cores)")
    mo.vstack([mo.md(f"Cache: `{CACHE.relative_to(lab.ROOT)}` "
                     f"({'found' if CACHE.exists() else 'not found, press the button'})"), run])
    return configs, run


@app.function
def label(cfg):
    keys = [k for k in cfg if k != "sample"]
    return "SZ2 default" if not keys else ", ".join(f"{k}={cfg[k]}" for k in keys)


@app.cell
def _(configs, run):
    mo.stop(not CACHE.exists() and not run.value, mo.md("No results yet."))
    with mo.status.progress_bar(total=len(configs), title="Measuring") as _bar:
        _rows = lab.grid(configs, CACHE, progress=lambda n, t: _bar.update())
    results = [(c, r) for c, r in zip(configs, _rows)]
    ref = {c["sample"]: r for c, r in results if label(c) == "SZ2 default"}
    table = [{"sample": c["sample"], "change": label(c), "bits/pixel": round(r["bpp"], 4),
              "size vs SZ2 (%)": round(100 * (r["bytes"] / ref[c["sample"]]["bytes"] - 1), 2),
              "encode time vs SZ2 (×)": round(r["t_enc"] / ref[c["sample"]]["t_enc"], 2)} for c, r in results]
    mo.accordion({"All measurements": mo.ui.table(table, selection=None)})
    return ref, results, table


@app.cell
def _(table):
    _samples = list(lab.SMALL)
    _changes = list(dict.fromkeys(t["change"] for t in table if t["change"] != "SZ2 default"))
    _y = np.arange(len(_changes))
    learn_fig, _axs = plt.subplots(1, len(_samples), figsize=(3.3 * len(_samples), 0.32 * len(_changes) + 1.6),
                                   sharey=True, layout="constrained")
    for _ax, _s in zip(_axs, _samples):
        _v = {t["change"]: t["size vs SZ2 (%)"] for t in table if t["sample"] == _s}
        _vals = [_v.get(c, np.nan) for c in _changes]
        _ax.barh(_y, _vals, color=[BLUE if (v == v and v < 0) else "#c3c2b7" for v in _vals], height=0.7)
        _ax.axvline(0, color="0.3", lw=0.8)
        _ax.set_title(_s, fontsize=10)
        _ax.set_xlabel("size vs SZ2 (%)")
        _ax.grid(axis="y", visible=False)
    _axs[0].set_yticks(_y, _changes, fontsize=8)
    _axs[0].invert_yaxis()
    learn_fig.suptitle("One change at a time · blue = smaller than SZ2 · empty = not applicable · each panel has its own scale", fontsize=10)
    learn_fig
    return


@app.cell
def _(results, ref):
    # For each setting, take the value with the best mean size change over the samples where it applies.
    _by = {}
    for _c, _r in results:
        for _k, _v in _c.items():
            if _k != "sample":
                _by.setdefault((_k, _v), []).append(100 * (_r["bytes"] / ref[_c["sample"]]["bytes"] - 1))
    best = {}
    for (_k, _v), _d in _by.items():
        if np.mean(_d) < 0 and (_k not in best or np.mean(_d) < best[_k][1]):
            best[_k] = (_v, float(np.mean(_d)))
    combined = {k: v for k, (v, _) in best.items()}
    mo.md("**Best single change per setting (mean over samples):** " +
          (", ".join(f"`{k}={v}` ({d:+.2f} %)" for k, (v, d) in best.items()) or "none is better than SZ2"))
    return (combined,)


@app.cell
def _(combined):
    def _fit(sample):  # n_prev cannot exceed bands - 1
        cfg = {"sample": sample, **combined}
        bands = len(lab.load(sample)[0])
        if "n_prev" in cfg:
            cfg["n_prev"] = min(cfg["n_prev"], bands - 1)
        if bands <= 2:
            cfg.pop("order", None)
        return cfg

    combo_configs = [_fit(s) for s in lab.SMALL]
    run_combo = mo.ui.run_button(label="Measure the combination")
    mo.vstack([mo.md("## Combine the best changes\n\n" + "\n".join(f"- {c['sample']}: `{label(c)}`"
                                                                     for c in combo_configs)), run_combo])
    return combo_configs, run_combo


@app.cell
def _(combo_configs, ref, run_combo):
    mo.stop(not (lab.cached(combo_configs, CACHE) or run_combo.value),
            mo.md("Press the button to measure the combination."))
    combo = lab.grid(combo_configs, CACHE)
    combo_rows = [{"sample": c["sample"], "settings": label(c), "bits/pixel": round(r["bpp"], 4),
                   "SZ2 bits/pixel": round(ref[c["sample"]]["bpp"], 4),
                   "size vs SZ2 (%)": round(100 * (r["bytes"] / ref[c["sample"]]["bytes"] - 1), 2),
                   "encode time vs SZ2 (×)": round(r["t_enc"] / ref[c["sample"]]["t_enc"], 2),
                   "decode time vs SZ2 (×)": round(r["t_dec"] / ref[c["sample"]]["t_dec"], 2)}
                  for c, r in zip(combo_configs, combo)]
    mo.ui.table(combo_rows, selection=None, pagination=False)
    return (combo_rows,)


@app.cell
def _():
    mo.md(r"""
    ## Findings

    - **Context from earlier bands is the main source of gain.** Without it (`n_prev=0`) the streams grow
      by 4 % to 22 %. The largest loss is on Sentinel-3 OLCI, where adjacent bands are very similar.
    - **More earlier bands help only products with many related grids.** `n_prev=3` reduces the LST stack
      by about 2 % and does nothing for 4-band Sentinel-2.
    - **More backpropagation steps help only Sentinel-3 OLCI** (−1 % to −1.7 %), at 1.5× to 3.5× the time.
    - **The SZ2 learning rate and width are already good.** A wider network does not help at this data size.
    - **The greedy band order is worse than the file order.** Fine-detail correlation is not a good
      proxy for how much one band predicts the next. The file order of these products is already sensible.
    - The best combination gains 0 % to 2 %, depending on the product. The best settings depend on the data,
      so the next step is to let the encoder choose them per image (the SZ3 header can carry them).
    - Time ratios of the combination come from a separate, less loaded run. Do not compare them directly
      with the single-change times.
    """)
    return


if __name__ == "__main__":
    app.run()
