import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")

with app.setup:
    import itertools

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    from satzip import lab

    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.alpha": 0.25, "lines.linewidth": 2})
    SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]  # categorical, fixed order


@app.cell
def _():
    mo.md(r"""
    # 07 · Big models on a GPU

    SZ2 uses a 6 000-parameter network. tufazip uses about 160 million parameters.
    This notebook makes the predictor much larger and measures what the extra capacity gives.

    Three options control the size. The stream records them, so the decoder builds the same network.

    | option | SZ2 | meaning |
    |---|---|---|
    | `hidden` | 64 | width of each layer |
    | `blocks` | 0 | residual blocks (LayerNorm, Linear, SiLU, Linear). 0 = the SZ2 network |
    | `context` | 0 | radius of a wider input window: causal pixels of the current band and a square of each earlier band. Radius 4 adds 170 inputs, radius 8 adds 678 |

    Large networks learn slowly from zero. Two settings help:
    a lower learning rate, and an **offline prior** trained on other Sentinel-2 tiles (as tufazip does).
    The prior is part of the decoder, like the tufazip decompressor. The table reports its size separately.

    **Bit-exact decoding.** The decoder must reproduce every float of the encoder. On a GPU the codec
    switches PyTorch to deterministic algorithms. Every result below is decoded and compared with the input.
    Encode and decode on the same GPU model, driver and PyTorch version.
    """)
    return


@app.cell
def _():
    _rows = [{"PyTorch": torch.__version__, "GPU available": torch.cuda.is_available(),
              "backend": f"ROCm {torch.version.hip}" if torch.version.hip else (f"CUDA {torch.version.cuda}" if torch.version.cuda else "CPU only")}]
    if torch.cuda.is_available():
        _p = torch.cuda.get_device_properties(0)
        _rows[0].update({"device": _p.name, "VRAM (GB)": round(_p.total_memory / 2**30, 1)})
    mo.vstack([mo.md("## Hardware"), mo.ui.table(_rows, selection=None, pagination=False),
               mo.md("No GPU found: install the ROCm build with `scripts/setup_rocm.sh` and open this notebook "
                     "with `.venv-rocm/bin/marimo edit notebooks/07_big_models.py`.")
               if not torch.cuda.is_available() else mo.md("")])
    return


@app.cell
def _():
    _gpu = torch.cuda.is_available()
    _devices = ["cuda", "cpu"] if _gpu else ["cpu"]
    # Start with one measured Radeon configuration or a smaller CPU preview.
    _hidden, _blocks = (["512"], ["4"]) if _gpu else (["128", "256"], ["2"])
    form = mo.md("""
    ## Settings

    {device} {samples}

    {hidden} {blocks} {context}

    {lr} {train_every} {prior}
    """).batch(
        device=mo.ui.dropdown(_devices, value=_devices[0], label="Device"),
        samples=mo.ui.multiselect(list(lab.SAMPLES), value=["S2 Toulouse"] if _gpu else ["S2 Toulouse", "S3 OLCI Biscay"], label="Samples"),
        hidden=mo.ui.multiselect(["64", "128", "256", "512", "1024", "2048", "4096"], value=_hidden, label="hidden"),
        blocks=mo.ui.multiselect(["0", "1", "2", "4", "8", "16"], value=_blocks, label="blocks"),
        context=mo.ui.multiselect(["0", "2", "4", "6", "8"], value=["0"] if _gpu else ["0", "4"], label="context"),
        lr=mo.ui.multiselect(["0.003", "0.001", "0.0003", "0.0001"], value=["0.0003"] if _gpu else ["0.001"], label="learning rate"),
        train_every=mo.ui.multiselect(["1", "2", "4", "8"], value=["1"], label="train_every"),
        prior=mo.ui.multiselect(["none", *lab.PRIORS], value=["none"], label="offline prior (Sentinel-2 samples only)"),
    )
    form
    return (form,)


@app.cell
def _(form):
    v = form.value
    device = v["device"]
    CACHE = lab.ROOT / "reports" / "07_big_models" / f"results-{device}.json"
    _sizes = sorted({(int(h), int(b), int(c)) for h, b, c in itertools.product(v["hidden"], v["blocks"], v["context"])})
    configs = [{"sample": s} for s in v["samples"]]  # SZ2 reference (CPU)
    for s, (h, b, c), lr, k, pr in itertools.product(v["samples"], _sizes, v["lr"], v["train_every"], v["prior"]):
        if pr != "none" and not s.startswith("S2"):
            continue
        cfg = {"sample": s, "hidden": h, "blocks": b, "context": c, "lr": float(lr), "train_every": int(k), "device": device}
        if pr != "none":
            cfg["prior"] = pr
        configs.append(cfg)
    configs = list({lab._key(c): c for c in configs}.values())
    return CACHE, configs, device


@app.cell
def _(configs, device):
    # Probe each model size once: memory and time of one training step, then a time estimate per sample.
    _sizes = sorted({(c["hidden"], c["blocks"], c["context"]) for c in configs if "hidden" in c})
    _probes = []
    for _h, _b, _c in mo.status.progress_bar(_sizes, title="Probing model sizes"):
        try:
            _p = lab.probe(_h, _b, _c, device=device)
        except torch.OutOfMemoryError:
            _p = {"hidden": _h, "blocks": _b, "context": _c, "ms per step": None, "peak MB": "out of memory"}
        _probes.append(_p)
    _fronts = {s: lab.fronts(s) for s in {c["sample"] for c in configs}}
    for _p in _probes:
        if _p["ms per step"]:
            # Each wavefront: one inference pass (about 1/3 of a step) and its share of a training step.
            _per_front = _p["ms per step"] * (1 / 3 + 1)
            _p["est. encode min (largest sample)"] = round(max(_fronts.values()) * _per_front / 6e4, 1)
    mo.vstack([mo.md(f"## Plan: {len(configs)} runs\n\nOne training step on 512 pixels for each model size. "
                     "Encoding and decoding each take about the estimated time, for every learning rate and prior."),
               mo.ui.table(_probes, selection=None, pagination=False)])
    return


@app.cell
def _(CACHE, configs):
    run = mo.ui.run_button(label=f"Run the {len(configs)} configurations (results are kept after each run)")
    mo.vstack([run, mo.md(f"Cache: `{CACHE.relative_to(lab.ROOT)}` · {'some results exist' if CACHE.exists() else 'empty'}")])
    return (run,)


@app.cell
def _(CACHE, configs, run):
    mo.stop(not (run.value or lab.cached(configs, CACHE)), mo.md("Press the button to run. Finished runs stay in the cache."))
    with mo.status.progress_bar(total=len(configs), title="Encode, decode, check") as _bar:
        rows = lab.grid(configs, CACHE, progress=lambda n, t: _bar.update())
    ref = {r["sample"]: r for c, r in zip(configs, rows) if len(c) == 1}
    table = [{"sample": r["sample"], "hidden": r.get("hidden", 64), "blocks": r.get("blocks", 0),
              "context": r.get("context", 0), "lr": r.get("lr", 0.003), "train_every": r.get("train_every", 1),
              "prior": r.get("prior") or "", "params (M)": round(r["params"] / 1e6, 3),
              "bits/pixel": round(r["bpp"], 4),
              "size vs SZ2 (%)": round(100 * (r["bytes"] / ref[r["sample"]]["bytes"] - 1), 2),
              "prior (KB)": round(r["prior bytes"] / 1e3, 1),
              "encode (s)": round(r["t_enc"], 1), "decode (s)": round(r["t_dec"], 1),
              "peak VRAM (MB)": round(r["peak VRAM MB"])} for r in rows]
    mo.ui.table(table, selection=None)
    return (table,)


@app.cell
def _(table):
    _samples = list(dict.fromkeys(t["sample"] for t in table))
    big_fig, _axs = plt.subplots(1, len(_samples), figsize=(4.6 * len(_samples), 3.8), squeeze=False,
                                 sharey=True, layout="constrained")
    _contexts = sorted({t["context"] for t in table})
    for _ax, _s in zip(_axs[0], _samples):
        for _ctx, _col in zip(_contexts, SERIES):
            _t = [t for t in table if t["sample"] == _s and t["context"] == _ctx and t["params (M)"] > 0.006]
            _ax.scatter([t["params (M)"] for t in _t], [t["size vs SZ2 (%)"] for t in _t], color=_col, s=22,
                        label=f"context {_ctx}", zorder=3)
        _ax.axhline(0, color="0.3", lw=0.8)
        _ax.set_xscale("log")
        _ax.set_title(_s, fontsize=10)
        _ax.set_xlabel("parameters (millions, log)")
    _axs[0][0].set_ylabel("size vs SZ2 (%), lower is better")
    _h, _l = _axs[0][0].get_legend_handles_labels()
    big_fig.legend(_h, _l, loc="outside upper center", ncol=len(_contexts), frameon=False, fontsize=8)
    big_fig
    return


@app.cell
def _(table):
    _best = {}
    for _t in table:
        if _t["params (M)"] > 0.006 and (_t["sample"] not in _best or _t["size vs SZ2 (%)"] < _best[_t["sample"]]["size vs SZ2 (%)"]):
            _best[_t["sample"]] = _t
    mo.vstack([mo.md("## Best configuration per sample"), mo.ui.table(list(_best.values()), selection=None, pagination=False)])
    return


if __name__ == "__main__":
    app.run()
