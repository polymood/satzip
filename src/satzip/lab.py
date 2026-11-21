"""Experiment helpers for the notebooks: fixed samples, one verified measurement, a cached parallel grid.

Every measurement encodes, decodes and compares with the input. Sizes are complete streams
plus any side information that the decoder needs and that the stream does not hold
(two-pass weights, band order).
"""

import hashlib
import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

import satzip

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
LST = DATA / "ESACCI-LST-L3C-LST-SLSTRB-0.01deg_1DAILY_DAY-20230101000000-fv4.00.nc"
LST_NAMES = ["satze", "sataz", "solze", "solaz", "lst", "t2m", "ndvi", "emis_0", "emis_1", "qual_flag",
             "lst_uncertainty", "lst_unc_ran", "lst_unc_loc_atm", "lst_unc_loc_sfc", "lst_unc_sys", "lcc", "n", "ncld"]

SAMPLES = {  # name -> cache file in data/ (scripts/download_data.py fetches them)
    "S2 Toulouse": "sentinel-2-l2a_1.440_43.600_512_2024-06-01_B02-B03-B04-B08.npz",
    "S2 Beauce": "sentinel-2-l2a_1.600_48.250_512_2024-06-01_B02-B03-B04-B08.npz",
    "S1 Toulouse": "sentinel-1-grd_1.440_43.600_512_2024-06-01_vv-vh.npz",
    "S3 OLCI Biscay": "sentinel-3-olci-wfr-l2-netcdf_-5.000_47.000_512_2024-07-01_oa03-reflectance-oa04-reflectance-"
                      "oa05-reflectance-oa06-reflectance-oa07-reflectance-oa08-reflectance.npz",
    "LST 18 grids": None,  # 512 x 512 centre of chunk (11, 19), all 18 integer grids
    # Large samples, for big models (more data to learn from, longer wavefronts for a GPU):
    "S2 Toulouse 2048": "sentinel-2-l2a_1.440_43.600_2048_2024-06-01_B02-B03-B04-B08.npz",
    "LST chunk 1000": None,  # all of chunk (11, 19): 18 grids x 1000 x 1000
}


def load(sample):
    """(img, fill) for a sample name."""
    if sample.startswith("LST"):
        img, _ = satzip.nc_chunk(LST, LST_NAMES, 11, 19)
        if sample == "LST 18 grids":
            img = img[:, 244:756, 244:756]
        return np.ascontiguousarray(img), -32768
    with np.load(DATA / SAMPLES[sample]) as z:
        return z["img"], (int(z["fill"].item()) if z["fill"].size else None)


PRIORS = {  # offline training sets (tufazip's "Cycle2 prior" idea); none overlaps a test sample
    "S2 Paris+Lyon+Nantes": ["sentinel-2-l2a_2.350_48.850_512_2024-06-01_B02-B03-B04-B08.npz",
                             "sentinel-2-l2a_4.830_45.760_512_2024-06-01_B02-B03-B04-B08.npz",
                             "sentinel-2-l2a_-1.550_47.220_512_2024-06-01_B02-B03-B04-B08.npz"],
}
TRAIN_KEYS = ("hidden", "lr", "train_every", "n_prev", "blocks", "context", "device")


def offline_prior(name, epochs=1, **codec):
    """Weights trained on the PRIORS[name] tiles with the codec's settings. Cached in reports/priors."""
    train = {k: codec[k] for k in TRAIN_KEYS if k in codec}
    key = hashlib.sha256(json.dumps([name, epochs, train], sort_keys=True).encode()).hexdigest()[:16]
    path = ROOT / "reports" / "priors" / f"{key}.pt"
    if path.exists():
        return torch.load(path)
    tiles = [np.load(DATA / f)["img"] for f in PRIORS[name]]
    prior = satzip.pretrain(tiles, epochs=epochs, **train)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(prior, path)
    return prior


def greedy_order(img, fill=None):
    """Band order where each band follows the band whose fine detail it correlates with most.
    Detail = value minus its 3 x 3 mean, on pixels valid in every band."""
    x = img.astype(np.float64)
    valid = np.ones(img.shape[1:], bool) if fill is None else (img != fill).all(0)
    pad = np.pad(x, ((0, 0), (1, 1), (1, 1)), mode="edge")
    blur = sum(pad[:, 1 + dy:1 + dy + x.shape[1], 1 + dx:1 + dx + x.shape[2]]
               for dy in (-1, 0, 1) for dx in (-1, 0, 1)) / 9
    d = (x - blur)[:, valid]
    if d.shape[1] < 2:
        return list(range(len(img)))
    c = np.abs(np.nan_to_num(np.corrcoef(d)))
    np.fill_diagonal(c, 0)
    order = [int(c.sum(1).argmax())]
    while len(order) < len(img):
        rest = [b for b in range(len(img)) if b not in order]
        order.append(max(rest, key=lambda b: c[order[-1], b]))
    return order


def measure(sample, *, two_pass=0, order="natural", prior=None, prior_epochs=1, **codec):
    """Encode, decode, check and time one configuration.

    two_pass = E > 0: the encoder first trains on the image for E epochs and ships the
    weights as float16 (counted in the size).
    prior = a PRIORS name: start from weights trained offline on other tiles. Like the
    tufazip decompressor, the decoder holds them; their size is reported, not added.
    codec: keyword arguments for satzip.compress (hidden, lr and the encoder options)."""
    img, fill = load(sample)
    gpu = torch.device(codec.get("device", "cpu")).type != "cpu"
    if gpu:
        torch.cuda.reset_peak_memory_stats()
    extra = 0
    if order == "greedy":
        img = img[greedy_order(img, fill)]
        extra += len(img)  # the permutation, one byte per band
    weights = offline_prior(prior, prior_epochs, **codec) if prior else None
    prior_bytes = sum(v.numel() * 2 for v in weights.values()) if weights else 0
    t0 = time.perf_counter()
    if two_pass:
        train = {k: codec[k] for k in TRAIN_KEYS if k in codec}
        weights = satzip.pretrain([img], epochs=two_pass, fill=fill, init=weights, **train)
        weights = {k: v.half().float() for k, v in weights.items()}  # what the decoder receives
        extra += sum(v.numel() * 2 for v in weights.values())
    elif weights:
        weights = {k: v.half().float() for k, v in weights.items()}
    data, _ = satzip.compress(img, prior=weights, fill=fill, **codec)
    t_enc = time.perf_counter() - t0
    t0 = time.perf_counter()
    if not np.array_equal(satzip.decompress(data, prior=weights), img):
        raise AssertionError(f"round trip failed: {sample} {codec}")
    t_dec = time.perf_counter() - t0
    size = len(data) + extra
    params = sum(v.numel() for v in satzip._initial_model(codec.get("hidden", 64), None, 2,
                                                           {k: v for k, v in codec.items()
                                                            if k in satzip.DEFAULTS}).state_dict().values())
    return {"sample": sample, "two_pass": two_pass, "order": order, **codec, "bytes": size,
            "side bytes": extra, "bpp": 8 * size / img.size, "t_enc": t_enc, "t_dec": t_dec,
            "pixels": img.size, "params": params, "prior": prior, "prior bytes": prior_bytes,
            "peak VRAM MB": torch.cuda.max_memory_allocated() / 2**20 if gpu else 0,
            "gpu": torch.cuda.get_device_name() if gpu else ""}


def _key(cfg):
    return json.dumps(cfg, sort_keys=True)


def _run(cfg):
    return measure(**cfg)


def cached(configs, cache):
    """True when every configuration already has a result in `cache`."""
    cache = Path(cache)
    have = {_key(r["config"]) for r in json.loads(cache.read_text())} if cache.exists() else set()
    return all(_key(c) in have for c in configs)


def grid(configs, cache, workers=None, progress=None):
    """Measure each configuration once, in parallel. Results persist in `cache` (JSON) after
    every finished job, so a stopped run resumes. Returns rows in the order of `configs`."""
    cache = Path(cache)
    done = {}
    if cache.exists():
        done = {_key(r["config"]): r for r in json.loads(cache.read_text())}
    todo = [c for c in configs if _key(c) not in done]
    if todo:
        cache.parent.mkdir(parents=True, exist_ok=True)
        on_gpu = any(c.get("device", "cpu") != "cpu" for c in todo)
        workers = workers or (1 if on_gpu else os.cpu_count())  # one GPU job at a time by default
        # Notebook probes may have initialized HIP already. GPU workers need a fresh runtime.
        ctx = multiprocessing.get_context("spawn") if on_gpu else None
        with ProcessPoolExecutor(min(workers, len(todo)), mp_context=ctx) as ex:
            futures = {ex.submit(_run, c): c for c in todo}
            for n, f in enumerate(as_completed(futures), 1):
                done[_key(futures[f])] = {"config": futures[f], "result": f.result()}
                cache.write_text(json.dumps(list(done.values()), indent=1))
                if progress:
                    progress(n, len(todo))
    return [done[_key(c)]["result"] for c in configs]


def probe(hidden, blocks, context=0, n_prev=2, batch=512, device="cuda", steps=5):
    """Time one online training step (forward, backward, Adam) of a model size and measure its
    memory. Use it to find the largest model that fits before a long run."""
    dev = torch.device(device)
    model = satzip.make_model(hidden, n_prev=n_prev, blocks=blocks, context=context).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    n_in = model[0].in_features if isinstance(model, torch.nn.Sequential) else model.inp.in_features
    f = torch.randn(batch, n_in, device=dev)
    if dev.type != "cpu":
        torch.cuda.reset_peak_memory_stats()
    for i in range(steps + 1):
        if i == 1:  # first step warms up kernels and allocator
            if dev.type != "cpu":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
        out = model(f)
        loss = out.square().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    if dev.type != "cpu":
        torch.cuda.synchronize()
    ms = 1e3 * (time.perf_counter() - t0) / steps
    return {"hidden": hidden, "blocks": blocks, "context": context, "inputs": n_in,
            "params (M)": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
            "ms per step": round(ms, 2),
            "peak MB": round(torch.cuda.max_memory_allocated() / 2**20) if dev.type != "cpu" else None}


def fronts(sample):
    """Network calls to code a sample: wavefronts per band times bands."""
    img, _ = load(sample)
    return img.shape[0] * (img.shape[2] + 2 * img.shape[1] - 2)


SMALL = [s for s in SAMPLES if "2048" not in s and "1000" not in s]  # the 512 x 512 samples of notebooks 04-06
