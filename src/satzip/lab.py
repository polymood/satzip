"""Experiment helpers for the notebooks: fixed samples, one verified measurement, a cached parallel grid.

Every measurement encodes, decodes and compares with the input. Sizes are complete streams
plus any side information that the decoder needs and that the stream does not hold
(two-pass weights, band order).
"""

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

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
}


def load(sample):
    """(img, fill) for a sample name."""
    if sample == "LST 18 grids":
        img, _ = satzip.nc_chunk(LST, LST_NAMES, 11, 19)
        return np.ascontiguousarray(img[:, 244:756, 244:756]), -32768
    with np.load(DATA / SAMPLES[sample]) as z:
        return z["img"], (int(z["fill"].item()) if z["fill"].size else None)


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


def measure(sample, *, two_pass=0, order="natural", **codec):
    """Encode, decode, check and time one configuration.

    two_pass = E > 0: the encoder first trains on the image for E epochs and ships the
    weights as float16 (counted in the size). codec: keyword arguments for satzip.compress."""
    img, fill = load(sample)
    extra = 0
    if order == "greedy":
        img = img[greedy_order(img, fill)]
        extra += len(img)  # the permutation, one byte per band
    prior = None
    t0 = time.perf_counter()
    if two_pass:
        train = {k: codec[k] for k in ("hidden", "lr", "train_every", "n_prev") if k in codec}
        weights = satzip.pretrain([img], epochs=two_pass, fill=fill, **train)
        prior = {k: v.half().float() for k, v in weights.items()}  # what the decoder receives
        extra += sum(v.numel() * 2 for v in prior.values())
    data, _ = satzip.compress(img, prior=prior, fill=fill, **codec)
    t_enc = time.perf_counter() - t0
    t0 = time.perf_counter()
    if not np.array_equal(satzip.decompress(data, prior=prior), img):
        raise AssertionError(f"round trip failed: {sample} {codec}")
    t_dec = time.perf_counter() - t0
    size = len(data) + extra
    return {"sample": sample, "two_pass": two_pass, "order": order, **codec, "bytes": size,
            "side bytes": extra, "bpp": 8 * size / img.size, "t_enc": t_enc, "t_dec": t_dec,
            "pixels": img.size}


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
        with ProcessPoolExecutor(min(workers or os.cpu_count(), len(todo))) as ex:
            futures = {ex.submit(_run, c): c for c in todo}
            for n, f in enumerate(as_completed(futures), 1):
                done[_key(futures[f])] = {"config": futures[f], "result": f.result()}
                cache.write_text(json.dumps(list(done.values()), indent=1))
                if progress:
                    progress(n, len(todo))
    return [done[_key(c)]["result"] for c in configs]
