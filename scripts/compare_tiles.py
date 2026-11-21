"""Benchmark downloaded Sentinel NPZ tiles and keep the smallest standalone satzip file per device."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile, ZipInfo

import numpy as np
import torch

import satzip
from satzip import lab

REPORT = lab.ROOT / "reports" / "tile_comparison"
ARCHIVES = lab.DATA / "compressed"
RUNS = ARCHIVES / "measurements"
PRIOR = "S2 Paris+Lyon+Nantes"
GPU = dict(hidden=512, blocks=4, context=0, lr=0.0003, train_every=1, device="cuda")
CPU_VARIANTS = {
    "S2 Toulouse": {"n_prev": 3}, "S2 Toulouse 2048": {"n_prev": 3},
    "S2 Beauce": {"steps": 2}, "S1 Toulouse": {"n_prev": 0},
    "S3 OLCI Biscay": {"steps": 4},
}


def compare(path, mode):
    with np.load(path) as z:
        img = z["img"]
        fill = int(z["fill"].item()) if z["fill"].size else None
    name = next((s for s, f in lab.SAMPLES.items() if f == path.name), path.stem)
    rows = []

    def record(codec, size, shared=0):
        rows.append(dict(tile=path.name, codec=codec, raw_bytes=img.nbytes, bytes=size,
                         bits_per_sample=round(8 * size / img.size, 5), shared_weight_bytes=shared))

    if mode == "standard":
        for codec, sizes in satzip.baselines(img).items():
            if codec != "raw" and None not in sizes:
                record(codec, sum(sizes))

        def layouts(stack, label, overhead=0):
            for layout, size in satzip.jpegxl_layouts(stack).items():
                record(f"JPEG XL 9 {layout}{label}", size + overhead)

        layouts(img, "")
        if fill is None:
            bands, packed = satzip.palette.prepare(img, np.ones_like(img, dtype=bool))
            tables, _ = satzip.palette.read(packed, 0, img.dtype, len(img))
            if any(t is not None for t in tables):
                restored = np.stack([b if t is None else t[b] for b, t in zip(bands, tables)])
                np.testing.assert_array_equal(restored, img)
                layouts(np.stack(bands), " + rank", len(packed))
    elif mode == "two-pass":
        # Learn from this tile first. The resulting archive pays for its own weights.
        opts = CPU_VARIANTS.get(name, {})
        weights = satzip.pretrain([img], epochs=4, fill=fill, n_prev=opts.get("n_prev", 2))
        half = {k: v.half().numpy() for k, v in weights.items()}
        prior = {k: torch.from_numpy(v).float() for k, v in half.items()}
        target = ARCHIVES / f"{path.stem}.two-pass.npz"
        best = None
        for settings in ([opts, {}] if "steps" in opts else [opts]):
            stream, _ = satzip.compress(img, fill=fill, prior=prior, **settings)
            payload = io.BytesIO()
            np.savez_compressed(payload, stream=np.frombuffer(stream, np.uint8), **half)
            payload.seek(0)
            with np.load(payload) as archive:
                restored_prior = {k: torch.from_numpy(archive[k]).float() for k in archive.files if k != "stream"}
                decoded = satzip.decompress(archive["stream"].tobytes(), prior=restored_prior)
            np.testing.assert_array_equal(decoded, img, strict=True)
            size = payload.getbuffer().nbytes
            steps = settings.get("steps", 1)
            record(f"CPU two-pass (steps={steps}, weights included)", size)
            if best is None or size < best:
                target.write_bytes(payload.getbuffer())
                best = size
    else:
        configs = [("CPU SZ1", dict(version=1)), ("CPU SZ2", {})] if mode == "cpu" else [("GPU online", GPU)]
        if mode == "cpu" and name in CPU_VARIANTS:
            opts = CPU_VARIANTS[name]
            option, value = next(iter(opts.items()))
            configs.append((f"CPU {option}={value}", opts))
        if mode == "gpu" and name.startswith("S2"):
            configs.append(("GPU shared prior", GPU))
        best = None
        for codec, opts in configs:
            weights = None
            shared_bytes = 0
            if codec == "GPU shared prior":
                prior_file = ARCHIVES / "sentinel2-prior.npz"
                half = {k: v.half().numpy() for k, v in lab.offline_prior(PRIOR, **GPU).items()}
                np.savez_compressed(prior_file, **half)
                with np.load(prior_file) as archive:
                    weights = {k: torch.from_numpy(archive[k]).float() for k in archive.files}
                shared_bytes = prior_file.stat().st_size
            stream, _ = satzip.compress(img, fill=fill, prior=weights, **opts)
            decoded = satzip.decompress(stream, prior=weights)
            np.testing.assert_array_equal(decoded, img, strict=True)
            record(codec, len(stream), shared_bytes)
            if weights or best is None or len(stream) < best:
                target = ARCHIVES / f"{path.stem}.{('shared' if weights else mode)}.sz"
                target.write_bytes(stream)
                if not weights:
                    best = len(stream)
    with (RUNS / f"{path.stem}.{mode}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    print(name, mode, min((r["bytes"], r["codec"]) for r in rows), flush=True)


def bundle(paths):
    """Pack the smallest independently decodable satzip result measured for each tile."""
    # Store larger tiles first so the nested Toulouse window can refer to its parent.
    paths = sorted(paths, key=lambda p: p.stat().st_size, reverse=True)
    manifest = []
    large = lab.DATA / lab.SAMPLES["S2 Toulouse 2048"]
    small = lab.DATA / lab.SAMPLES["S2 Toulouse"]
    target = ARCHIVES / "sentinel-tiles.zip"
    temp = target.with_suffix(".zip.part")
    with ZipFile(temp, "w") as z:
        for path in paths:
            with np.load(path) as tile:
                fill = tile["fill"]
                entry = dict(file=path.name, fill=fill.tolist(), fill_dtype=fill.dtype.str,
                             sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                if path == small and large in paths:
                    with np.load(large) as parent:
                        center = parent["img"][:, 768:1280, 768:1280]
                    if np.array_equal(tile["img"], center):
                        entry.update(crop_of=large.name, window=[768, 768, 512, 512])
            if "crop_of" not in entry:
                archive = min((ARCHIVES / f"{path.stem}.{ending}" for ending in
                               ("cpu.sz", "gpu.sz", "two-pass.npz")), key=lambda f: f.stat().st_size)
                entry["payload"] = archive.name
                z.write(archive, archive.name)
            manifest.append(entry)
        z.writestr(ZipInfo("manifest.json"), json.dumps(manifest, separators=(",", ":")))
    temp.replace(target)
    print(f"{target}: {target.stat().st_size:,} bytes")


def write_measurements(paths):
    measurements = []
    for path in paths:
        for mode in ("cpu", "gpu", "two-pass", "standard"):
            with (RUNS / f"{path.stem}.{mode}.csv").open() as f:
                measurements.extend(csv.DictReader(f))
    with (REPORT / "measurements.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=measurements[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(measurements)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["cpu", "gpu", "standard", "two-pass", "bundle"])
    p.add_argument("paths", nargs="*", type=Path)
    args = p.parse_args()
    if args.mode == "gpu" and not (torch.version.hip and torch.cuda.is_available()):
        p.error("Radeon comparison requires a ROCm device")
    RUNS.mkdir(exist_ok=True, parents=True)
    REPORT.mkdir(exist_ok=True, parents=True)
    paths = [p.resolve() for p in args.paths] or sorted(lab.DATA.glob("sentinel-*.npz"), key=lambda p: p.stat().st_size)
    if args.mode == "bundle":
        write_measurements(paths)
        bundle(paths)
    else:
        for path in paths:
            compare(path, args.mode)
