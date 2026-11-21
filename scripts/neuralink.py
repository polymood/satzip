"""Train a prior on the Neuralink challenge recordings and benchmark the audio codec on held-out files.

Every fifth file (sorted by name) is held out for the benchmark, like the challenge, which scores
unseen recordings. One training file in twenty is kept aside to choose models (validation).

    python scripts/neuralink.py train reports/neuralink/small.pt --hidden 256 --depth 2 --steps 3000
    python scripts/neuralink.py bench reports/neuralink/small.pt --workers 4
"""
import argparse
import csv
import lzma
import subprocess
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch

from satzip import audio, lab

DATA = lab.DATA / "neuralink" / "data"
REPORT = lab.ROOT / "reports" / "neuralink"


def split():
    """(train, validation, test) file lists."""
    files = sorted(DATA.glob("*.wav"))
    return ([f for i, f in enumerate(files) if i % 5 and i % 20 != 1],
            [f for i, f in enumerate(files) if i % 20 == 1], files[::5])


def _wavpack(wav: bytes) -> int:
    with tempfile.TemporaryDirectory() as d:
        src, dst = Path(d) / "a.wav", Path(d) / "a.wv"
        src.write_bytes(wav)
        subprocess.run(["wavpack", "-q", "-hh", "-x6", str(src), "-o", str(dst)], check=True)
        return dst.stat().st_size


def _measure(job):
    path, prior_path = job
    torch.set_num_threads(1)
    wav = path.read_bytes()
    prior = torch.load(prior_path, weights_only=False)
    t0 = time.perf_counter()
    stream = audio.compress(wav, prior)
    return dict(file=path.name, wav=len(wav), satzip=len(stream), encode_s=round(time.perf_counter() - t0, 2),
                lzma=len(lzma.compress(wav, preset=9 | lzma.PRESET_EXTREME)), wavpack=_wavpack(wav)), stream


def _decode(job):
    streams, prior_path = job
    torch.set_num_threads(1)
    return audio.decompress_many(streams, torch.load(prior_path, weights_only=False))


def bench(prior_path, workers, limit):
    test = split()[2][:limit or None]
    with ProcessPoolExecutor(workers) as ex:
        results = list(ex.map(_measure, [(f, prior_path) for f in test]))
        rows, streams = [r for r, _ in results], [s for _, s in results]
        t0 = time.perf_counter()
        groups = [streams[i:i + audio.LANES] for i in range(0, len(streams), audio.LANES)]
        decoded = [w for g in ex.map(_decode, [(g, prior_path) for g in groups]) for w in g]
    for f, w in zip(test, decoded):
        if w != f.read_bytes():
            raise AssertionError(f"round trip failed: {f.name}")
    total = {k: sum(r[k] for r in rows) for k in ("wav", "satzip", "lzma", "wavpack")}
    REPORT.mkdir(parents=True, exist_ok=True)
    out = REPORT / f"{Path(prior_path).stem}-bench.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0], lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    samples = (total["wav"] - 44 * len(rows)) / 2
    print(f"{len(rows)} held-out files, all decoded exactly ({time.perf_counter() - t0:.0f}s decode)")
    for k in ("satzip", "wavpack", "lzma"):
        print(f"{k:8} {total[k]:>11,} bytes  {total['wav'] / total[k]:.3f}x  {8 * total[k] / samples:.3f} bits/sample")
    print(f"per-file results: {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["train", "bench"])
    p.add_argument("prior", type=Path)
    p.add_argument("--context", type=int, default=32)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=0, help="bench only the first N held-out files")
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    if a.mode == "train":
        train_files, val_files, _ = split()
        table = audio.value_table([f.read_bytes() for f in sorted(DATA.glob("*.wav"))])
        t0 = time.time()
        prior = audio.train([f.read_bytes() for f in train_files], table, context=a.context, hidden=a.hidden,
                            depth=a.depth, steps=a.steps, batch=a.batch, lr=a.lr, device=a.device,
                            val=[f.read_bytes() for f in val_files],
                            log=lambda s, b, v: print(f"step {s:6}  train {b:.3f}  validation {v:.4f} bits/sample  "
                                                      f"{time.time() - t0:.0f}s", flush=True))
        a.prior.parent.mkdir(parents=True, exist_ok=True)
        torch.save(prior, a.prior)
    else:
        bench(a.prior, a.workers, a.limit)
