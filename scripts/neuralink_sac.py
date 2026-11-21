"""Benchmark Sac (github.com/slmdev/sac) on the held-out Neuralink recordings, beside satzip's results.

Build Sac first (on Linux without static libc, link dynamically):
    git clone https://github.com/slmdev/sac && make -C sac -j LDFLAGS=-s
    python scripts/neuralink_sac.py sac/sac
"""
import csv
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from neuralink import REPORT, split

PROFILES = ("--high", "--veryhigh")


def run(job):
    sac, path = job
    wav, row = path.read_bytes(), {"file": path.name}
    with tempfile.TemporaryDirectory() as d:
        stream, decoded = Path(d) / "a.sac", Path(d) / "a.wav"
        for p in PROFILES:
            subprocess.run([sac, p, str(path), str(stream)], check=True, capture_output=True)
            subprocess.run([sac, "--decode", str(stream), str(decoded)], check=True, capture_output=True)
            if decoded.read_bytes() != wav:
                raise AssertionError(f"Sac {p} round trip failed: {path.name}")
            row[p.strip("-")] = stream.stat().st_size
    return row


if __name__ == "__main__":
    sac = str(Path(sys.argv[1]).resolve())
    with ProcessPoolExecutor() as ex:
        rows = list(ex.map(run, [(sac, f) for f in split()[2]]))
    ours = {r["file"]: r for r in csv.DictReader((REPORT / "c128-h512-d3-bench.csv").open())}
    for r in rows:
        r |= {"wav": int(ours[r["file"]]["wav"]), "satzip": int(ours[r["file"]]["satzip"])}
    with (REPORT / "sac-bench.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "wav", "satzip", "high", "veryhigh"])
        w.writeheader()
        w.writerows(rows)
    total = {k: sum(r[k] for r in rows) for k in ("wav", "satzip", "high", "veryhigh")}
    samples = (total["wav"] - 44 * len(rows)) / 2
    print(f"{len(rows)} held-out files, every Sac stream decoded exactly")
    for k in ("satzip", "high", "veryhigh"):
        wins = "" if k == "satzip" else f"  smaller than satzip on {sum(r[k] < r['satzip'] for r in rows)} files"
        print(f"{k:9} {total[k]:>10,} bytes  {total['wav'] / total[k]:.4f}x  {8 * total[k] / samples:.4f} bits/sample{wins}")
