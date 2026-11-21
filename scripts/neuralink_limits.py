"""Information-theoretic limits for lossless compression of the Neuralink recordings.

All quantities are computed on the 149 held-out recordings in ADC rank units (quantization step 1).

    python scripts/neuralink_limits.py measure   # needs the WAVs; writes the CSVs
    python scripts/neuralink_limits.py plot      # renders the figures from the CSVs only
"""
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # one BLAS thread per worker process
os.environ.setdefault("OMP_NUM_THREADS", "1")

import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "neuralink_limits"
RATE = 19531  # Hz, from the WAV headers
ORDERS = (1, 4, 32, 128, 512)
WINDOWS = (8, 32, 128, 512, 2048)
LAGS = (1, 2, 5, 10, 50)
PLATEAU_HZ = (8000, 9000)  # flattest band below the anti-alias roll-off
SEG = 1024


def gaussian_bits(var):
    """Entropy of a Gaussian of variance `var` quantized with step 1, in bits."""
    return 0.5 * np.log2(2 * np.pi * np.e * var)


def levinson(r, p):
    """Prediction coefficients a (x_t ~ a @ [x_{t-1}, ..., x_{t-p}]) and the error variance of orders 0..p."""
    a, err, errs = np.zeros(0), r[0], [r[0]]
    for k in range(1, p + 1):
        g = (r[k] - a @ r[k - 1:0:-1]) / err
        a = np.r_[a - g * a[::-1], g]
        err *= 1 - g * g
        errs.append(err)
    return a, np.array(errs)


def autocorrelation(x, lags):
    n = len(x)
    return np.fft.irfft(np.abs(np.fft.rfft(x, 2 * n)) ** 2)[:lags + 1] / n


def psd(x):
    """Two-sided power spectral density per cycle (white noise of variance v gives v), Welch with Hann windows."""
    w = np.hanning(SEG)
    segs = np.lib.stride_tricks.sliding_window_view(x, SEG)[::SEG // 2]
    segs = (segs - segs.mean(1, keepdims=True)) * w
    return np.fft.rfftfreq(SEG, 1 / RATE), (np.abs(np.fft.rfft(segs, axis=1)) ** 2).mean(0) / (w @ w)


def residual_tests(x, prefix):
    """Nonlinear structure left in the order-128 linear prediction residual."""
    a, _ = levinson(autocorrelation(x, 128), 128)
    e = x[128:] - np.lib.stride_tricks.sliding_window_view(x, 128)[:-1, ::-1] @ a
    t = np.arange(max(WINDOWS), len(e))
    fixed = gaussian_bits(e[t].var())
    out = {}
    c = np.r_[0, np.cumsum(e * e)]
    for w in WINDOWS:  # Gaussian code length with the variance of the previous w residuals
        v = (c[t] - c[t - w]) / w
        adaptive = np.mean(0.5 * np.log2(2 * np.pi * v) + e[t] ** 2 / (2 * v * np.log(2)))
        out[f"{prefix}_volatility_gain_w{w}"] = fixed - adaptive
    m = np.abs(e) - np.abs(e).mean()
    for k in LAGS:
        out[f"{prefix}_abs_corr_lag{k}"] = (m[k:] @ m[:-k]) / (m @ m)
    return out


def measure_one(job):
    path, table = job
    from satzip import audio
    x = np.searchsorted(table, audio.read_wav(path.read_bytes())[1]).astype(float)
    x -= x.mean()
    r = autocorrelation(x, max(ORDERS))
    a, errs = levinson(r, 128)
    e = x[128:] - np.lib.stride_tricks.sliding_window_view(x, 128)[:-1, ::-1] @ a
    _, counts = np.unique(np.rint(e), return_counts=True)
    q = counts / counts.sum()
    _, errs = levinson(r, max(ORDERS))
    freq, s = psd(x)
    band = (freq >= PLATEAU_HZ[0]) & (freq < PLATEAU_HZ[1])
    row = {"file": path.name, "samples": len(x), "std_steps": np.sqrt(r[0]), "no_prediction": gaussian_bits(r[0]),
           **{f"oracle_linear_p{p}": gaussian_bits(errs[p]) for p in ORDERS},
           "oracle_linear_p128_histogram": -(q * np.log2(q)).sum(),
           "noise_std_steps": np.sqrt(np.median(s[band])),
           "noise_floor": gaussian_bits(np.median(s[band])),
           "noise_floor_minimum": gaussian_bits(s[freq > 300].min())}
    rng = np.random.default_rng(len(x))
    spectrum = np.fft.rfft(x)
    surrogate = np.fft.irfft(np.abs(spectrum) * np.exp(2j * np.pi * rng.random(len(spectrum))), len(x))
    row |= residual_tests(x, "real") | residual_tests(surrogate, "surrogate")
    return row, s


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0])
        w.writeheader()
        w.writerows({k: f"{v:.6g}" if isinstance(v, float) else v for k, v in r.items()} for r in rows)


def measure():
    sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
    from neuralink import split
    from satzip import audio
    train, _, test = split()
    table = audio.value_table([f.read_bytes() for f in train])
    with ProcessPoolExecutor(os.cpu_count()) as ex:
        results = list(ex.map(measure_one, [(f, table) for f in test]))
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "per-recording.csv", [r for r, _ in results])
    spectra = np.array([s for _, s in results])
    freq = np.fft.rfftfreq(SEG, 1 / RATE)
    write_csv(OUT / "spectrum.csv", [{"frequency_hz": f, "mean_psd": m, "p10_psd": lo, "p90_psd": hi}
                                     for f, m, lo, hi in zip(freq, spectra.mean(0), *np.percentile(spectra, [10, 90], 0))])
    spacing = np.diff(table.astype(int))
    print(f"{len(test)} recordings; value table {len(table)} levels, spacing {spacing.min()}-{spacing.max()} "
          f"(median {np.median(spacing):.0f})")
    rows = [r for r, _ in results]
    for k in rows[0]:
        if k != "file":
            print(f"{k:40} {np.mean([r[k] for r in rows]):10.5f}")


def read(name):
    with (OUT / name).open() as f:
        return list(csv.DictReader(f))


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "axes.titlelocation": "left", "axes.labelsize": 10})
    rows = read("per-recording.csv")
    mean = lambda k: np.mean([float(r[k]) for r in rows])
    bench = {r["file"]: r for r in csv.DictReader((ROOT / "reports/neuralink/c128-h512-d3-bench.csv").open())}
    sac = {r["file"]: r for r in csv.DictReader((ROOT / "reports/neuralink/sac-bench.csv").open())}
    samples = sum((int(r["wav"]) - 44) // 2 for r in bench.values())
    rate = lambda table, k: 8 * sum(int(r[k]) for r in table.values()) / samples
    ladder = [("No prediction (Gaussian, signal variance)", mean("no_prediction"), "#999999"),
              ("LZMA, measured", rate(bench, "lzma"), "#D55E00"),
              ("Sac --veryhigh, measured", rate(sac, "veryhigh"), "#D55E00"),
              ("satzip neural codec, measured", rate(bench, "satzip"), "#0072B2"),
              ("Oracle linear predictor, order 512", mean("oracle_linear_p512"), "#007F64"),
              ("Oracle order 128, true residual histogram", mean("oracle_linear_p128_histogram"), "#007F64"),
              ("Physics floor: white noise 8-9 kHz", mean("noise_floor"), "#6A3D9A"),
              ("Physics floor: lowest PSD bin", mean("noise_floor_minimum"), "#6A3D9A"),
              ("200x target", 16 / 200, "#000000")]
    fig, ax = plt.subplots(figsize=(7.6, 4.2), layout="constrained")
    y = np.arange(len(ladder))[::-1]
    for yi, (name, bits, color) in zip(y, ladder):
        ax.hlines(yi, 0, bits, color=color, lw=2.5)
        ax.plot(bits, yi, "o", color=color)
        ax.text(bits + 0.08, yi, f"{bits:.2f} bits  ({16 / bits:.1f}x)", va="center", fontsize=8.5)
    ax.set_yticks(y, [n for n, _, _ in ladder], fontsize=9)
    ax.set(xlim=(0, 7.6), xlabel="Bits per sample (16 bits = uncompressed)",
           title="Codecs and bounds, 149 held-out recordings")
    ax.grid(axis="x", alpha=.2)
    fig.savefig(OUT / "limits.png", dpi=200)
    plt.close(fig)

    spec = read("spectrum.csv")
    f = np.array([float(r["frequency_hz"]) for r in spec])
    m, lo, hi = (np.array([float(r[k]) for r in spec]) for k in ("mean_psd", "p10_psd", "p90_psd"))
    floor = np.mean([float(r["noise_std_steps"]) ** 2 for r in rows])
    fig, ax = plt.subplots(figsize=(7.6, 3.8), layout="constrained")
    ax.fill_between(f[1:], lo[1:], hi[1:], color="#0072B2", alpha=.15, lw=0, label="10th-90th percentile of recordings")
    ax.loglog(f[1:], m[1:], color="#0072B2", label="Mean over recordings")
    ax.axhline(floor, color="#6A3D9A", ls="--", label=f"Mean 8-9 kHz level, {floor:.1f} steps$^2$ (upper estimate of white noise)")
    ax.axvspan(*PLATEAU_HZ, color="#6A3D9A", alpha=.08)
    ax.set(xlabel="Frequency (Hz)", ylabel="PSD (rank steps$^2$ per cycle/sample)",
           title="Power spectrum of the held-out recordings")
    ax.legend(frameon=False, fontsize=8.5)
    fig.savefig(OUT / "spectrum.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    {"measure": measure, "plot": plot}[sys.argv[1] if len(sys.argv) > 1 else "measure"]()
