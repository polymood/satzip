"""Measure zero-lag correlation of first differences on equal-length recording prefixes."""
import csv

import numpy as np

from neuralink import DATA, REPORT
from satzip.audio import read_wav


def main():
    recordings = [(p.name, read_wav(p.read_bytes())[1]) for p in sorted(DATA.glob("*.wav"))]
    length = int(np.median([len(x) for _, x in recordings]))
    selected = [(name, x[:length]) for name, x in recordings if len(x) >= length]
    differences = np.diff(np.stack([x for _, x in selected]).astype(float))
    differences -= differences.mean(axis=1, keepdims=True)
    differences /= differences.std(axis=1, keepdims=True)
    corr = differences @ differences.T / differences.shape[1]
    np.fill_diagonal(corr, 0)
    common = differences.mean(axis=0)
    r_squared = (differences @ common / len(common)) ** 2 / common.var()
    with (REPORT / "correlation.csv").open("w") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["file", "prefix_samples", "best_partner", "abs_r", "common_mode_r_squared"])
        for i, (name, _) in enumerate(selected):
            j = np.abs(corr[i]).argmax()
            writer.writerow([name, length, selected[j][0], abs(corr[i, j]), r_squared[i]])


if __name__ == "__main__":
    main()
