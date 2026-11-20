"""Reversible rank transform for bands with gaps in their integer alphabet."""

import struct
import zlib

import numpy as np


def prepare(img, masks):
    bands, tables = [], []
    for band, valid in zip(img, masks):
        values, counts = np.unique(band[valid], return_counts=True)
        table = b""
        if values.size:
            lo, hi = int(values[0]), int(values[-1])
            # The predictor and range coder work on small integer coordinates.
            required = lo < -(1 << 23) or hi >= 1 << 23
            if required or hi - lo + 1 > len(values):
                deltas = np.diff(values, prepend=np.zeros(1, dtype=values.dtype))
                candidate = zlib.compress(deltas.astype(img.dtype).tobytes())
                # Estimate the wasted bits from the spacing of adjacent observed values.
                gaps = np.diff(values.astype(np.float64))
                savings = np.dot(counts[1:] + counts[:-1], np.log2(np.maximum(1, gaps))) / 2
                if required or savings > 8 * len(candidate) + 0.05 * counts.sum():
                    table = candidate
                    band = np.searchsorted(values, band).astype(np.int32)
        bands.append(band)
        tables.append(struct.pack("<I", len(table)) + table)
    return bands, b"".join(tables)


def read(data, off, dtype, count):
    tables = []
    for _ in range(count):
        size, = struct.unpack_from("<I", data, off)
        off += 4
        if size:
            deltas = np.frombuffer(zlib.decompress(data[off:off + size]), dtype)
            tables.append(np.cumsum(deltas, dtype=dtype.type))
        else:
            tables.append(None)
        off += size
    return tables, off
