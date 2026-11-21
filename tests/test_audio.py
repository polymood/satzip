import io
import wave

import numpy as np

from satzip import audio


def _wav(samples, channels=1):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(19531)
        w.writeframes(np.asarray(samples, "<i2").tobytes())
    return buf.getvalue()


def _signal(rng, n, table):
    walk = np.cumsum(rng.normal(0, 3, n)) + rng.normal(0, 4, n) + len(table) // 2
    return table[np.clip(np.rint(walk), 0, len(table) - 1).astype(int)]


def test_round_trip_in_any_grouping():
    rng = np.random.default_rng(0)
    table = (np.arange(1023) * 64.1 - 32768).astype(np.int16)
    files = [_wav(_signal(rng, n, table)) for n in (2000, 1500, 2500)]
    prior = audio.train(files, audio.value_table(files), context=8, hidden=32, depth=1, steps=30, batch=256)
    streams = [audio.compress(f, prior) for f in files]
    assert all(s[:3] == audio.MAGIC and len(s) < len(f) for s, f in zip(streams, files))
    assert audio.decompress(streams[1], prior) == files[1]
    assert audio.decompress_many(streams, prior) == files
    assert audio.decompress_many(streams[::-1], prior) == files[::-1]  # same file, other lane


def test_fallback_for_unknown_values_and_formats():
    rng = np.random.default_rng(1)
    table = (np.arange(1023) * 64.1 - 32768).astype(np.int16)
    good = _wav(_signal(rng, 500, table))
    prior = audio.train([good], table, context=4, hidden=16, depth=0, steps=2, batch=64)
    odd = _wav(np.r_[_signal(rng, 400, table), 7])  # 7 is not in the table
    stereo = _wav(_signal(rng, 400, table), channels=2)
    for f in (odd, stereo, b"not a wav"):
        s = audio.compress(f, prior)
        assert s[:3] == audio.RAW_MAGIC and audio.decompress(s, prior) == f
