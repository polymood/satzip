import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")

with app.setup:
    import base64
    import tempfile
    from pathlib import Path

    import h5py
    import marimo as mo
    import numpy as np

    import satzip


@app.cell
def _():
    mo.md("""
    # Codec checks

    Run the notebook to check the codec on synthetic images and a temporary NetCDF file.
    The table reports each result. The cells below contain the individual assertions.
    """)
    return


@app.cell
def _(
    test_benchmark_edge_tiles_and_stored_sizes,
    test_decode_original_sz1_stream,
    test_old_prior_selects_sz1,
    test_quadrature_preserves_fill_and_wraps_for_all_byte_values,
    test_residual_tokens_at_both_boundaries,
    test_roundtrip,
    test_roundtrip_fill,
    test_roundtrip_prior_and_constant_band,
    test_sparse_wide_palette_preserves_endianness_and_fill,
    test_sz3_options_roundtrip,
    test_wavefronts_cover_each_pixel_once_after_its_neighbours,
):
    _checks = {
        "Lossless round trip and estimated bit cost": test_roundtrip,
        "Pretrained prior and constant band": test_roundtrip_prior_and_constant_band,
        "Causal wavefront order": test_wavefronts_cover_each_pixel_once_after_its_neighbours,
        "Shared, differing and empty fill masks": test_roundtrip_fill,
        "SZ3 options: batched training, steps, previous bands": test_sz3_options_roundtrip,
        "Original SZ1 byte stream": test_decode_original_sz1_stream,
        "SZ1 prior selection": test_old_prior_selects_sz1,
        "Wide integer palette and byte order": test_sparse_wide_palette_preserves_endianness_and_fill,
        "Residual tokens at both band boundaries": test_residual_tokens_at_both_boundaries,
        "Uncertainty transform over all byte values": test_quadrature_preserves_fill_and_wraps_for_all_byte_values,
        "NetCDF edge tiles, stored sizes and cache": test_benchmark_edge_tiles_and_stored_sizes,
    }
    check_results = []
    for _name, _check in _checks.items():
        try:
            _check()
        except Exception as _error:
            check_results.append({"Check": _name, "Result": "Failed", "Detail": repr(_error)})
        else:
            check_results.append({"Check": _name, "Result": "Passed", "Detail": ""})
    mo.ui.table(check_results, selection=None, pagination=False)
    return (check_results,)


@app.function
def sample_image(seed=0):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[:40, :48]
    base = 1000 + 300 * np.sin(x / 7) * np.cos(y / 5)
    bands = [base * k + rng.normal(0, 20, base.shape) for k in (1.0, 1.3, 0.8)]
    return np.clip(np.stack(bands), 0, 65535).astype(np.uint16)


@app.function
def test_roundtrip():
    img = sample_image()
    data, bits = satzip.compress(img)
    assert np.array_equal(satzip.decompress(data), img)
    assert len(data) < img.nbytes / 2
    assert abs(bits.sum() / 8 - len(data)) / len(data) < 0.05  # estimated bits match coded size


@app.function
def test_roundtrip_prior_and_constant_band():
    img = sample_image(1)
    img[1] = 7  # constant band: min == max
    prior = satzip.pretrain([sample_image(2)])
    data, _ = satzip.compress(img, prior=prior)
    assert np.array_equal(satzip.decompress(data, prior=prior), img)

    opts = {"hidden": 32, "blocks": 2, "context": 2}
    tile = img[:, :10, :9]
    prior = satzip.pretrain([tile], **opts)
    data, _ = satzip.compress(tile, prior=prior, **opts)
    assert np.array_equal(satzip.decompress(data, prior=prior), tile)


@app.function
def test_wavefronts_cover_each_pixel_once_after_its_neighbours():
    h, w = 5, 7
    t = np.full((h, w), -1)
    for i, (y, x) in enumerate(satzip.wavefronts(h, w)):
        assert (t[y, x] == -1).all()
        t[y, x] = i
    assert (t >= 0).all()
    for y in range(h):
        for x in range(w):
            for dy, dx in satzip.CUR:
                if 0 <= y + dy < h and 0 <= x + dx < w:
                    assert t[y + dy, x + dx] < t[y, x]


@app.function
def test_roundtrip_fill():
    img = sample_image(3).astype(np.int16)
    img[:, 5:20, 10:30] = -32768  # swath gap shared by all bands
    img[2, :, :4] = -32768  # extra gap in one band
    img[0, 30:, :] = -32768
    data, _ = satzip.compress(img, fill=-32768)
    assert np.array_equal(satzip.decompress(data), img)
    empty = np.full((2, 9, 9), -32768, np.int16)
    assert np.array_equal(satzip.decompress(satzip.compress(empty, fill=-32768)[0]), empty)


@app.function
def test_sz3_options_roundtrip():
    img = sample_image(4).astype(np.int16)
    img[:, 3:9, 5:12] = -32768
    assert satzip.compress(img)[0][:3] == b"SZ2"  # default settings keep the SZ2 format
    for opts in ({"train_every": 4}, {"steps": 2}, {"n_prev": 1}, {"n_prev": 3, "train_every": 3, "steps": 2},
                 {"blocks": 2, "hidden": 32, "context": 3}, {"context": 8, "steps": 0}):
        data, _ = satzip.compress(img, fill=-32768, **opts)
        assert data[:3] == b"SZ3", opts
        assert np.array_equal(satzip.decompress(data), img), opts
    prior = satzip.pretrain([sample_image(5)], train_every=2, n_prev=3)
    data, _ = satzip.compress(img, prior=prior, fill=-32768, train_every=2, n_prev=3)
    assert np.array_equal(satzip.decompress(data, prior=prior), img)


@app.function
def test_decode_original_sz1_stream():
    # Produced by the original encoder, including two different fill masks.
    data = base64.b64decode(
        "U1oxPGkyAgAAAAMAAAADAAAACAAAAKabRDsB/P/////////9/////////zcAAAAAAAAAEwAAAAAAAADVA5ZB"
        "//////////8MAAAAAAAAAAYAAAAAAAAANPeCQAAAAAAAAAAAATwAAAD9N3pYWgAABObWtEYCACEBHAAAABDP"
        "WMwBAAF3gAAAAAk7IXLCuudWAAEaAtwupX4ftvN9AQAAAAAEWVoBPAAAAP03elhaAAAE5ta0RgIAIQEc"
        "AAAAEM9YzAEAAfeAAAAA/Jt1pAjLHMYAARoC3C6lfh+2830BAAAAAARZWv4LAABU2Lr3vK8qgsr1tFE=")
    img = np.array([[[-4, 7, 30], [5, -4, -3], [8, 29, 55]],
                    [[12, -1, 3], [8, -4, 8], [3, 9, 10]]], dtype="<i2")
    assert np.array_equal(satzip.decompress(data), img)
    assert satzip.compress(img, fill=-4, hidden=8, version=1)[0] == data


@app.function
def test_old_prior_selects_sz1():
    img = sample_image()[:, :8, :9]
    prior = satzip.pretrain([img], hidden=8, version=1)
    data, _ = satzip.compress(img, prior=prior, hidden=8)
    assert data.startswith(b"SZ1")
    assert np.array_equal(satzip.decompress(data, prior=prior), img)


@app.function
def test_sparse_wide_palette_preserves_endianness_and_fill():
    levels = np.array([0, 11, 1 << 63, (1 << 64) - 1], dtype=">u8")
    img = levels[np.random.default_rng(4).integers(0, 4, (3, 12, 15))]
    img[1] = 1 << 63
    img[2] = levels[-1]
    data, bits = satzip.compress(img, hidden=8, fill=int(levels[-1]))
    out = satzip.decompress(data)
    assert out.dtype == img.dtype and np.array_equal(out, img)
    assert not bits[1:].any()


@app.function
def test_residual_tokens_at_both_boundaries():
    import constriction
    from satzip.residual import ResidualModel

    values = np.arange(-300, 401, dtype=np.int32)
    enc = constriction.stream.queue.RangeEncoder()
    writer = ResidualModel(-300, 400)
    scale = np.linspace(0.05, 30, len(values))
    centers = [-400, -200, 50, 350, 500]
    for center in centers:
        costs = writer.encode(enc, values, np.full(len(values), center), scale)
        assert np.isfinite(costs).all()
    dec = constriction.stream.queue.RangeDecoder(enc.get_compressed())
    reader = ResidualModel(-300, 400)
    for center in centers:
        assert np.array_equal(reader.decode(dec, np.full(len(values), center), scale), values)


@app.function
def test_quadrature_preserves_fill_and_wraps_for_all_byte_values():
    values = np.arange(256, dtype=np.uint8).reshape(16, 16)
    img = np.stack([values, values[::-1], values.T])
    recipe = (0, (1, 2))
    for fill in (None, 0, 128, 255):
        transformed = satzip.uncertainty.transform(img, recipe, fill)
        if fill is not None:
            assert np.array_equal(img == fill, transformed == fill)
        assert np.array_equal(satzip.uncertainty.transform(transformed, recipe, fill, inverse=True), img)
        data, _ = satzip.compress(img, fill=fill, quadrature=recipe, hidden=8)
        restored = satzip.decompress(data)
        assert restored.dtype == img.dtype and np.array_equal(restored, img)


@app.function
def test_benchmark_edge_tiles_and_stored_sizes():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "small.nc"
        with h5py.File(path, "w") as f:
            ds = f.create_dataset("emis", shape=(1, 2, 9, 13), dtype="i2", chunks=(1, 1, 4, 5),
                                  compression="gzip", fillvalue=-32768)
            ds[:, 0, :8, :7] = np.arange(56).reshape(8, 7)
            ds[:, 1, 2:, 3:12] = 14
            stored = ds.id.get_storage_size()

        names = ["emis_0", "emis_1"]
        rows = satzip.nc_benchmark(path, names, workers=1, cs=6)
        assert len(rows) == 6
        assert np.isclose(sum(sum(r["NetCDF file"]) for r in rows), stored)
        assert sum(sum(r["raw"]) for r in rows) == 2 * 9 * 13 * 2
        assert satzip.nc_benchmark(path, names, workers=1, cs=6) == rows
        assert satzip.nc_cache_path(path, names, 6) != satzip.nc_cache_path(path, names[::-1], 6)
        full_cache = satzip.nc_cache_path(path, names, 6, full=True)
        assert full_cache != satzip.nc_cache_path(path, names, 6)
        full = satzip.nc_benchmark(path, names, workers=1, cs=6, full=True)
        assert all(r["SZ1"] > 0 and min(r["JPEG XL layouts"].values()) > 0 for r in full)
        assert [r["satzip"] for r in full] == [r["satzip"] for r in rows]
        full_cache.unlink()
        assert satzip.nc_benchmark(path, names, workers=1, cs=6, full=True) == full


if __name__ == "__main__":
    app.run()
