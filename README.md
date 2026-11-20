# satzip

satzip is an experimental lossless compressor for integer satellite image stacks. It uses the online learning idea from [tufazip](https://github.com/Tufalabs/tufazip). Encoder and decoder train the same small neural network after each wavefront of pixels. Each band can use two earlier bands as side information.

## How SZ2 works

1. For a band with gaps between observed integer levels, the encoder estimates the bits those gaps cost. When the estimated saving covers the compressed lookup table and a small margin, it codes each level by its rank. The stream carries the table. Wide integer coordinates also use this mapping to keep the predictor inputs small.
2. The network starts from the MED spatial predictor. In each of the two previous bands, it estimates a local linear relationship from six pairs of already-decoded neighbours. The resulting spectral predictions, their fit errors, and the spatial context help it predict a value and scale.
3. The encoder rounds and clips the predicted value to the band range and codes the signed integer error. Small errors get their own tokens. Larger errors share a token for their leading bits and carry the remaining bits separately. The predicted scale selects a histogram. Both sides update the token counts after each wavefront.
4. The network learns from the reconstructed wavefront using a quantized Laplace loss. A supplied fill value gets a separate mask, and a constant band takes its value from the header.

## Install and use

```sh
uv sync
```

## Get the test data

The repository does not contain data. Run the download script to put the test data in `data/`.

```sh
uv run python scripts/download_data.py            # all data, about 410 MB
uv run python scripts/download_data.py --no-lst   # Sentinel samples only, about 50 MB
```

The Sentinel-1, Sentinel-2 and Sentinel-3 windows come from Microsoft Planetary Computer.
The ESA CCI LST file (SLSTR-B, 2023-01-01, daytime, 359 MB) comes from the CEDA archive.
Neither source needs an account.

```python
import satzip

data, bits = satzip.compress(img, fill=-32768)  # img has shape (bands, height, width)
img2 = satzip.decompress(data)
```

`len(data)` includes the header, lookup tables, masks and coded pixels. `bits` estimates the symbol cost at each pixel. The decoder reads SZ1 and SZ2. Use `compress(img, version=1)` to reproduce the original encoder. Passing an original SZ1 prior also selects that encoder by default.

For packed uncertainty grids on the same scale, SZ2 can predict the total from its components using their rounded root-sum-square. Pass `quadrature=(total_index, component_indices)` to `compress`. The stream carries the indices, and the decoder restores the original total from its coded error. The transform preserves fill values and skips locations with a missing component. The NetCDF benchmark applies it when all five ESA CCI LST uncertainty variables are selected.

## Measured sizes

Each entry is a complete stream in bytes, encoded without an offline prior and decoded back to the input values. JPEG XL uses the smallest measured effort-9 output across independent bands, a planar stack, groups of up to three channels, and RGBA for four-band inputs. Each JPEG XL total includes eight bytes per band for the offset used to map values to uint16.

| Input | Original SZ1 | SZ2 | JPEG XL | SZ2 saving vs JPEG XL |
|---|---:|---:|---:|---:|
| Sentinel-2 Toulouse, 4 bands | 1,205,287 | 1,156,784 | 1,196,493 | 3.3% |
| Sentinel-2 Beauce farmland, 4 bands | 943,775 | 827,577 | 854,381 | 3.1% |
| Sentinel-1 Toulouse, 2 bands | 410,215 | 401,491 | 404,559 | 0.8% |
| Sentinel-3 OLCI Bay of Biscay, 6 bands | 1,245,733 | 1,194,265 | 1,258,193 | 5.1% |
| ESA CCI LST, chunk (11, 19), 18 grids | 3,787,741 | 3,699,702 | 3,913,647 | 5.5% |

The Sentinel windows above are 512 × 512. The LST measurement covers one 1000 × 1000 chunk of the January 1, 2023 SLSTR-B daytime product. On that chunk SZ2 took about 68 seconds to encode and 64 seconds to decode on this host. JPEG XL is a much faster codec on the Sentinel tiles. The original per-band effort-7 comparison gave 1,409,439 bytes for OLCI. Allowing JPEG XL to code bands together closes much of that gap.

On the LST chunk, enabling the uncertainty transform reduces SZ2 to **3,533,282 bytes**, 9.7% smaller than the best ordinary JPEG XL layout. Giving JPEG XL the same transform yields **3,708,891 bytes**, including the recipe and fill information. SZ2 retains a 4.7% advantage in that comparison. Scanning the full file found 25,037,505 valid totals. In every case the rounded prediction was within one packed count, and in 12,804,114 cases it was exact. The compressed sizes above are chunk measurements.

On the sparser LST chunk (1, 25), the original SZ1 stream used 316,853 bytes and plain SZ2 used 291,678. With the uncertainty transform, SZ2 used 281,878 bytes and the smallest tested JPEG XL layout used 312,843 bytes. Each stream decoded exactly.

On Beauce farmland, giving JPEG XL the same rank transform produces 854,135 bytes including the tables, compared with 854,381 bytes on the original values. SZ2 still uses 827,577 bytes.

## Run the comparisons and tests in Marimo

From the repository root, open the comparison notebook.

```sh
uv run marimo edit notebooks/03_compare.py
```

Select one or more cached Sentinel tiles or a local LST file, choose the JPEG XL effort and press **Run comparison**. To reproduce the LST rows, use chunk (11, 19) or (1, 25) with the prefilled band order. The table shows complete stream sizes, bits per pixel, satzip timings and the applicable preprocessing controls.

Open the test notebook to run all ten checks and see the result of each one. It uses synthetic inputs and a temporary NetCDF file.

```sh
uv run marimo edit tests/test_roundtrip.py
```

For the original Sentinel experiments or the full-file NetCDF benchmark, open the corresponding notebook.

```sh
uv run marimo edit notebooks/01_copernicus.py
uv run marimo edit notebooks/02_netcdf.py
```

Notebook 01 downloads the Sentinel samples from Microsoft Planetary Computer with anonymous access. In notebook 03, select the cached Sentinel tiles to plot the size differences and download the measurements.

Notebook 02 runs all codecs over the full integer grid of the LST product. The full suite includes SZ1, plain SZ2, the uncertainty transform and effort-9 JPEG XL in multiple layouts. It chooses the smallest JPEG XL layout over the whole product for each variant. The same layout is used in every chunk of that comparison.

For the size benchmark, workers can reuse a JPEG XL payload for an identical array with the same shape and encoder settings. Every occurrence contributes its full byte count and is decoded and compared with its source. This matters for fill-only bands and for bands shared between the plain and transformed variants.

Set the worker count, then press **Run the benchmark**. Each verified chunk is checkpointed next to the input, so an interrupted run can resume. The completed run opens from its `.full-v1.bench.json` cache. The totals include fill-only chunks. The histograms default to populated chunks and let you select a reference codec or include the fill-only chunks. **Save figures and measurement tables** exports a PDF, PNGs, CSVs and settings. The uncertainty scan can run separately.

## Improvement experiments (CPU)

Notebooks 04 to 06 test changes to SZ2 on five samples: Sentinel-2 Toulouse, Sentinel-2 Beauce,
Sentinel-1 Toulouse, Sentinel-3 OLCI and a 512 × 512 crop of the 18 LST grids.
Each notebook opens on its saved results in `reports/`. A button computes the missing results on all cores.
Every measurement encodes, decodes and compares with the input.

```sh
uv run marimo edit notebooks/04_speed.py      # train every k wavefronts, smaller networks
uv run marimo edit notebooks/05_learning.py   # Adam steps, learning rate, width, earlier bands, band order
uv run marimo edit notebooks/06_two_pass.py   # train on the image first and send the weights
```

The experiments use three encoder options. The stream records them (format SZ3), so the decoder needs no extra input.
With the default values the encoder writes the same SZ2 stream as before.

| option | default | meaning |
|---|---|---|
| `train_every` | 1 | one training update for every k wavefronts, on all their pixels |
| `steps` | 1 | Adam steps for each update. 0 freezes the model (no online training). |
| `n_prev` | 2 | number of earlier bands that the network reads |

Main results:

- Online learning is worth 5 % to 28 % of the stream. Context from earlier bands is worth 4 % to 22 %.
- Less frequent training is up to 2.5× faster, but it costs 1 % to 13 % in size.
- `n_prev=3` makes the LST stack about 2 % smaller. More Adam steps make Sentinel-3 OLCI up to 1.7 % smaller.
- A model trained on the whole image and then frozen is worse than online learning.
  Two passes followed by online learning make the pixel stream 0.3 % to 5 % smaller.
  The weights (14 KB) pay for themselves only on large images.

## Limits

The codec requires matching floating-point computations on both sides. Use the same CPU type and PyTorch version for encoding and decoding. The input grids must contain integers.
