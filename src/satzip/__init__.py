"""satzip: lossless multispectral image compression with an online-learned predictor.

Same recipe as tufazip / NNCP / cmix, scaled down to a CPU:

1. A neural net predicts a probability distribution for the next pixel value.
2. A range coder (arithmetic coding) spends about -log2(p) bits on the true value.
3. The net takes one gradient step on the pixels it just coded ("online learning").

The decoder repeats the same computations in the same order, so it gets the same
probabilities and can invert the coder. Nothing but the bit stream (and an
optional offline "prior" of initial weights) is shared.

Pixel order is a wavefront t = x + 2y. The causal neighbours W, NW, N, NE, WW, NN of
a pixel all lie on earlier wavefronts, so one wavefront is coded as one batch. That
gives about W + 2H network calls per band instead of H * W.
"""

import hashlib
import io
import os
import json
import lzma
import math
import struct
import urllib.request
import zlib
from collections import OrderedDict
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # deterministic GPU matmuls (must precede CUDA init)

import constriction
import numpy as np
import torch

from . import palette, uncertainty
from .residual import ResidualModel

PAD = 8  # border of the working copy; also the largest `context` radius
CUR = [(0, -1), (-1, 0), (-1, -1), (-1, 1), (0, -2), (-2, 0)]  # W N NW NE WW NN
WIN = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]  # 3x3 in an earlier band
N_PREV = 2  # earlier bands used as side information (default)
N_IN = len(CUR) + 2 + N_PREV * (len(WIN) + 2)
N_EXTRA = 4 + 5 * N_PREV
# Encoder options. SZ2 streams use the defaults; SZ3 / SQ3 streams store the others as JSON.
DEFAULTS = {"train_every": 1, "steps": 1, "n_prev": N_PREV, "blocks": 0, "context": 0, "device": "cpu"}
_SZ2_CUR = set(CUR) | {(-1, -2), (-2, -1), (-2, 1), (-2, 2)}
MAGIC = b"SZ2"
LEGACY_MAGIC = b"SZ1"
QUADRATURE_MAGIC = b"SQ2"


def n_inputs(version=2, n_prev=N_PREV):
    base = len(CUR) + 2 + n_prev * (len(WIN) + 2)
    return base if version == 1 else base + 4 + 5 * n_prev


def _cur_offsets(r):
    """Causal offsets in the current band within radius r, beyond the SZ2 set. dx + 2dy < 0
    means an earlier wavefront, so the decoder already has the value."""
    return [(dy, dx) for dy in range(-r, 1) for dx in range(-r, r + 1)
            if dx + 2 * dy < 0 and (dy, dx) not in _SZ2_CUR]


def _win_offsets(r):
    """Offsets in an earlier band within radius r, beyond its 3 x 3 window."""
    return [(dy, dx) for dy in range(-r, r + 1) for dx in range(-r, r + 1) if max(abs(dy), abs(dx)) > 1]


def n_context(context, n_prev=N_PREV):
    return 0 if context < 2 else len(_cur_offsets(context)) + n_prev * len(_win_offsets(context))


class ResMLP(torch.nn.Module):
    """Wider and deeper predictor: input layer, `blocks` pre-norm residual blocks, zero-initialised head."""

    def __init__(self, n_in, hidden, blocks):
        super().__init__()
        self.inp = torch.nn.Linear(n_in, hidden)
        self.blocks = torch.nn.ModuleList(torch.nn.Sequential(
            torch.nn.LayerNorm(hidden), torch.nn.Linear(hidden, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden)) for _ in range(blocks))
        self.out = torch.nn.Linear(hidden, 2)
        torch.nn.init.zeros_(self.out.weight)  # start as the plain MED predictor
        torch.nn.init.zeros_(self.out.bias)

    def forward(self, x):
        x = self.inp(x)
        for block in self.blocks:
            x = x + block(x)
        return self.out(torch.nn.functional.silu(x))


def make_model(hidden: int = 64, *, version=2, n_prev=N_PREV, blocks=0, context=0) -> torch.nn.Module:
    """blocks=0: the SZ1/SZ2 network (two hidden layers). blocks>0: ResMLP."""
    torch.manual_seed(0)
    n_in = n_inputs(version, n_prev) + n_context(context, n_prev)
    if blocks:
        return ResMLP(n_in, hidden, blocks)
    m = torch.nn.Sequential(
        torch.nn.Linear(n_in, hidden), torch.nn.SiLU(),
        torch.nn.Linear(hidden, hidden), torch.nn.SiLU(),
        torch.nn.Linear(hidden, 2),
    )
    torch.nn.init.zeros_(m[-1].weight)  # start as the plain MED (LOCO-I) predictor
    torch.nn.init.zeros_(m[-1].bias)
    return m


def wavefronts(h: int, w: int):
    for t in range(w + 2 * h - 2):
        y = np.arange(max(0, (t - w + 2) // 2), min(h - 1, t // 2) + 1)
        yield y, t - 2 * y


def _ctx(P, p, yy, xx):
    """MED (LOCO-I / JPEG-LS) prediction and local gradient activity of band p from its
    causal neighbours."""
    g = lambda dy, dx: P[p][yy + dy, xx + dx]
    w, n, nw, ne, ww, nn = g(0, -1), g(-1, 0), g(-1, -1), g(-1, 1), g(0, -2), g(-2, 0)
    lo, hi = np.minimum(w, n), np.maximum(w, n)
    ref = np.where(nw >= hi, lo, np.where(nw <= lo, hi, w + n - nw))
    act = (abs(w - nw) + abs(n - nw) + abs(n - ne) + abs(w - ww) + abs(n - nn)) / 5 + 1
    return ref, act


def _features(P, norm, b, y, x, version, n_prev=N_PREV, context=0):
    """Inputs scaled by local activity, so the net sees the same numbers in flat and in busy areas."""
    yy, xx = y + PAD, x + PAD
    ref, act = _ctx(P, b, yy, xx)
    mu, sig = norm[b]
    cols = [(P[b][yy + dy, xx + dx] - ref) / act for dy, dx in CUR] + [np.log(act), (ref - mu) / sig]
    for k in range(1, n_prev + 1):
        if b - k >= 0:
            ref_p, act_p = _ctx(P, b - k, yy, xx)
            cols += [(P[b - k][yy + dy, xx + dx] - ref_p) / act_p for dy, dx in WIN] + [np.log(act_p), np.ones(len(y))]
        else:
            cols += [np.zeros(len(y))] * (len(WIN) + 2)
    if version == 2:
        cols.extend((P[b][yy + dy, xx + dx] - ref) / act
                    for dy, dx in ((-1, -2), (-2, -1), (-2, 1), (-2, 2)))
        cur = np.stack([P[b][yy + dy, xx + dx] for dy, dx in CUR])
        avg = cur.mean(0)
        centered = cur - avg
        for k in range(1, n_prev + 1):
            if b < k:
                cols.extend([np.zeros(len(y))] * 5)
                continue
            p = b - k
            v = P[p][yy, xx]
            prev = np.stack([P[p][yy + dy, xx + dx] for dy, dx in CUR])
            prev_avg = prev.mean(0)
            prev -= prev_avg
            # Fit a local spectral slope on the six shared, already decoded neighbours.
            slope = (prev * centered).sum(0) / ((prev ** 2).sum(0) + len(CUR))
            level = (v - norm[p][0]) / norm[p][1]
            cols.extend([level, (level * sig + mu - ref) / act,
                         (avg + slope * (v - prev_avg) - ref) / act, slope,
                         np.abs(centered - slope * prev).mean(0) / act])
    if context >= 2:  # wider windows, same normalisation as the SZ2 inputs
        cols.extend((P[b][yy + dy, xx + dx] - ref) / act for dy, dx in _cur_offsets(context))
        win = _win_offsets(context)
        for k in range(1, n_prev + 1):
            if b - k >= 0:
                ref_p, act_p = _ctx(P, b - k, yy, xx)
                cols.extend((P[b - k][yy + dy, xx + dx] - ref_p) / act_p for dy, dx in win)
            else:
                cols.extend([np.zeros(len(y))] * len(win))
    t = lambda a: torch.from_numpy(np.asarray(a, np.float32))
    return t(np.arcsinh(np.stack(cols, 1))), t(ref), t(act)  # asinh: class edges give huge ratios


def laplace_nll(s, mean, scale):
    """-ln P(s) for a Laplace(mean, scale) quantized to integer bins (stable form)."""
    a = (s - mean).abs()
    # clamp per branch: torch.where still backprops the unused branch, an inf there gives NaN grads
    at, ac = a.clamp_min(0.5), a.clamp_max(0.5)
    tail = math.log(0.5) - (at - 0.5) / scale + torch.log(-torch.expm1(-1 / scale))
    center = 1 - 0.5 * torch.exp(-(0.5 - ac) / scale) - 0.5 * torch.exp(-(0.5 + ac) / scale)
    return -torch.where(a >= 0.5, tail, torch.log(center.clamp_min(1e-12)))


def _predict(model, f, ref, act):
    out = model(f)
    return ref + act * out[:, 0], act * torch.nn.functional.softplus(out[:, 1]) + 0.05


def _train(model, opt, batch, steps):
    """`steps` Adam steps on the pooled pixels of the last wavefronts."""
    f, ref, act, s = (torch.cat(t) for t in zip(*batch))
    for _ in range(steps):
        mean, scale = _predict(model, f, ref, act)
        opt.zero_grad()
        laplace_nll(s, mean, scale).mean().backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()


def _run(P, M, stats, model, lr, code, version, laplace_bits=None, opts=None):
    """Walk all bands and wavefronts. `code` returns the true symbols of a batch:
    the encoder reads and encodes them, the decoder decodes them. M holds the
    valid-pixel masks, known to both sides; fill pixels are never coded.

    With default options this takes one step per wavefront and reuses the prediction
    graph, exactly as SZ1/SZ2 streams expect. Other options use _run_batched."""
    if opts and opts != DEFAULTS:
        return _run_batched(P, M, stats, model, lr, code, version, opts)
    torch.set_num_threads(1)  # ponytail: fixed thread count keeps float sums bit-identical across encode/decode
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    B, H, W = M.shape
    norm = [(s[2], s[3]) for s in stats]
    for b in range(B):
        if version == 2 and (stats[b][0] == stats[b][1] or not M[b].any()):
            P[b].fill(stats[b][2])
            continue
        for y, x in wavefronts(H, W):
            k = M[b, y, x]
            if not k.any():
                continue
            y, x = y[k], x[k]
            f, ref, act = _features(P, norm, b, y, x, version)
            out = model(f)
            mean = ref + act * out[:, 0]
            scale = act * torch.nn.functional.softplus(out[:, 1]) + 0.05
            s = code(b, y, x, mean.detach().double().numpy(), scale.detach().double().numpy())
            P[b][y + PAD, x + PAD] = s
            nll = laplace_nll(torch.from_numpy(s.astype(np.float32)), mean, scale)
            opt.zero_grad()
            nll.mean().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if laplace_bits is not None:
                laplace_bits[b, y, x] = nll.detach().numpy() / math.log(2)


def _run_batched(P, M, stats, model, lr, code, version, opts):
    """Predict without a graph, then train on the pixels of `train_every` wavefronts at once.
    Fewer optimizer calls means less Python overhead per pixel. Runs the network on
    opts["device"]; features and the range coder stay on the CPU."""
    train_every, steps, n_prev, context = opts["train_every"], opts["steps"], opts["n_prev"], opts["context"]
    dev = torch.device(opts["device"])
    torch.set_num_threads(1)
    if dev.type != "cpu":
        torch.use_deterministic_algorithms(True)  # the decoder must reproduce every float
    model.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    B, H, W = M.shape
    norm = [(s[2], s[3]) for s in stats]
    for b in range(B):
        if stats[b][0] == stats[b][1] or not M[b].any():
            P[b].fill(stats[b][2])
            continue
        batch = []
        for y, x in wavefronts(H, W):
            k = M[b, y, x]
            if not k.any():
                continue
            y, x = y[k], x[k]
            f, ref, act = (t.to(dev) for t in _features(P, norm, b, y, x, version, n_prev, context))
            with torch.no_grad():
                mean, scale = _predict(model, f, ref, act)
            s = code(b, y, x, mean.cpu().double().numpy(), scale.cpu().double().numpy())
            P[b][y + PAD, x + PAD] = s
            if steps == 0:  # frozen model: decoder runs inference only
                continue
            batch.append((f, ref, act, torch.from_numpy(s.astype(np.float32)).to(dev)))
            if len(batch) == train_every:
                _train(model, opt, batch, steps)
                batch = []
        if batch:  # end of band: learn from the remainder before the next band
            _train(model, opt, batch, steps)


def _band_stats(img, M):
    stats = []
    for c, m in zip(img, M):
        v = c[m]
        stats.append((int(v.min()), int(v.max()), int(round(v.mean())), np.float32(max(v.std(), 1.0)))
                     if v.size else (0, 1, 0, np.float32(1.0)))
    return stats


def _padded(img, M, stats):
    """float64 working copy; fill pixels and the border take the band mean."""
    return [np.pad(np.where(m, c, mu).astype(np.float64), PAD, constant_values=mu)
            for c, m, (_, _, mu, _) in zip(img, M, stats)]


def _masks(img, fill):
    return np.ones(img.shape, bool) if fill is None else img != fill


def _pack_masks(M) -> bytes:
    out = b""
    for i, m in enumerate(M):
        if i and np.array_equal(m, M[i - 1]):
            out += b"\0"  # same mask as the previous band
        else:
            c = lzma.compress(np.packbits(m).tobytes(), preset=9 | lzma.PRESET_EXTREME)
            out += b"\1" + struct.pack("<I", len(c)) + c
    return out


def _unpack_masks(data, off, B, H, W):
    M = np.empty((B, H, W), bool)
    for i in range(B):
        flag, off = data[off], off + 1
        if flag == 0:
            M[i] = M[i - 1]
        else:
            (n,), off = struct.unpack_from("<I", data, off), off + 4
            M[i] = np.unpackbits(np.frombuffer(lzma.decompress(data[off:off + n]), np.uint8), count=H * W).reshape(H, W)
            off += n
    return M, off


def prior_hash(prior) -> bytes:
    if prior is None:
        return bytes(8)
    return hashlib.sha256(b"".join(v.numpy().tobytes() for v in prior.values())).digest()[:8]


HEAD = "<4if?q"  # bands, height, width, hidden, lr, has_fill, fill
BAND = "<3qf"  # min, max, mean, std


def _version(prior, version):
    if version is None:
        return 1 if prior is not None and prior["0.weight"].shape[1] == N_IN else 2
    if version not in (1, 2):
        raise ValueError("supported satzip versions are 1 and 2")
    return version


def _initial_model(hidden, prior, version, opts=None):
    o = {**DEFAULTS, **(opts or {})}
    model = make_model(hidden, version=version, n_prev=o["n_prev"], blocks=o["blocks"], context=o["context"])
    if prior is not None:
        model.load_state_dict(prior)
    return model


def _options(version, **given):
    """Validated encoder options (all keys of DEFAULTS)."""
    unknown = set(given) - set(DEFAULTS)
    if unknown:
        raise TypeError(f"unknown options: {sorted(unknown)}")
    o = {**DEFAULTS, **given}
    o["device"] = torch.device(o["device"]).type  # "cuda:1" -> "cuda": the stream records the kind only
    if o != DEFAULTS and version != 2:
        raise ValueError("encoder options need SZ2 or later")
    if not (1 <= o["train_every"] and 0 <= o["steps"] and 0 <= o["n_prev"] and 0 <= o["blocks"]):
        raise ValueError("train_every >= 1; steps, n_prev, blocks >= 0 (steps=0: no online training)")
    if o["context"] not in (0, *range(2, PAD + 1)):
        raise ValueError(f"context must be 0 or 2..{PAD}")
    return o


def _entropy_models(stats, version):
    if version == 1:
        return [constriction.stream.model.QuantizedLaplace(lo, max(hi, lo + 1)) for lo, hi, _, _ in stats]
    return [ResidualModel(lo, hi) for lo, hi, _, _ in stats]


def compress(img: np.ndarray, prior=None, hidden: int = 64, lr: float = 3e-3,
             fill: int | None = None, *, version: int | None = None, quadrature=None, **options):
    """Return the complete stream and estimated symbol costs for an integer (bands, H, W) image.

    The default is SZ2. Supply version=1 to reproduce SZ1, or pass an SZ1 prior
    to select its original codec automatically. For a packed uncertainty product,
    quadrature=(total_index, component_indices) stores a reversible prediction recipe.

    options (stored in the stream, format SZ3 / SQ3 when any differs from DEFAULTS):
    train_every, steps, n_prev, blocks (ResMLP depth, 0 = SZ2 network),
    context (wider window radius, 0 or 2..8), device ("cpu", "cuda", ...).
    """
    assert img.ndim == 3 and img.dtype.kind in "ui", "expected (bands, H, W) integer array"
    version = _version(prior, version)
    opts = _options(version, **options)
    if quadrature is not None:
        if version != 2:
            raise ValueError("quadrature prediction requires SZ2")
        img = uncertainty.transform(img, quadrature, fill)
    M = _masks(img, fill)
    bands, tables = palette.prepare(img, M) if version == 2 else (img, b"")
    stats = _band_stats(bands, M)
    lr = struct.unpack("<f", struct.pack("<f", lr))[0]
    model = _initial_model(hidden, prior, version, opts)
    enc = constriction.stream.queue.RangeEncoder()
    families = _entropy_models(stats, version)
    bits = np.zeros(img.shape, np.float32)

    def code(b, y, x, mean, scale):
        s = bands[b][y, x].astype(np.int32)
        if version == 1:
            enc.encode(s, families[b], mean, scale)
        else:
            bits[b, y, x] = families[b].encode(enc, s, mean, scale)
        return s

    _run(_padded(bands, M, stats), M, stats, model, lr, code, version, bits if version == 1 else None, opts)
    magic = MAGIC if version == 2 else LEGACY_MAGIC
    if quadrature is not None:
        magic = QUADRATURE_MAGIC
        tables = uncertainty.pack(quadrature) + tables
    extra = b""
    if opts != DEFAULTS:  # SZ3 / SQ3: SZ2 layout plus the non-default options as JSON
        blob = json.dumps({k: v for k, v in opts.items() if v != DEFAULTS[k]}, sort_keys=True).encode()
        magic, extra = magic[:2] + b"3", struct.pack("<H", len(blob)) + blob
    stored_fill = int(fill or 0)
    if version == 2 and img.dtype.kind == "u" and stored_fill >= 1 << 63:
        stored_fill -= 1 << 64
    head = magic + img.dtype.str.encode() + struct.pack(HEAD, *img.shape, hidden, lr, fill is not None, stored_fill)
    head += extra + b"".join(struct.pack(BAND, *s) for s in stats) + prior_hash(prior) + tables
    if fill is not None:
        head += _pack_masks(M)
    return head + enc.get_compressed().tobytes(), bits


def stream_options(data: bytes) -> dict:
    """Encoder options recorded in a stream (DEFAULTS for SZ1 / SZ2)."""
    if data[2:3] != b"3":
        return dict(DEFAULTS)
    off = 6 + struct.calcsize(HEAD)
    (n,) = struct.unpack_from("<H", data, off)
    return {**DEFAULTS, **json.loads(data[off + 2:off + 2 + n])}


def decompress(data: bytes, prior=None, device: str | None = None) -> np.ndarray:
    """Decode a stream. `device` overrides the recorded device kind (the floats must still match)."""
    magic = data[:3]
    if magic not in (MAGIC, LEGACY_MAGIC, QUADRATURE_MAGIC, b"SZ3", b"SQ3"):
        raise ValueError("not a satzip stream")
    version = 1 if magic == LEGACY_MAGIC else 2
    dtype = np.dtype(data[3:6].decode())
    B, H, W, hidden, lr, has_fill, fill = struct.unpack_from(HEAD, data, 6)
    off = 6 + struct.calcsize(HEAD)
    opts = stream_options(data)
    if magic[2:] == b"3":
        off += 2 + struct.unpack_from("<H", data, off)[0]
    if device is not None:
        opts["device"] = device
    stats = [struct.unpack_from(BAND, data, off + i * struct.calcsize(BAND)) for i in range(B)]
    stats = [(lo, hi, m, np.float32(s)) for lo, hi, m, s in stats]
    off += struct.calcsize(BAND) * B
    if data[off:off + 8] != prior_hash(prior):
        raise ValueError("prior does not match the one used to compress")
    off += 8
    quadrature, off = uncertainty.unpack(data, off) if magic[:2] == b"SQ" else (None, off)
    tables, off = palette.read(data, off, dtype, B) if version == 2 else ([None] * B, off)
    M, off = _unpack_masks(data, off, B, H, W) if has_fill else (np.ones((B, H, W), bool), off)
    model = _initial_model(hidden, prior, version, opts)
    dec = constriction.stream.queue.RangeDecoder(np.frombuffer(data[off:], np.uint32).copy())
    families = _entropy_models(stats, version)
    P = _padded(np.zeros((B, H, W), np.int32), M, stats)

    def code(b, y, x, mean, scale):
        if version == 1:
            return dec.decode(families[b], mean, scale)
        return families[b].decode(dec, mean, scale)

    _run(P, M, stats, model, lr, code, version, opts=opts)
    out = np.empty((B, H, W), dtype)
    for band, padded, table in zip(out, P, tables):
        values = padded[PAD:-PAD, PAD:-PAD]
        band[:] = values if table is None else table[values.astype(np.intp)]
    if has_fill and not M.all():
        if version == 2 and dtype.kind == "u" and fill < 0:
            fill += 1 << 64
        out[~M] = fill
    if quadrature is not None:
        return uncertainty.transform(out, quadrature, fill if has_fill else None, inverse=True)
    return out


def pretrain(tiles, hidden: int = 64, lr: float = 3e-3, epochs: int = 1,
             fill: int | None = None, *, version: int = 2, quadrature=None, init=None, **options):
    """Train initial weights with the same predictor, rank transform and options as the codec.
    Returns CPU tensors (a prior for compress / decompress)."""
    version = _version(None, version)
    if quadrature is not None and version != 2:
        raise ValueError("quadrature prediction requires SZ2")
    opts = _options(version, **options)
    model = _initial_model(hidden, init, version, opts)
    for _ in range(epochs):
        for img in tiles:
            if quadrature is not None:
                img = uncertainty.transform(img, quadrature, fill)
            M = _masks(img, fill)
            bands = palette.prepare(img, M)[0] if version == 2 else img
            stats = _band_stats(bands, M)
            _run(_padded(bands, M, stats), M, stats, model, lr,
                 lambda b, y, x, *_: bands[b][y, x].astype(np.int32), version, opts=opts)
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def prior_nbytes(prior) -> int:
    return sum(v.numel() * 4 for v in prior.values()) if prior is not None else 0


# ---------------------------------------------------------------- baselines

def _shuffle(b):  # HDF5 / NetCDF byte shuffle filter
    return np.ascontiguousarray(b).view(np.uint8).reshape(-1, b.itemsize).T.tobytes()


def _unshuffle(r, like):
    return np.frombuffer(r, np.uint8).reshape(like.itemsize, -1).T.copy().view(like.dtype).reshape(like.shape)


def _u16(b):
    """Map a band onto uint16 for image codecs (offset by its min, 8 bytes to store). None if it does not fit."""
    lo = b.min()
    if int(b.max()) - int(lo) > 65535:
        return None
    return np.subtract(b, lo, dtype=np.uint64, casting="unsafe").astype(np.uint16)


def _codecs():
    import imagecodecs as ic

    pre = ic.AEC.FLAG.DATA_PREPROCESS
    byte = [  # (name, encode, decode) on the raw band
        ("shuffle+zlib-9", lambda b: zlib.compress(_shuffle(b), 9), lambda c, b: _unshuffle(zlib.decompress(c), b)),
        ("shuffle+zstd-19", lambda b: ic.zstd_encode(_shuffle(b), level=19), lambda c, b: _unshuffle(ic.zstd_decode(c), b)),
        ("lzma-9e", lambda b: lzma.compress(b.tobytes(), preset=9 | lzma.PRESET_EXTREME),
         lambda c, b: np.frombuffer(lzma.decompress(c), b.dtype).reshape(b.shape)),
    ]
    image = [  # on the uint16-mapped band
        ("CCSDS-121 (libaec)", lambda u: ic.aec_encode(u, bitspersample=16, flags=pre),
         lambda c, u: np.frombuffer(ic.aec_decode(c, bitspersample=16, flags=pre), np.uint16,
                                    count=u.size).reshape(u.shape)),
        ("PNG", lambda u: ic.png_encode(u, level=9), lambda c, u: ic.png_decode(c)),
        ("JPEG 2000", lambda u: ic.jpeg2k_encode(u, level=0, reversible=True), lambda c, u: ic.jpeg2k_decode(c)),
        ("JPEG XL", lambda u: ic.jpegxl_encode(u, lossless=True, effort=7), lambda c, u: ic.jpegxl_decode(c)),
    ]
    return byte, image


def baselines(img: np.ndarray) -> dict[str, list[int | None]]:
    """Per-band compressed size in bytes for standard lossless codecs, each round-trip checked.
    None = codec cannot hold this band (value range over 16 bits)."""
    byte, image = _codecs()
    out = {"raw": [b.nbytes for b in img]}
    for name, enc, dec in byte:
        out[name] = []
        for b in img:
            c = enc(b)
            assert np.array_equal(dec(c, b), b), name
            out[name].append(len(c))
    for name, enc, dec in image:
        out[name] = []
        for b in img:
            u = _u16(b)
            if u is None:
                out[name].append(None)
                continue
            c = enc(u)
            assert np.array_equal(dec(c, u), u), name
            out[name].append(len(c) + 8)
    return out


def jpegxl_layouts(img: np.ndarray, effort: int = 9, *, stream_cache=None) -> dict[str, int]:
    """Round-trip JPEG XL in several layouts, charging 8 bytes per band for its offset."""
    import imagecodecs as ic

    mapped = [_u16(b) for b in img]
    if any(b is None for b in mapped):
        raise ValueError("JPEG XL comparison requires each band to span at most 16 bits")
    stack = np.stack(mapped)
    offsets = img.min(axis=(1, 2))[:, None, None]
    layouts = {"independent": 1}
    if len(stack) > 1:
        layouts["planar"] = len(stack)
    if len(stack) >= 3:
        layouts["RGB groups"] = 3
    if len(stack) == 4:
        layouts["RGBA"] = 4
    sizes = {}
    for layout, group in layouts.items():
        size = 8 * len(stack)
        for i in range(0, len(stack), group):
            bands = stack[i:i + group]
            planar = layout == "planar"
            options = {}
            if planar:
                pixels = bands
                options = {"planar": True, "photometric": "GRAY"}
            elif len(bands) == 1:
                pixels = bands[0]
            else:
                pixels = np.moveaxis(bands, 0, -1)
            pixels = np.ascontiguousarray(pixels)
            if stream_cache is None:
                stream = ic.jpegxl_encode(pixels, lossless=True, effort=effort, **options)
            else:
                # All pixels here are uint16. Reuse identical encoder inputs across
                # fill bands and prefilter variants. Every stream is still decoded below.
                key = (pixels.shape, planar, effort, hashlib.sha256(pixels).digest())
                stream = stream_cache.get(key)
                if stream is None:
                    stream = ic.jpegxl_encode(pixels, lossless=True, effort=effort, **options)
                    stream_cache[key] = stream
                    if len(stream_cache) > 64:
                        stream_cache.popitem(last=False)
                else:
                    stream_cache.move_to_end(key)
            decoded = ic.jpegxl_decode(stream)
            if not planar:
                decoded = decoded[None] if len(bands) == 1 else np.moveaxis(decoded, -1, 0)
            restored = decoded.astype(img.dtype) + offsets[i:i + group]
            if not np.array_equal(restored, img[i:i + group]):
                raise ValueError(f"JPEG XL round trip failed for {layout}")
            size += len(stream)
        sizes[layout] = size
    return sizes


# ---------------------------------------------------------------- NetCDF / HDF5 chunks

_benchmark_jxl_streams = OrderedDict()  # bounded, process-local, sizes only benchmark

def _nc_split(name):  # "emis_1" -> ("emis", 1)
    return (name[:-2], int(name[-1])) if len(name) > 2 and name[-2] == "_" and name[-1].isdigit() else (name, None)


def _stored_bytes(ds, prefix, y0, x0, height, width):
    """Allocate each physical chunk's bytes by the fraction of its grid cells in this tile."""
    if ds.chunks is None:
        return ds.id.get_storage_size() * height * width / ds.size
    origin = tuple(p // c * c for p, c in zip(prefix, ds.chunks))
    planes = math.prod(min(c, n - p) for c, n, p in zip(ds.chunks, ds.shape, origin))
    ch, cw = ds.chunks[-2:]
    total = 0
    for cy in range(y0 // ch * ch, y0 + height, ch):
        h = min(ch, ds.shape[-2] - cy)
        rows = min(cy + h, y0 + height) - max(cy, y0)
        for cx in range(x0 // cw * cw, x0 + width, cw):
            w = min(cw, ds.shape[-1] - cx)
            cols = min(cx + w, x0 + width) - max(cx, x0)
            size = ds.id.get_chunk_info_by_coord(origin + (cy, cx)).size
            total += size * rows * cols / (planes * h * w)
    return total


def nc_chunk(path, names, i, j, cs=1000):
    """Chunk (i, j) of 2-D grid variables as a (bands, cs, cs) stack, plus the bytes each one
    takes in the file as stored (a chunk shared by two bands is split in half)."""
    import h5py

    bands, stored = [], []
    with h5py.File(path) as f:
        for nm in names:
            var, k = _nc_split(nm)
            ds = f[var]
            yx = (slice(i * cs, (i + 1) * cs), slice(j * cs, (j + 1) * cs))
            prefix = (0, k) if k is not None else (0,)
            band = ds[prefix + yx]
            bands.append(band)
            stored.append(_stored_bytes(ds, prefix, i * cs, j * cs, *band.shape))
    return np.stack(bands), stored


def nc_chunk_bench(path, names, i, j, fill=-32768, cs=1000, prior=None, full=False):
    """All codecs on one chunk. satzip is decoded and checked too."""
    import time

    img, stored = nc_chunk(path, names, i, j, cs)
    t = time.perf_counter()
    recipe = uncertainty.lst_recipe(names) if _version(prior, None) == 2 else None
    data, bits = compress(img, fill=fill, prior=prior, quadrature=recipe)
    t_enc = time.perf_counter() - t
    t = time.perf_counter()
    restored = decompress(data, prior=prior)
    if restored.dtype != img.dtype or not np.array_equal(restored, img):
        raise ValueError(f"satzip round trip failed on chunk {i},{j}")
    t_dec = time.perf_counter() - t
    row = {"i": i, "j": j, "valid": [int(v) for v in (img != fill).sum((1, 2))],
            "NetCDF file": stored, "satzip": len(data), "satzip bits": [float(v) for v in bits.sum((1, 2))],
            "t_enc": t_enc, "t_dec": t_dec}
    del restored, bits, data
    row.update(baselines(img))
    if full:
        versions = [(1, "SZ1")]
        if recipe:
            versions.append((2, "SZ2 plain"))
        for version, label in versions:
            stream, _bits = compress(img, fill=fill, version=version)
            decoded = decompress(stream)
            if decoded.dtype != img.dtype or not np.array_equal(decoded, img):
                raise ValueError(f"{label} round trip failed on chunk {i},{j}")
            row[label] = len(stream)
            del decoded, _bits, stream
        row.setdefault("SZ2 plain", row["satzip"])
        row["JPEG XL layouts"] = jpegxl_layouts(img, stream_cache=_benchmark_jxl_streams)
        if recipe:
            transformed = uncertainty.transform(img, recipe, fill)
            if not np.array_equal(uncertainty.transform(transformed, recipe, fill, inverse=True), img):
                raise ValueError(f"uncertainty transform round trip failed on chunk {i},{j}")
            overhead = len(uncertainty.pack(recipe)) + 9
            row["JPEG XL transformed layouts"] = {
                k: v + overhead for k, v in jpegxl_layouts(transformed, stream_cache=_benchmark_jxl_streams).items()
            }
    return row


def nc_cache_path(path, names, cs=1000, *, full=False):
    """Separate benchmark results for different products, band selections, orders and tile sizes."""
    stat = Path(path).stat()
    key = json.dumps([list(names), cs, stat.st_size, stat.st_mtime_ns]).encode()
    suite = "full-v1" if full else "sz2"
    return Path(f"{path}.{hashlib.sha256(key).hexdigest()[:12]}.{suite}.bench.json")


def nc_benchmark(path, names, cache=None, workers=None, cs=1000, *, full=False, progress=None):
    """Run nc_chunk_bench on every chunk in parallel; results cached as JSON."""
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from functools import partial
    from tempfile import NamedTemporaryFile

    import h5py

    cache = Path(cache) if cache is not None else nc_cache_path(path, names, cs, full=full)
    if cache.exists():
        return json.loads(cache.read_text())
    with h5py.File(path) as f:
        H, W = f[_nc_split(names[0])[0]].shape[-2:]
    jobs = [(i, j) for i in range((H + cs - 1) // cs) for j in range((W + cs - 1) // cs)]
    parts = cache.with_name(cache.name + ".chunks")
    parts.mkdir(exist_ok=True)

    def save(destination, value):
        with NamedTemporaryFile("w", dir=destination.parent, delete=False) as tmp:
            json.dump(value, tmp)
        os.replace(tmp.name, destination)

    rows = {}
    missing = []
    for i, j in jobs:
        part = parts / f"{i}-{j}.json"
        if part.exists():
            rows[i, j] = json.loads(part.read_text())
        else:
            missing.append((i, j))
    if missing:
        with ProcessPoolExecutor(workers or os.cpu_count()) as ex:
            bench = partial(nc_chunk_bench, path, names, cs=cs, full=full)
            pending = [ex.submit(bench, i, j) for i, j in missing]
            for future in as_completed(pending):
                row = future.result()
                i, j = row["i"], row["j"]
                save(parts / f"{i}-{j}.json", row)
                rows[i, j] = row
                if progress:
                    progress(len(rows), len(jobs))
    res = [rows[key] for key in jobs]
    save(cache, res)
    return res


# ---------------------------------------------------------------- data

PC = "https://planetarycomputer.microsoft.com/api"


def _http_json(url, body=None):
    req = urllib.request.Request(url, body and json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=120))


def fetch(collection: str, lon: float, lat: float, assets, size: int = 512,
          dates: str = "2024-06-01T00:00:00Z/2024-09-30T23:59:59Z", query: dict | None = None,
          cache: str | Path = "data"):
    """(bands, size, size) window of a Copernicus product from Microsoft Planetary Computer
    (anonymous access). Returns (img, fill). Cached as .npz.

    COG assets (Sentinel-1 GRD, Sentinel-2 L2A): window at the point, or at the image
    centre when the raster has no CRS (Sentinel-1 radar geometry).
    NetCDF assets (Sentinel-3): each file is downloaded once, the centre window is read."""
    cache = Path(cache)
    path = cache / f"{collection}_{lon:.3f}_{lat:.3f}_{size}_{dates[:10]}_{'-'.join(assets)}.npz"
    if path.exists():
        z = np.load(path)
        return z["img"], (None if z["fill"].size == 0 else int(z["fill"]))
    import h5py
    import rasterio
    from rasterio.warp import transform
    from rasterio.windows import Window

    q = {"collections": [collection], "intersects": {"type": "Point", "coordinates": [lon, lat]},
         "datetime": dates, "limit": 1, **({"query": query} if query else {})}
    item = _http_json(f"{PC}/stac/v1/search", q)["features"][0]
    token = _http_json(f"{PC}/sas/v1/token/{collection}")["token"]
    bands, fill = [], None
    for name in assets:
        href = item["assets"][name]["href"]
        if href.endswith(".nc"):
            local = cache / item["id"] / href.rsplit("/", 1)[1]
            if not local.exists():
                local.parent.mkdir(parents=True, exist_ok=True)
                urllib.request.urlretrieve(f"{href}?{token}", local)
            with h5py.File(local) as f:
                ds = max((d for d in f.values() if isinstance(d, h5py.Dataset) and d.ndim == 2), key=lambda d: d.size)
                r, c = ds.shape[0] // 2, ds.shape[1] // 2
                bands.append(ds[r - size // 2:r + size // 2, c - size // 2:c + size // 2])
                fill = int(ds.attrs["_FillValue"][0]) if "_FillValue" in ds.attrs else fill
        else:
            with rasterio.open(f"{href}?{token}") as src:
                if src.crs:
                    xs, ys = transform("EPSG:4326", src.crs, [lon], [lat])
                    r, c = src.index(xs[0], ys[0])
                else:
                    r, c = src.height // 2, src.width // 2
                bands.append(src.read(1, window=Window(c - size // 2, r - size // 2, size, size)))
    img = np.stack(bands)
    cache.mkdir(parents=True, exist_ok=True)
    np.savez(path, img=img, fill=np.array([] if fill is None else fill))
    print(f"fetched {item['id']}")
    return img, fill
