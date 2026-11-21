"""Lossless codec for electrode recordings stored as 16-bit mono WAV (Neuralink compression challenge).

Pipeline, after tufazip (github.com/Tufalabs/tufazip):

1. Tokens. The ADC produces about 1023 distinct 16-bit values. Samples become ranks in a value
   table that ships with the decoder, like tufazip's vocabulary.
2. A causal network reads the recent samples and running noise statistics and outputs a softmax
   over every possible next rank. Its logits are indexed by the offset from the previous sample,
   so one distribution shape applies at every signal level.
3. A range coder spends -log2 p bits per sample.
4. The network is trained offline on other recordings (tufazip's prior) and ships with the decoder.

Bit-exact decoding. Every network call has the shape (LANES, features) and rows never interact.
The encoder fills the rows with consecutive positions of one file, the decoder with LANES files at
the same position. Features come from integer sums and exact lookups, so both sides see identical
floats. Coding runs on the CPU with one thread.
"""

import lzma
import struct

import constriction
import numpy as np
import torch
from torch import nn

LANES = 64
MAGIC, RAW_MAGIC = b"NZ1", b"NZ0"
CATEGORICAL = constriction.stream.model.Categorical(perfect=False)


def read_wav(data: bytes):
    """(header bytes, int16 samples, trailing bytes) for 16-bit mono PCM WAV, else None."""
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return None
    pos, fmt = 12, None
    try:
        while pos + 8 <= len(data):
            cid, size = data[pos:pos + 4], struct.unpack_from("<I", data, pos + 4)[0]
            if cid == b"fmt ":
                fmt = struct.unpack_from("<HHIIHH", data, pos + 8)
            if cid == b"data":
                start, end = pos + 8, min(pos + 8 + size, len(data))
                if fmt is None or fmt[:2] != (1, 1) or fmt[5] != 16 or (end - start) % 2:
                    return None
                return data[:start], np.frombuffer(data[start:end], "<i2"), data[end:]
            pos += 8 + size + (size & 1)
    except struct.error:
        return None
    return None


class Predictor(nn.Module):
    """Residual MLP: recent samples relative to the last one, the last level and windowed mean
    |diff| -> logits over the offset of the next rank from the last one."""

    def __init__(self, vocab, context=32, hidden=256, depth=2, windows=(16, 256)):
        super().__init__()
        self.vocab, self.context, self.windows = vocab, context, tuple(windows)
        self.inp = nn.Linear(context + len(windows), hidden)
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
            for _ in range(depth))
        self.out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 2 * vocab - 1))
        # asinh of every possible offset, so features never call a transcendental function at code time
        self.register_buffer("rel", torch.asinh(torch.arange(1 - vocab, vocab, dtype=torch.float64) / 4).float(),
                             persistent=False)

    def features(self, hist, sums):
        """hist: (n, context) int64 ranks before the target; sums: (n, len(windows)) int64."""
        prev = hist[:, -1:]
        rel = self.rel[hist[:, :-1] - prev + self.vocab - 1]
        level = prev.float() * (2 / (self.vocab - 1)) - 1
        noise = sums.float() / torch.tensor([8.0 * w for w in self.windows], device=hist.device)
        return torch.cat([rel, level, noise], 1)

    def forward(self, hist, sums):
        """Logits over the vocab for each row."""
        h = self.inp(self.features(hist, sums))
        for block in self.blocks:
            h = h + block(h)
        logits = self.out(h)
        idx = torch.arange(self.vocab, device=hist.device) - hist[:, -1:] + self.vocab - 1
        return logits.gather(1, idx)


def _padded(ranks, context, vocab):
    """Ranks after `context` copies of the middle rank, and cumulative |diff| (csum[p] = sum of
    |seq[j] - seq[j-1]| for j < p)."""
    seq = np.concatenate([np.full(context, vocab // 2, np.int64), ranks.astype(np.int64)])
    csum = np.concatenate([[0], np.cumsum(np.abs(np.diff(seq, prepend=seq[0])))])
    return seq, csum


def _inputs(seq, csum, p, context, windows, start=0):
    """hist and sums to predict seq[p] (p: int array), never reading before `start`."""
    hist = seq[p[:, None] + np.arange(-context, 0)]
    sums = np.stack([csum[p] - csum[np.maximum(p - w, start)] for w in windows], 1)
    return torch.from_numpy(hist), torch.from_numpy(sums)


def make_model(prior):
    model = Predictor(len(prior["table"]), **prior["config"])
    model.load_state_dict(prior["state"])
    return model.eval()


def _probabilities(model, hist, sums):
    with torch.no_grad():
        return model(hist, sums).double().softmax(1).numpy()


class _OneThread:
    def __enter__(self):
        self.n = torch.get_num_threads()
        torch.set_num_threads(1)

    def __exit__(self, *exc):
        torch.set_num_threads(self.n)


def compress(wav: bytes, prior) -> bytes:
    """Stream for one WAV file. Files that are not 16-bit mono PCM, or hold values outside the
    prior's table, fall back to LZMA."""
    parsed = read_wav(wav)
    table = prior["table"]
    if parsed is not None:
        header, samples, trailer = parsed
        ranks = np.searchsorted(table, samples)
        if not np.array_equal(table[np.minimum(ranks, len(table) - 1)], samples):
            parsed = None
    if parsed is None:
        return RAW_MAGIC + lzma.compress(wav, preset=9 | lzma.PRESET_EXTREME)
    model = make_model(prior)
    k, windows = model.context, model.windows
    seq, csum = _padded(ranks, k, len(table))
    enc = constriction.stream.queue.RangeEncoder()
    positions = np.arange(k, len(seq))
    with _OneThread():
        for i in range(0, len(positions), LANES):
            p = positions[i:i + LANES]
            rows = np.concatenate([p, np.full(LANES - len(p), p[0])])  # fixed shape; extra rows ignored
            probs = _probabilities(model, *_inputs(seq, csum, rows, k, windows))[:len(p)]
            enc.encode(seq[p].astype(np.int32), CATEGORICAL, probs)
    words = enc.get_compressed()
    return (MAGIC + struct.pack("<IHI", len(ranks), len(header), len(trailer)) + header + trailer
            + words.astype("<u4").tobytes())


def decompress_many(streams, prior) -> list[bytes]:
    """Decode streams LANES at a time: one network call per sample position for a whole group."""
    out = [None] * len(streams)
    coded = []
    for i, s in enumerate(streams):
        if s[:3] == RAW_MAGIC:
            out[i] = lzma.decompress(s[3:])
        elif s[:3] == MAGIC:
            coded.append(i)
        else:
            raise ValueError("not a satzip audio stream")
    if not coded:
        return out
    model = make_model(prior)
    table, k, windows = prior["table"], model.context, model.windows
    with _OneThread():
        for g in range(0, len(coded), LANES):
            group = coded[g:g + LANES]
            heads, decoders, lengths = [], [], []
            for i in group:
                n, hl, tl = struct.unpack_from("<IHI", streams[i], 3)
                body = 13 + hl + tl
                heads.append((streams[i][13:13 + hl], streams[i][13 + hl:body]))
                decoders.append(constriction.stream.queue.RangeDecoder(
                    np.frombuffer(streams[i][body:], "<u4").astype(np.uint32)))
                lengths.append(n)
            longest = max(lengths)
            seq = np.full((LANES, k + longest), len(table) // 2, np.int64)
            csum = np.zeros((LANES, k + longest + 1), np.int64)
            lane = np.arange(LANES)
            for p in range(k, k + longest):
                hist = torch.from_numpy(seq[:, p - k:p].copy())
                sums = torch.from_numpy(np.stack([csum[:, p] - csum[:, max(p - w, 0)] for w in windows], 1))
                probs = _probabilities(model, hist, sums)
                for j, (dec, n) in enumerate(zip(decoders, lengths)):
                    if p - k < n:
                        seq[j, p] = dec.decode(CATEGORICAL, probs[j:j + 1])[0]
                csum[lane, p + 1] = csum[lane, p] + np.abs(seq[lane, p] - seq[lane, p - 1])
            for j, i in enumerate(group):
                samples = table[seq[j, k:k + lengths[j]]].astype("<i2")
                out[i] = heads[j][0] + samples.tobytes() + heads[j][1]
    return out


def decompress(stream: bytes, prior) -> bytes:
    return decompress_many([stream], prior)[0]


def value_table(wavs) -> np.ndarray:
    """Sorted distinct sample values of the 16-bit mono files among `wavs`."""
    parsed = (read_wav(w) for w in wavs)
    return np.unique(np.concatenate([p[1] for p in parsed if p is not None])).astype(np.int16)


def _stack(wavs, table, context, device):
    """All files' padded ranks and file-local cumulative |diff| in one tensor each, the positions
    to predict and the start of the file holding each one."""
    seqs, sums, positions, starts = [], [], [], []
    offset = 0
    for w in wavs:
        parsed = read_wav(w)
        if parsed is None:
            continue
        seq, csum = _padded(np.searchsorted(table, parsed[1]), context, len(table))
        seqs.append(seq)
        sums.append(csum[:-1])
        positions.append(offset + np.arange(context, len(seq)))
        starts.append(np.full(len(seq) - context, offset))
        offset += len(seq)
    return [torch.from_numpy(np.concatenate(a)).to(device) for a in (seqs, sums, positions, starts)]


def _batch(model, data, pick):
    """(hist, sums, target) for the positions data[2][pick]."""
    seq, csum, positions, starts = data
    p, start = positions[pick], starts[pick]
    hist = seq[p[:, None] + torch.arange(-model.context, 0, device=p.device)]
    sums = torch.stack([csum[p] - csum[torch.maximum(p - w, start)] for w in model.windows], 1)
    return hist, sums, seq[p]


def _evaluate(model, data, chunk=1 << 15):
    """Mean ideal code length in bits per sample over every position of `data`."""
    total, n = 0.0, len(data[2])
    model.eval()
    with torch.no_grad():
        for i in range(0, n, chunk):
            hist, sums, target = _batch(model, data, torch.arange(i, min(i + chunk, n), device=data[2].device))
            total += nn.functional.cross_entropy(model(hist, sums), target, reduction="sum").item()
    model.train()
    return total / n / np.log(2)


def train(wavs, table, *, context=32, hidden=256, depth=2, windows=(16, 256), steps=2000, batch=4096,
          lr=1e-3, device="cpu", seed=0, val=(), log=None, log_every=500):
    """Offline prior: Adam on random positions of `wavs`, cosine learning-rate decay.
    log(step, train bits, held-out bits or None) runs every `log_every` steps; held-out bits are
    measured on `val` (WAV bytes) when given."""
    torch.manual_seed(seed)
    gen = torch.Generator(device).manual_seed(seed)
    model = Predictor(len(table), context, hidden, depth, windows).to(device)
    data = _stack(wavs, table, context, device)
    held = _stack(val, table, context, device) if val else None
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for step in range(steps):
        hist, sums, target = _batch(model, data, torch.randint(len(data[2]), (batch,), device=device, generator=gen))
        loss = nn.functional.cross_entropy(model(hist, sums), target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if log and (step % log_every == 0 or step == steps - 1):
            log(step, loss.item() / np.log(2), _evaluate(model, held) if held else None)
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return {"config": dict(context=context, hidden=hidden, depth=depth, windows=tuple(windows)),
            "table": table, "state": state}


def bits_per_sample(wavs, prior, device="cpu") -> float:
    """Ideal code length under the prior, in bits per sample over all `wavs` (what compress
    spends, without stream overhead)."""
    model = make_model(prior).to(device)
    return _evaluate(model, _stack(wavs, prior["table"], model.context, device))
