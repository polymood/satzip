"""Adaptive coding of integer errors around the neural prediction."""

import constriction
import numpy as np


def _token(value):
    # Keep the top three bits and code the remaining low bits separately.
    shift = np.maximum(np.frexp(value)[1] - 3, 0)
    return (value >> shift) + 4 * shift, shift


class ResidualModel:
    def __init__(self, lo, hi):
        self.lo, self.hi = lo, hi
        n_tokens = int(_token(hi - lo)[0]) + 1
        self.counts = np.full((64, max(n_tokens, 2)), 0.1)
        self.categorical = constriction.stream.model.Categorical(perfect=False)
        self.uniform = constriction.stream.model.Uniform()

    def _context(self, mean, scale):
        center = np.clip(np.rint(mean), self.lo, self.hi).astype(np.int64)
        context = np.clip(np.rint(3 * np.log2(scale)) + 8, 0, 63).astype(np.intp)
        return center, np.minimum(center - self.lo, self.hi - center), context

    def encode(self, enc, values, mean, scale):
        center, radius, context = self._context(mean, scale)
        delta = values - center
        magnitude = np.abs(delta)
        # Alternate signs until the nearer band boundary, then use consecutive codes.
        folded = np.where(magnitude <= radius, 2 * magnitude - (delta < 0), magnitude + radius)
        token, shift = _token(folded)
        probabilities = self.counts[context]
        enc.encode(token.astype(np.int32), self.categorical, probabilities)
        low = shift > 0
        sizes = 1 << shift[low]
        enc.encode((folded[low] % sizes).astype(np.int32), self.uniform, sizes)
        bits = np.log2(probabilities.sum(axis=1) / self.counts[context, token]) + shift
        np.add.at(self.counts, (context, token), 1)
        return bits

    def decode(self, dec, mean, scale):
        center, radius, context = self._context(mean, scale)
        token = dec.decode(self.categorical, self.counts[context])
        shift = np.maximum(token // 4 - 1, 0)
        folded = (token.astype(np.int64) - 4 * shift) << shift
        low = shift > 0
        folded[low] += dec.decode(self.uniform, 1 << shift[low])
        np.add.at(self.counts, (context, token), 1)
        delta = np.where(folded % 2, -(folded + 1) // 2, folded // 2)
        tail_sign = np.where(center - self.lo < self.hi - center, 1, -1)
        return center + np.where(folded <= 2 * radius, delta, tail_sign * (folded - radius))
