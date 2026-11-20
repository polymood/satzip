"""Reversible root-sum-square prediction for packed integer uncertainty grids."""

import struct

import numpy as np


def lst_recipe(names):
    """Locate the total and four component uncertainties in the ESA CCI LST product."""
    total = "lst_uncertainty"
    components = ("lst_unc_ran", "lst_unc_loc_atm", "lst_unc_loc_sfc", "lst_unc_sys")
    if total in names and all(name in names for name in components):
        return names.index(total), tuple(names.index(name) for name in components)
    return None


def pack(recipe):
    target, components = recipe
    return struct.pack(f"<II{len(components)}I", target, len(components), *components)


def unpack(data, off):
    target, count = struct.unpack_from("<II", data, off)
    components = struct.unpack_from(f"<{count}I", data, off + 8)
    return (target, components), off + 8 + 4 * count


def transform(img, recipe, fill=None, *, inverse=False):
    """Rotate the target's valid integer alphabet by the rounded root-sum-square.

    The permutation fixes the fill value. A missing component leaves that sample
    untouched, so the same operation works with the mask-aware codec and JPEG XL.
    """
    target, components = recipe
    if img.dtype.kind not in "ui" or img.dtype.itemsize > 2:
        raise ValueError("quadrature prediction requires packed 8-bit or 16-bit integers")
    if (not components or target in components or len(set(components)) != len(components)
            or any(i < 0 or i >= len(img) for i in (target, *components))):
        raise ValueError("quadrature requires distinct target and component band indices")
    bounds = np.iinfo(img.dtype)
    has_fill = fill is not None and bounds.min <= fill <= bounds.max
    valid = img[target] != fill if has_fill else np.ones(img.shape[1:], bool)
    squared = np.zeros(img.shape[1:], np.int64)
    for i in components:
        if has_fill:
            valid &= img[i] != fill
        squared += np.square(img[i], dtype=np.int64)
    shift = np.rint(np.sqrt(squared[valid])).astype(np.int64)
    values = img[target][valid].astype(np.int64)
    rank = values - bounds.min
    if has_fill:
        rank -= values > fill
    modulus = int(bounds.max) - int(bounds.min) + 1 - int(has_fill)
    rotated = (rank + shift if inverse else rank - shift) % modulus + bounds.min
    if has_fill:
        rotated += rotated >= fill
    out = img.copy()
    out[target, valid] = rotated
    return out
