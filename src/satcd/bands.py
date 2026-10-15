"""Sentinel-2 band bookkeeping and input validation.

The most dangerous failure in this pipeline is silent: feed the tokenizer the
wrong bands in the wrong order, or data at the wrong scale, and nothing
crashes -- the output is just quietly wrong. Every tensor entering or leaving
the codec goes through an assertion here.
"""

from __future__ import annotations

import torch

# CaBuAr's 12-band order, confirmed in week 1: rendering indices (3, 2, 1) as
# RGB produced plausible natural-colour imagery, which fixes B04/B03/B02 and
# therefore the whole sequence. This also matches the order of TerraMind's
# 12-element pretraining statistics.
#
# B10 (cirrus) is absent: it is consumed by atmospheric correction and does
# not appear in L2A products. A 13-band L1C source would need it dropped.
BANDS = (
    "B01", "B02", "B03", "B04", "B05", "B06",
    "B07", "B08", "B8A", "B09", "B11", "B12",
)

IDX = {name: i for i, name in enumerate(BANDS)}

# Natural colour: red, green, blue.
RGB = (IDX["B04"], IDX["B03"], IDX["B02"])

# Normalised Burn Ratio: (NIR - SWIR2) / (NIR + SWIR2).
NBR_BANDS = (IDX["B08"], IDX["B12"])

# Normalised Difference Vegetation Index: (NIR - Red) / (NIR + Red).
NDVI_BANDS = (IDX["B08"], IDX["B04"])

PATCH_SIZE = 16          # tokenizer patch size; tile sides must be a multiple
N_BANDS = len(BANDS)


class BandError(ValueError):
    """Raised when a tensor is not what the pipeline expects."""


def _check_shape(x: torch.Tensor, name: str) -> tuple[int, int, int]:
    if x.ndim < 3:
        raise BandError(f"{name}: expected at least (C, H, W), got {tuple(x.shape)}")
    c, h, w = x.shape[-3], x.shape[-2], x.shape[-1]
    if c != N_BANDS:
        raise BandError(f"{name}: expected {N_BANDS} bands, got {c}")
    if h % PATCH_SIZE or w % PATCH_SIZE:
        raise BandError(
            f"{name}: tile is {h}x{w}; the tokenizer patch size is "
            f"{PATCH_SIZE}, so both sides must be a multiple of it"
        )
    if torch.isnan(x).any():
        raise BandError(f"{name}: contains NaN")
    return c, h, w


NODATA = 0.0


def nodata_fraction(x: torch.Tensor) -> float:
    """Fraction of pixels equal to the nodata value.

    Sentinel-2 patches cut near a swath edge or scene boundary are zero-filled.
    Such tiles compress trivially and carry no signal, so they are excluded at
    index time -- see satcd.data.CaBuAr.
    """
    return float((x == NODATA).float().mean())


def assert_raw_dn(x: torch.Tensor, name: str = "tensor",
                  max_nodata: float = 0.05) -> None:
    """Validate a tensor of raw Sentinel-2 L2A digital numbers.

    Use BEFORE standardization, on anything read from disk.
    """
    _check_shape(x, name)
    lo, hi = float(x.min()), float(x.max())

    nd = nodata_fraction(x)
    if nd > max_nodata:
        raise BandError(
            f"{name}: {100 * nd:.1f}% of pixels are zero (nodata). Tiles at a "
            "swath or scene edge are zero-filled and should be excluded at "
            "index time, not reach here."
        )
    if hi <= 1.5:
        raise BandError(
            f"{name}: value range [{lo:.3f}, {hi:.3f}] looks 0-1 normalized. "
            "This pipeline expects raw L2A DN in the thousands -- check you "
            "are reading CaBuAr's raw/ files and not normalized/."
        )
    if lo < 0:
        raise BandError(f"{name}: negative reflectance at {lo:.1f}")


def assert_standardized(x: torch.Tensor, name: str = "tensor") -> None:
    """Validate a tensor about to enter the tokenizer encoder.

    Use AFTER standardization. Week 1 established that raw-DN input saturates
    the bounded decoder output in 14-21% of pixels, while standardized input
    does not saturate at all -- so values here should sit near zero.
    """
    _check_shape(x, name)
    if float(x.abs().max()) > 50:
        raise BandError(
            f"{name}: max |value| is {float(x.abs().max()):.1f}. Standardized "
            "input should be within a few units of zero -- was standardize() "
            "applied?"
        )


def summarize(x: torch.Tensor) -> str:
    """Per-band min/max/mean, for logging and debugging."""
    if x.ndim == 4:
        x = x[0]
    flat = x.reshape(x.shape[0], -1)
    rows = [
        f"  {name}: min={float(flat[i].min()):9.2f} "
        f"max={float(flat[i].max()):9.2f} mean={float(flat[i].mean()):9.2f}"
        for i, name in enumerate(BANDS)
    ]
    return "\n".join(rows)