"""CaBuAr dataset loading and the single shared preprocessing path.

One rule governs this module: every experimental arm consumes the output of
`prepare()` and nothing else. The reference arm takes it directly; the
compressed arm passes it through the codec first. Any difference in outcome is
then attributable to compression alone.

Tensors leaving this module are RAW L2A digital numbers. Standardization for
the tokenizer happens inside satcd.codec; normalization for the detector
happens in the training script, applied identically to every arm.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .bands import N_BANDS, assert_raw_dn

REPO = "DarthReca/california_burned_areas"
HDF5 = "raw/patched/512x512.hdf5"

SOURCE_SIZE = 512
TILE = 224
N_FOLDS = 5

# A 2x2 non-overlapping grid centred in the 512x512 source, dropping a 32 px
# border. Chosen over a centre crop (too few tiles to train on) and over an
# overlapping grid (double-counts pixels at evaluation).
_BORDER = (SOURCE_SIZE - 2 * TILE) // 2          # 32
TILE_ORIGINS = [
    (_BORDER, _BORDER),
    (_BORDER, _BORDER + TILE),
    (_BORDER + TILE, _BORDER),
    (_BORDER + TILE, _BORDER + TILE),
]
TILES_PER_SCENE = len(TILE_ORIGINS)

CACHE_VERSION = 2


@dataclass
class Sample:
    """One bitemporal tile pair."""
    t1: torch.Tensor        # (12, 224, 224) raw DN, pre-fire
    t2: torch.Tensor        # (12, 224, 224) raw DN, post-fire
    mask: torch.Tensor      # (224, 224) int64, 0 = unburned, 1 = burned
    scene: str
    fold: int
    tile: int


def prepare(array: np.ndarray) -> torch.Tensor:
    """THE preprocessing path. Every arm goes through here.

    CaBuAr stores (H, W, C) uint16; PyTorch wants (C, H, W) float. No scaling
    is applied -- the output is raw L2A digital numbers.
    """
    if array.ndim != 3 or array.shape[-1] != N_BANDS:
        raise ValueError(
            f"expected (H, W, {N_BANDS}) from CaBuAr, got {array.shape}"
        )
    return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1))).float()


def prepare_mask(array: np.ndarray) -> torch.Tensor:
    """(H, W, 1) uint16 -> (H, W) int64 with values in {0, 1}."""
    if array.ndim == 3:
        array = array[..., 0]
    return torch.from_numpy(np.ascontiguousarray(array)).long()


def crop(x: torch.Tensor, origin: tuple[int, int], size: int = TILE) -> torch.Tensor:
    """Take one tile at `origin` from a (..., H, W) tensor."""
    top, left = origin
    return x[..., top:top + size, left:left + size].contiguous()


def default_path() -> str:
    """Download (or find in cache) the CaBuAr patched archive."""
    from huggingface_hub import hf_hub_download
    return hf_hub_download(REPO, filename=HDF5, repo_type="dataset")


class CaBuAr(Dataset):
    """Sentinel-2 L2A pre/post fire pairs with CAL FIRE burned-area masks.

    The archive holds 534 scenes, but only 356 carry BOTH acquisitions. The
    other 178 have post_fire and mask only: those support burned-area
    delineation from a single image, not bitemporal change detection, and are
    skipped. Tiles that are substantially zero-filled (nodata at a swath or
    scene edge, or a band missing from the product) are skipped as well --
    they compress trivially and would flatter the codec.

    Fold assignment comes from the HDF5 group attribute, so all tiles of a
    scene stay in the same fold and there is no leakage between train and test.

    Args:
        fold: which fold is held out.
        split: "train" uses the other four folds, "test" uses `fold` alone.
    """

    REQUIRED = ("pre_fire", "post_fire", "mask")
    MAX_NODATA = 0.05        # drop a tile if either date is >5% zero-filled
    SCAN_STRIDE = 4          # spatial subsampling for the nodata scan

    def __init__(self, path: str | Path | None = None, fold: int = 0,
                 split: str = "train", validate: bool = False,
                 quiet: bool = False, cache: str | Path | None = None):
        if split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")
        self.path = str(path or default_path())
        self.fold, self.split, self.validate = fold, split, validate
        self.quiet = quiet
        self.cache = Path(cache or "results/valid_tiles.json")
        self._file: h5py.File | None = None
        self.skipped_scenes = 0
        self.skipped_tiles = 0
        self._index = self._build_index()

    # ------------------------------------------------------------------
    # nodata screening
    # ------------------------------------------------------------------

    def _scan_valid_tiles(self) -> dict[str, list[int]]:
        """Which tiles of which scenes are usable, screened for nodata.

        Checks ALL bands -- a whole band can be zero while band 0 is fine,
        which is how a tile reaches ~8% nodata (1/12) with no spatial gap at
        all. Subsamples spatially by SCAN_STRIDE to keep the scan cheap.

        Deterministic for a given archive, so the result is cached. The cache
        records the parameters it was built with and is rebuilt if they change.
        """
        if self.cache.exists():
            try:
                with open(self.cache) as fh:
                    blob = json.load(fh)
                if (blob.get("version") == CACHE_VERSION
                        and blob.get("max_nodata") == self.MAX_NODATA
                        and blob.get("tile") == TILE):
                    return blob["valid"]
            except (json.JSONDecodeError, KeyError):
                pass        # unreadable or stale -- rebuild

        if not self.quiet:
            print("[CaBuAr] scanning tiles for nodata (one-off, cached)...")

        s = self.SCAN_STRIDE
        valid: dict[str, list[int]] = {}

        with h5py.File(self.path, "r") as f:
            for key in sorted(f.keys()):
                g = f[key]
                if not all(d in g for d in self.REQUIRED):
                    continue
                keep = []
                for t, (top, left) in enumerate(TILE_ORIGINS):
                    ok = True
                    for date in ("pre_fire", "post_fire"):
                        patch = g[date][top:top + TILE:s, left:left + TILE:s, :]
                        if (patch == 0).mean() > self.MAX_NODATA:
                            ok = False
                            break
                    if ok:
                        keep.append(t)
                valid[key] = keep

        self.cache.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache, "w") as fh:
            json.dump({
                "version": CACHE_VERSION,
                "max_nodata": self.MAX_NODATA,
                "tile": TILE,
                "stride": s,
                "valid": valid,
            }, fh)
        return valid

    def _build_index(self) -> list[tuple[str, int, int]]:
        """(scene key, fold, tile index) for every usable tile in this split.

        Two exclusions, both reported:
          - scenes lacking either acquisition (bitemporal CD needs both)
          - tiles that are substantially zero-filled
        """
        valid = self._scan_valid_tiles()
        entries = []

        with h5py.File(self.path, "r") as f:
            for key in sorted(f.keys()):
                g = f[key]
                if not all(d in g for d in self.REQUIRED):
                    self.skipped_scenes += 1
                    continue
                fold = int(g.attrs["fold"])
                keep = (fold == self.fold) if self.split == "test" else (fold != self.fold)
                if not keep:
                    continue
                usable = valid.get(key, list(range(TILES_PER_SCENE)))
                self.skipped_tiles += TILES_PER_SCENE - len(usable)
                entries.extend((key, fold, t) for t in usable)

        if not self.quiet:
            if self.skipped_scenes:
                print(f"[CaBuAr] skipped {self.skipped_scenes} scenes lacking "
                      f"{' / '.join(self.REQUIRED)}")
            if self.skipped_tiles:
                print(f"[CaBuAr] skipped {self.skipped_tiles} tiles with "
                      f">{100 * self.MAX_NODATA:.0f}% nodata")
        if not entries:
            raise RuntimeError(
                f"no scenes for fold={self.fold} split={self.split!r}; "
                "check the fold attribute values in the archive"
            )
        return entries

    # ------------------------------------------------------------------

    def _handle(self) -> h5py.File:
        # Opened lazily so each DataLoader worker gets its own handle --
        # h5py file objects do not survive being forked.
        if self._file is None:
            self._file = h5py.File(self.path, "r")
        return self._file

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, i: int) -> Sample:
        key, fold, tile = self._index[i]
        g = self._handle()[key]

        origin = TILE_ORIGINS[tile]
        t1 = crop(prepare(np.asarray(g["pre_fire"])), origin)
        t2 = crop(prepare(np.asarray(g["post_fire"])), origin)
        mask = crop(prepare_mask(np.asarray(g["mask"])), origin)

        if self.validate:
            assert_raw_dn(t1, f"{key}[{tile}] pre_fire", self.MAX_NODATA)
            assert_raw_dn(t2, f"{key}[{tile}] post_fire", self.MAX_NODATA)

        return Sample(t1=t1, t2=t2, mask=mask, scene=key, fold=fold, tile=tile)

    def __del__(self):
        if self._file is not None:
            self._file.close()


def collate(batch: list[Sample]) -> dict[str, torch.Tensor | list]:
    """Stack samples into the shape FCSiamDiff expects.

    Returns `image` of shape (B, 2, 12, 224, 224) and `mask` of (B, 224, 224).
    """
    return {
        "image": torch.stack([torch.stack([s.t1, s.t2]) for s in batch]),
        "mask": torch.stack([s.mask for s in batch]),
        "scene": [s.scene for s in batch],
    }


def class_balance(ds: CaBuAr, limit: int | None = None) -> tuple[float, float]:
    """Fraction of unburned and burned pixels. Needed for loss weighting.

    Burned pixels are a small minority, so training with unweighted
    cross-entropy produces a model that predicts 'unburned' everywhere and
    scores high accuracy while being useless.
    """
    n = len(ds) if limit is None else min(limit, len(ds))
    burned = total = 0
    for i in range(n):
        m = ds[i].mask
        burned += int(m.sum())
        total += m.numel()
    return 1.0 - burned / total, burned / total
