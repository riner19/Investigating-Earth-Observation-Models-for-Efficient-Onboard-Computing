"""Verify the data layer before building anything on top of it.

Trains nothing, writes nothing. Checks that tiles load, shapes are what the
detector expects, masks are binary, and reports the class balance that sets
the loss weighting.
"""

import sys
from pathlib import Path

from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from satcd.data import CaBuAr, collate

FOLD = 0
SAMPLE_TILES = 200          # how many tiles to scan for mask statistics


def main() -> None:
    train = CaBuAr(fold=FOLD, split="train", validate=True)
    test = CaBuAr(fold=FOLD, split="test", validate=True, quiet=True)

    total = len(train) + len(test)
    print(f"\nfold {FOLD}:  train {len(train)} tiles   test {len(test)} tiles")
    print(f"total {total} tiles  ({total // 4} usable scenes)")

    # ---- fold sizes, over the usable subset only -------------------------
    print("\nper-fold tile counts (usable scenes only):")
    for k in range(5):
        n = len(CaBuAr(fold=k, split="test", quiet=True))
        print(f"  fold {k}: {n:5d} tiles  ({n // 4} scenes)")

    # ---- one sample ------------------------------------------------------
    s = train[0]
    print(f"\nsample 0: scene {s.scene}  fold {s.fold}  tile {s.tile}")
    print(f"  t1   {tuple(s.t1.shape)} {s.t1.dtype}  "
          f"DN [{float(s.t1.min()):.0f}, {float(s.t1.max()):.0f}]")
    print(f"  t2   {tuple(s.t2.shape)} {s.t2.dtype}  "
          f"DN [{float(s.t2.min()):.0f}, {float(s.t2.max()):.0f}]")
    print(f"  mask {tuple(s.mask.shape)} {s.mask.dtype}")

    # ---- batching --------------------------------------------------------
    batch = next(iter(DataLoader(train, batch_size=4, collate_fn=collate)))
    shape = tuple(batch["image"].shape)
    expected = (4, 2, 12, 224, 224)
    print(f"\nbatched image {shape}  mask {tuple(batch['mask'].shape)}")
    print(f"  expected {expected} -> {'OK' if shape == expected else 'MISMATCH'}")

    # ---- mask statistics over many tiles, not just one -------------------
    n = min(SAMPLE_TILES, len(test))
    print(f"\nscanning {n} test tiles for mask statistics...")

    values: set[int] = set()
    burned = pixels = 0
    empty_tiles = 0

    for i in range(n):
        m = test[i].mask
        values.update(m.unique().tolist())
        b = int(m.sum())
        burned += b
        pixels += m.numel()
        if b == 0:
            empty_tiles += 1

    print(f"  mask values present  : {sorted(values)}")
    print(f"  tiles with no burn   : {empty_tiles} / {n} "
          f"({100 * empty_tiles / n:.0f}%)")
    print(f"  burned pixel fraction: {burned / pixels:.5f}")

    if burned == 0:
        print("  WARNING: no burned pixels found at all -- check mask loading")
    else:
        print(f"  suggested positive class weight: {(pixels - burned) / burned:.1f}")

    if values - {0, 1}:
        print(f"  WARNING: unexpected mask values {sorted(values - {0, 1})} "
              "-- thresholding needed before use as labels")


if __name__ == "__main__":
    main()
