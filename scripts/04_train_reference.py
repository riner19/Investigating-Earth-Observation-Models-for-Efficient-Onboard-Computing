"""Train the reference change detector on UNCOMPRESSED data, then freeze it.

This produces the ceiling every later experiment is measured against. The
resulting checkpoint is never retrained: weeks 3-6 all load this exact file.

Checkpoint selection uses a validation split carved from the TRAINING folds,
split by scene so no scene straddles the boundary. The test fold is evaluated
exactly once, at the end, on the selected checkpoint. Selecting on the test
fold would make the reported ceiling best-of-N on noisy data, and an inflated
ceiling inflates every later delta.

Usage:
    python3 scripts/04_train_reference.py                      # fold 0
    python3 scripts/04_train_reference.py --pos-weight 1.0     # balance test
    for f in 0 1 2 3 4; do python3 scripts/04_train_reference.py --fold $f; done
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from satcd.codec import band_stats, standardize
from satcd.data import TILE, TILE_ORIGINS, CaBuAr, collate
from satcd.detector import (BCEDiceLoss, build_detector, confusion,
                            load_detector, metrics, save_checkpoint)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def burn_flags(ds: CaBuAr) -> list[bool]:
    """Whether each indexed tile contains any burned pixel.

    Reads only the mask region via h5py rather than whole cubes.
    """
    import h5py

    flags = []
    with h5py.File(ds.path, "r") as f:
        for key, _fold, tile in ds._index:
            top, left = TILE_ORIGINS[tile]
            flags.append(bool(f[key]["mask"][top:top + TILE, left:left + TILE, 0].any()))
    return flags


def scene_split(ds: CaBuAr, val_frac: float, seed: int) -> tuple[list[int], list[int]]:
    """Split dataset indices into train and validation BY SCENE.

    Splitting by tile would put tiles of the same scene on both sides, which
    leaks: adjacent tiles share fire, terrain and acquisition conditions.
    """
    scenes = sorted({key for key, _, _ in ds._index})
    random.Random(seed).shuffle(scenes)
    n_val = max(1, round(len(scenes) * val_frac))
    val_scenes = set(scenes[:n_val])

    train_idx = [i for i, (k, _, _) in enumerate(ds._index) if k not in val_scenes]
    val_idx = [i for i, (k, _, _) in enumerate(ds._index) if k in val_scenes]
    return train_idx, val_idx


def make_sampler(flags: list[bool], indices: list[int]) -> WeightedRandomSampler:
    """Oversample tiles containing burn so about half of each batch is positive.

    Around 70% of CaBuAr tiles contain no burned pixel. Left unbalanced, a
    quarter of batches at size 4 would carry no positive signal at all.
    TRAINING ONLY -- validation and test keep their natural distribution,
    otherwise the numbers are neither meaningful as a ceiling nor comparable
    to published results.
    """
    sub = [flags[i] for i in indices]
    n_pos = sum(sub)
    n_neg = len(sub) - n_pos
    w_pos = 0.5 / max(n_pos, 1)
    w_neg = 0.5 / max(n_neg, 1)
    weights = [w_pos if f else w_neg for f in sub]
    return WeightedRandomSampler(weights, num_samples=len(sub), replacement=True)


@torch.no_grad()
def evaluate(model, loader, mean, std) -> dict[str, float]:
    model.eval()
    tp = fp = fn = tn = 0
    for batch in loader:
        x = standardize(batch["image"].to(DEVICE), mean[None], std[None])
        y = batch["mask"].to(DEVICE)
        a, b, c, d = confusion(model(x), y)
        tp, fp, fn, tn = tp + a, fp + b, fn + c, tn + d
    return metrics(tp, fp, fn, tn)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--pos-weight", type=float, default=3.0)
    ap.add_argument("--val-frac", type=float, default=0.12)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    tag = f"_{args.tag}" if args.tag else ""
    print(f"device {DEVICE}  fold {args.fold}  seed {args.seed}  "
          f"pos_weight {args.pos_weight}")

    full_train = CaBuAr(fold=args.fold, split="train")
    test_ds = CaBuAr(fold=args.fold, split="test", quiet=True)

    flags = burn_flags(full_train)
    train_idx, val_idx = scene_split(full_train, args.val_frac, args.seed)

    print(f"train {len(train_idx)} tiles   val {len(val_idx)} tiles   "
          f"test {len(test_ds)} tiles")
    print(f"  train tiles with burn: {sum(flags[i] for i in train_idx)} "
          f"/ {len(train_idx)}")
    print(f"  val   tiles with burn: {sum(flags[i] for i in val_idx)} "
          f"/ {len(val_idx)}")

    train_loader = DataLoader(
        Subset(full_train, train_idx), batch_size=args.batch_size,
        sampler=make_sampler(flags, train_idx), collate_fn=collate,
        num_workers=args.workers, drop_last=True,
    )
    val_loader = DataLoader(
        Subset(full_train, val_idx), batch_size=args.batch_size,
        shuffle=False, collate_fn=collate, num_workers=args.workers,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate, num_workers=args.workers,
    )

    # Statistics are (1, C, 1, 1); detector input is (B, 2, C, H, W), so an
    # extra leading axis makes them broadcast over the date dimension too.
    mean, std = band_stats(DEVICE)

    model = build_detector().to(DEVICE)
    print(f"detector parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = BCEDiceLoss(pos_weight=args.pos_weight).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    ckpt = Path(f"results/reference_fold{args.fold}{tag}.pt")
    best_val_f1, best_epoch = 0.0, -1
    history = []

    for epoch in range(args.epochs):
        model.train()
        running, t0 = 0.0, time.time()

        for batch in train_loader:
            x = standardize(batch["image"].to(DEVICE), mean[None], std[None])
            y = batch["mask"].to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item()

        scheduler.step()
        v = evaluate(model, val_loader, mean, std)
        history.append({"epoch": epoch, "loss": running / len(train_loader),
                        **{f"val_{k}": val for k, val in v.items()}})

        flag = ""
        if v["f1"] > best_val_f1:
            best_val_f1, best_epoch = v["f1"], epoch
            save_checkpoint(
                model, ckpt,
                fold=args.fold, epoch=epoch, seed=args.seed,
                val_f1=v["f1"], pos_weight=args.pos_weight,
                lr=args.lr, batch_size=args.batch_size,
            )
            flag = "  <- saved"

        print(f"epoch {epoch:3d}  loss {running / len(train_loader):.4f}  "
              f"val P {v['precision']:.3f}  R {v['recall']:.3f}  "
              f"F1 {v['f1']:.3f}  IoU {v['iou']:.3f}  "
              f"[{time.time() - t0:.0f}s]{flag}")

    # ---- the test fold is touched exactly once, here ---------------------
    print(f"\nbest val F1 {best_val_f1:.4f} at epoch {best_epoch}")
    print("evaluating the selected checkpoint on the held-out test fold...")

    frozen = load_detector(ckpt, DEVICE)
    t = evaluate(frozen, test_loader, mean, std)

    print(f"\nREFERENCE (fold {args.fold}, uncompressed)")
    print(f"  precision {t['precision']:.4f}")
    print(f"  recall    {t['recall']:.4f}")
    print(f"  F1        {t['f1']:.4f}")
    print(f"  IoU       {t['iou']:.4f}")

    out = Path(f"results/reference_fold{args.fold}{tag}.json")
    out.write_text(json.dumps({
        "fold": args.fold,
        "test": t,
        "best_val_f1": best_val_f1,
        "best_epoch": best_epoch,
        "checkpoint": str(ckpt),
        "args": vars(args),
        "history": history,
    }, indent=2))

    print(f"\ncheckpoint {ckpt}")
    print(f"metrics    {out}")

    ratio = t["precision"] / t["recall"] if t["recall"] else 0.0
    if t["f1"] < 0.4:
        print("\nGATE FAILED: reference F1 below 0.4. Every later number is "
              "measured against this, so debug before continuing -- check "
              "label alignment, band order, and loss weighting.")
    else:
        print("\nGATE PASSED. Freeze this checkpoint; do not retrain it.")
        if ratio < 0.8:
            print(f"NOTE: precision/recall = {ratio:.2f}. The detector "
                  "over-predicts burn. A trigger-happy detector keeps firing "
                  "on degraded input and UNDERSTATES the compression cost -- "
                  "try --pos-weight 1.0 and compare.")


if __name__ == "__main__":
    main()
