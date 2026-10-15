"""Week 1 gate: round-trip one real CaBuAr tile and look at it.

Encodes a Sentinel-2 L2A tile to 196 tokens, decodes it back, and writes a
side-by-side RGB comparison. Both panels use a stretch computed on the
original, so the comparison is fair rather than flattering.
"""

import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from satcd.bands import RGB, assert_raw_dn, summarize
from satcd.codec import (band_stats, bits_per_tile, compression_ratio,
                         load_tokenizer, rate_bpppb, roundtrip)

REPO = "DarthReca/california_burned_areas"
HDF5 = "raw/patched/512x512.hdf5"
TILE = 224
TIMESTEPS = 10
SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def center_crop(a: torch.Tensor, size: int) -> torch.Tensor:
    h, w = a.shape[-2:]
    top, left = (h - size) // 2, (w - size) // 2
    return a[..., top:top + size, left:left + size]


def stretch(img: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip((img - lo) / (hi - lo + 1e-9), 0, 1)


def main() -> None:
    path = hf_hub_download(REPO, filename=HDF5, repo_type="dataset")

    with h5py.File(path, "r") as f:
        key = list(f.keys())[0]
        pre = np.asarray(f[key]["pre_fire"])          # (512, 512, 12) uint16
        fold = int(f[key].attrs["fold"])

    print(f"scene {key}  fold {fold}  shape {pre.shape} {pre.dtype}")

    x = torch.from_numpy(pre.transpose(2, 0, 1)).float()[None]
    x = center_crop(x, TILE).to(DEVICE)
    assert_raw_dn(x, "pre_fire tile")

    print(f"tile {tuple(x.shape)}  DN range [{float(x.min()):.0f}, {float(x.max()):.0f}]")
    print("per-band input:")
    print(summarize(x))

    model = load_tokenizer(DEVICE)
    mean, std = band_stats(DEVICE)

    recon, tokens = roundtrip(model, x, mean, std, timesteps=TIMESTEPS, seed=SEED)

    # ---- rate ------------------------------------------------------------
    bits = bits_per_tile(tokens)
    print(f"\ntokens {tuple(tokens.shape)}  range [{int(tokens.min())}, {int(tokens.max())}]")
    print(f"bits per tile   : {bits:.0f}  ({bits / 8:.0f} bytes)")
    print(f"rate            : {rate_bpppb(tokens, x.shape):.5f} bpppb")
    print(f"ratio vs 12-bit : {compression_ratio(tokens, x.shape, 12):.0f}x")
    print(f"ratio vs 16-bit : {compression_ratio(tokens, x.shape, 16):.0f}x")

    # ---- fidelity --------------------------------------------------------
    print(f"\nrecon DN range [{float(recon.min()):.0f}, {float(recon.max()):.0f}]")
    print("per-band MAE (DN):")
    for i, m in enumerate((recon - x).abs().mean(dim=(0, 2, 3))):
        print(f"  band {i:2d}: {float(m):8.1f}")
    print(f"  overall : {float((recon - x).abs().mean()):8.1f}")

    # ---- figure ----------------------------------------------------------
    a, b = x[0].cpu().numpy(), recon[0].cpu().numpy()
    orig = np.stack([a[i] for i in RGB], -1)
    rec = np.stack([b[i] for i in RGB], -1)
    lo, hi = np.percentile(orig, 2), np.percentile(orig, 98)

    fig, ax = plt.subplots(1, 2, figsize=(11, 5.5))
    ax[0].imshow(stretch(orig, lo, hi))
    ax[0].set_title("original")
    ax[1].imshow(stretch(rec, lo, hi))
    ax[1].set_title(f"reconstruction ({TIMESTEPS} diffusion steps)")
    for a_ in ax:
        a_.axis("off")
    fig.suptitle(f"{key}  |  {bits:.0f} bits per tile  |  "
                 f"{compression_ratio(tokens, x.shape, 12):.0f}x")
    fig.tight_layout()

    Path("results").mkdir(exist_ok=True)
    out = "results/roundtrip_compare.png"
    fig.savefig(out, dpi=140)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()