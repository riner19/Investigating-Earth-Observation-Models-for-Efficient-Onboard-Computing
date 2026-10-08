"""Week 1 gate: round-trip one real CaBuAr tile and look at it."""

import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
from terratorch.registry import FULL_MODEL_REGISTRY

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from satcd.codec import band_stats, standardize, destandardize

REPO = "DarthReca/california_burned_areas"
TILE = 224
TIMESTEPS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# assumed order: B01 B02 B03 B04 B05 B06 B07 B08 B8A B09 B11 B12
RGB = (3, 2, 1)


def center_crop(a, size):
    h, w = a.shape[-2:]
    top, left = (h - size) // 2, (w - size) // 2
    return a[..., top:top + size, left:left + size]


def stretch(img, lo, hi):
    return np.clip((img - lo) / (hi - lo + 1e-9), 0, 1)


def main():
    path = hf_hub_download(REPO, filename="raw/patched/512x512.hdf5", repo_type="dataset")

    with h5py.File(path, "r") as f:
        key = list(f.keys())[0]
        pre = np.asarray(f[key]["pre_fire"])      # (512, 512, 12)
        fold = int(f[key].attrs["fold"])

    print(f"scene {key}  fold {fold}  shape {pre.shape} {pre.dtype}")

    x = torch.from_numpy(pre.transpose(2, 0, 1)).float()[None]   # (1, 12, 512, 512)
    x = center_crop(x, TILE).to(DEVICE)
    print(f"tile: {tuple(x.shape)}  DN range [{float(x.min()):.0f}, {float(x.max()):.0f}]")

    model = FULL_MODEL_REGISTRY.build("terramind_v1_tokenizer_s2l2a", pretrained=True)
    model = model.to(DEVICE).eval()
    mean, std = band_stats(DEVICE)

    with torch.no_grad():
        z = standardize(x, mean, std)
        tokens = model.encode(z)[-1]
        z_hat = model.decode_tokens(tokens, timesteps=TIMESTEPS)
        recon = destandardize(z_hat, mean, std)

    print(f"tokens: {tuple(tokens.shape)}  range [{int(tokens.min())}, {int(tokens.max())}]")
    print(f"decoder output saturated: {float((z_hat.abs() > 0.99).float().mean()):.3f}")
    print(f"recon DN range [{float(recon.min()):.0f}, {float(recon.max()):.0f}]")

    print("\nper-band MAE (DN):")
    for i, m in enumerate((recon - x).abs().mean(dim=(0, 2, 3))):
        print(f"  band {i:2d}: {float(m):8.1f}")
    print(f"  overall : {float((recon - x).abs().mean()):8.1f}")

    a = x[0].cpu().numpy()
    b = recon[0].cpu().numpy()
    orig = np.stack([a[i] for i in RGB], -1)
    rec = np.stack([b[i] for i in RGB], -1)
    lo, hi = np.percentile(orig, 2), np.percentile(orig, 98)

    fig, ax = plt.subplots(1, 2, figsize=(11, 5.5))
    ax[0].imshow(stretch(orig, lo, hi)); ax[0].set_title("original")
    ax[1].imshow(stretch(rec, lo, hi)); ax[1].set_title(f"reconstruction ({TIMESTEPS} steps)")
    for a_ in ax:
        a_.axis("off")
    fig.suptitle(f"{key}  |  2726 bits per tile  |  2651x")
    fig.tight_layout()

    Path("results").mkdir(exist_ok=True)
    out = "results/roundtrip_compare.png"
    fig.savefig(out, dpi=140)
    print(f"\nwrote {out} -- open it")


if __name__ == "__main__":
    main()