"""Determine the value scale the TerraMind S2L2A tokenizer expects and returns.

Open question: the decoder's output is bounded in [-1, 1], and every synthetic
input so far saturates at the bounds. Two candidate conventions:

  A. raw DN in          -> recon * std + mean out
  B. (x - mean) / std in -> recon * std + mean out

This script reports saturation under both. Low saturation is the signal that
the input convention is right. Synthetic data cannot settle it conclusively --
a real tile does that -- but it narrows it to one candidate.

Run:  python scripts/00b_scale_check.py
"""

import torch
from terratorch.registry import FULL_MODEL_REGISTRY
from terratorch.models.backbones.terramind.model import terramind_register as reg

TIMESTEPS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def band_means(t: torch.Tensor) -> list[float]:
    return [round(float(v), 1) for v in t.mean(dim=(0, 2, 3))]


def saturation(t: torch.Tensor, thr: float = 0.99) -> float:
    """Fraction of output pixels pinned at the bounds."""
    return float((t.abs() > thr).float().mean())


def roundtrip(model, x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        tokens = model.encode(x)[-1]
        return model.decode_tokens(tokens, timesteps=TIMESTEPS)


def report(name: str, x_in: torch.Tensor, recon: torch.Tensor) -> float:
    sat = saturation(recon)
    print(f"  {name}")
    print(f"    input  means : {band_means(x_in)}")
    print(f"    decode means : {band_means(recon)}")
    print(f"    decode range : [{float(recon.min()):.3f}, {float(recon.max()):.3f}]")
    print(f"    saturated    : {sat:.3f}")
    return sat


def main() -> None:
    print(f"device: {DEVICE}")

    model = FULL_MODEL_REGISTRY.build("terramind_v1_tokenizer_s2l2a", pretrained=True)
    model = model.to(DEVICE).eval()

    mean = torch.tensor(reg.v1_pretraining_mean["untok_sen2l2a@224"]).view(1, 12, 1, 1).to(DEVICE)
    std = torch.tensor(reg.v1_pretraining_std["untok_sen2l2a@224"]).view(1, 12, 1, 1).to(DEVICE)

    print("=" * 70)
    print("pretraining statistics")
    print("  mean:", [round(float(v), 1) for v in mean.flatten()])
    print("  std :", [round(float(v), 1) for v in std.flatten()])

    # ------------------------------------------------------------------
    # Convention A -- raw DN input
    # ------------------------------------------------------------------
    print("=" * 70)
    print("convention A: raw DN input")

    flat_dn = mean.expand(1, 12, 224, 224).contiguous()
    sat_a_flat = report("flat image at per-band mean", flat_dn, roundtrip(model, flat_dn))

    noisy_dn = mean + std * torch.randn(1, 12, 224, 224, device=DEVICE)
    sat_a_noise = report("gaussian noise N(mean, std)", noisy_dn, roundtrip(model, noisy_dn))

    # ------------------------------------------------------------------
    # Convention B -- standardized input
    # ------------------------------------------------------------------
    print("=" * 70)
    print("convention B: standardized input")

    flat_z = torch.zeros(1, 12, 224, 224, device=DEVICE)
    sat_b_flat = report("flat image at zero", flat_z, roundtrip(model, flat_z))

    noisy_z = torch.randn(1, 12, 224, 224, device=DEVICE)
    recon_b_noise = roundtrip(model, noisy_z)
    sat_b_noise = report("gaussian noise N(0, 1)", noisy_z, recon_b_noise)

    # ------------------------------------------------------------------
    # De-standardization check, convention B
    # ------------------------------------------------------------------
    print("=" * 70)
    print("de-standardization, convention B noise case")
    print("  recon * std + mean :", band_means(recon_b_noise * std + mean))
    print("  (compare against the pretraining means above)")

    # ------------------------------------------------------------------
    # One-call path -- note it uses its own default timestep count
    # ------------------------------------------------------------------
    print("=" * 70)
    print("one-call model(x), convention B noise input")
    with torch.no_grad():
        full = model(noisy_z)
    print("  means :", band_means(full))
    print(f"  range : [{float(full.min()):.3f}, {float(full.max()):.3f}]")
    print(f"  saturated: {saturation(full):.3f}")
    print("  note: model(x) uses its own default timestep count, not TIMESTEPS.")
    print("  always call decode_tokens with timesteps= explicitly.")

    # ------------------------------------------------------------------
    print("=" * 70)
    print("summary -- lower saturation means the input convention is closer")
    print(f"  A  raw DN      : flat {sat_a_flat:.3f}   noise {sat_a_noise:.3f}")
    print(f"  B  standardized: flat {sat_b_flat:.3f}   noise {sat_b_noise:.3f}")
    print()
    if sat_b_noise < sat_a_noise - 0.05:
        print("  -> convention B looks right: standardize going in.")
    elif sat_a_noise < sat_b_noise - 0.05:
        print("  -> convention A looks right: raw DN going in.")
    else:
        print("  -> no clear difference on synthetic data.")
    print()
    print("Synthetic inputs have no spatial structure, so heavy saturation is")
    print("expected under BOTH conventions. Treat this as a hint only.")
    print("Settle it on a real CaBuAr tile: run both conventions, compute")
    print("per-band MAE against the original, and look at the RGB side by side.")


if __name__ == "__main__":
    main()