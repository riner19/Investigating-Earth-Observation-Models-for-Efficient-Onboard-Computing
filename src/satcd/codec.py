"""TerraMind S2L2A tokenizer wrapper and rate accounting.

Encapsulates the value-scale convention established in week 1: the encoder
expects standardized input, the decoder returns standardized output, and the
pretraining statistics for both live in the terratorch register. Nothing
outside this module should touch those statistics directly.
"""

from __future__ import annotations

import math

import torch
from terratorch.registry import FULL_MODEL_REGISTRY
from terratorch.models.backbones.terramind.model import terramind_register as reg

from .bands import N_BANDS, assert_raw_dn, assert_standardized

MODEL_NAME = "terramind_v1_tokenizer_s2l2a"
STATS_KEY = "untok_sen2l2a@224"

CODEBOOK_SIZE = 15360
BITS_PER_TOKEN = math.log2(CODEBOOK_SIZE)      # 13.907


def load_tokenizer(device: str = "cpu"):
    """Build the pretrained tokenizer and put it in eval mode."""
    model = FULL_MODEL_REGISTRY.build(MODEL_NAME, pretrained=True)
    return model.to(device).eval()


def band_stats(device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    """Per-band pretraining mean and std, shaped for broadcasting."""
    mean = torch.tensor(reg.v1_pretraining_mean[STATS_KEY]).view(1, N_BANDS, 1, 1)
    std = torch.tensor(reg.v1_pretraining_std[STATS_KEY]).view(1, N_BANDS, 1, 1)
    return mean.to(device), std.to(device)


def standardize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Raw L2A DN -> tokenizer input space."""
    return (x - mean) / std


def destandardize(z: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Decoder output -> raw L2A DN."""
    return z * std + mean


@torch.no_grad()
def encode(model, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor,
           validate: bool = True) -> torch.Tensor:
    """Raw L2A DN -> token indices, shape (B, 14, 14) int32.

    `model.encode` returns a three-tuple (latent, scalar, tokens); only the
    tokens are the compressed representation.
    """
    if validate:
        assert_raw_dn(x, "encode input")
    z = standardize(x, mean, std)
    if validate:
        assert_standardized(z, "encoder input")
    return model.encode(z)[-1]


@torch.no_grad()
def decode(model, tokens: torch.Tensor, mean: torch.Tensor, std: torch.Tensor,
           timesteps: int = 10, seed: int | None = None) -> torch.Tensor:
    """Token indices -> raw L2A DN.

    The decoder is a diffusion model and therefore stochastic: identical tokens
    decoded twice with different seeds give different images. Pass `seed` to fix
    the sample. That seed-to-seed difference is exactly what the null-pair
    experiment measures.

    Note `model(x)` would run encode and decode in one call but ignores
    `timesteps`, defaulting to 50 -- which is why this module never uses it.
    """
    if seed is not None:
        torch.manual_seed(seed)
    z_hat = model.decode_tokens(tokens, timesteps=timesteps)
    return destandardize(z_hat, mean, std)


@torch.no_grad()
def roundtrip(model, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor,
              timesteps: int = 10, seed: int | None = None):
    """Encode and decode in one call. Returns (reconstruction, tokens)."""
    tokens = encode(model, x, mean, std)
    recon = decode(model, tokens, mean, std, timesteps=timesteps, seed=seed)
    return recon, tokens


# --------------------------------------------------------------------------
# Rate accounting -- these numbers go into thesis section 3.3
# --------------------------------------------------------------------------

def bits_per_tile(tokens: torch.Tensor) -> float:
    """Bits for one tile's tokens, assuming a uniform code over the codebook.

    This is an upper bound: the empirical token distribution is not uniform, so
    entropy coding the token stream would reduce it further. Not pursued, since
    it would improve the rate without changing information content.
    """
    return tokens[0].numel() * BITS_PER_TOKEN


def rate_bpppb(tokens: torch.Tensor, tile_shape) -> float:
    """Bits per pixel per band."""
    c, h, w = tile_shape[-3:]
    return bits_per_tile(tokens) / (c * h * w)


def compression_ratio(tokens: torch.Tensor, tile_shape, source_bits: int = 12) -> float:
    """Ratio against the ORIGINAL tile at the given bit depth.

    Never compute this against an upsampled tensor -- that inflates the figure.
    """
    c, h, w = tile_shape[-3:]
    return (c * h * w * source_bits) / bits_per_tile(tokens)