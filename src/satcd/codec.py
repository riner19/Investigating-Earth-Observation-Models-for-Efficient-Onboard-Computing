import torch
from terratorch.registry import FULL_MODEL_REGISTRY
from terratorch.models.backbones.terramind.model import terramind_register as reg

_KEY = "untok_sen2l2a@224"
CODEBOOK_SIZE = 15360


def band_stats(device="cpu"):
    mean = torch.tensor(reg.v1_pretraining_mean[_KEY]).view(1, 12, 1, 1).to(device)
    std = torch.tensor(reg.v1_pretraining_std[_KEY]).view(1, 12, 1, 1).to(device)
    return mean, std


def standardize(x, mean, std):
    """Raw L2A DN -> tokenizer input space."""
    return (x - mean) / std


def destandardize(z, mean, std):
    """Decoder output -> raw L2A DN."""
    return z * std + mean