"""Change detection model, loss, and metrics.

The detector is trained ONCE on uncompressed data and then frozen. Every
experimental arm -- reference, compressed, asymmetric, both nulls -- loads the
same checkpoint. A retrained detector would measure domain adaptation rather
than information loss, which is a different question.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bands import N_BANDS


def build_detector(in_channels: int = N_BANDS, classes: int = 2) -> nn.Module:
    """FC-Siam-diff: a fully convolutional siamese network with difference skip
    connections (Caye Daudt et al., 2018).

    Input  (B, 2, C, H, W) -- two co-registered dates
    Output (B, classes, H, W) -- per-pixel logits
    """
    from torchgeo.models import FCSiamDiff
    return FCSiamDiff(in_channels=in_channels, classes=classes)


class DiceLoss(nn.Module):
    """Soft Dice on the positive (burned) class.

    Dice is insensitive to the large unburned background, which makes it a
    good partner for cross-entropy when roughly 4% of pixels are positive.
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prob = logits.softmax(dim=1)[:, 1]
        t = target.float()
        inter = (prob * t).sum(dim=(1, 2))
        total = prob.sum(dim=(1, 2)) + t.sum(dim=(1, 2))
        dice = (2 * inter + self.smooth) / (total + self.smooth)
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    """Weighted cross-entropy plus soft Dice.

    `pos_weight` is kept mild because the WeightedRandomSampler already
    rebalances at the tile level. Applying the full inverse-frequency weight
    (~25) on top of balanced sampling over-corrects and the model predicts
    burn almost everywhere.
    """

    def __init__(self, pos_weight: float = 3.0, dice_weight: float = 1.0):
        super().__init__()
        self.register_buffer("w", torch.tensor([1.0, float(pos_weight)]))
        self.dice = DiceLoss()
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, weight=self.w.to(logits.device))
        return ce + self.dice_weight * self.dice(logits, target)


@torch.no_grad()
def confusion(logits: torch.Tensor, target: torch.Tensor) -> tuple[int, int, int, int]:
    """(tp, fp, fn, tn) for the positive class."""
    pred = logits.argmax(dim=1)
    p, t = pred == 1, target == 1
    return (
        int((p & t).sum()),
        int((p & ~t).sum()),
        int((~p & t).sum()),
        int((~p & ~t).sum()),
    )


def metrics(tp: int, fp: int, fn: int, tn: int) -> dict[str, float]:
    """Change-class precision, recall, F1 and IoU.

    Accuracy is deliberately not reported: at ~4% positive pixels a model that
    predicts 'unburned' everywhere scores 96% and detects nothing.
    """
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "iou": iou}


def save_checkpoint(model: nn.Module, path: str | Path, **meta) -> None:
    """Persist the frozen reference detector."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)


def load_detector(path: str | Path, device: str = "cpu",
                  in_channels: int = N_BANDS, classes: int = 2) -> nn.Module:
    """Load the frozen reference detector. Always returns it in eval mode.

    This is how every experimental arm gets its detector. Do not fine-tune the
    result -- the whole comparison rests on it being identical across arms.
    """
    blob = torch.load(path, map_location=device, weights_only=False)
    model = build_detector(in_channels, classes)
    model.load_state_dict(blob["state_dict"])
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model
