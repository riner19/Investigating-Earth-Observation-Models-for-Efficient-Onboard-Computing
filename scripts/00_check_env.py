"""Environment and tokenizer check. Run before anything else."""

import math
import torch

print("=" * 60)
print("torch          :", torch.__version__)
print("cuda available :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda device    :", torch.cuda.get_device_name(0))

import terratorch
print("terratorch     :", getattr(terratorch, "__version__", "unknown"))

print("=" * 60)
print("building tokenizer (first run downloads weights from HF)...")

from terratorch.registry import FULL_MODEL_REGISTRY

model = FULL_MODEL_REGISTRY.build("terramind_v1_tokenizer_s2l2a", pretrained=True)
model.eval()

n_params = sum(p.numel() for p in model.parameters())
print(f"parameters     : {n_params:,}")

print("=" * 60)
print("synthetic encode/decode, shapes only")

x = torch.rand(1, 12, 224, 224)

with torch.no_grad():
    out = model.encode(x)

print("encode returned:", type(out), "len =", len(out) if isinstance(out, tuple) else "n/a")
for i, item in enumerate(out):
    shape = tuple(item.shape) if torch.is_tensor(item) else type(item)
    dtype = item.dtype if torch.is_tensor(item) else ""
    print(f"  [{i}] {shape} {dtype}")

tokens = out[-1]
print("tokens shape   :", tuple(tokens.shape))
print("tokens dtype   :", tokens.dtype)
print("tokens min/max :", int(tokens.min()), int(tokens.max()))

CODEBOOK = 15360
bits = tokens.numel() * math.log2(CODEBOOK)
samples = x.numel()
print(f"bits per tile  : {bits:.0f}")
print(f"bpppb (12-bit) : {bits / samples:.5f}")
print(f"ratio vs 12bit : {samples * 12 / bits:.0f}x")

with torch.no_grad():
    recon = model.decode_tokens(tokens, timesteps=10)

print("recon shape    :", tuple(recon.shape))
print("recon min/max  :", float(recon.min()), float(recon.max()))

print("=" * 60)
print("pretraining statistics (needed for scaling in step 7)")
try:
    from terratorch.models.backbones.terramind.model import terramind_register as reg
    mean = reg.v1_pretraining_mean["untok_sen2l2a@224"]
    std = reg.v1_pretraining_std["untok_sen2l2a@224"]
    print("mean:", [round(float(v), 1) for v in mean])
    print("std :", [round(float(v), 1) for v in std])
except Exception as e:
    print("could not read stats from this path:", e)
    print("find them yourself with: python -c \"import terratorch, os; print(os.path.dirname(terratorch.__file__))\"")
    print("then grep for 'pretraining_mean'")