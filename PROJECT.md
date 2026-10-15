# The project so far

**Investigating Earth Observation Models for Efficient Onboard Computing**
BSc thesis · BME VIK, Department of Control Engineering and Information Technology
Ринат Еркінбек · supervisor Barcza Bende · BMEVIIIAT03
Submission: 11 December 2026, 12:00

Written 15 October 2026, end of week 2.

---

## 1. The question

Earth observation satellites photograph far more of the planet than they can
send back. The sensor runs continuously; the radio link only works during the
few minutes per orbit when a ground station is in view. Everything acquired in
between has to be compressed, and compressed hard.

Conventional onboard compression (CCSDS 122, JPEG2000) throws away detail but
keeps the picture essentially intact, operating at roughly 1–4 bits per pixel
per band.

A newer idea is categorically different. Run the image through a neural model
that converts it into a short description of *what is in the scene*, transmit
only that, and have a second model on the ground paint a picture back from the
description. This is far smaller than any conventional codec can reach.

But the ground no longer receives your image. It receives a **plausible image
synthesised from a description of yours**. So:

> **Does enough real information survive for the task the imagery was
> collected to perform?**

This thesis answers that for one concrete case: compress Sentinel-2 imagery
with the TerraMind S2L2A tokenizer, reconstruct it, run an off-the-shelf
change detection model on the reconstruction, and measure how far the result
falls short of the same detector run on the original.

---

## 2. How the compression works

TerraMind is a large multimodal Earth observation model developed by IBM, ESA
and Forschungszentrum Jülich. **TerraMind itself is not a codec.** The codec is
a component inside it: the **S2L2A tokenizer**.

### Encoding

A 224×224 tile across 12 spectral bands is divided into 16×16 patches, giving
a 14×14 grid. Each patch is matched to one entry in a fixed vocabulary of
15,360 "visual words", jointly across all 12 bands at once.

Output: **196 integers.** That is the entire compressed image.

### The rate

```
bits per token   = log2(15360)            ≈ 13.907
bits per tile    = 196 × 13.907           ≈ 2726 bits = 341 bytes
samples per tile = 12 × 224 × 224         = 602,112
rate             = 2726 / 602,112         ≈ 0.00453 bpppb
```

| Reference | Original tile | Ratio |
|---|---|---|
| 12-bit (Sentinel-2 radiometric depth) | 903,168 B | **2651×** |
| 16-bit (uint16, as CaBuAr stores it) | 1,204,224 B | **3534×** |

341 bytes for a 2.24 km × 2.24 km patch of Earth in twelve spectral bands.
Each token describes 160 m × 160 m of ground — roughly two football pitches —
in under two bytes.

Against the operational 1–4 bpppb, this sits **two to three orders of
magnitude lower**. Conventional codecs cannot reach this rate at all; pushed
there, they output unusable mush. That is what makes the question real rather
than a benchmark exercise: not "is this better compression?" but "at a rate
where conventional compression is not an option, does the task still work?"

### Decoding — and why it matters

The decoder is **diffusion-based**. It does not reconstruct; it *samples* an
image consistent with the 196 tokens. Two consequences shape the entire
experimental design:

1. Pixel-fidelity metrics (PSNR, MSE) will look bad even where the semantics
   survive, because the decoder was never trying to minimise pixel error.
2. The same tokens decoded twice with different random seeds give **different
   images**.

Point 2 is not a nuisance. It is a measurable failure mode, and measuring it
is one of the thesis's contributions.

### What runs where

Only the **encoder** flies. The diffusion decoder runs on the ground, where
compute is effectively free. This matters for the title: "efficient onboard
computing" constrains the encoder alone.

---

## 3. The experimental design

The change detector always receives a pair of images. Every configuration
differs only in **what goes in each slot**.

| Configuration | Slot 1 | Slot 2 | Real change? |
|---|---|---|---|
| **Reference** | original t1 | original t2 | yes |
| **Fully compressed** | reconstruction t1 | reconstruction t2 | yes |
| **Asymmetric** | original t1 | reconstruction t2 | yes |
| **Null A** | reconstruction t1, seed A | reconstruction t1, seed B | **no** |
| **Null B** | original t1 | reconstruction t1 | **no** |

### What each one isolates

**Reference** is the ceiling — what the detector achieves when nothing is
degraded. Without it no other number is interpretable.

**Fully compressed** is the harshest case and the headline result. Both dates
have been through encode and decode, so the detector sees two independently
hallucinated images.

**Asymmetric** is the operationally realistic case. In a deployed system the
archive reference was downlinked at full quality months ago; only the new
acquisition is compressed. Expect a smaller drop. The gap between this and
fully-compressed isolates the cost of degrading the second image.

**Null A** measures **decoder stochasticity**. One acquisition, encoded once,
decoded twice with different seeds. Ground truth change is zero by
construction, so anything the detector reports is pure invention.

**Null B** measures the **domain gap**. An original paired with its own
reconstruction, zero elapsed time. Anything detected here comes from the
systematic difference between real and reconstructed imagery — artefacts,
smoothing, lost high frequencies — rather than from randomness.

### Why two nulls

Each null matches the input structure of one real arm:

- Fully compressed has reconstruction in both slots → matched by **Null A**
- Asymmetric has original then reconstruction → matched by **Null B**

Each real arm gets a floor measured under its own conditions. Subtracting a
floor derived from a different input configuration would not be a fair
subtraction.

### How the numbers combine

- Reference − Compressed = total cost of compressing both dates
- Reference − Asymmetric = cost of compressing one date
- Compressed − Null A = real change signal above the stochastic floor
- Asymmetric − Null B = real change signal above the domain-gap floor

The last two go in the abstract. *"F1 of 0.31 against a floor of 0.04"* is a
result; *"F1 of 0.31"* on its own is not.

---

## 4. What has been built

```
src/satcd/
  bands.py      band order, index constants, input assertions
  codec.py      tokenizer wrapper, scale convention, rate accounting
  data.py       the single shared preprocessing path, CaBuAr Dataset
  detector.py   FC-Siam-diff, loss, metrics, freeze/load
scripts/
  00_check_env.py      versions, CUDA, tokenizer API, bitrate
  00b_scale_check.py   determines the input/output value scale
  01_roundtrip.py      the week 1 gate: encode, decode, compare
  02_explore_data.py   HF repo listing, HDF5 structure
  03_check_data.py     data layer verification
  04_train_reference.py  trains and freezes the reference detector
results/
  roundtrip_compare.png     the gate figure
  valid_tiles.json          cached nodata screening
  reference_fold{0..4}.pt   FROZEN detector checkpoints
  reference_fold{0..4}.json metrics
```

One rule governs the structure: **every experimental arm calls the same
`prepare()` function and nothing else.** The reference arm takes its output
directly; the compressed arm passes it through the codec first. Any difference
in outcome is then attributable to compression alone. If a second
preprocessing path ever appears, that is the bug.

---

## 5. What has been established

### 5.1 The tokenizer API (undocumented, found by measurement)

```python
from terratorch.registry import FULL_MODEL_REGISTRY
model = FULL_MODEL_REGISTRY.build("terramind_v1_tokenizer_s2l2a", pretrained=True)
latent, scalar, tokens = model.encode(x)          # 3-tuple; tokens last
recon = model.decode_tokens(tokens, timesteps=10)
```

- `encode` returns three things, not one: a (1, 5, 14, 14) latent — five being
  the FSQ projection dimension — a scalar, and the (1, 14, 14) int32 tokens.
- **`model(x)` ignores `timesteps` and silently uses 50.** The two paths are
  not equivalent. Always call `decode_tokens` with `timesteps=` explicit.
- 289,290,693 parameters, but that is encoder **plus** decoder. Not an onboard
  figure.
- CUDA gives ~10× on decoding: 21.9 it/s versus 2.05 on CPU.

### 5.2 The value scale convention (the week's main investigation)

Satellite data arrives as integers in the thousands; neural networks want
values near zero. Nothing documented whether the tokenizer standardises
internally. Getting it wrong would not crash — it would silently produce poor
reconstructions, and the thesis would conclude "neural compression performs
badly" when the real cause was malformed input.

Two candidate conventions were tested by measuring the fraction of decoder
output pixels pinned at its bounds:

| Encoder input | flat field | Gaussian noise |
|---|---|---|
| raw DN | 0.211 | 0.144 |
| standardised | **0.000** | **0.000** |

Unambiguous. The convention is:

```
encoder input:  z = (x - mean) / std
decoder output: x = z * std + mean
```

using `v1_pretraining_mean/std["untok_sen2l2a@224"]` from the terratorch
register. Confirmed on real data at 0.005 saturation. Lives in `codec.py`;
these numbers appear nowhere else in the codebase.

This belongs in the thesis as a small methodological contribution — it was
established by experiment, not read from documentation.

### 5.3 The round trip works

Scene `012b8863…_0`, 224 centre crop, 10 diffusion steps, seed 0:

| | |
|---|---|
| input DN range | [87, 5912] |
| reconstruction DN range | [318, 4473] |
| overall MAE | **219.6 DN** (≈10% of typical values) |
| per-band MAE | 55.7 (B01) to 311.7 (B11) |
| decoder saturation | 0.005 |

The reconstruction is recognisably the same terrain — ridge lines, drainage
network, vegetated spurs all in the right places. Blurred, but unmistakably
the same place. → `results/roundtrip_compare.png`

### 5.4 The dataset

**CaBuAr**: Sentinel-2 L2A pre/post fire pairs over California wildfires with
burned-area masks from CAL FIRE. Using `raw/patched/512x512.hdf5` (5.88 GB).

**Not `normalized/`** — those are scaled with the dataset authors' own
statistics, which would compose with TerraMind's standardisation and put the
input outside the distribution the tokenizer was trained on.

| | |
|---|---|
| groups in archive | 534 |
| with both acquisitions | **356** (the other 178 are post-fire only) |
| tiles after 2×2 tiling | 1424 |
| after nodata screening | **1327** (97 dropped, 6.8%) |
| per-fold scenes | 64 / 54 / 62 / 85 / 65 |
| burned pixel fraction | 0.0389 |
| tiles with no burn at all | **70%** |

The 178 post-only scenes support burned-area *delineation* from a single
image, not bitemporal change detection. Nodata tiles — zero-filled at swath
edges, or with a whole band missing — are excluded because they encode to
almost nothing and would flatter the codec.

### 5.5 The reference arm — frozen

FC-Siam-diff, 24,464,738 parameters, `in_channels=12`. Trained once on
uncompressed data per fold, then frozen.

| fold | P | R | F1 | IoU |
|---|---|---|---|---|
| 0 | 0.779 | 0.850 | 0.813 | 0.685 |
| 1 | 0.555 | 0.625 | 0.588 | 0.416 |
| 2 | 0.510 | 0.934 | 0.660 | 0.492 |
| 3 | 0.798 | 0.896 | 0.844 | 0.730 |
| 4 | 0.694 | 0.908 | 0.787 | 0.649 |

**F1 = 0.738 ± 0.110    IoU = 0.595 ± 0.134**

---

## 6. Decisions, and why

**Tiling: 2×2 non-overlapping, centred, dropping a 32 px border.** A centre
crop alone gives 356 tiles, too few to train on. An overlapping 3×3 grid gives
more data but double-counts pixels at evaluation, which makes F1 harder to
interpret. No leakage: folds are per scene, so all four tiles stay together.

**Sampling: `WeightedRandomSampler` targeting ~50% positive tiles, training
only.** With 70% of tiles empty, about a quarter of batches at size 4 would
carry no positive signal at all. Validation and test keep their natural
distribution — otherwise F1 is neither meaningful as a ceiling nor comparable
to published results.

**`pos_weight = 1.0`, settled empirically.** Fold 0 test metrics:

| pos_weight | P | R | F1 | IoU |
|---|---|---|---|---|
| 3.0 | 0.709 | 0.715 | 0.712 | 0.553 |
| **1.0** | **0.798** | **0.901** | **0.846** | **0.734** |

The sampler alone rebalances correctly; extra pixel weighting over-corrects.
This matters beyond tidiness: a detector biased toward predicting burn keeps
firing on degraded input and would **understate** the compression cost.

**Checkpoint selection on a scene-level validation split, not the test fold.**
The first run selected on test, making the reported F1 best-of-30 on noisy
data. An inflated ceiling inflates every later ΔF1. Now 12% of the training
folds carry selection, split by scene (adjacent tiles share fire, terrain and
acquisition conditions), and the test fold is touched exactly once at the end.

---

## 7. Findings that are already thesis content

**Dynamic range is compressed.** [87, 5912] → [318, 4473]. Both tails lost.
Burn scars are *defined* by extremes — very dark in NIR, very bright in SWIR.
If the codec flattens the signal the task depends on, that is a result.

**Hallucination is visible on the very first tile.** A small dark feature
returns as a differently-shaped ellipse; thin linear features — roads, stream
networks — vanish entirely. Not blur: invention. This is the direct motivation
for the null-pair experiment.

**Decoder stochasticity is measurable and non-trivial.** Same tile, same 196
tokens, two diffusion samples: MAE 235.7 versus 219.6. About **7% of the
reconstruction error is the decoder choosing a different plausible image**,
not information loss.

**A week 1 claim was retracted.** Week 1 reported that NIR degrades worst.
It does not survive a seed change — B08 and B11 swap places. What holds is
only the weaker claim: smooth low-resolution bands survive best. Needs many
tiles and all five folds before it enters chapter 5.

---

## 8. The constraint that shapes week 3

The reference arm's fold-to-fold spread is **±0.110**, with F1 ranging 0.588
to 0.844. That is plausibly **larger than the compression effect being
measured**.

So the comparison must be **paired**: same fold, same frozen checkpoint,
reference versus compressed, ΔF1 computed *within* each fold and then averaged
across the five. Paired differences cancel fold difficulty, so their variance
is far smaller than the raw F1 variance.

A pooled comparison — reference mean against compressed mean — would bury a
0.05 effect inside fold-to-fold noise and show nothing. Discovering this now
rather than in week 4, when the numbers would simply have come out muddy, is
worth more than the reference F1 itself.

---

## 9. Open items

**Three literature figures still unsourced**, needed for §2.3 and §2.4:
Sentinel-2's actual onboard compression scheme and rate; a cited lossless
compression ratio for multispectral imagery; the operational bpppb range for
CCSDS 122. The 1–4 bpppb figure is already asserted in the task description
and needs a reference behind it.

**Deferred measurements:** encoder-only parameter count, memory and latency
(§3.5); Jetson Orin Nano profiling (week 5).

**Known bug:** the `pos_weight` warning in `04_train_reference.py` advises
trying 1.0 even when already at 1.0. Make it conditional.

---

## 10. Where the thesis stands

**Engineering: a week ahead.** Week 1's gate was met on day one; week 2's
reference arm is frozen with four days to spare.

**Writing: behind.** §2.1–2.4 is 19 paragraphs due 21 October. One is drafted.

| | |
|---|---|
| Chapter 2 total | 33 paragraphs, 12 pages, due 28 Oct |
| Experiments freeze | 25 November — no new experiments after |
| Full draft to Barcza | 29 November |
| Submission | **10 December** (the deadline is 11 Dec at **noon**) |

### Next — week 3

- `compress.py`: encode → tokens → decode, as an arm
- the compressed and asymmetric arms
- both nulls
- **the floor number** — the gate that can reshape the thesis

If the floor turns out close to the compressed-arm signal, the finding becomes
*"generative codecs manufacture as much apparent change as they preserve"*.
That is a complete and defensible negative result, and it would be far better
to discover it in week 3 than in week 8.
