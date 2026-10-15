# Work log

Thesis: Investigating Earth Observation Models for Efficient Onboard Computing
BME VIK IIT · BMEVIIIAT03 · supervisor Barcza Bende · submission 11 Dec 2026, 12:00

---

## 2026-10-08 — Week 1

### Environment
- Project folder created, venv on Python 3.14, CUDA available.
- torch 2.14.1+cu130, GPU: NVIDIA RTX 3050 Ti Laptop.
- terratorch installed; `__version__` reports "unknown" — record the real one
  from `pip show terratorch`.
- Note: Python 3.14 is very new. `torch.jit` emits a FutureWarning (harmless).
  If a geospatial wheel fails to install later, rebuild the venv on 3.12.

### TerraMind S2L2A tokenizer — API
```python
from terratorch.registry import FULL_MODEL_REGISTRY
model = FULL_MODEL_REGISTRY.build("terramind_v1_tokenizer_s2l2a", pretrained=True)
latent, scalar, tokens = model.encode(x)     # 3-tuple, tokens last
recon = model.decode_tokens(tokens, timesteps=10)
```
- Weights ~1.09 GB, downloaded from HF on first build.
- 289,290,693 parameters TOTAL (encoder + diffusion decoder).
  Do NOT quote this as an onboard figure — only the encoder flies.
- `encode()` returns: latent (1,5,14,14) float32 — the 5 FSQ dimensions;
  a (1,) float32 scalar; tokens (1,14,14) int32.
- `model(x)` runs encode+decode but uses its own default of 50 timesteps.
  Always call `decode_tokens` with `timesteps=` explicitly.
- CUDA gives ~10x on decoding: 21.9 it/s vs 2.05 it/s on CPU.

### Bitrate (thesis numbers, chapter 3)
- codebook 15360 → log2(15360) = 13.907 bits per token
- 196 tokens per 224×224×12 tile → **2726 bits per tile**
- **0.00453 bits per pixel per band**
- **2651× vs 12-bit source** (≈3530× vs 16-bit storage)
- Computed against the original tile, not an upsampled tensor.

### Scale convention — SETTLED
Decoder output is bounded in [-1, 1] (tanh). Tested raw-DN input vs
standardized input, measuring saturation fraction (|x| > 0.99):

| input convention | flat image | gaussian noise |
|---|---|---|
| raw DN | 0.211 | 0.144 |
| standardized | **0.000** | **0.000** |

**Convention: `(x - mean) / std` going in, `recon * std + mean` coming out.**
Stats from `terramind_register.v1_pretraining_{mean,std}["untok_sen2l2a@224"]`.

mean: [1390.5, 1503.3, 1718.2, 1853.9, 2199.1, 2780.0, 2987.0, 3083.2, 3132.2, 3163.0, 2424.9, 1857.6]
std : [2106.8, 2141.1, 2039.0, 2134.1, 2085.3, 1889.9, 1820.3, 1871.9, 1753.8, 1797.4, 1434.3, 1334.3]

Confirmed on real data: saturation 0.005 on a real tile.
Lives in `src/satcd/codec.py`. Never inline these numbers anywhere else.

### Dataset — CaBuAr
- Source: `DarthReca/california_burned_areas` on HF.
- Using **`raw/patched/512x512.hdf5`** (5.88 GB).
  NOT `normalized/` — those are pre-scaled with the authors' own statistics,
  which would collide with TerraMind's standardization.
- 534 top-level groups named `<uuid>_<patch_index>`.
- Each group: `pre_fire` (512,512,12) uint16, `post_fire` (512,512,12) uint16,
  `mask` (512,512,1) uint16 with values {0,1}.
- Axis order is (H, W, C) → transpose to (C, H, W).
- **12 bands, maps directly onto TerraMind S2L2A. No B10 to drop.**
- Fold assignment is an HDF5 group attr: `attrs["fold"]`. Also `attrs["comments"]`.
- `metadata.parquet` (65340 rows) is per-patch provenance — uuid, scene name,
  x, y, patch, crs (UTM 32610/32611), timestamp, post flag. Not needed for
  the five-fold split.

### WEEK 1 GATE — PASSED
Scene `012b8863-976c-44e1-a491-9adf19c1cbba_0`, fold 1, 224 centre crop.
Reconstruction is recognisably the same terrain: ridge lines, drainage
network and vegetated spurs all preserved. → `results/roundtrip_compare.png`

- overall MAE **235.7 DN** (≈10% of typical DN), at 2651× compression
- per-band MAE: B01 66.0 (lowest), B08 329.5 (highest)
- tokens in range [248, 15330], decoder saturation 0.005
- RGB from indices (3,2,1) looks like natural-colour California →
  band order assumption validated

### Early observations for the thesis
1. **Dynamic range is compressed.** Original [87, 5912] → recon [318, 4473].
   Both tails are lost. Relevant because burn scars live in the extremes —
   dark in NIR, bright in SWIR. Watch for reduced sensitivity downstream.
2. **Band degradation is uneven and not random.** The smooth 60 m aerosol band
   survives best; NIR degrades worst. The vegetation-sensitive bands, i.e. the
   ones carrying the change signal, are the ones being damaged. Figure for ch. 5.
3. **Hallucination visible on the first tile.** A small dark feature comes back
   as a differently-shaped ellipse; thin linear features (roads, stream network)
   are smoothed away entirely. Keep this PNG for the discussion chapter.

### Next — Week 2
- `src/satcd/bands.py` — band constants, input assertions
- `src/satcd/data.py` — single `prepare()`, CaBuAr Dataset using the fold attr
- Train FC-Siam-diff (`in_channels=12`) on uncompressed data, then FREEZE
- Gate: reference F1 in a sane range (~0.5–0.8) for burned area
- Send Barcza the round-trip figure
---

## 2026-10-15 — Week 2

### Refactor
Codec logic moved out of the scripts into `src/satcd/`:
- `bands.py` — band order, RGB/NBR/NDVI indices, `assert_raw_dn`,
  `assert_standardized`, `summarize`
- `codec.py` — `load_tokenizer`, `band_stats`, `standardize`/`destandardize`,
  `encode`/`decode`/`roundtrip`, and rate accounting
  (`bits_per_tile`, `rate_bpppb`, `compression_ratio`)
- `data.py` — `prepare()` (the single shared path), `CaBuAr` Dataset,
  `collate`, `class_balance`

`01_roundtrip.py` now calls these rather than doing the work inline.

### Decoder stochasticity, measured
Same tile, same 196 tokens, two different diffusion samples:

| run | MAE (DN) |
|---|---|
| seed unset | 235.7 |
| seed 0 | 219.6 |

≈7% of the reconstruction error on this tile is the decoder choosing a
different plausible image, not information loss. First direct evidence of the
effect the null-pair experiment is built to quantify.

`codec.decode()` now takes an explicit `seed`: fixed for the main experiments,
deliberately varied for the null-pair arm in week 3.

### CORRECTION to the week 1 band-degradation claim
Week 1 stated NIR (B08) degrades worst. Not supported — the ordering moves
between seeds:

| band | seed unset | seed 0 |
|---|---|---|
| B01 | 66.0 | 55.7 |
| B08 | 329.5 | 290.0 |
| B11 | 315.5 | 311.7 |

B01 is lowest in both, so "smooth low-resolution bands survive best" holds.
Which high-resolution band degrades *worst* is unresolved on one tile at one
seed. Needs many tiles and the five folds before it can go in ch. 5.
Soften §7.2 of the week 1 report accordingly.

### CORRECTION to the week 1 dataset description
Week 1 §5.3 generalised the HDF5 structure from the first group. It is not
uniform:

| groups | datasets present |
|---|---|
| 356 | `mask`, `post_fire`, `pre_fire` |
| 178 | `mask`, `post_fire` |

The 178 post-only scenes support burned-area **delineation** from a single
image, not bitemporal change detection. They are skipped.

**Usable: 356 scenes × 4 tiles = 1424 tiles.** This number and the reason for
it belong in §4.2.

Fold distribution over all 534 groups: {0: 130, 1: 104, 2: 99, 3: 109, 4: 92}
— a 40% spread before filtering. Per-fold counts over the usable subset are
reported by `03_check_data.py`; if the spread is large, say so in §4.2 rather
than let a reader notice the error bars rest on uneven sample sizes.

### Tiling decision
512 → 224 as a **2×2 non-overlapping grid, centred, dropping a 32 px border**.
Four tiles per scene.

Alternatives rejected: a centre crop gives only 356 tiles, too few to train
on; an overlapping 3×3 grid gives more data but double-counts pixels at
evaluation, making F1 harder to interpret. No leakage risk — folds are
assigned per scene, so all four tiles of a scene stay together.

### Nodata tiles — discovered by an assertion, not by inspection
`assert_raw_dn` fired on a tile whose full range was [0.000, 0.000]. Not a
scaling problem: the tile is **entirely zero**. Sentinel-2 patches cut near a
swath edge or scene boundary are zero-filled, and CaBuAr tiles inherit that.

A later failure at 9.2% nodata exposed a second form: 1/12 ≈ 8.3%, i.e. a
whole *band* zero while the others carry data. The first version of the scan
checked only band 0 and missed it. The scan now reads **all 12 bands**,
subsampled 4× spatially.

Why this matters beyond the crash: an all-zero tile encodes to almost nothing
and would flatter the codec in week 3, and contributes nothing but noise to
training. Exclusion is not optional.

**Threshold: a tile is dropped if either date is >5% zero-filled.** That choice
needs defending in §4.3 along with the count it removes. Results cached to
`results/valid_tiles.json`, which records its own parameters and rebuilds if
they change.

### Data layer, verified
- **1327 usable tiles** from 356 scenes
  (97 tiles, 6.8%, dropped for nodata; 178 scenes dropped for lacking `pre_fire`)
- per-fold scenes: 64 / 54 / 62 / 85 / 65 — a **57% spread**, note in §4.2
- mask values exactly {0, 1}
- burned pixel fraction **0.0389**
- **70% of tiles contain no burned pixel at all**

Note the per-split counts printed by the script are per-split, not totals, and
its "usable scenes" line is just `tiles // 4` — use the figures above.

The 70% figure has real consequences: at batch size 4, about a quarter of
batches would carry no positive signal whatsoever.

→ **training** uses a `WeightedRandomSampler` targeting ~50% positive tiles
→ **validation and test** keep their natural distribution, otherwise F1 is
  neither meaningful as a ceiling nor comparable to published results

### Detector
`src/satcd/detector.py`: FC-Siam-diff via torchgeo, `in_channels=12`,
**24,464,738 parameters**. Loss is weighted cross-entropy + soft Dice. Metrics
are change-class precision, recall, F1 and IoU — accuracy is deliberately not
reported, since at ~4% positive pixels a model predicting "unburned"
everywhere scores 96% and detects nothing.

`load_detector()` always returns the model in eval mode with
`requires_grad=False`, so an arm cannot accidentally fine-tune it.

### Checkpoint selection — a bias found and removed
The first training run selected the checkpoint on the **test** fold, making
the reported F1 best-of-30 on noisy data (epoch-to-epoch swing was 0.72–0.82).
An inflated ceiling inflates every later ΔF1.

Now: a **scene-level** validation split, 12% of the training folds, carries
checkpoint selection. Splitting by scene rather than tile matters — adjacent
tiles of one scene share fire, terrain and acquisition conditions. The test
fold is evaluated **exactly once**, at the end, on the already-selected
checkpoint.

val→test gaps confirm this was necessary: validation shares scenes and regions
with training, so it is systematically easier.

### pos_weight — settled empirically (§4.4)
Fold 0, test metrics:

| pos_weight | P | R | F1 | IoU |
|---|---|---|---|---|
| 3.0 | 0.709 | 0.715 | 0.712 | 0.553 |
| **1.0** | **0.798** | **0.901** | **0.846** | **0.734** |

The sampler alone rebalances correctly; the extra pixel weight over-corrects.
`pos_weight=1.0` wins on every metric. Both JSON files kept — this is a
justified choice for §4.4, not an arbitrary one.

### REFERENCE ARM — complete and frozen

5 folds, `pos_weight=1.0`, 30 epochs, ~25 s/epoch.

| fold | P | R | F1 | IoU | test tiles |
|---|---|---|---|---|---|
| 0 | 0.779 | 0.850 | 0.813 | 0.685 | 258 |
| 1 | 0.555 | 0.625 | 0.588 | 0.416 | 219 |
| 2 | 0.510 | 0.934 | 0.660 | 0.492 | 248 |
| 3 | 0.798 | 0.896 | 0.844 | 0.730 | 340 |
| 4 | 0.694 | 0.908 | 0.787 | 0.649 | 262 |

**F1 = 0.738 ± 0.110    IoU = 0.595 ± 0.134**

Checkpoints: `results/reference_fold{0..4}.pt` — **frozen, never retrained.**

Fold-to-fold spread is large (0.588 to 0.844): different fires, different
regions. Fold 1 is hardest, with the biggest val→test gap (0.919 → 0.588) and
the smallest test set. Fold 2 over-predicts heavily (P 0.510, R 0.934) despite
`pos_weight=1.0`.

### CONSEQUENCE FOR WEEK 3 — the comparison must be paired
The ±0.110 fold spread is plausibly **larger than the compression effect being
measured**. A pooled comparison of reference mean against compressed mean would
bury a 0.05 effect inside fold-to-fold variance.

→ compute ΔF1 **within** each fold (same frozen checkpoint, reference vs
  compressed), then average the five paired differences and report their own
  standard deviation. Paired differences cancel fold difficulty. Goes in §4.6.

### Known issue
The `--pos-weight` warning in `04_train_reference.py` misfires: it advises
trying `pos_weight 1.0` even when already at 1.0. Fired on folds 2 and 4.
Make it conditional on `args.pos_weight > 1.0`.

### Week 2 status
- ✅ `bands.py`, `codec.py`, `data.py`, `detector.py`
- ✅ data layer verified
- ✅ reference arm trained and frozen, 5 folds
- ❌ **Writing: §2.1–2.4, 19 paragraphs, due 21 Oct — 1 drafted**

### Next — Week 3
- `compress.py`: encode → tokens → decode as an arm
- compressed arm (both dates), asymmetric arm (date 2 only)
- null A (two decoder seeds), null B (original vs its own reconstruction)
- **the floor number** — the gate that can reshape the thesis
- all comparisons paired within fold
