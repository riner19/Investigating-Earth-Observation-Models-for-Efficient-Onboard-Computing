# Thesis structure — every chapter, every section

**Investigating Earth Observation Models for Efficient Onboard Computing**
BME VIK IIT · BMEVIIIAT03 · submission 11 December 2026, 12:00

Status as of 15 October 2026 (end of week 2).

---

## The headline

**29 of ~51 pages are writable right now** — all of chapter 2, all of chapter 3
except the profiling section, and all of chapter 4. None of it depends on an
experiment you have not yet run.

| Chapter | Pages | Writable today |
|---|---|---|
| 1 Introduction | 4 | no — write last |
| 2 Compression background | 12 | **yes, all of it** |
| 3 TerraMind and the S2L2A tokenizer | 8 | **yes, except §3.5** |
| 4 Method and experimental design | 10 | **yes, all of it** |
| 5 Results | 12 | no — weeks 3–6 |
| 6 Discussion and conclusions | 5 | no |

The writing is the thing that is behind, and it is not blocked on anything.

---

## File layout

Delete the template's demo chapters and create these in `content/`:

| File | Chapter |
|---|---|
| `introduction.tex` | 1 |
| `background.tex` | 2 |
| `terramind.tex` | 3 |
| `method.tex` | 4 |
| `results.tex` | 5 |
| `conclusion.tex` | 6 |

In `thesis.tex`:

```latex
\include{content/introduction}
\include{content/background}
\include{content/terramind}
\include{content/method}
\include{content/results}
\include{content/conclusion}
```

Label everything: `\label{sec:...}`, `\label{eq:...}`, `\label{fig:...}`,
`\label{tab:...}`. Then `\autoref{}` and `\eqref{}` work and you never type a
number by hand.

---

# Chapter 1 — Introduction

**4 pages · 5 sections · write LAST, in week 8**

Written last because the contributions list must match what you actually
delivered, not what you hoped to.

| § | Title | ¶ | Content |
|---|---|---|---|
| 1.1 | Motivation | 3 | Acquisition volume versus downlink capacity. Concrete Sentinel-2 figures. The bottleneck as the problem the thesis addresses. |
| 1.2 | Onboard compression today | 2 | CCSDS 122, JPEG2000, the 1–4 bpppb operating range. One page, no detail — ch. 2 does the theory. |
| 1.3 | Semantic compression as an alternative | 2 | Foundation models encode to discrete tokens orders of magnitude smaller. Pose the question. |
| 1.4 | Objectives and contributions | 1 | Numbered list, 4–5 items, one sentence each. Write only after ch. 5 exists. |
| 1.5 | Structure of the thesis | 1 | One sentence per chapter. |

---

# Chapter 2 — Compression background

**12 pages · 7 sections · 33 paragraphs · ALL WRITABLE NOW**

Pure literature. Nothing here can change based on your results, which is
exactly why it goes first.

### §2.1 Information and entropy — 1.5 pages, 5 ¶ — *1 drafted*

| ¶ | Content |
|---|---|
| 1 | Framing: compression as representation; lossless vs lossy; satellite data forces lossy |
| 2 | Entropy definition, two readings (expected bits, uncertainty) |
| 3 | **Source coding theorem — DRAFTED.** $\bar{L} \geq H(X)$, uniquely decodable codes, practical entropy coders |
| 4 | Notation table for chapters 3–5 |
| 5 | Why entropy alone is insufficient — the lossless bound sits above the downlink |

Equations: `eq:entropy`, `eq:source-coding-bound`. Cite `shannon1948`.

### §2.2 Source models — 1.5 pages, 5 ¶

| ¶ | Content |
|---|---|
| 1 | The memoryless assumption fails; every codec exploits correlation |
| 2 | Markov sources, conditional entropy, $H(X_n \mid X_{n-1}) \le H(X_n)$ |
| 3 | Spatial correlation in satellite imagery at 10 m |
| 4 | **Inter-band correlation** — often skipped, matters here. Forward-reference that the tokenizer encodes all 12 bands jointly, which is part of why its rate is reachable |
| 5 | Where structure runs out |

### §2.3 Lossy compression and rate–distortion — 1.5 pages, 4 ¶

| ¶ | Content |
|---|---|
| 1 | The rate–distortion function; the operating point is a design choice |
| 2 | Quantization as the only irreversible step — sets up §2.5 |
| 3 | MSE/PSNR vs task-level distortion. **The distinction the thesis turns on.** |
| 4 | Why lossless is not an option — needs a sourced ratio |

### §2.4 Classical onboard compression — 1.5 pages, 5 ¶

| ¶ | Content |
|---|---|
| 1 | Constraints onboard: power, radiation-tolerant CPUs, real-time, no retry. Note the encoder/decoder asymmetry — forward-reference §3.5 |
| 2 | CCSDS 122.0-B: wavelet + bit-plane encoder |
| 3 | JPEG2000 and Sentinel-2 — **verify the real onboard rate** |
| 4 | The 1–4 bpppb operating regime. **This is what makes 0.00453 interpretable.** |
| 5 | Scope statement: this thesis does not evaluate classical codecs, and why |

§2.5 ¶5 matters tonally. Framed as a scoping decision with a reason —
classical codecs cannot reach this rate, so matched-rate comparison is
impossible — it reads as judgement. Framed apologetically it reads as a gap.

### §2.5 Learned compression with discrete latents — 3 pages, 6 ¶

| ¶ | Content |
|---|---|
| 1 | The autoencoder framing; learned transform instead of a fixed one |
| 2 | VQ-VAE: codebook, nearest-neighbour lookup, commitment loss, straight-through estimator |
| 3 | Codebook collapse and the practical difficulties of VQ |
| 4 | **FSQ**: fixed grid in a low-dimensional projection, no learned codebook, no commitment loss |
| 5 | VQ vs FSQ compared — a table works well here |
| 6 | **State plainly that TerraMind uses FSQ, not VQ.** The single most common misreading of the model |

Cite `vandenoord2017`, `mentzer2023`.

### §2.6 Feature compression for split inference — 1.5 pages, 4 ¶

| ¶ | Content |
|---|---|
| 1 | The device–edge split: compress features, not images, when the receiver only needs the task output |
| 2 | BottleNet++ |
| 3 | FOOL applied to the satellite downlink |
| 4 | Position this work: the image-reconstruction path is taken here; feature-space compression is future work (§6.3) |

### §2.7 Change detection — 1.5 pages, 4 ¶

| ¶ | Content |
|---|---|
| 1 | The task; bitemporal co-registered pairs |
| 2 | Siamese fully-convolutional architectures: FC-EF, FC-Siam-conc, FC-Siam-diff |
| 3 | Burned-area detection specifically; spectral indices (NBR, dNBR, BAIS2) |
| 4 | Metrics under heavy class imbalance — why change-class F1 and IoU, not accuracy |

---

# Chapter 3 — TerraMind and the S2L2A tokenizer

**8 pages · 5 sections · §3.1–3.4 WRITABLE NOW, §3.5 blocked until week 5**

### §3.1 The model — 1.5 pages, 4 ¶

| ¶ | Content |
|---|---|
| 1 | TerraMind as a multimodal any-to-any generative EO model (IBM / ESA / FZ Jülich) |
| 2 | Modalities supported; pretraining on TerraMesh |
| 3 | **The backbone is not a codec; the tokenizers are.** This distinction organises the chapter |
| 4 | Why the tiny/base/large variants are irrelevant here — only the tokenizer is used |

### §3.2 The S2L2A tokenizer — 2 pages, 5 ¶

| ¶ | Content |
|---|---|
| 1 | FSQ with five dimensions, codebook of 15,360, pretrained 100 epochs on 9M S-2 L2A images |
| 2 | Patch size 16; 224×224 → 14×14 grid → 196 tokens, jointly across 12 bands |
| 3 | One token covers 160 m × 160 m — recurs in ch. 5 on detectable change size |
| 4 | The API as established: `encode` returns (latent (1,5,14,14), scalar, tokens (1,14,14) int32) |
| 5 | **The `model(x)` trap**: it ignores `timesteps` and uses 50. Worth stating — it would invalidate a timing comparison |

### §3.3 Rate analysis — 1.5 pages, 5 ¶ — **all numbers in hand**

| ¶ | Content |
|---|---|
| 1 | What the tokenizer produces |
| 2 | Bit cost derivation: $\log_2 15360 \approx 13.907$; 2726 bits = 341 bytes |
| 3 | Normalised rate: 0.00453 bpppb. Define bpppb formally |
| 4 | Ratios table: 2651× vs 12-bit, 3534× vs 16-bit. **Caveat: computed against the original tile, not an upsampled tensor** |
| 5 | Operational context: two to three orders of magnitude below 1–4 bpppb |

Also state the uniform-code assumption: 2726 bits is an upper bound, entropy
coding the token stream would reduce it, and this is deliberately not pursued
because it changes the rate without changing information content.

**Table 3.1** — rate and ratio summary.

### §3.4 The decoder — 1.5 pages, 4 ¶

| ¶ | Content |
|---|---|
| 1 | Diffusion-based decoding; `timesteps` controls detail |
| 2 | **It samples rather than reconstructs** — a plausible image consistent with the tokens |
| 3 | Consequence 1: pixel metrics understate what survives. Forward-reference §5.1 |
| 4 | Consequence 2: identical tokens, different seeds → different images. **Measured: MAE 235.7 vs 219.6 DN on the same tokens, ≈7% of the error.** This paragraph justifies the null-pair experiment |

### §3.5 Computational cost — 1.5 pages, 4 ¶ — **BLOCKED until week 5**

| ¶ | Content |
|---|---|
| 1 | Encoder vs decoder asymmetry: only the encoder flies |
| 2 | **Encoder-only** parameter count, memory, latency. Note 289,290,693 is encoder+decoder and must not be quoted as an onboard figure |
| 3 | Jetson Orin Nano measurements, fp16 and int8 |
| 4 | Relation to a realistic onboard compute budget |

---

# Chapter 4 — Method and experimental design

**10 pages · 7 sections · ALL WRITABLE NOW**

This is the reproducibility chapter and the one where your week-1 and week-2
decisions pay off. Everything here is settled.

### §4.1 Pipeline overview — 1 page, 3 ¶

One figure plus prose. Source imagery → shared preprocessing → arm-specific
compression → frozen detector → metrics. Emphasise that both arms traverse one
shared preprocessing function, so any difference is attributable to
compression alone.

**Figure 4.1** — the pipeline, all five arms.

### §4.2 Data — 1.5 pages, 5 ¶ — **all counts in hand**

| ¶ | Content |
|---|---|
| 1 | CaBuAr: Sentinel-2 L2A pre/post fire pairs, CAL FIRE masks. Cite `regecambrin2024` |
| 2 | **`raw/` not `normalized/`** — the authors' scaling would compose with TerraMind's standardisation and put the input off-distribution |
| 3 | 534 groups, **356 with both acquisitions**; the other 178 are post-fire only and support delineation, not bitemporal CD |
| 4 | Nodata screening: 1424 tiles → **1327** (97 dropped, 6.8%); threshold 5%, two forms (spatial edges and whole-band zeros) |
| 5 | Folds: 64 / 54 / 62 / 85 / 65 scenes — **a 57% spread; state that error bars rest on uneven sample sizes** |

**Table 4.1** — dataset composition after each exclusion.

### §4.3 Preprocessing — 1.5 pages, 7 ¶ — **the reproducibility section**

| ¶ | Content |
|---|---|
| 1 | Axis order (H,W,C) → (C,H,W); uint16 DN, no scaling in `prepare()` |
| 2 | Band selection: 12 bands map directly to S2L2A; B10 absent from L2A. A 13-band L1C source would need it dropped |
| 3 | Tiling: 2×2 non-overlapping, centred, 32 px border dropped. Alternatives rejected and why |
| 4 | **Value scaling** — the key paragraph. $z = (x-\mu)/\sigma$ in, $\hat{x} = \hat{z}\sigma + \mu$ out |
| 5 | The saturation evidence: 0.211 / 0.144 for raw DN vs **0.000 / 0.000** standardised; 0.005 on real data |
| 6 | State that this was undocumented and established experimentally |
| 7 | The shared-path guarantee: every arm consumes one `prepare()` |

**Table 4.2** — the 12-element pretraining mean and std vectors.
**Table 4.3** — the saturation comparison.

¶5–6 are where this section earns marks. Most theses would write "the input is
standardised" and move on.

### §4.4 Change detection model — 1.5 pages, 6 ¶ — **all numbers in hand**

| ¶ | Content |
|---|---|
| 1 | FC-Siam-diff via torchgeo, `in_channels=12`, 24,464,738 parameters. Cite `daudt2018` |
| 2 | Loss: weighted cross-entropy + soft Dice |
| 3 | **Class imbalance**: 3.89% burned pixels, 70% of tiles entirely unburned. WeightedRandomSampler targeting ~50% positive, **training only** |
| 4 | **`pos_weight` settled empirically**: 3.0 → F1 0.712, 1.0 → F1 0.846. The sampler alone rebalances; extra weighting over-corrects |
| 5 | **Checkpoint selection on a scene-level validation split**, 12% of train folds. Split by scene because adjacent tiles share fire, terrain and acquisition. Test touched once |
| 6 | The checkpoint is frozen and shared by every arm. A retrained detector measures domain adaptation, not information loss |

**Table 4.4** — the `pos_weight` comparison.

¶4 is a justified choice rather than an arbitrary one. ¶5 is a bias you found
and removed — say so.

### §4.5 Experimental configurations — 2 pages, 7 ¶

| ¶ | Content |
|---|---|
| 1 | The five-arm table (slot 1, slot 2, real change present) |
| 2 | Reference — the ceiling |
| 3 | Fully compressed — harshest, the headline |
| 4 | Asymmetric — the operationally realistic case; archive at full quality, only the new acquisition compressed |
| 5 | Null A — decoder stochasticity, two seeds on identical tokens |
| 6 | Null B — the domain gap, original vs its own reconstruction |
| 7 | **The pairing**: Null A matches fully-compressed's input structure, Null B matches asymmetric's. Each arm gets a floor measured under its own conditions |

**Table 4.5** — the configuration table.
**Figure 4.2** — example tiles from each arm.

¶7 is the subtle part and the thing a careless reader would get wrong.

### §4.6 Metrics — 1.5 pages, 5 ¶

| ¶ | Content |
|---|---|
| 1 | Change-class precision, recall, F1, IoU — formally defined. Why not accuracy |
| 2 | Absolute ΔF1 and retention ratio $F1_{\text{comp}} / F1_{\text{ref}}$ |
| 3 | Floor-corrected comparison: each arm against its matched null |
| 4 | **Paired within fold.** Reference F1 spread is ±0.110 across folds, plausibly larger than the compression effect. A pooled comparison would bury a 0.05 effect in fold variance |
| 5 | Fidelity metrics: per-band MAE, NBR error. Why not PSNR alone |

¶4 is the methodological point that justifies the whole analysis design.

### §4.7 Rate–quality parameters — 1 page, 4 ¶

| ¶ | Content |
|---|---|
| 1 | Why an internal rate axis is needed given no classical baseline |
| 2 | Diffusion step count: same bitrate, varying reconstruction quality |
| 3 | Input resolution: 224 → 196 tokens, 112 → 49 tokens. Note 112 is off the training resolution |
| 4 | Token reduction / masking |

---

# Chapter 5 — Results

**12 pages · 7 sections · BLOCKED until weeks 3–6**

Experiments freeze 25 November. Nothing enters this chapter afterwards.

| § | Title | Pages | Needs |
|---|---|---|---|
| 5.1 | Reconstruction quality | 2 | week 3 — partial data exists (one tile) |
| 5.2 | **The false-change floor** | 2 | week 3 |
| 5.3 | **Primary comparison** | 3 | weeks 3–4 |
| 5.4 | Rate–quality parameters | 2 | week 5 |
| 5.5 | Computational profiling | 1 | week 5 |
| 5.6 | Second dataset or fine-tuning | 1 | week 6 |
| 5.7 | Summary of findings | 1 | week 7 |

**§5.2 goes before §5.3.** The floor has to be on the page before any accuracy
result, because everything afterwards is read against it.

**§5.7** is 3–5 numbered findings, one sentence each with its supporting
number. These become the contributions list in §1.4.

Already in hand for §5.1: the round-trip figure, per-band MAE, the dynamic
range compression [87,5912] → [318,4473], and the hallucination example.

---

# Chapter 6 — Discussion and conclusions

**5 pages · 4 sections · write in week 8**

### §6.1 Interpretation — 1.5 pages

Is semantic tokenization viable for change detection at this rate, and under
what conditions? **Give a direct answer.** Examiners reward a clear verdict
over hedging.

### §6.2 Limitations — 1.5 pages

Be thorough and unprompted; this section earns marks:

- L2A as a proxy for onboard data — atmospheric correction happens on the
  ground, so a deployed system would compress L0/L1 radiometry
- the generative decoder makes reconstruction non-deterministic
- a single primary dataset with one change type (coarse, high-contrast burn)
- small test sets, uneven folds, large fold-to-fold variance
- one detector architecture — results may not generalise
- no classical codec comparison (scope, §2.4)

### §6.3 Future work — 1 page

Feature-space compression following BottleNet++/FOOL, skipping image
reconstruction entirely. A tokenizer trained on L0/L1 radiometry. Encoder
quantization for flight hardware. Multi-temporal tokenization.

### §6.4 Conclusion — 1 page

Restate the question, state the answer, state the evidence.

---

# Front and back matter

| Item | Where | Notes |
|---|---|---|
| **Kivonat** (Hungarian) | `abstract.tex` | **Mandatory**, ~250 words. Get a native speaker to check it |
| **Abstract** (English) | `abstract.tex` | ~250 words. Write this first, then translate |
| Student declaration | `include/declaration_EN.tex` | Already wired in; name comes from `thesis.tex` |
| **Generative AI declaration** | `content/appendices.tex` | **Required.** Already included via `\input{include/\AIdeclaration}` |
| Acknowledgement | `acknowledgement.tex` | Half a page |
| Appendix | `appendices.tex` | Delete the demo TeXstudio screenshot and the "42" equations |

---

# Figures and tables

### Existing

| Ref | Content | Where |
|---|---|---|
| Fig 5.1 | `results/roundtrip_compare.png` | §5.1 |

### To make — writable now

| Ref | Content | § |
|---|---|---|
| Fig 2.1 | Rate–distortion curve sketch | 2.3 |
| Fig 2.2 | VQ vs FSQ quantization diagram | 2.5 |
| Fig 2.3 | Siamese architecture | 2.7 |
| Fig 3.1 | Tokenizer encode/decode block diagram | 3.2 |
| Fig 3.2 | A tile with its 14×14 token grid | 3.2 |
| Fig 4.1 | Full pipeline, five arms | 4.1 |
| Fig 4.2 | Example tiles per arm | 4.5 |
| Tab 3.1 | Rate and compression ratios | 3.3 |
| Tab 4.1 | Dataset composition after exclusions | 4.2 |
| Tab 4.2 | Pretraining mean/std vectors | 4.3 |
| Tab 4.3 | Saturation comparison | 4.3 |
| Tab 4.4 | `pos_weight` comparison | 4.4 |
| Tab 4.5 | Experimental configurations | 4.5 |
| Tab 4.6 | Reference arm, five folds | 4.4 or 5.3 |

### Blocked

Fig 5.2 hallucinated change · Fig 5.3 primary results with error bars ·
Fig 5.4 rate–quality curves · Tab 5.1 floor · Tab 5.2 primary comparison ·
Tab 5.3 profiling

---

# Writing order

The plan's schedule, with what is actually unblocked:

| Week | Dates | Write |
|---|---|---|
| 2 | by 21 Oct | **§2.1–2.4** (19 ¶) |
| 3 | 22–28 Oct | **§2.5–2.7** (14 ¶) |
| 4 | 29 Oct–4 Nov | **§3.1–3.4** |
| 5 | 5–11 Nov | **§4.1–4.4** |
| 6 | 12–18 Nov | **§4.5–4.7**, §3.5 |
| 7 | 19–25 Nov | §5.1–5.4 |
| 8 | 26 Nov–2 Dec | §5.5–5.7, ch. 6, ch. 1, abstracts — **draft to Barcza by 29 Nov** |
| 9 | 3–9 Dec | Revision, AI declaration, formatting |
| — | 10 Dec | **Submit** |

Chapters 2, 3 and 4 are all unblocked today. If a week frees up, pull work
forward — every page written before the results arrive is a page you are not
writing in December.

---

# Three figures still to source

Needed before §2.3 and §2.4 can be finished:

1. Sentinel-2's actual onboard compression scheme and bitrate — Sentinel-2
   User Handbook
2. A cited lossless compression ratio for multispectral satellite imagery
3. The operational bpppb range for CCSDS 122

The 1–4 bpppb figure is already asserted in your task description and needs a
reference behind it. About an hour of reading.
