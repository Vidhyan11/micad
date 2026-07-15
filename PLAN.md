# Implementation Plan — Faithful-by-Concept (MICAD 2026)

Scope: **Full** (contributions 1–3). Target environment: **Kaggle-native** (notebooks driving a reusable `src/fbc` package). Everything runs on one Kaggle GPU with **frozen encoders** — we only ever train small MLP heads.

This document maps each proposal contribution to concrete code modules, data flows, and paper tables. It is the source of truth for the build.

---

## 0. Design principles (non-negotiable)

1. **True bottleneck.** The diagnosis head sees **only concept scores**, never the raw embedding. This is what makes faithfulness meaningful — if the label could leak through the embedding, "concept explanations" would be theatre.
2. **Frozen encoders.** Encoder weights never update. Embeddings are extracted **once** and cached to disk (`.npy`/`.pt`). All training reads cached embeddings → runs are minutes, not hours.
3. **Leakage-clean from day one.** Lesion-/patient-level splits + near-duplicate removal *before* any training. No metric is trusted until splits are audited.
4. **Inference-only faithfulness.** The counterfactual test flips a concept *in concept space* and re-runs the tiny diagnosis head. No image generator, no diffusion, no backbone gradients required.
5. **Reproducible.** Fixed seeds, config-driven runs, results written to CSV + LaTeX so tables regenerate deterministically.

---

## 1. Repository structure

```
micad/
  PROPOSAL.md
  PLAN.md                     # this file
  README.md                   # quickstart + Kaggle instructions
  requirements.txt
  pyproject.toml              # so `pip install -e .` works locally & on Kaggle
  .gitignore                  # artifacts/, *.npy, *.pt, __pycache__

  src/fbc/                    # the package (attach to Kaggle as a utility dataset)
    config.py                 # paths, concept vocab, dataset slugs, hyperparams
    data/
      concepts.py             # canonical concept vocabulary + cross-dataset mapping
      derm7pt.py              # loader -> records(image, concepts_gt, dx, group, split)
      ph2.py
      fitzpatrick17k.py
      pad_ufes.py
      splits.py               # patient/lesion splits + dedup (SelfClean-style hashing)
    encoders/
      base.py                 # Encoder API: embed(images)->(N,d); (optional) zeroshot_concepts
      dinov2.py               # fallback general encoder
      monet.py                # derm CLIP; also zero-shot concept scoring for pseudo-labels
      dermlip.py              # derm CLIP (ICCV'25) — primary
    models/
      concept_head.py         # MLP: emb -> concept logits
      diagnosis_head.py       # MLP: concept scores -> dx logits  (BOTTLENECK)
      cbm.py                  # wiring + baselines (image-only, post-hoc CBM)
    train/
      losses.py               # concept BCE + dx CE (+ optional faithfulness reg)
      train_cbm.py            # sequential/joint training loop over cached embeddings
    faithfulness/
      importance.py           # stated importance: |weight|, grad, leave-one-out
      counterfactual.py       # intervene on concept c, measure decision change
      metric.py               # per-case + dataset faithfulness score (rank correlation)
    fairness/
      audit.py                # per-Fitzpatrick concept-acc + faithfulness gaps
      calibrate.py            # group-conditional concept calibration (mitigation)
    eval/
      metrics.py              # AUROC/F1/bal-acc/concept-AP/ECE
      tables.py               # emit CSV + LaTeX for every paper table
      figures.py              # qualitative counterfactual panels
    utils/
      seed.py, io.py, logging.py

  notebooks/                  # thin drivers; each imports fbc and produces one artifact
    01_extract_embeddings.ipynb
    02_pseudolabels.ipynb
    03_train_cbm.ipynb
    04_faithfulness.ipynb
    05_fairness.ipynb
    06_tables_figures.ipynb

  experiments/configs/*.yaml  # one config per encoder/ablation
  artifacts/                  # embeddings cache, checkpoints, results (gitignored)
```

**Kaggle workflow:** push `src/fbc` as a Kaggle *utility dataset* (or `pip install` from a wheel), attach the 4 data datasets, and run notebooks `01→06`. Each notebook writes to `/kaggle/working` and can be chained via "notebook output as input".

---

## 2. Data layer

### 2.1 Canonical concept vocabulary (`data/concepts.py`)
Define one shared concept vocabulary so every dataset maps into the same space. Anchor on the **7-point checklist** (derm7pt is our GT source):

| # | Concept | derm7pt | PH2 | Foundation pseudo-label |
|---|---|---|---|---|
| 1 | Pigment network (typical/atypical) | GT | GT | zero-shot |
| 2 | Blue-whitish veil | GT | GT | zero-shot |
| 3 | Vascular structures | GT | – | zero-shot |
| 4 | Pigmentation | GT | – | zero-shot |
| 5 | Streaks | GT | GT | zero-shot |
| 6 | Dots & globules | GT | GT | zero-shot |
| 7 | Regression structures | GT | GT | zero-shot |
| (+) | Asymmetry (ABCD) | – | GT | zero-shot |

`concepts.py` exposes `CONCEPTS` (ordered list), per-dataset column maps, and helpers to build a fixed-length concept vector with a mask for "unavailable in this dataset."

### 2.2 Per-dataset loaders
Each loader returns a uniform record list / DataFrame with columns:
`image_path, concepts_gt (vector+mask), diagnosis, fitzpatrick (or NaN), patient_id/lesion_id, source`.

- **derm7pt** (`menakamohanakumar/derm7pt`): primary. Parse the official metadata CSV → 7-pt concept GT + diagnosis; paired dermoscopic+clinical images. Use the **official train/val/test split** if present, else patient-level split.
- **PH2** (`jamesgoydos/...ph2-data`): 200 images, dermoscopic feature annotations + masks → concept validation set.
- **Fitzpatrick17k** (`mobaswiralfarabi/fitzpatrick17k`): FST I–VI labels, 114 conditions → fairness. No concept GT → pseudo-labels.
- **PAD-UFES-20** (`orvile/pad-ufes-20`): 2.3k smartphone clinical images + rich metadata incl. Fitzpatrick and clinical flags → fairness + real dark-skin cases.

### 2.3 Splits & dedup (`data/splits.py`)
- **Patient/lesion-level** grouping so the same lesion never spans train/test (Cassidy 2021).
- **Near-duplicate removal** via perceptual/embedding hashing (SelfClean-style): compute cosine similarity on encoder embeddings, drop near-dupes across split boundaries.
- Emit a `splits_report.json` (counts per class, per FST group, dropped dupes) that we cite in the paper's protocol section.

**Milestone D:** loaders + splits + dedup pass an audit notebook cell (no ID overlap, dup rate reported).

---

## 3. Encoders (frozen) — `encoders/`

Common `Encoder` API:
```python
class Encoder:
    dim: int
    def embed(self, images) -> np.ndarray            # (N, dim), L2-normalized option
    def zeroshot_concepts(self, images, prompts) -> np.ndarray   # optional (CLIP-like)
```

- **DermLIP** (primary), **MONET** (primary + zero-shot pseudo-labeler), **DINOv2** (fallback/general baseline). Load HF/torch-hub weights, `eval()`, `no_grad`.
- `01_extract_embeddings.ipynb`: for each dataset × encoder, extract and cache `emb_{dataset}_{encoder}.npy` + aligned `meta_{dataset}.parquet`. **Run once.**

**Milestone E:** cached embeddings for all datasets × {DINOv2, MONET, DermLIP}.

---

## 4. Pseudo-labels (`02_pseudolabels.ipynb`, uses `encoders/monet.py`)

For datasets without concept GT (Fitzpatrick17k, PAD-UFES-20, and to augment others):
- Use MONET/DermLIP **zero-shot concept scoring**: for each concept, a positive/negative text prompt pair → similarity → concept probability.
- Calibrate prompt quality against derm7pt/PH2 GT (report pseudo-label AUROC vs GT) — this becomes the **"pseudo-label quality" ablation** in the paper.
- Cache `pseudo_concepts_{dataset}.npy`.

**Milestone P:** pseudo-label vs GT agreement table (validates the bootstrapping claim, contribution 2).

---

## 5. Model — `models/`

- **`concept_head`**: MLP `emb (d) → hidden → concept_logits (k)`. Trained with masked BCE (mask = concept-available).
- **`diagnosis_head`**: MLP `concept_probs (k) → hidden → dx_logits (C)`. **Input is concepts only** — the bottleneck.
- **`cbm.py`** wires them and defines training modes:
  - **Sequential** (default, most faithful): train concept head to convergence, freeze it, then train diagnosis head on predicted concepts.
  - **Joint** (ablation): train both with `L = λ·concept_BCE + dx_CE`.
- **Baselines** (for tables):
  - **Image-only** MLP: `emb → dx` (no concepts) — accuracy ceiling, zero interpretability.
  - **Plain CBM**: same architecture, trained for accuracy only (no faithfulness consideration).
  - **Post-hoc CBM** (Yuksekgonul ICLR'23 style): fit concept vectors post-hoc on a frozen image-only model — the natural faithfulness comparison.

**Milestone M:** `03_train_cbm.ipynb` produces checkpoints + a diagnosis/concept-accuracy table competitive with image-only (Experiment 1).

---

## 6. Contribution 1 — Concept-counterfactual faithfulness — `faithfulness/`

This is the headline metric. Two ingredients per case, per concept `c`:

**(a) Stated importance `I_c`** — what the model *claims* it used. Compute via (configurable, report ≥2):
- learned weight magnitude of the diagnosis head w.r.t. `c`,
- gradient `∂ŷ/∂ĉ_c` at the case,
- leave-one-concept-out drop in predicted class prob.

**(b) Measured counterfactual effect `E_c`** — intervene in concept space: set `ĉ_c` to its counterfactual value (flip present↔absent; or push to the opposite class-conditional mean), **re-run the diagnosis head**, measure change in predicted class probability / decision flip.

**Faithfulness score** (`metric.py`):
- Per case: rank correlation (Spearman/Kendall) between `{I_c}` and `{E_c}` across concepts → **the concepts the model says matter are the ones that actually move the decision**.
- Dataset-level: mean per-case correlation + a **"decisive-concept hit rate"** (does the top-stated concept, when flipped, actually flip/most-change the decision?).
- Also report a **sufficiency/comprehensiveness** pair (ERASER-style): erasing top-k stated concepts vs erasing random-k.

**Experiment 2 (faithfulness table):** Faithful-by-Concept vs plain CBM vs post-hoc CBM vs a Grad-CAM proxy — show ours scores measurably higher.

Optional (nice-to-have, contribution 2 depth): add a **faithfulness regularizer** during training that aligns `I_c` with `E_c`, and show it improves the metric without hurting accuracy.

**Milestone F1:** `04_faithfulness.ipynb` emits Experiment-2 table + per-case records for figures.

---

## 7. Contribution 3 — Fairness of reasoning — `fairness/`

**Audit (`audit.py`):** group cases by Fitzpatrick {I–II, III–IV, V–VI} (Fitzpatrick17k, PAD-UFES-20). For each group report:
- diagnosis accuracy (context),
- **concept accuracy** (vs GT where available, else vs pseudo-GT with the caveat stated),
- **faithfulness score** (Section 6).
Report the **gap** (worst-group vs best-group) for concept accuracy and faithfulness — this is the novel finding: does the *reasoning* degrade on dark skin, not just accuracy?

**Mitigation (`calibrate.py`):** **group-conditional concept calibration** — fit per-group temperature/Platt scaling (or per-group affine recalibration) on the concept head outputs using a held-out calibration split, so concept probabilities are equally reliable across groups. Then re-run the audit and show the gap shrinks.

**Experiment 3 (fairness-of-reasoning table):** concept accuracy + faithfulness per FST group, before/after mitigation.

**Milestone F3:** `05_fairness.ipynb` emits Experiment-3 table (pre/post mitigation).

---

## 8. Evaluation, tables, figures — `eval/`

- **`metrics.py`**: AUROC, macro-F1, balanced accuracy (diagnosis); concept AP / AUROC; ECE (calibration); faithfulness metrics.
- **`tables.py`**: every paper table → CSV + LaTeX. Tables map 1:1 to Proposal §6:
  1. Diagnosis + concept accuracy (vs image-only, plain CBM).
  2. Faithfulness (vs post-hoc CBM, Grad-CAM proxy).
  3. Fairness-of-reasoning (per FST group, pre/post mitigation).
  4. Ablations (encoder {MONET/DermLIP/DINOv2}, pseudo-label quality, concept-set size).
  5. Qualitative panels.
- **`figures.py`**: per-case panel — predicted concepts, the counterfactual concept that flips the label, NL explanation; a faithful vs unfaithful contrast.

**Milestone R:** `06_tables_figures.ipynb` regenerates all 5 tables + qualitative figure from cached results.

---

## 9. Experiment matrix (what actually gets run)

| Exp | Datasets | Encoders | Output |
|---|---|---|---|
| E1 Diagnosis/concept acc | derm7pt (+PH2) | DermLIP (main) | Table 1 |
| E2 Faithfulness | derm7pt (+PH2) | DermLIP | Table 2 |
| E3 Fairness-of-reasoning | Fitzpatrick17k, PAD-UFES-20 | DermLIP | Table 3 |
| E4a Encoder ablation | derm7pt | MONET, DermLIP, DINOv2 | Table 4a |
| E4b Pseudo-label quality | derm7pt/PH2 (GT vs pseudo) | MONET/DermLIP | Table 4b |
| E4c Concept-set size | derm7pt | DermLIP | Table 4c |
| E5 Qualitative | all | DermLIP | Figure |

All seeded; each row is a config in `experiments/configs/`.

---

## 10. Milestones & order of work

1. **M0 Scaffold** — repo structure, `pyproject.toml`, `config.py`, `requirements.txt`, git init, `.gitignore`. *(no GPU)*
2. **MD Data** — 4 loaders + concept vocab + splits/dedup + audit. *(CPU; needs datasets attached)*
3. **ME Embeddings** — 3 encoders, cache embeddings for all datasets. *(GPU, run once)*
4. **MP Pseudo-labels** — zero-shot concept scoring + GT-agreement validation.
5. **MM Model** — concept/diagnosis heads + baselines + Experiment 1.
6. **MF1 Faithfulness** — importance + counterfactual + metric + Experiment 2.
7. **MF3 Fairness** — audit + calibration + Experiment 3.
8. **MR Reporting** — tables + figures, all regenerable.

Each milestone ends with a runnable notebook cell that produces a checkable artifact.

---

## 11. Open questions to resolve at each dataset (verify, don't assume)

- Exact column names / label encodings in each Kaggle dataset (confirm on first load).
- derm7pt: official split availability + how paired clinical/dermoscopic images are keyed.
- Fitzpatrick17k: current image availability (some URLs historically dead) — confirm the Kaggle mirror ships images.
- PAD-UFES-20: which metadata fields encode Fitzpatrick and any usable clinical concept flags.
- DermLIP/MONET: exact HF repo IDs, input preprocessing, and whether weights are downloadable inside a Kaggle no-internet session (may need to attach weights as a dataset).

These are resolved empirically in **MD/ME**, not guessed.

---

## 12. Risk register

| Risk | Mitigation |
|---|---|
| Foundation weights not reachable in offline Kaggle | Attach MONET/DermLIP weights as a Kaggle dataset; DINOv2 as fallback |
| Pseudo-labels too noisy | Validate vs GT (MP); fall back to GT-only concepts for core tables, pseudo for fairness only |
| Bottleneck hurts accuracy too much | Report the gap honestly; joint-training ablation; concept-set tuning |
| Fitzpatrick V–VI too few samples | Pool PAD-UFES-20 + Fitzpatrick17k; report CIs; state as limitation |
| Faithfulness metric contested | Report ≥2 importance definitions + ERASER suff/comp as triangulation |
```

Next steps are laid out as milestones M0→MR.

