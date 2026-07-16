# RUNBOOK — how to run Faithful-by-Concept on Kaggle

Everything runs in **one Kaggle Notebook**. You paste each cell below **in order**.
Code lives on GitHub (github.com/Vidhyan11/micad); the notebook clones it and pulls
updates. This file is the single source of truth for *what to run*.

---

## 0. One-time Kaggle setup (before any code)

Do this once in the Kaggle notebook editor (right-hand panel):

1. **Accelerator → GPU** (Settings ⚙ → Accelerator → GPU T4 x2 or P100).
2. **Internet → On** (Settings ⚙ → Internet → On). Needed to clone the repo and
   download encoder weights.
3. **Add data** — attach the 3 datasets (⊕ Add Input → search → Add):
   - `menakamohanakumar/derm7pt`
   - `orvile/pad-ufes-20`
   - `nazmusresan/fitzpatrick17k`
   (PH2 is not used — don't need it.)

> The datasets mount under `/kaggle/input/...`. Our code finds them automatically
> by marker files, so the exact folder nesting doesn't matter.

---

## 1. Setup cell — clone the code (run once per session)

Run this first, every time you start/restart the notebook session:

```python
!rm -rf /kaggle/working/micad
!git clone -q https://github.com/Vidhyan11/micad.git /kaggle/working/micad
import sys; sys.path.insert(0, "/kaggle/working/micad/src")
import fbc, fbc.config as C
C.ensure_dirs()
print("fbc", fbc.__version__, "| ON_KAGGLE:", C.ON_KAGGLE)
print("datasets found:", C.available_datasets())
```

✅ Expect: `datasets found: {'derm7pt': True, ..., 'fitzpatrick17k': True, 'pad_ufes_20': True}`.

**To pull my latest code updates later** (without re-cloning), run:
```python
!cd /kaggle/working/micad && git pull -q
```

---

## 2. Verify the data loaders (quick sanity check)

```python
!python /kaggle/working/micad/scripts/verify_data.py
```

✅ Expect: each dataset loads, images resolve on disk, sane class/Fitzpatrick counts.

---

## 3. Extract frozen-encoder embeddings (ME) — GPU

**3a. Smoke test** (8 images/dataset — confirms GPU + model load):
```python
!python /kaggle/working/micad/scripts/extract_embeddings.py --encoder dinov2 --limit 8
```
✅ Expect: `device=cuda`, shapes `(8, 768)`, `failed images=0`.

**3b. Full extraction** (all images; a few minutes — Fitzpatrick is ~16.5k):
```python
!python /kaggle/working/micad/scripts/extract_embeddings.py --encoder dinov2
```
✅ Expect shapes ≈ `(1011,768)`, `(2298,768)`, `(16574,768)`.

> Later we'll also run `--encoder dermlip` (primary derm encoder). One line each;
> embeddings cache separately per encoder.

---

## 4. Leakage-clean splits + dedup (MD part 3)

```python
!python /kaggle/working/micad/scripts/make_splits.py --encoder dinov2
```
✅ Expect: near-duplicates dropped, train/val/test counts, and a
`split × Fitzpatrick group` crosstab (confirms dark-skin cases in the test split).

---

## 5. Saving your work (IMPORTANT)

Outputs (embeddings, splits) are written to `/kaggle/working/artifacts/`, which is
**wiped when the session restarts** unless you save a notebook version:

- Click **Save Version → Save & Run All (Commit)** to persist `/kaggle/working`.
- Cached embeddings then survive, so you never re-extract.
- To chain notebooks, attach a committed notebook's output via **Add Input →
  Notebook Output**.

Re-extraction is only ~5 min, so this is convenience, not critical.

---

## 6. DermLIP embeddings — primary derm encoder (GPU)

DermLIP is a CLIP-style dermatology model (needed for pseudo-labels; also our
primary CBM encoder). Install open_clip first, then extract:

```python
!pip install -q open_clip_torch
!cd /kaggle/working/micad && git pull -q
!python /kaggle/working/micad/scripts/extract_embeddings.py --encoder dermlip
```
✅ Expect shapes like `(1011, 512)` etc. (DermLIP dim differs from DINOv2's 768).
Then re-run splits so DermLIP artifacts carry the same protocol:
```python
!python /kaggle/working/micad/scripts/make_splits.py --encoder dermlip
```

## 7. Foundation pseudo-labels + validation (MP)

```python
!python /kaggle/working/micad/scripts/pseudolabel.py --encoder dermlip
```
✅ Expect: a **derm7pt zero-shot concept recovery** table (AUROC/AP per dermoscopic
concept vs real GT — the evidence that bootstrapping works), plus clinical
pseudo-concept positive-rates for Fitzpatrick/PAD. Paste this output.

---

## 8. Train the two CBMs — Experiment 1 (MM)

```python
!cd /kaggle/working/micad && git pull -q
!python /kaggle/working/micad/scripts/train.py --encoder dermlip
```
Flags: `--joint` (joint-mode ablation), `--multiclass` (full multiclass instead of
the default concept-aligned binary detection).

✅ Expect an **Experiment 1 table**. Primary task is concept-aligned **binary
detection** — Model A: melanoma vs rest (what the 7-pt checklist is for); Model B:
malignant vs rest. Rows: image-only, CBM-sequential, and (Model A) CBM-oracle
(GT concepts → dx = concept-set ceiling). The CBM should now track image-only
closely. Paste this output.

---

## 9. Concept-counterfactual faithfulness — Experiment 2 (MF1)

```python
!cd /kaggle/working/micad && git pull -q
!python /kaggle/working/micad/scripts/faithfulness.py --encoder dermlip
```
Flags: `--importance {gradient,loo}`, `--cf_mode {flip,ablate}`.

✅ Expect an **Experiment 2 table** comparing the PURE bottleneck vs a LEAKY CBM on:
`ccf_corr` (stated↔causal agreement), `decisive_hit`, `comprehensiveness`,
`sufficiency` (lower better), `reliance` (how much the decision depends on
concepts). The pure bottleneck should show **much higher reliance /
comprehensiveness** — its decisions are genuinely concept-driven; the leaky model's
are not. Paste this output.

---

## 10. Fairness-of-reasoning audit + mitigation — Experiment 3 (MF3)

Two arms (run both — they are the paper's fairness story):

```python
!cd /kaggle/working/micad && git pull -q
# Arm 1 — BIASED training (light skin only): the realistic deployment failure mode
!python /kaggle/working/micad/scripts/fairness.py --encoder dermlip --train_groups I-II
# Arm 2 — DIVERSE training (all skin tones): the fix
!python /kaggle/working/micad/scripts/fairness.py --encoder dermlip
```

✅ Each prints a per-Fitzpatrick-group table (I-II / III-IV / V-VI) of diagnosis
accuracy AND reasoning faithfulness (reliance, comprehensiveness), BEFORE and AFTER
group-conditional concept calibration, with worst-vs-best GAPs. Story: Arm 1 should
show a gap on unseen dark skin; Arm 2 (diverse training) closes it; calibration is
the lightweight mitigation when retraining isn't possible. Paste both outputs.

---

## 11. Paper tables + qualitative figure (MR)

Run after E1/E2/E3 (needs their CSVs). Regenerates LaTeX tables + a
concept-counterfactual figure:
```python
!cd /kaggle/working/micad && git pull -q
!python /kaggle/working/micad/scripts/make_report.py --encoder dermlip
```
✅ Writes `results/*_table.tex` and `figures/qualitative_melanoma.png`. To download
from Kaggle: they're under `/kaggle/working/artifacts/` (Save Version to persist).

---

## 12. Encoder ablation — Experiment 4a

Prereq: DINOv2 embeddings + splits already done (§3–4); DermLIP done (§6);
pseudo-labels done (§7). Then:
```python
!cd /kaggle/working/micad && git pull -q
!python /kaggle/working/micad/scripts/ablation_encoder.py --encoders dinov2 dermlip
```
✅ Table of both models × both encoders — diagnosis, concept AUROC (Model A), and
faithfulness (reliance/comprehensiveness). Shows whether the derm foundation
encoder (DermLIP) beats the general one (DINOv2) on accuracy *and* faithful
reasoning. Paste this output.

---

## Steps coming next

- **Paper draft** — MICAD manuscript from the numbers above.

This RUNBOOK is updated as each step lands — `git pull` and re-read the bottom.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `datasets found: ... False` | Dataset not attached, or wrong slug. Re-add via ⊕ Add Input. |
| `git clone` auth error | Repo is private → make it public (Settings → Visibility), or use a token. |
| `device=cpu` | GPU not enabled → Settings → Accelerator → GPU, then restart session. |
| `ModuleNotFoundError: fbc...` | Re-run the Setup cell (Section 1). |
| Outputs gone after reopening | You didn't Save Version. Re-run from Section 3. |
