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

## Steps coming next (not yet runnable — I'll add cells here)

- **MP** — foundation pseudo-labels: bring up DermLIP, validate pseudo-concepts vs
  derm7pt GT, generate clinical concept labels for Fitzpatrick/PAD.
- **MM** — train the two CBMs (concept head → diagnosis bottleneck) + baselines
  (Experiment 1: diagnosis + concept accuracy).
- **MF1** — concept-counterfactual faithfulness (Experiment 2).
- **MF3** — fairness-of-reasoning audit + mitigation (Experiment 3).
- **MR** — tables + figures.

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
