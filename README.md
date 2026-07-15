# Faithful-by-Concept

Verifiable & equitable concept reasoning for skin-lesion diagnosis. Target venue: **MICAD 2026**.

> A concept-bottleneck dermatology diagnostic that (1) diagnoses through true dermatologist
> concepts, (2) *measures* whether those concepts causally drive the decision
> (concept-counterfactual faithfulness), and (3) audits + mitigates whether that faithfulness
> holds equally across skin tones.

See **[PROPOSAL.md](PROPOSAL.md)** for the research idea and **[PLAN.md](PLAN.md)** for the
implementation plan.

---

## Layout
```
src/fbc/        reusable package (data, encoders, models, faithfulness, fairness, eval)
notebooks/      thin Kaggle drivers 01..06, each produces one artifact
experiments/    per-run YAML configs
artifacts/      embeddings cache, checkpoints, results (gitignored)
```

## Running on Kaggle (internet ON)
1. Attach the 4 datasets to the notebook (see slugs below). They mount under `/kaggle/input/`.
2. Attach this repo (as a utility dataset) **or** `!pip install -e /kaggle/working/micad`.
3. Run notebooks in order: `01_extract_embeddings → 02_pseudolabels → 03_train_cbm →
   04_faithfulness → 05_fairness → 06_tables_figures`. Chain them via
   "Add data → Notebook output" so cached embeddings/checkpoints flow forward.

### Datasets (Kaggle slugs → mount path)
| Dataset | Slug | Mounts at |
|---|---|---|
| derm7pt | `menakamohanakumar/derm7pt` | `/kaggle/input/derm7pt` |
| PH2 | `jamesgoydos/melanoma-skin-lesion-id-ph2-data` | `/kaggle/input/melanoma-skin-lesion-id-ph2-data` |
| Fitzpatrick17k | `mobaswiralfarabi/fitzpatrick17k` | `/kaggle/input/fitzpatrick17k` |
| PAD-UFES-20 | `orvile/pad-ufes-20` | `/kaggle/input/pad-ufes-20` |

## Local dev
```bash
pip install -e .
python -c "import fbc; print(fbc.__version__)"
```

Config auto-detects Kaggle vs local (see `src/fbc/config.py`).
