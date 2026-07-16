# Faithful-by-Concept

**Verifiable & equitable concept reasoning for skin-lesion diagnosis.** Target venue: **MICAD 2026**.

> An AI that diagnoses skin lesions (e.g. melanoma) **and explains itself with the clues a
> dermatologist uses** (pigment network, blue-whitish veil, asymmetry, …) — then goes further:
> it **proves** those clues actually drive the decision, and **checks the reasoning is fair
> across skin tones**. Frozen foundation encoder + tiny trained heads → runs on one Kaggle GPU.

New here? Read **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** — a plain-English overview for beginners.

## What it does (3 contributions)
1. **Concept-counterfactual faithfulness** — flips a concept in concept space (no image
   generator) and measures whether the diagnosis changes as the explanation implies.
2. **Faithful-by-Concept model** — a concept bottleneck whose diagnosis head sees *only*
   concepts, with concept supervision bootstrapped from a dermatology foundation model.
3. **Fairness-of-reasoning audit** — the first test of whether concept faithfulness holds
   equally across Fitzpatrick skin tones.

## Headline results (honest)
- **Diagnosis:** the interpretable bottleneck **matches/beats** a black-box image model on
  melanoma detection (AUROC **0.85** vs 0.83) — interpretability at no accuracy cost.
- **Faithfulness:** our model genuinely relies on its concepts (**reliance 0.22**) while a
  "fake-interpretable" leaky model relies on them **≈ 0** — a statistically significant gap.
- **Fairness:** reasoning is **statistically equitable** across skin tones; **diverse training**
  (not post-hoc calibration) is the effective lever for dark-skin faithfulness.
- **Encoder:** a derm foundation encoder yields more faithful concepts than a general one.

## Documentation
| File | What |
|---|---|
| [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) | Beginner-friendly 7-part overview |
| [PROPOSAL.md](PROPOSAL.md) | Original research idea |
| [PLAN.md](PLAN.md) | Implementation plan / milestones |
| [RUNBOOK.md](RUNBOOK.md) | **Exact ordered cells to run on Kaggle** |
| [paper/main.tex](paper/main.tex) | Draft MICAD manuscript |

## Layout
```
src/fbc/        reusable package: data, encoders, models, faithfulness, fairness, eval
scripts/        pipeline drivers (verify_data, extract_embeddings, make_splits,
                pseudolabel, train, faithfulness, fairness, ablation_encoder, make_report)
experiments/    per-run YAML configs
paper/          LaTeX manuscript draft
artifacts/      embeddings cache, checkpoints, results, figures (gitignored)
```

## Running it
Everything runs in one Kaggle notebook (GPU + Internet ON). Follow **[RUNBOOK.md](RUNBOOK.md)**;
the short version:
```python
!git clone -q https://github.com/Vidhyan11/micad.git /kaggle/working/micad
import sys; sys.path.insert(0, "/kaggle/working/micad/src")
# then: verify_data → extract_embeddings → make_splits → pseudolabel →
#       train → faithfulness → fairness → ablation_encoder → make_report
```

### Datasets (Kaggle)
| Dataset | Role |
|---|---|
| `menakamohanakumar/derm7pt` | primary — real 7-point concept ground truth (Model A) |
| `nazmusresan/fitzpatrick17k` | fairness backbone — Fitzpatrick I–VI (Model B) |
| `orvile/pad-ufes-20` | clinical fairness check + partial concept GT |

*(PH2 was dropped — no mirror ships images and concept labels together.)*

## Local dev
```bash
pip install -e .
python -c "import fbc; print(fbc.__version__)"
```
Config auto-detects Kaggle vs local (`src/fbc/config.py`).
