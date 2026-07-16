# Paper draft — Faithful-by-Concept (MICAD 2026)

`main.tex` is the manuscript draft, populated with our real experiment numbers.

## Compile
- **Overleaf (recommended):** new project → upload `main.tex` → set compiler to
  pdfLaTeX. For the real submission, switch `\documentclass` to Springer's
  **LNEE/llncs** class (Overleaf has the Springer template; drop `main.tex`'s body in).
- The qualitative figure is referenced at `../artifacts/figures/qualitative_melanoma.png`
  — generate it with `scripts/make_report.py` and copy it next to `main.tex` (or fix
  the path) before compiling.

## Remaining `\todo`s (fill from Kaggle runs)
- Table 2: 95% CIs from `faithfulness.py`.
- Table 3: V–VI vs I–II bootstrap difference CIs from `fairness.py`.
- E4a: encoder-ablation table from `ablation_encoder.py`.

## Numbers currently in the draft (observed)
- E1 (melanoma detection): image-only AUROC 0.830; CBM 0.852; oracle 0.914; concept AUROC 0.827 (vs 0.644 zero-shot).
- E2: pure reliance 0.223 vs leaky ~0; ccf_corr magnitude-blindness noted.
- E3: biased V–VI bal-acc 0.682/reliance 0.178 → diverse 0.725/0.217.
