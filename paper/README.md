# Paper draft — Faithful-by-Concept (MICAD 2026)

`main.tex` is the manuscript draft, populated with our real experiment numbers.

## Compile
- **Overleaf (recommended):** new project → upload `main.tex` → set compiler to
  pdfLaTeX. For the real submission, switch `\documentclass` to Springer's
  **LNEE/llncs** class (Overleaf has the Springer template; drop `main.tex`'s body in).
- The qualitative figure is referenced at `../artifacts/figures/qualitative_melanoma.png`
  — generate it with `scripts/make_report.py` and copy it next to `main.tex` (or fix
  the path) before compiling.

## Status: all `\todo`s filled (final numbers with 95% CIs are in the draft).

## Key numbers in the draft (observed, with 95% CI)
- E1 (melanoma detection): image-only AUROC 0.830; CBM 0.852; oracle 0.914; concept AUROC 0.827 (vs 0.644 zero-shot).
- E2: pure reliance 0.223 [0.212,0.236] vs leaky 0.000; Δ significant. ccf_corr magnitude-blindness noted.
- E3 (honest): NO significant within-regime dark-vs-light faithfulness gap (equitable). Diverse
  training significantly improves V–VI faithfulness vs biased (0.178→0.217, non-overlapping CIs).
  Group-conditional calibration was NOT effective (reported as a negative result).
- E4a: DermLIP > DINOv2 on concepts (0.827 vs 0.782) and faithfulness (0.223 vs 0.155) at equal dx AUROC.

## Honesty note
The fairness story is "reasoning is equitable; data diversity — not calibration — is the lever."
Do NOT reintroduce any claim that the calibration mitigation shrinks the gap; it does not.
