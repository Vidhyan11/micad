# Response to Reviewers — Faithful-by-Concept (MICAD 2026)

We thank both reviewers for their careful and constructive comments. All points are
addressed in the camera-ready; changes are summarized below with their locations.

---

## Reviewer 1

**R1.1 — Clarify how bootstrapped concepts are validated and how pseudo-concept errors
influence diagnosis and fairness.**
Added a *Pseudo-concept validation and error propagation* paragraph (Sec. 5.1). We now
state explicitly: (i) where ground truth exists, zero-shot scoring recovers dermoscopic
concepts at mean AUROC 0.64 and a supervised head at 0.827±0.001; (ii) the only
externally checkable clinical concept, PAD-UFES *elevation*, validates at AUROC 0.63;
(iii) pseudo-concept errors inflate Model B *concept-accuracy* numbers (which we
therefore never report as a validity measure), but our faithfulness metric is
*model-internal* and thus unaffected by pseudo-label noise; (iv) the one place such
noise could bias conclusions is fairness, which we now discuss in Sec. 6.

**R1.2 — State the equity conclusion cautiously; absence of significance may be low
power.**
Reworded throughout (Abstract, Sec. 5.3, Sec. 6, Conclusion). We now report "no
*detectable* disparity on the evaluated datasets" and explicitly note that the
darker-skin subgroup is small (n_test = 407) with higher variance, so the null may
reflect limited statistical power rather than proven equity.

**R1.3 — Report subgroup sizes, confidence intervals, and the Fitzpatrick assignment
procedure.**
- Subgroup test sizes added to Table 3 (I–II 1485, III–IV 1211, V–VI 407).
- Dark-vs-light differences now reported with bootstrap CIs (reliance −0.011
  [−0.022, 0.001] diverse; −0.012 [−0.025, 0.001] biased), both including zero.
- Fitzpatrick assignment procedure stated (Sec. 3.4): dataset-provided
  `fitzpatrick_scale` labels from Fitzpatrick17k, binned I–II / III–IV / V–VI, with a
  note that these labels are crowd/algorithm-derived and noisy.

**R1.4 — Clinical plausibility of independently flipping concepts.**
Added to Sec. 6: independent single-concept flips are a *mechanistic* probe of the
model's internal dependence, not a claim about clinical co-occurrence; dermoscopic
criteria are correlated, so some counterfactuals are clinically unusual. This does not
affect what the test measures, but the counterfactuals should be read mechanistically;
joint, correlation-aware interventions are noted as future work.

**R1.5 — Make the Model A / Model B distinction easier to follow.**
Added a compact summary table (Table 1, Sec. 3.1) contrasting the two models by domain,
dataset, concept vocabulary, label source, and role, and refer to it consistently.

---

## Reviewer 2

**R2.1 — Fairness findings are evidence from the evaluated datasets, not proof of
general equity.**
Reworded (Abstract, Sec. 5.3, Sec. 6, Conclusion) to frame the result as "no detectable
disparity on the evaluated datasets" and to call for replication on larger, cleaner
skin-tone cohorts before generalizing.

**R2.2 — Discuss noisy skin-tone labels, dataset imbalance, and concept correlations.**
Added to Sec. 6: Fitzpatrick17k skin-tone labels are dataset-provided and noisy; the
tone groups are imbalanced; and if label noise or diagnostic-concept correlations were
skin-tone-dependent the audit could be confounded. We also note the concept-correlation
caveat under R1.4.

**R2.3 — Explain the practical meaning of the reported reliance values for clinicians.**
Added to Sec. 6: a reliance of 0.19 means neutralizing the concepts shifts the predicted
malignancy probability by ~0.19 on average, so the concepts materially drive the
decision and clinician correction of a concept is actionable; the leaky model's ~0
reliance means the same correction would change nothing.

---

We believe these revisions directly address the reviewers' concerns while keeping every
claim consistent with our multi-seed results (mean±std, bootstrap/Wilcoxon significance).
