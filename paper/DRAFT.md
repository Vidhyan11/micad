# Faithful-by-Concept: Verifiable and Equitable Concept Reasoning for Skin-Lesion Diagnosis

*Rough full-prose draft (≈10 pages). Review here, then we port to the Springer LNEE/llncs template.*
*Authors: anonymized for review.*

---

## Abstract

Dermatology AI increasingly *claims* to explain its decisions through clinical concepts — the 7-point dermoscopy checklist, the ABCD rule — yet almost no work tests whether those concepts *causally* drive the decision, and none asks whether such faithfulness holds equally across skin tones. We present **Faithful-by-Concept**, a concept-bottleneck framework that (i) diagnoses through dermatologist concepts, with concept supervision bootstrapped from a frozen dermatology foundation model where ground truth is unavailable; (ii) introduces a **concept-counterfactual faithfulness** test that intervenes directly in concept space — no image generator — and measures whether the concepts an explanation cites are the ones that actually move the decision; and (iii) conducts the first **fairness-of-reasoning** audit, testing whether concept faithfulness degrades across Fitzpatrick skin tones. On derm7pt, our bottleneck matches or exceeds an image-only baseline for melanoma detection (AUROC 0.852 vs. 0.830) while remaining fully interpretable. We show that a common "leaky" concept model appears interpretable yet is causally unfaithful (concept reliance ≈ 0 vs. 0.22 for the pure bottleneck; gap significant at 95% CI), and that rank-correlation faithfulness metrics are magnitude-blind and can be fooled by such models. Auditing across skin tones, we find reasoning faithfulness is statistically *equitable* within a training regime, and that skin-tone-diverse training — not post-hoc calibration — significantly improves dark-skin faithfulness. All experiments run on a single GPU with frozen encoders.

**Keywords:** concept bottleneck · faithfulness · algorithmic fairness · dermoscopy · explainable AI

---

## 1. Introduction

Concept-based explanations are attractive for clinical AI because they speak the clinician's language: rather than a heatmap, the model reports *why* in terms a dermatologist already trusts — an atypical pigment network, a blue-whitish veil, irregular streaks. A growing body of dermatology work therefore routes predictions through such concepts and presents the concept activations as an explanation of the diagnosis.

But a concept explanation is only worth trusting if it is **faithful** — if the concepts it cites are the ones that genuinely determine the model's decision — and only worth deploying if that faithfulness is **equitable** — if it holds across the patient groups the model will actually serve. Both properties are routinely assumed and almost never tested. Concept-bottleneck papers assert faithfulness by architecture; fairness studies in dermatology measure *accuracy* gaps across skin tone but never gaps in *reasoning*. The result is a literature of models that look interpretable and look fair, with neither claim verified.

We close both gaps with three contributions.

1. **A concept-counterfactual faithfulness metric for dermatology.** Because our diagnosis head consumes *only* concepts, we can intervene directly in concept space — flip a concept from present to absent — and measure the causal effect on the decision, with no generative model of images. We use this to test, per case, whether the concepts the model reports as important are the ones whose removal actually changes the diagnosis, and we show that popular rank-correlation faithfulness measures are magnitude-blind and can rate a concept-ignoring model as "faithful."

2. **Faithful-by-Concept**, a frozen-encoder concept bottleneck optimized for this property, with concept supervision bootstrapped from a dermatology foundation model where human concept labels do not exist. It is cheap — only two small heads are trained — and, for the clinically-aligned detection task, it matches or beats an image-only model.

3. **The first fairness-of-reasoning audit** in dermatology AI: we measure concept faithfulness, not just accuracy, across Fitzpatrick skin tones, and we test a group-conditional calibration mitigation. Our findings are reported honestly, including a negative result for calibration.

A leakage-clean protocol — lesion-level splits and embedding-based de-duplication — underpins every result.

---

## 2. Related Work

**Concept-bottleneck and concept-alignment models.** Concept Bottleneck Models [Koh et al., 2020] predict human-specified concepts and then the label from those concepts. In dermatology, image–concept alignment approaches such as MICA [Bie et al., 2024] and CLIP-based dermoscopic concept predictors [Patrício et al., 2024] provide concept explanations and support test-time intervention. None of these works *tests* whether the concepts are causally responsible for the decision, and none examines fairness of the explanation.

**Post-hoc concept methods.** Post-hoc Concept Bottleneck Models [Yuksekgonul et al., 2023] fit concept directions onto a fixed classifier, giving concept attributions without retraining. These are general-purpose and were not designed to certify causal faithfulness in a clinical setting.

**Foundation concept annotators.** MONET [Kim et al., 2024] uses an image–text foundation model to score dermatological concepts automatically, enabling concept-level auditing. MONET is a labeling and auditing tool rather than a faithful end-to-end diagnostic system; we use exactly this style of foundation scoring to *bootstrap* our concept supervision.

**Counterfactual and shortcut audits.** Pixel-level generative counterfactual audits [DeGrave et al., 2023] reveal shortcuts by editing images, but require an image generator and operate outside a concept vocabulary. Our counterfactuals live in concept space and are inference-only.

**Faithfulness measurement.** Faithfulness of explanations is a long-standing concern [Jacovi & Goldberg, 2020; Adebayo et al., 2018]. ERASER [DeYoung et al., 2020] operationalizes it via comprehensiveness and sufficiency — erasing or retaining the tokens an explanation deems important. We adapt these to concept space and show they succeed where rank-correlation fails.

**Skin-tone fairness.** Daneshjou et al. [2022] document accuracy disparities across skin tone on a curated diverse benchmark; conformal approaches target fair coverage across tone [Lu et al., 2022]. All prior work concerns *accuracy or coverage*; none measures whether the model's *reasoning* is equally faithful across tones.

**Our white space.** To our knowledge, no prior work measures whether a skin-AI's concept explanations are causally faithful, nor whether that faithfulness is equitable across skin tones. That two-way intersection is our contribution.

---

## 3. Method

### 3.1 Two-model, shared-metric design

Dermoscopic 7-point concepts (pigment network, blue-whitish veil, …) describe the magnified sub-surface structure visible under a dermatoscope; they are simply not present in an ordinary clinical photograph. A single concept model cannot honestly span both domains, and applying dermoscopic concepts to clinical images would confound any downstream analysis. We therefore instantiate the framework twice, sharing one architecture and one faithfulness metric:

- **Model A (dermoscopic):** trained on derm7pt, whose images carry expert 7-point concept ground truth. Model A validates the faithfulness metric against *trustworthy* concepts.
- **Model B (clinical):** trained on Fitzpatrick17k clinical photographs, with clinical concepts (ABCD + visible morphology) bootstrapped from a foundation model. Model B hosts the skin-tone fairness audit.

### 3.2 Concept bottleneck

A frozen encoder *f* (DermLIP; DINOv2 as an ablation) maps an image *x* to an embedding *z = f(x)*. A concept head *g* predicts concept probabilities **ĉ = σ(g(z)) ∈ [0,1]^K**. A diagnosis head *h* predicts the label from concepts *only*: **ŷ = h(ĉ)**. Only *g* and *h* — small MLPs — are trained; the encoder is never updated, so embeddings are extracted once and reused. The purity of the bottleneck (the diagnosis head never sees *z*) is what makes the counterfactual test in §3.3 causally meaningful: the decision can change *only* through the concepts. We train sequentially (concept head to convergence, then diagnosis head on predicted concepts) with class-balanced cross-entropy and balanced-accuracy model selection.

**Bootstrapping concepts.** Where concept ground truth is unavailable (Fitzpatrick17k, PAD-UFES-20), we score each concept zero-shot from the foundation model using an ensemble of positive/negative text prompts, and use the resulting probabilities as soft supervision. We validate this bootstrap where any ground truth exists (§5.1).

### 3.3 Concept-counterfactual faithfulness

For a case with predicted concepts ĉ and predicted class ŷ, we compare:

- **Stated importance** *I_c* — the model's own attribution of the decision to concept *c*, taken as the gradient of *P(ŷ)* with respect to ĉ_c (the family of local attributions XAI methods produce).
- **Causal effect** *E_c* — the true effect of intervening on concept *c*: flip ĉ_c → 1 − ĉ_c, re-run *h*, and record |ΔP(ŷ)|.

From these we report:

- **Reliance** — |P(ŷ | ĉ) − P(ŷ | ĉ = reference)|, the drop when *all* concepts are neutralized to a reference value. This measures how much the decision truly depends on concepts; it is ≈1 in spirit for a pure bottleneck and ≈0 for a model that ignores concepts.
- **Comprehensiveness / sufficiency** (ERASER-style) — the decision change when the top-*k* *stated-important* concepts are erased (comprehensiveness; higher is better) or are the *only* ones retained (sufficiency; lower is better), averaged over *k*.
- **ccf_corr** — the per-case rank correlation Spearman(*I, E*), reported for completeness.

A central methodological point: **ccf_corr is magnitude-blind.** A model whose concepts are causally inert has *I_c ≈ 0* and *E_c ≈ 0* for all *c*, but these near-zero vectors can be perfectly *rank*-correlated. Faithfulness must therefore be judged by magnitude-aware measures (reliance, comprehensiveness), not rank agreement alone — a point we demonstrate empirically in §5.2.

### 3.4 Fairness of reasoning

On Fitzpatrick17k we compute both diagnosis accuracy and faithfulness (reliance, comprehensiveness) per skin-tone group (I–II, III–IV, V–VI), report worst-vs-best gaps, and test the dark-vs-light difference with a bootstrap. As a retraining-free mitigation we fit a per-group temperature on the concept logits (group-conditional calibration) on a held-out split and re-audit. We also contrast two *training regimes* — light-skin-only vs. skin-tone-diverse — to test whether the deployment condition known to cause accuracy disparities also affects reasoning.

---

## 4. Datasets and Protocol

**derm7pt** [Kawahara et al., 2019] — 1,011 cases with expert 7-point-checklist concept annotations and diagnosis; we use its official 413/203/395 train/val/test split. Concepts are binarized to the suspicious (score-positive) variant of each criterion. Primary source for Model A.

**Fitzpatrick17k** [Groh et al., 2021] — 16,574 clinical images labeled with Fitzpatrick skin type I–VI (V–VI: 2,168 images), coarse three-partition diagnosis (benign / malignant / non-neoplastic). No concept ground truth. Fairness backbone for Model B.

**PAD-UFES-20** [Pacheco et al., 2020] — 2,298 smartphone clinical lesions with rich metadata; we use its `elevation` field as partial clinical concept ground truth to validate the bootstrap.

**Leakage-clean protocol.** We remove near-duplicates by clustering images at high cosine similarity on the frozen embeddings (≈600 removed from Fitzpatrick17k) and form group-aware, diagnosis-stratified splits so no patient/lesion crosses a boundary. PH2 was excluded: no available mirror ships images and concept labels together.

---

## 5. Experiments

Encoder: DermLIP (frozen), unless noted. All heads are 1-hidden-layer MLPs; seeds fixed; 95% CIs are 2,000-sample bootstraps over the test set.

### 5.1 Diagnosis and concept accuracy (E1)

We evaluate the clinically-aligned *detection* task (Model A: melanoma vs. rest; Model B: malignant vs. rest), the task the concepts are designed for.

| Model | Variant | dx bal-acc | dx AUROC | concept AUROC |
|---|---|---|---|---|
| A | image-only | 0.763 | 0.830 | — |
| A | **CBM (ours)** | **0.777** | **0.852** | 0.827 |
| A | CBM-oracle (GT concepts) | 0.854 | 0.914 | — |
| B | image-only | 0.788 | 0.886 | — |
| B | CBM (ours) | 0.720 | 0.792 | (pseudo) |

**Findings.** For melanoma detection the concept bottleneck *matches and slightly exceeds* the image-only model (AUROC 0.852 vs. 0.830) — interpretability at no accuracy cost. The oracle (ground-truth concepts → diagnosis) reaches 0.914, so the concept set is highly diagnostic; the oracle→CBM gap is exactly the concept-prediction error, and concepts are predicted well (mean AUROC 0.827). Crucially, supervised concept prediction (0.827) far exceeds zero-shot (0.644), quantifying the value of the foundation bootstrap. On the harder clinical task, Model B's bottleneck trails image-only by a modest, honestly-reported margin (AUROC 0.792 vs. 0.886); its concept "AUROC" against its own pseudo-labels is not an accuracy measure and we do not report it as one. Externally, the clinical bootstrap validates against PAD-UFES `elevation` at AUROC 0.63 (absolute lesion size was not recoverable from crops — 0.55 — and was dropped from the vocabulary).

### 5.2 Faithfulness (E2)

We compare the pure bottleneck against a **leaky CBM** whose diagnosis head sees concepts *and* the embedding.

| Model | Variant | reliance | comprehensiveness | sufficiency | ccf_corr |
|---|---|---|---|---|---|
| A | **pure bottleneck** | **0.223** [0.212, 0.236] | **0.129** [0.123, 0.136] | 0.073 | 0.605 |
| A | leaky | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 | 0.819 |
| B | **pure bottleneck** | **0.208** [0.203, 0.212] | **0.172** [0.169, 0.175] | 0.074 | 0.553 |
| B | leaky | 0.108 [0.104, 0.113] | 0.103 [0.099, 0.107] | 0.020 | 0.780 |

**Findings.** The pure bottleneck's decisions genuinely depend on its concepts (reliance 0.223), whereas the leaky model — which *displays the same concepts* — is causally inert (reliance 0.000): flipping or erasing its concepts does not move the decision, which is driven by the embedding. The pure−leaky gap is significant (Model A Δreliance = 0.223 [0.211, 0.236]; Model B = 0.100 [0.094, 0.106]; non-overlapping CIs). Strikingly, the leaky model scores *higher* rank-agreement (ccf_corr 0.819 vs. 0.605) — a concrete demonstration that rank-correlation faithfulness is magnitude-blind and would certify an unfaithful model. This is the case current derm concept papers never check, and our magnitude-aware metrics catch it.

### 5.3 Fairness of reasoning (E3)

On Fitzpatrick17k (Model B) we audit accuracy and faithfulness per skin tone, and contrast light-only vs. diverse training.

| Regime | Group | dx bal-acc | reliance | comprehensiveness |
|---|---|---|---|---|
| Diverse | I–II | 0.713 | 0.197 | 0.167 |
| Diverse | III–IV | 0.720 | 0.221 | 0.180 |
| Diverse | V–VI (dark) | 0.725 | 0.217 | 0.175 |
| Biased (I–II only) | V–VI (dark) | 0.682 | 0.178 | 0.139 |

**Findings.** *(i) Reasoning is equitable within a regime.* Under any single training regime the dark-vs-light faithfulness difference is not significant (e.g., biased training, ΔrelianceV–VI−I–II = −0.003 [−0.016, 0.009], n.s.). To our knowledge this is the first evidence that concept-based reasoning need not be less faithful on dark skin. *(ii) Data diversity is the effective lever.* Skin-tone-diverse training significantly improves dark-skin faithfulness over light-dominant training: V–VI reliance 0.178 [0.167, 0.189] → 0.217 [0.205, 0.230] and comprehensiveness 0.139 → 0.175, both with non-overlapping CIs. *(iii) Post-hoc calibration is not sufficient.* Group-conditional concept calibration did not significantly change the gaps (reliance gap 0.019 → 0.019); we report this as a negative result rather than a mitigation. Effect sizes are modest, consistent with the derm foundation encoder being partially skin-tone-robust.

### 5.4 Encoder ablation (E4a)

| Encoder | dx bal-acc | dx AUROC | concept AUROC | reliance |
|---|---|---|---|---|
| DINOv2 (general) | 0.759 | 0.849 | 0.782 | 0.155 |
| **DermLIP (derm)** | 0.777 | 0.852 | **0.827** | **0.223** |

At near-identical diagnosis AUROC, the dermatology foundation encoder yields substantially better concept prediction (0.827 vs. 0.782) and more faithful reasoning (reliance 0.223 vs. 0.155). Domain pretraining buys *faithful concepts*, not merely accuracy.

### 5.5 Qualitative (E5)

For a representative melanoma case we visualize the predicted concept probabilities alongside each concept's counterfactual effect; the concept whose flip most reduces the melanoma probability (the decisive concept) is highlighted, giving a per-case, human-readable, causally-grounded rationale. *(Figure to be inserted.)*

---

## 6. Discussion and Limitations

Our results support a simple message: interpretability claims should be *tested*, not assumed, and the test is cheap when the model is a genuine bottleneck. The faithful-vs-leaky contrast is the clearest evidence — a model can present concepts while ignoring them, and only magnitude-aware causal metrics reveal it.

On fairness, our honest finding is that reasoning faithfulness was *equitable across skin tones in every regime tested*, and that the actionable lever was training-data diversity, not post-hoc calibration (which was ineffective). This is reassuring but comes with modest effect sizes, partly because the derm foundation encoder is already fairly skin-tone-robust. Clinical concept pseudo-labels are noisy and only partially validated (PAD-UFES elevation, AUROC 0.63); we therefore treat pseudo-label quality as a studied variable rather than ground truth. Our faithfulness metric is defined within the concept bottleneck; extending it to end-to-end models, and to multi-way diagnosis beyond binary detection, is future work. Finally, real 7-point concept ground truth exists at limited scale (derm7pt, ~1k cases); broader concept-annotated data would strengthen Model A.

---

## 7. Conclusion

Faithful-by-Concept makes concept-based dermatology AI **verifiable** and lets us ask whether it is **equitable**. It tests, rather than assumes, that the concepts an explanation cites causally drive the diagnosis; it shows that a widely-used class of "interpretable" models fails this test while a pure bottleneck passes it; and it delivers the first audit of whether that faithfulness is fair across skin tones, finding reasoning to be equitable and data diversity — not calibration — to be the lever. The framework is cheap, reproducible, and single-GPU.

---

## References

*(Draft bibliography — verify every entry against the original before submission.)*

1. Koh, P.W. et al. **Concept Bottleneck Models.** ICML, 2020.
2. Bie, Y. et al. **MICA: Towards Explainable Skin Lesion Diagnosis via Multi-Level Image–Concept Alignment.** AAAI, 2024.
3. Patrício, C. et al. **Towards Concept-based Interpretability of Skin Lesion Diagnosis using Vision-Language Models.** ISBI, 2024.
4. Yuksekgonul, M., Wang, M., Zou, J. **Post-hoc Concept Bottleneck Models.** ICLR, 2023.
5. Kim, C. et al. **Transparent Medical Image AI via an Image–Text Foundation Model Grounded in Medical Literature (MONET).** Nature Medicine, 2024.
6. DeGrave, A.J. et al. **Auditing the Inference Processes of Medical-Image Classifiers by Leveraging Generative AI and the Expertise of Physicians.** Nature Biomedical Engineering, 2023.
7. DeYoung, J. et al. **ERASER: A Benchmark to Evaluate Rationalized NLP Models.** ACL, 2020.
8. Jacovi, A., Goldberg, Y. **Towards Faithfully Interpretable NLP Systems.** ACL, 2020.
9. Adebayo, J. et al. **Sanity Checks for Saliency Maps.** NeurIPS, 2018.
10. Daneshjou, R. et al. **Disparities in Dermatology AI Performance on a Diverse, Curated Clinical Image Set (DDI).** Science Advances, 2022.
11. Lu, C. et al. **Fair Conformal Predictors for Skin-Tone-Equitable Coverage.** AAAI, 2022.
12. Groh, M. et al. **Evaluating Deep Neural Networks Trained on Clinical Images in Dermatology with the Fitzpatrick 17k Dataset.** CVPR Workshops, 2021.
13. Kawahara, J. et al. **Seven-Point Checklist and Skin Lesion Classification using Multi-task Multi-modal Neural Networks.** IEEE J. Biomedical and Health Informatics, 2019.
14. Pacheco, A.G.C. et al. **PAD-UFES-20: A Skin Lesion Dataset Composed of Patient Data and Clinical Images Collected from Smartphones.** Data in Brief, 2020.
15. Oquab, M. et al. **DINOv2: Learning Robust Visual Features without Supervision.** TMLR, 2024.
16. **DermLIP / Derm1M: A Million-scale Vision–Language Dataset and Model for Dermatology.** ICCV, 2025.
17. Guo, C. et al. **On Calibration of Modern Neural Networks.** ICML, 2017.
18. Cassidy, B. et al. **Analysis of the ISIC Image Datasets: Usage, Benchmarks and Recommendations.** Medical Image Analysis, 2022.
