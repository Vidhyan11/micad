# Faithful-by-Concept — Verifiable & Equitable Concept Reasoning for Skin Lesion Diagnosis

**Target venue:** MICAD 2026 (Springer LNEE, ≤10 pages, single-blind)
**Status:** Novel idea, literature-verified, feasible on one Kaggle GPU
**Working title options:** "Faithful-by-Concept" / "DermCCF" / "Do Skin-AI Explanations Reason Like — and Fairly Like — a Dermatologist?"

---

## 0. Why the previous idea was dropped
A four-part literature sweep (SOTA methods, trustworthiness/UQ, interpretability/causal,
MICAD scope) showed that multimodal fusion + conformal prediction + skin-tone accuracy
fairness are each already published on HAM10000/ISIC (Lu AAAI'22, Fayyad 2023, CE-ViTs
2025, MICA AAAI'24, Daneshjou Sci.Adv.'22). A reviewer would call that combination
incremental. We pivoted to a verified gap.

## 1. The novel idea (one paragraph)
Dermatology AI papers now *claim* to explain themselves with clinical concepts (7-point
checklist, ABCD), but **almost none test whether the stated concepts actually cause the
model's decision**, and **none ask whether that faithfulness holds equally across skin
tones**. **Faithful-by-Concept** does three things nobody has combined: (1) it diagnoses
skin lesions through a **concept bottleneck** of true dermatologist criteria, with concept
supervision *bootstrapped from a frozen dermatology foundation model* (MONET/DermLIP) to
beat the annotation bottleneck; (2) it introduces a **concept-counterfactual faithfulness
score** — flip a clinical concept *in concept space* (no image generator needed) and
measure whether the diagnosis changes as the model claimed it would; (3) it runs the first
**fairness audit of reasoning itself** — testing whether concept accuracy *and* faithfulness
degrade on darker (Fitzpatrick V–VI) skin, then mitigating the gap. The result is a CAD
system that is interpretable, *verifiably* faithful, and *equitably* faithful.

## 2. Precise contributions (what we claim)
1. **A concept-counterfactual faithfulness metric** for dermatology — measures, per case,
   whether the concept the model says it used is the concept that flips its decision.
   (Novel: faithfulness is asserted but not tested in derm CBM papers.)
2. **Faithful-by-Concept model** — foundation-bootstrapped concept bottleneck that is
   optimized for this faithfulness, not just accuracy. Cheap: frozen encoder + small heads.
3. **Fairness-of-reasoning audit + mitigation** — first study of whether concept accuracy
   and explanation faithfulness are equitable across skin tones; a concept-calibration
   mitigation that shrinks the gap. (Novel intersection: concept × faithfulness × fairness.)
4. **Leakage-clean protocol** — lesion-/patient-level splits and de-duplication (per
   Cassidy 2021, Abhishek 2025, SelfClean 2024), which most derm papers skip.

## 3. Why this is genuinely novel — comparison with prior work

| Prior work | What it did | What it did NOT do (our gap) |
|---|---|---|
| MICA (Bie et al., AAAI 2024) | Image–concept alignment CBM, test-time intervention | Never *tests* faithfulness causally; no fairness |
| Patrício et al. (ISBI 2024) | CLIP predicts dermoscopic concepts | No counterfactual faithfulness; no skin-tone audit |
| Post-hoc CBM (Yuksekgonul, ICLR 2023) | Concept vectors post-hoc | General, not derm faithfulness/fairness |
| MONET (Kim et al., Nat. Med. 2024) | Foundation model auto-scores concepts | A tool, not a faithful/fair diagnostic system |
| DeGrave et al. (Nat. BME 2023) | Generative counterfactual *audit* of shortcuts | Pixel-level, expensive; no concept bottleneck; no fairness metric |
| Lu et al. (AAAI 2022) | Fair *conformal coverage* across skin tone | Fairness of *coverage*, not of *concept reasoning/faithfulness* |
| Daneshjou et al. (Sci. Adv. 2022) | Skin-tone *accuracy* gap (DDI) | Accuracy only — never reasoning/explanation fairness |
| ProtoPNet-derm (Correia, MICCAI-W 2023) | "This looks like that" prototypes | No clinical-concept faithfulness score; no fairness |

**No existing paper measures whether a skin-AI's concept explanations are causally faithful,
and whether that faithfulness is fair across skin tones.** That two-way intersection is our
white space.

## 4. Method (all single-GPU, NO generative training)
```
 Image --> Frozen foundation encoder (DermLIP / MONET / DINOv2)  --> embedding
                                   |
                     Concept head (small MLP) --> 7-pt / ABCD concept scores  <-- interpretable
                                   |
                     Diagnosis head (small MLP over concepts) --> label
                                   |
        Concept-counterfactual test: flip concept c in the bottleneck,
        re-run diagnosis head, measure decision change  = faithfulness of c
                                   |
        Fairness audit: repeat concept accuracy + faithfulness per Fitzpatrick group
```
- **Concept labels:** ground truth on derm7pt; pseudo-labels from the frozen foundation
  model (MONET/DermLIP zero-shot concept scoring) on datasets lacking concept GT.
- **Faithfulness score:** correlation between the model's stated concept importance and the
  measured counterfactual decision change (higher = more faithful). Purely inference.
- **Fairness:** compute concept accuracy + faithfulness for Fitzpatrick I–II vs III–IV vs
  V–VI; report gaps; mitigate with group-conditional concept calibration.

## 5. Datasets — all free on Kaggle
| Dataset | Kaggle slug | Role | Why |
|---|---|---|---|
| **derm7pt** | `menakamohanakumar/derm7pt` | Primary | True 7-point-checklist concept ground truth (dermoscopic + clinical paired) |
| **PH2** | `jamesgoydos/melanoma-skin-lesion-id-ph2-data` | Concept validation | Dermoscopic-feature annotations + masks |
| **Fitzpatrick17k** | `mobaswiralfarabi/fitzpatrick17k` | Fairness | Skin-tone (FST I–VI) labels |
| **PAD-UFES-20** | `orvile/pad-ufes-20` | Fairness + metadata | Fitzpatrick + rich metadata, clinical smartphone images, real dark-skin cases |

Foundation models (frozen, free weights): **MONET** (HF), **DermLIP/Derm1M** (ICCV'25),
or **DINOv2** as a fallback general encoder. All run frozen — only small heads train.

## 6. Experiments (paper tables)
1. **Diagnosis + concept accuracy** vs image-only CNN and a plain CBM (we stay competitive).
2. **Faithfulness table** — our faithfulness score vs. baselines (post-hoc CBM, Grad-CAM
   proxy); show Faithful-by-Concept is measurably more faithful.
3. **Fairness-of-reasoning table** — concept accuracy + faithfulness per Fitzpatrick group,
   before/after mitigation; show the gap and that we shrink it.
4. **Ablations** — encoder choice (MONET vs DermLIP vs DINOv2), pseudo-label quality,
   concept-set size.
5. **Qualitative** — per-case: predicted concepts, the counterfactual that flips the label,
   and a natural-language explanation; contrast a faithful vs an unfaithful example.

## 7. Feasibility on Kaggle (why this fits a free GPU)
- Foundation encoder is **frozen** → we only extract embeddings once and train tiny MLP
  heads. No diffusion, no LLM fine-tuning, no huge backbone training.
- derm7pt (~1k cases), PH2 (200), PAD-UFES-20 (2.3k) are small → minutes per run.
- Everything is inference + light training → comfortably inside Kaggle session limits.

## 8. MICAD fit
- Hits tracks T1 (AI imaging), T4 (CAD & decision support), and the CFP's explicit
  **explainable-AI** emphasis.
- Only 3 dermatology papers have ever appeared at MICAD (all 2024), none on
  concept-explainability → little internal competition, high topical freshness.
- Rubric = originality, significance, clarity, correctness → this hits originality hard
  while staying rigorous and reproducible.

## 9. Honest odds
No one can guarantee acceptance (single-blind, 2–3 reviewers, ~1/3 selected). But this is
built to maximize odds: a verified-novel intersection, a concrete new metric, a real
clinical framing, cheap reproducible experiments, and leakage-clean rigor.

## 10. Two build sizes
- **Core (lower risk):** contributions 1–2 on derm7pt (+PH2) — faithfulness metric + model.
- **Full (recommended):** add contribution 3 (fairness-of-reasoning on Fitzpatrick17k /
  PAD-UFES-20) — this is what makes it a standout 10/10.
