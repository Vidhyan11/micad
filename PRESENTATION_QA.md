# Faithful-by-Concept — Understanding & Presentation Prep (Q&A)

A plain-English study sheet for explaining and defending the project.

---

## 1. The project in two lines

- **What it is:** A skin-lesion diagnosis AI that reasons through dermatologist concepts (pigment network, asymmetry, etc.) and — unlike prior work — actually *tests* whether those concepts causally drive the decision and whether that reasoning is fair across skin tones.
- **Input → Output:** *Input* = a skin-lesion image. *Output* = a diagnosis (e.g. melanoma vs. not) **plus** the concept scores behind it **plus** a faithfulness check (flip a concept → does the decision change as claimed?).
- **What we infer:** A concept-bottleneck model can be as accurate as a black box *and* genuinely faithful (its concepts causally drive decisions: reliance ≈ 0.22 vs. ≈ 0 for a "fake-interpretable" leaky model), and its reasoning is statistically equitable across skin tones — with **training-data diversity, not post-hoc calibration**, being the lever that keeps it fair.

---

## 2. The bottleneck vs. faithfulness (simple terms)

> These are **not two models**. The **bottleneck is the model** (how it works); **faithfulness is a test** we run on it (how we check it's honest).

### The concept bottleneck (the model)
A normal AI jumps straight from image to "melanoma" — a black box. Our model works in **two steps**:
1. Look at the image and score the dermatologist's clues (atypical pigment network? blue-whitish veil? asymmetry?).
2. Make the diagnosis using **only those clue scores** — nothing else.

It's a "bottleneck" because *all* information must squeeze through that narrow list of clues before a decision — the model can't peek at the raw image. So the clues **are** the reasoning.

🩺 *Analogy:* a doctor who must fill in a checklist and may only diagnose from the checklist, not from an unexplainable gut feeling.

### Faithfulness (the test)
A model can **display** clues while secretly **ignoring** them. Faithfulness checks whether it *actually uses* them. The test: **flip a clue and see if the answer changes.**
- Diagnosis changes when the clue is flipped → the model really used it = **faithful** ✅
- Diagnosis barely moves → the clue was decoration = **unfaithful** ❌

🩺 *Analogy:* the doctor says "melanoma, because of the blue veil." Cover the veil and ask again — if they still confidently say melanoma, they weren't really using it.

**Key finding:** the pure bottleneck is faithful; a "leaky" model that also shows clues is fake (flip a clue → nothing happens) — and nobody in dermatology AI was testing for this.

---

## 3. Reliance, usefulness, and applications

### What is "reliance"?
**Reliance = how much the diagnosis actually leans on the concepts.** We wipe out the concepts (set them to a neutral "no information" value) and measure how much the melanoma probability moves.
- Big drop → high reliance (decision was genuinely built on concepts).
- No drop → ~zero reliance (concepts were just for show).

Our numbers: pure bottleneck **0.22**; leaky model **≈ 0**.

🧑‍⚖️ *Analogy:* remove all the evidence and see if the verdict collapses. If the judge still convicts with zero evidence, the evidence wasn't the real reason.

### How useful in the real world?
- **Clinician trust:** proves the stated reason is real, so a doctor can cross-check it.
- **Catching fake explanations:** flags "interpretable" models that secretly ignore their own reasons.
- **Regulation:** EU AI Act / FDA increasingly require *meaningful* explanations for high-risk medical AI.
- **Equity:** checks the reasoning works as well on dark skin as light (dermatology AI has a documented history of failing on darker skin).

### Where are the applications used?
- Skin-cancer screening / teledermatology (a phone or clinic tool that flags lesions *with a checkable reason*).
- Other medical imaging: radiology, pathology, ophthalmology.
- Regulatory / audit tools (certify a vendor's "explainable AI" claim is genuine).
- Fairness audits across patient groups (skin tone, age, sex).
- Medical education / second opinion.
- Beyond medicine: any high-stakes concept-based decision (e.g. a loan model that lists the factors it "used" — now testable).

**One sentence:** it turns "the AI says it's interpretable and fair" into "we *checked*, and here's the proof."

---

## 4. Novelty claim & MICAD fit

### The novelty claim (sharp)
> **First to causally *test* whether a dermatology AI's concept explanations are faithful — and first to audit whether that faithfulness is *equitable across skin tones*.**

Supporting novelties: (a) a cheap concept-space counterfactual test (no image generator); (b) the finding that rank-correlation faithfulness is *magnitude-blind* and must be replaced by magnitude-aware metrics; (c) foundation-bootstrapped concepts, externally validated.

**Honest caveat:** the *ingredients* (CBMs, ERASER faithfulness, skin-tone fairness) each exist. The novelty is the **intersection + derm-specific causal test + fairness-of-reasoning angle + the leaky-model demonstration** — a *verified-gap + new-measurement* paper, **not a new algorithm**.

### Is it worth MICAD (at that level)?
**Yes — well-matched, not a stretch.**
- MICAD is a respectable Springer-proceedings venue with an explicit **explainable-AI** focus — not top-tier (MICCAI/CVPR/Nature). That level fits this work exactly.
- **Fits because:** on-theme, few derm/concept papers there (low competition), rigorous, reproducible, single-GPU, and honest (reports CIs and a negative result).
- **Would not fit higher venues:** modest fairness effect, no new algorithm.
- **Verdict:** competitive, honest, on-theme submission at the right level — not a guaranteed accept (single-blind, ~⅓ accepted; fairness is the softer half), but a legitimate contribution.

---

## 5. Anticipated counter-questions (with answers)

### Faithfulness metric
- **"Isn't faithfulness trivially true since the head only sees concepts?"** → For our *pure* model yes — and that's the point: we quantify *how much* (0.22) and show the common leaky model fails it (≈0). The metric distinguishes faithful from fake.
- **"Flipping to `1−c` makes unrealistic inputs — is the counterfactual valid?"** → We also report ERASER ablation-to-reference (milder, in-distribution) and get the same ranking. Fully in-distribution counterfactuals are future work.
- **"Why is reliance only ~0.22?"** → It's a *relative* contrast (0.22 vs ≈0); the absolute is deflated by gentle mean-ablation and averaging. The gap is what's significant.

### The leaky baseline
- **"Is the leaky model a strawman?"** → No — a head seeing concepts + embedding is the documented "leakage" failure and the default if you don't enforce the bottleneck. It's *just as accurate*, so people would deploy it unaware its explanations are hollow.

### Fairness (the softest area)
- **"Gaps are tiny and calibration didn't work — what's the contribution?"** → The first audit + a reassuring (equitable) result + a CI-backed finding that data diversity, not calibration, is the lever. We report the calibration negative openly.
- **"You artificially biased training to find a gap — contrived?"** → It mirrors the real deployment condition (light-dominant training, diverse patients) and shows diverse training fixes it.
- **"Fitzpatrick17k labels are noisy web images."** → Acknowledged; it's the largest source with skin-tone labels (2,168 V–VI). We report CIs to reflect the uncertainty.

### Concepts / bootstrapping
- **"Pseudo-labels come from the same model you use as encoder — circular?"** → Partly, and we're explicit. We validate externally on PAD-UFES `elevation` (AUROC 0.63) and lean the fairness claim on faithfulness (GT-free), not concept accuracy.
- **"Model B concept AUROC 0.99 is suspicious."** → That's fit to pseudo-labels, not accuracy — flagged explicitly; the real check is the external PAD-UFES number.

### Data / setup
- **"derm7pt is only ~1,000 cases."** → It's the standard benchmark with *real* 7-point concept GT (rare); official split, CIs reported; it's the concept-validation set, not a large-scale claim.
- **"Why binary detection, not multiclass?"** → The concepts are *designed* for melanoma detection; multiclass with non-melanocytic classes bottlenecks poorly (we show it) — an honest concept-task-alignment finding.
- **"Why two models?"** → Dermoscopic concepts aren't visible in clinical photos; one model across both domains would confound everything. The framework (not the weights) is the contribution.

### Novelty
- **"CBMs / faithfulness / fairness all exist — what's new?"** → The intersection, the cheap concept-space causal test, and the magnitude-blindness insight. Upfront: it's a verified-gap + new-measurement paper, not a new backbone.

### Clinical
- **"Would a dermatologist trust it?"** → It gives a *checkable* rationale for cross-checking; it's decision-support, not autonomous diagnosis; no clinical-readiness claim.
- **"DermLIP is frozen — would fine-tuning change the conclusions?"** → Maybe the accuracy numbers, but the faithfulness methodology and pure-vs-leaky contrast are architecture-level and would hold. Frozen keeps it single-GPU and reproducible.

### Two meta-tips for the defense
1. Lead with the strong result (faithfulness, E2); be pre-emptively honest about the soft one (fairness, E3). Owning a limitation beats dodging it.
2. One-line thesis to keep ready: *"Everyone claims concept explanations; we're the first to test they're real and fair — and we show a model that looks interpretable but isn't."*
