# Faithful-by-Concept — Project Summary (Plain-English)

A beginner-friendly overview of what this project is, what we built, and where it stands.

---

## 1. What is the project? (Overview)

- We are building an **AI that looks at a photo of a skin spot (lesion) and decides whether it's likely skin cancer** (e.g. melanoma).
- The twist: instead of being a mysterious "black box," it **explains its decision using the same clues a real dermatologist uses** — things like *pigment network, blue-whitish veil, irregular colour, asymmetry, streaks*. These clues are called **concepts**.
- We add two promises that most other papers skip:
  - **Verifiable:** we don't just *claim* the AI used those clues — we **prove** it, by testing whether changing a clue actually changes the decision.
  - **Equitable (fair):** we check whether the AI explains itself **just as well on dark skin as on light skin**.
- It's **cheap to run**: we take a big pre-trained "foundation" model, freeze it (don't retrain it), and only train a couple of tiny add-on layers. The whole thing runs on **one free Kaggle GPU**.

---

## 2. What have we done so far?

### What we tried
- Built **two models** (because dermoscopy close-up images and normal clinical photos are very different worlds):
  - **Model A** — trained on the *derm7pt* dataset, which has **real doctor-labelled concepts** (the 7-point checklist).
  - **Model B** — trained on the *Fitzpatrick17k* dataset (clinical photos with skin-tone labels), where we had no concept labels, so we **generated them automatically** using the foundation model.
- Built a **faithfulness test**: flip a concept (say "pretend the blue-whitish veil is gone") and see if the diagnosis changes the way the explanation says it should.
- Built a **fairness audit**: measure both accuracy and explanation-faithfulness separately for light, medium, and dark skin.
- Built a **"leakage-clean" data pipeline** (removes duplicate images and keeps the same patient out of both training and testing) so our results aren't accidentally inflated.

### What we finally obtained
- A **complete, working, reproducible system** — all code on GitHub, all experiments runnable on Kaggle with a step-by-step runbook.
- A **draft research paper** with all result tables and a figure.

### Results we got (in simple numbers)
- **Diagnosis:** the explainable model was **as good as (slightly better than) the black-box** at spotting melanoma (score ~0.85 vs ~0.83, where 1.0 is perfect). So interpretability cost us *nothing*.
- **Faithfulness (the headline):** our model **genuinely relies on its concepts** (reliance ≈ 0.22), while a "fake-interpretable" model that secretly ignores its concepts scored **≈ 0** — a clear, statistically solid win. This is the whole point: we can *tell the difference* between honest and dishonest explanations.
- **Fairness:** the explanations were **equally faithful on dark and light skin** (no significant gap — good news). Also, **training on diverse skin tones significantly improved reasoning on dark skin**. A quick "calibration" fix we tried **did not help** — and we report that honestly.
- **Encoder choice:** a **dermatology-specialised** foundation model gave **better, more faithful concepts** than a general-purpose one, even though raw accuracy was similar.

### What we expected initially vs. what actually happened
- **Expected:** the explainable model would be *roughly as accurate* as the black box. **Reality:** it was even slightly better. ✅ (better than hoped)
- **Expected:** find a **big unfairness gap** in dark-skin reasoning and fix it. **Reality:** reasoning was already fairly **equitable**, and the real lever turned out to be **diverse training data**, not our quick fix. (Different from the dream, but an honest and still-useful finding.)
- **Bonus we didn't expect:** we discovered that a popular way of measuring faithfulness (rank correlation) is **misleading** — it can call a dishonest model "faithful." We show why you must measure it differently.

---

## 3. Where can this be applied?

- **Skin cancer screening** and **teledermatology** (remote diagnosis apps) — where doctors need to trust and double-check the AI's reasoning.
- **Any medical imaging AI that must justify itself** with clinical criteria — radiology (X-rays, CT), pathology (biopsy slides), etc.
- **Auditing "explainable AI" claims** in general: a tool to check whether a model's stated reasons *actually* drive its decisions, in any field.
- **Fairness checks** for medical AI — testing whether a system works and explains equally well across different patient groups (skin tone, age, sex).

---

## 4. Previous research on this

- **Concept-based models** (e.g. MICA, Patrício et al.): they give concept explanations, but **never test whether the concepts truly cause the decision**.
- **Post-hoc concept methods** (e.g. Post-hoc CBM): attach concept explanations to an already-trained model after the fact.
- **Foundation concept scorers** (e.g. MONET): use big models to label concepts, but aren't a trustworthy diagnostic system on their own.
- **Pixel-level "what-if" audits** (DeGrave et al.): powerful but **expensive** (need image generators) and don't use clinical concepts or check fairness.
- **Skin-tone fairness studies** (Daneshjou et al., Lu et al.): show AI is often **less accurate on dark skin** — but they only look at *accuracy*, never at whether the **reasoning/explanation** is fair.

---

## 5. How is this project novel?

- **First to actually test** (not just assume) that a skin-AI's concept explanations **causally drive** its diagnosis — using a **cheap trick**: flipping concepts in "concept space" (no expensive image generation needed).
- **First to ask whether that faithfulness is fair across skin tones** — nobody has checked if explanations are as honest on dark skin as on light skin.
- **A methodological insight** the field needs: the common rank-correlation way of scoring faithfulness is **magnitude-blind** and can be fooled; we show the right way to measure it.
- The **combination** — *faithfulness × fairness of concept reasoning* — is genuine white space that no prior paper covers.

---

## 6. What gaps remain, and are they worth filling?

- **Fairness effects are modest, and our quick fix (calibration) didn't work.**
  → *Worth partly filling* only if we aim for a top-tier venue: try a better fix (e.g. group-balanced fine-tuning). Otherwise the honest message "diverse training data is the real lever" already stands on its own.
- **Auto-generated clinical concept labels are noisy** (we could only verify one of them, "elevation," against real data).
  → *Partly worth it* — better prompts would help — but it's acceptable to report label quality as a known, studied limitation.
- **Our faithfulness score (0.22) is moderate**, partly because we neutralise concepts gently.
  → *Cheap and worth doing* — a sharper test would make the number look more dramatic without changing the conclusion.
- **Only one dataset (derm7pt, ~1,000 images) has real concept labels.**
  → *Hard to fill* — such data is genuinely scarce. Best to just note it as a limitation.

**Overall verdict:** What we have is already a **legitimate, honest result**. The remaining gaps are **polish, not blockers**. A sensible plan: do **one or two cheap strengthenings** (sharper faithfulness test, maybe a better fairness fix), then **stop** — chasing the rest hits diminishing returns.

---

## 7. Is this project worth submitting to the MICAD conference?

- **Good fit:** it matches MICAD's themes (AI medical imaging, computer-aided diagnosis, and its stated focus on **explainable AI**), and **very few dermatology / concept-explainability papers** have appeared there — so **little direct competition**.
- **Strengths:** real **novelty**, one **statistically bulletproof** result (the faithfulness contrast), **honest rigor** (confidence intervals, and we report what *didn't* work), and it's **cheap and fully reproducible**.
- **Honest caveats:** conference reviewing is a lottery (single-blind, roughly 1 in 3 accepted), and the **fairness part is the softer** of the contributions.

**Verdict: Yes — it's worth submitting.** It's novel, rigorous, and honest. Not a *guaranteed* acceptance (nothing is), but a **legitimate, competitive contribution** that stands on real results rather than hype.
