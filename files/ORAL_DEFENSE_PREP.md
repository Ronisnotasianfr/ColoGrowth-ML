# ScienceMontgomery Oral Defense Prep

## LEAD WITH CALIBRATION (first 10 seconds of every answer)

The judge must hear "calibration benchmark" or "cross-platform probability alignment" in your first sentence. Practice this:

> "This is a computer science project about **solving the cross-platform probability calibration problem** for cancer transcriptomics. I systematically compared 5 calibration strategies across 4 model classes — no published study has done that."

## 5 Likely Judge Questions

### 1. "What makes this a computer science project (not just biology)?"
**OPENING (memorize):** "This is a **calibration benchmarking** project with a novel stability-based feature selector. The biology — colon cancer proliferation — is the application domain, not the contribution."

**Supporting points:**
- **StabilitySelector**: Instead of single-shot SelectKBest, I built an algorithm that resamples the training data 100 times, computes feature importance each time, and keeps only features selected in >=50% of resamples. This is novel — Meinshausen & Buhlmann (2010) proposed the concept but no sklearn implementation exists.
- **5 calibration methods x 4 models = 20 conditions**: Systematic comparison of Platt, Isotonic, QN+Platt, QN-only, and None, with bootstrap 95% CIs on ECE and Brier scores. No clinical study does this.
- **Nested CV pipeline**: Encapsulated preprocessing inside folds — data leakage prevention is an ML engineering contribution.
- **Drug sensitivity screen**: 295 x 969 = 285,855 computational comparisons with Bonferroni correction. Pure CS.

### 2. "How would you extend this to help patients?"
**OPENING:** "My calibration framework makes the models safe for clinical use — calibrated probabilities enable risk-stratified decision making."

**Supporting points:**
- Calibrated probabilities at decision thresholds: ECE < 0.04 for RF means when the model says 90% probability, ~90% of patients are actually high proliferation. Uncalibrated models can be off by 20+ points.
- Prospective trial: stratify by predicted class, randomize surveillance intensity. Schoenfeld power analysis sizes it at N~1,200.
- Drug repurposing: Trametinib (p=1.8e-12, Bonferroni survives) could be tested in existing trial cohorts with RNA-seq data.

### 3. "Only 10 genes? That's a small signature."
**OPENING:** "The 10 genes define the **target**, not the features. I remove them to prevent leakage. The models train on ~20,000 other genes and use StabilitySelector to find the ~500 most robustly associated ones."

**Supporting points:**
- 10 genes define the target. ~20,000 genes minus 10 are the features.
- StabilitySelector: bootstraps 100x, keeps features stable across resamples.
- Ki-67 (MKI67) is one of those 10 genes, yet model predictions correlate with MKI67 at r=0.59 despite it being removed. This proves the model learns genuine biology.

### 4. "What was the hardest bug or problem you solved?"
**OPENING:** "Cross-platform calibration. RNA-seq and microarray data have fundamentally different distributions — a classifier trained on one fails on the other without correction."

**Supporting points:**
- QN solves the distribution mismatch but doesn't fix probability calibration.
- Combined QN+Platt: Logistic Regression AUC goes from 0.97 to 0.97 with calibrated probabilities (ECE drops from 0.12 to 0.04).
- Target leakage: accidentally left proliferation genes in features, got AUC 0.99+. Removing them dropped to 0.78, recovered to 0.97 after proper training. This became the leakage-prevention methodology.
- StabilitySelector parallelization: 100 bootstrap resamples on 20,000 features is computationally heavy. Parallelized with joblib to run in <30s on 8 cores.

### 5. "How is this different from existing cancer classifiers?"
**OPENING:** "Existing cancer ML papers report AUC. My project reports **calibration** — the gap between predicted probability and observed frequency. That's what matters for clinical decisions."

**Supporting points:**
- Most CRC classifiers predict molecular subtype or survival, not proliferation with calibration assessment.
- Calibration benchmark: RF needs no calibration (ECE 0.032), LR needs QN+Platt (ECE 0.041). Without this analysis, you'd use the wrong method.
- Novel algorithm: StabilitySelector is my own sklearn-compatible transformer.
- Drug sensitivity link: validated 10-gene signature against 295 drugs — all top 5 hits converge on MAPK/ERK pathway.

## Opening Script (memorize for poster session)

When a judge approaches, lead with this (15 seconds):

> "Hi — this is a **calibration benchmarking** project for cancer ML. Most cancer classifiers report AUC but ignore calibration — whether predicted probabilities match reality. I systematically compared 5 calibration methods across 4 model classes, and built a **novel stability-based feature selector** that resamples data 100 times to find robust biomarkers. Random Forest without any calibration achieves AUC 0.973 on cross-platform validation. Logistic Regression needs QN+Platt to get there. The code is public, fully reproducible with one command."

## Quick Reference Card (for poster session)

| Question | Key phrase (max 6 words) |
|----------|--------------------------|
| CS contribution? | Calibration benchmark + StabilitySelector |
| Clinical relevance? | Calibrated probabilities enable safe decisions |
| Only 10 genes? | Target genes, not features |
| Hardest bug? | Cross-platform probability shift |
| Novelty? | 5-method calibration + bootstrap feature selection |
