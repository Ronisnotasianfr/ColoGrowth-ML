# Cover Letter

**Rohan Saindane**
Independent Researcher
Date: July 2026

---

**To the Editorial Board,**
*Genome Medicine*

I am writing to submit our manuscript titled:

**"Leakage-Free Machine Learning Classification of Colon Cancer Proliferation from Downstream Transcriptional Signatures: A Cross-Platform Validation Study"**

for consideration for publication in *Genome Medicine*.

## Study Overview

Tumor proliferation rate is among the most clinically important determinants of colon cancer aggressiveness, prognosis, and chemotherapy sensitivity. While Ki-67 immunohistochemistry is the current clinical standard, it suffers from substantial inter-observer variability. We present a rigorously designed machine learning framework that classifies high vs. low proliferation status in colon adenocarcinoma (COAD) from transcriptomic profiles alone, with complete methodological safeguards against the systematic data leakage errors that affect many similar published studies.

## Key Novel Contributions

1. **Leakage-free feature construction**: The ten hallmark proliferation genes (MKI67, PCNA, TOP2A, MCM2, MCM6, AURKA, BUB1, CCNB1, CDK1, BIRC5) used to construct the binary target class were completely excluded from the feature space prior to model training. Our models therefore learn from downstream transcriptional cascades — a biologically meaningful and non-trivial discrimination task that has not been previously validated at this rigor level.

2. **Cross-platform dual cohort validation**: We demonstrate generalizable predictions across two independent platform technologies — Affymetrix microarray (GEO GSE39582, n=585) and Illumina HiSeq RNA-seq (TCGA-COAD, n=322). A three-way validation design (GEO-train/GEO-holdout and TCGA-calibration/TCGA-evaluation) ensures that no TCGA data informed model fitting or feature selection.

3. **Comprehensive clinical utility assessment**: Clinical Decision Curve Analysis (DCA) demonstrates that all classifiers offer statistically superior net benefit over "Treat All" and "Treat None" strategies across a broad range of clinical thresholds.

4. **Independent prognostic validation**: Kaplan-Meier survival analysis and multivariate Cox proportional hazards modeling confirm that our proliferation class labels correlate significantly with overall survival (log-rank p < 0.05 in both cohorts) independent of age, sex, and tumor stage.

## Fit for *Genome Medicine*

This work falls squarely within *Genome Medicine*'s scope of genomic tools with direct clinical translational relevance. The study models are reproducible, the code is publicly available, and the results provide actionable computational tools for risk stratification in colorectal cancer. We respectfully request peer review.

Sincerely,
**Rohan Saindane**
https://github.com/Ronisnotasianfr/colon-cancer-predictor
