# ScienceMontgomery Oral Defense Prep

## 5 Likely Judge Questions

### 1. "What makes this a computer science project (not just biology)?"
**Answer framework:**
- Cross-platform calibration: solved the feature distribution mismatch problem (RNA-seq vs microarray) using quantile normalization. This is a CS contribution — data alignment across platforms.
- Systematic comparison of 4 calibration methods + 5 configurations (Platt, Isotonic, QN+Platt, QN Only, None) quantified which method minimizes expected calibration error. No clinical study I found does this.
- Nested cross-validation pipeline: encapsulated preprocessing inside folds to guarantee no data leakage. This is an ML engineering contribution.
- The drug sensitivity screen is a computational analysis — 295 drugs x 969 cell lines = 285,855 drug-cell line comparisons. Not wet-lab biology.

### 2. "How would you extend this to actually help patients?"
**Answer framework:**
- Prospective clinical trial: stratify patients by predicted proliferation class, randomize to standard vs intensified surveillance, use Schoenfeld power analysis to size the trial (N~1,200 for HR=0.78).
- Drug repurposing: the GDSC2 screen suggests colon cancer patients may selectively benefit from MEK inhibitors like Trametinib. Could be tested in existing trial cohorts with RNA-seq data (re-analysis).
- Multiple myeloma collaboration: the same 10-gene signature has shown prognostic value in MM. Already discussed with a clinical collaborator.

### 3. "How did you train 4 models on only 10 genes?"
**Answer:**
- The 10 proliferation genes are REMOVED from features (to prevent target leakage). The models train on the remaining ~20,000 genes, then use SelectKBest to pick the top K most informative features that correlate with proliferation indirectly (downstream transcriptional cascades).
- 10 genes define the *target*. ~20,000 genes (minus 10) are the *features*.
- Dimensionality reduced via ANOVA F-test feature selection inside the pipeline.

### 4. "What was the hardest bug or problem you solved?"
**Answer framework:**
- Cross-platform feature mismatch: TCGA (RNA-seq) and GEO (Affymetrix) have different gene symbols, different ranges, different distributions. Quantile normalization solved the distribution problem but we had to manually curate gene symbol mappings (e.g. Affymetrix probe IDs to gene names).
- Target leakage: originally the 10 proliferation genes were accidentally included in features, inflating AUC to ~0.98+. Detected this after Ki-67 correlation analysis was suspiciously perfect. Removing them dropped AUC to ~0.78 then recovered to ~0.97 after proper training.

### 5. "How is this different from existing cancer classifiers?"
**Answer framework:**
- Most published CRC classifiers predict molecular subtype or survival. This one specifically predicts *proliferation class* — a dynamic growth phenotype.
- Most use single-platform data. We validate across 3 independent cohorts spanning RNA-seq and 2 microarray platforms.
- Most don't address calibration. We show that calibration method choice can shift predicted risk by 20+ percentage points, which matters at clinical decision thresholds.
- Novel drug sensitivity link: connecting 10-gene proliferation signature to drug response in GDSC2 bridges transcriptomic proliferation prediction to therapeutic targeting.

## Quick Reference Card (for poster session)

| Question | Key phrase (max 6 words) |
|----------|--------------------------|
| CS contribution? | Cross-platform calibration benchmark |
| Clinical relevance? | Stratify + drug repurposing |
| Only 10 genes? | Target genes, not features |
| Hardest bug? | Target leakage detection |
| Novelty? | Proliferation-specific + calibration + drugs |
