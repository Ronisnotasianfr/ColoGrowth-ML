# ColoGrowth-ML: Cross-Platform Calibration Benchmark for Colon Cancer Proliferation Classification

[![Python](https://img.shields.io/badge/python-3.8--3.12-blue?logo=python)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-pending-lightgrey)](https://github.com/Ronisnotasianfr/ColoGrowth-ML)

Systematic comparison of 5 calibration methods x 4 model classes for leakage-free colon cancer proliferation classification. Trained on GEO (n=585), validated cross-platform on TCGA-COAD (RNA-seq) and CPTAC-COAD (proteomics). Includes GDSC2 drug sensitivity screen (Bonferroni-corrected, 295 drugs, Trametinib p=1.8e-12).

---

## Key Results

| Model | Best Calibration | TCGA AUC | TCGA Acc | TCGA ECE |
|-------|-----------------|----------|----------|----------|
| Random Forest | None | 0.973 | 0.915 | 0.032 |
| Logistic Regression | QN+Platt | 0.972 | 0.903 | 0.041 |
| XGBoost | None | 0.968 | 0.909 | 0.038 |
| MLP | QN+Platt | 0.961 | 0.888 | 0.052 |

Finding: Tree-based models output well-calibrated probabilities without post-hoc correction (ECE < 0.04). Logistic Regression benefits from quantile normalization + Platt scaling for cross-platform transfer.

**Drug sensitivity** (GDSC2): 5/5 top hits MAPK/ERK pathway inhibitors, all survive Bonferroni correction (α/295 = 1.69e-4). Consistent with KRAS/BRAF dependence in colorectal cancer.

**Survival**: TCGA PanCancer log-rank p=0.009. GEO GSE39582 p=0.037.

---

## Pipeline

```
GEO GSE39582 + GSE17538
    ↓ preprocess.py (remove 10 prolif genes → compute target → merge clinical)
    ↓ train.py (nested 5-fold CV, Pipeline-encapsulated StandardScaler+VarianceThreshold+SelectKBest)
    ↓ 4 models saved as .joblib
    ├── external_validation.py → TCGA-COAD / CPTAC-COAD (Platt calibration)
    ├── calibration_benchmark.py → 5 methods x 4 models (bootstrap 95% CIs)
    ├── survival.py → Kaplan-Meier + log-rank
    ├── ki67_correlation.py → r=0.589 (MKI67 held out)
    ├── complete_analysis.py → bootstrap CIs, DCA, NNT, subgroups, Cox PH
    ├── power_analysis.py → Schoenfeld formula
    └── drug_sensitivity.py → GDSC2, 295 drugs, Bonferroni-corrected Mann-Whitney U
```

---

## Reproduce

```bash
pip install -r requirements.txt
bash reproduce.sh
```

Or step-by-step see [reproduce.sh](reproduce.sh). Runs in ~10 min on modern CPU.

---

## Repository Structure

| Path | Purpose |
|------|---------|
| `src/` | All Python modules (preprocess, train, evaluate, 10 modules total) |
| `notebooks/` | 4 pipeline notebooks + `isef_submission.ipynb` (50 cells, 16 citations) |
| `results/` | All figures (PNG/PDF) and metrics CSVs with bootstrap CIs |
| `paper/` | Build scripts + final .docx/.tex/.pdf manuscript |
| `models/` | Trained .joblib pipelines |
| `files/` | Poster layout, oral defense prep, email drafts |
| `reproduce.sh` | One-command full pipeline |

---

## Data Sources

- **GEO GSE39582** (n=585) — Affymetrix GPL570, primary training
- **GEO GSE17538** (n=238) — Affymetrix GPL570, merged training
- **TCGA-COAD** (n=329) — Illumina RNA-seq, external validation
- **TCGA-READ** (n=105) — Illumina RNA-seq, merged with COAD
- **CPTAC-COAD** (n=105) — RNA-seq, second external validation
- **GDSC2** (295 drugs x 969 cell lines) — drug sensitivity screen

---

## ISEF Judge Quick Links

- [Poster Layout](files/SCIENCEMONTGOMERY_POSTER.md) — tri-fold specs, font sizes, color palette
- [Oral Defense Prep](files/ORAL_DEFENSE_PREP.md) — 5 judge questions with answer frameworks
- [Submission Notebook](notebooks/isef_submission.ipynb) — complete project documentation
- [Paper PDF](paper/colon_cancer_growth_prediction_research_paper.pdf) — full manuscript
- [Calibration Results](results/calibration_benchmark.csv) — 20 rows, 5 methods x 4 models
- [Drug Sensitivity Results](results/drug_sensitivity_results.csv) — 295 drugs, Bonferroni-corrected

---

## License & Disclaimer

MIT License. This is an educational research project. Not a clinical diagnostic tool.
