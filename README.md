# ColoGrowth-ML: Leakage-Free Colon Cancer Proliferation Classification

[![Python](https://img.shields.io/badge/python-3.8--3.12-blue?logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange?logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?logo=open-source-initiative&logoColor=white)](LICENSE)

Classifies colon cancer tumors as high or low proliferation from gene expression, trained on one platform (microarray) and validated on two others (RNA-seq, proteomics). The main design constraint is leakage control: the ten cell-cycle genes used to define the label are removed from the feature matrix before any split, and every preprocessing step runs inside a pipeline with nested cross-validation.

Four models are compared (logistic regression, random forest, XGBoost, MLP), always with calibration (Platt, isotonic, quantile normalization, QN+Platt), and checked against survival data, Ki-67 expression, a GDSC2 drug screen, and synthetic ground truth.

The honest notes on what went wrong and what I would do differently are in [NOTES.md](NOTES.md). The full manuscript lives in `overleaf/` as LaTeX.

## Key Findings

| Model | BCV | Hold-out ROC-AUC | Calibrated TCGA AUC | TCGA ECE |
|---|---|---|---|---|
| Logistic Regression | 0.9801 ± 0.0092 | 0.9850 | 0.941 (isotonic) | 0.037 |
| Random Forest | 0.9832 ± 0.0094 | 0.9871 | 0.973 (Platt) | 0.043 |
| XGBoost | 0.9756 ± 0.0120 | 0.9910 | 0.968 (Platt) | 0.038 |
| MLP | 0.9711 ± 0.0184 | 0.9812 | 0.935 (isotonic) | 0.029 |

Other results:

- Hold-out ROC-AUC 0.981 to 0.991 across the four models (XGBoost best).
- External transfer: random forest AUC 0.973 / accuracy 0.921 on TCGA-COAD RNA-seq, 0.949 / 0.868 on CPTAC-COAD proteomics.
- Platt scaling cut ECE roughly 3x for random forest (0.115 to 0.043, non-overlapping 95% CIs). QN+Platt was the best cross-platform protocol for logistic regression.
- Model scores correlate with Ki-67 (MKI67) expression even though MKI67 was held out of training: r=0.589 (GEO), r=0.543 (TCGA).
- High proliferation predicts shorter survival before staging (log-rank p=0.037 GEO, p=0.009 pan-cancer), but not after Cox adjustment for stage (HR 0.84, p=0.31). Details, including why the null is a real null and not underpowering, are in the paper.
- Drug screen (295 GDSC2 compounds, Bonferroni corrected): the top five hits all target the MAPK/ERK axis, Trametinib first at p=1.8x10^-12.
- Synthetic ground-truth experiment: all four models recovered 20/20 planted signal genes (7.7x enrichment).
- A LASSO minimal panel of 83 genes matches the full 500-feature pipeline (hold-out AUC 0.9849 vs 0.9834). Most stable genes: KIF23, DNA2, MCM3.
- Feature selection is stable: the same top-20 genes are selected in all 5 outer folds.

Two caveats that matter when reading the AUCs: the classes are separated by a Cohen's d of 2.3, so the task is easier than typical clinical ML problems, and the Cox result shows the score adds no independent survival information beyond stage in these cohorts.

## Data

| Dataset | Platform | Size | Role |
|---|---|---|---|
| GEO GSE39582 | Affymetrix GPL570 | n=585 | Training + hold-out |
| GEO GSE17538 | Affymetrix GPL570 | n=238 | Survival power check |
| TCGA-COAD | Illumina RNA-seq | n=329 | External validation |
| TCGA-READ | Illumina RNA-seq | n=105 | Pan-cancer survival |
| CPTAC-COAD | Mass spectrometry | n=105 | Proteomics validation |
| GDSC2 | Drug screen | 295 drugs x 969 lines | Drug sensitivity |

Downloads happen in `src/preprocess.py`/`reproduce.sh`; raw data is not committed.

## Methods in one line each

- Target: mean z-score of ten cell-cycle genes (Whitfield et al. 2002), binarized at the cohort median.
- Leakage control: those ten genes are dropped before any split; feature selection lives inside nested CV folds.
- Selector: bootstrap consensus (StabilitySelector), a practical take on Meinshausen and Buhlmann (2010); kept when present in the top-K of at least 50% of resamples.
- Calibration: Platt, isotonic, QN, QN+Platt, compared with bootstrap ECE intervals.
- Statistics: 1000-resample bootstrap CIs, log-rank, Cox PH, Schoenfeld power, Mann-Whitney U with Bonferroni correction for drugs.

## Reproduce

```bash
pip install -r requirements.txt
bash reproduce.sh
# roughly 5-10 minutes on a modern CPU
```

Docker:

```bash
docker compose up --build
```

Tests:

```bash
pytest tests/ -v
```

## Repository layout

```
├── src/             Pipeline modules: preprocess, train, evaluate, external_validation,
│                    calibration_benchmark, survival, ki67_correlation, complete_analysis,
│                    power_analysis, drug_sensitivity, synthetic_validation, ablation_study,
│                    lasso_minimal_model
├── notebooks/       EDA, preprocessing, training, evaluation
├── results/         Metrics CSVs (tracked), figures regenerated by the pipeline
├── models/          Trained .joblib pipelines (not committed, reproducible)
├── overleaf/        LaTeX manuscripts, each self-contained
│   ├── research_paper/     flagship paper (natbib, references.bib)
│   ├── poster/             beamerposter, 48x36in
│   ├── one_page_summary/   one-page summary for non-experts
│   └── synthica_submission/  journal submission form rewrite (IEEEtran refs)
├── scripts/         Utility scripts
├── tests/           Leakage and pipeline integrity tests
├── reproduce.sh     Full pipeline orchestrator
├── Dockerfile       Containerized reproducibility
├── docker-compose.yml
└── requirements.txt
```

## Citation

If you use this work, please cite:

```bibtex
@software{cologrowth_ml_2026,
  author = {Saindane, Rohan},
  title = {{ColoGrowth-ML}: Leakage-Free Colon Cancer Proliferation Classification},
  year = {2026},
  url = {https://github.com/Ronisnotasianfr/ColoGrowth-ML}
}
```

## License

MIT License. See [LICENSE](LICENSE).