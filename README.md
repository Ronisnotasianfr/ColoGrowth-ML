# ColoGrowth-ML: Cross-Platform Calibration Benchmark for Colon Cancer Proliferation Classification

[![Python](https://img.shields.io/badge/python-3.8--3.12-blue?logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange?logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?logo=open-source-initiative&logoColor=white)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-pending-lightgrey?logo=doi&logoColor=white)](https://github.com/Ronisnotasianfr/ColoGrowth-ML)

A systematic benchmark evaluating **5 calibration methods** across **4 model classes** for leakage-free colon cancer proliferation classification. Models trained on GEO microarray cohorts (n=823), validated cross-platform on TCGA-COAD RNA-seq (n=329) and CPTAC-COAD proteomics (n=105), and tested for clinical relevance via survival analysis, Ki-67 correlation, and GDSC2 drug sensitivity screening (295 drugs, Bonferroni-corrected).

---

## Key Findings

| Model | Optimal Calibration | TCGA AUC | TCGA Accuracy | TCGA ECE |
|---|---|---|---|---|
| Random Forest | Platt Scaling | 0.973 | 0.921 | **0.043** |
| XGBoost | Platt Scaling | 0.968 | 0.903 | **0.038** |
| Logistic Regression | None | 0.936 | 0.855 | 0.082 |
| MLP Neural Network | Isotonic Regression | 0.935 | 0.848 | **0.029** |

- **Platt Scaling** reduces ECE for tree-based models by 3× (RF: 0.115 → 0.043; 95% CIs non-overlapping, p<0.05)
- **Isotonic Regression** minimizes ECE for neural networks (MLP: 0.029)
- **Cross-platform generalizability** maintained across microarray → RNA-seq → proteomics platforms
- **Ki-67 biological validation**: predicted proliferation correlates with MKI67 expression (r=0.589)
- **Drug screen**: 5/5 top hits target the MAPK/ERK pathway — Trametinib (p=1.8×10⁻¹²), PD0325901 (p=5.9×10⁻¹²), SCH772984 (p=1.1×10⁻¹⁰) — all surviving Bonferroni correction (α/295=1.69×10⁻⁴)
- **Survival stratification**: high-proliferation patients show significantly worse OS (TCGA PanCancer log-rank p=0.009; GEO GSE39582 p=0.037)

---

## Overview

![Calibration Benchmark](results/calibration_benchmark.png)

```
GEO GSE39582 (n=585) ───────────────────┐
GEO GSE17538 (n=238) ──── merge ───► preprocess ──► train (nested 5-fold CV) ──► 12 models
                                        │                                           │
                                        │  ┌─ external_validation.py ──► TCGA / CPTAC
                                        │  ├─ calibration_benchmark.py ──► 5×4 comparison
                                        │  ├─ survival.py ──► KM + log-rank
                                        │  ├─ ki67_correlation.py ──► biological validation
                                        │  ├─ complete_analysis.py ──► DCA, Cox PH, subgroups
                                        │  ├─ synthetic_validation.py ──► ground-truth recovery
                                        │  ├─ ablation_study.py ──► feature selection comparison
                                        │  ├─ power_analysis.py ──► Schoenfeld power
                                        │  └─ drug_sensitivity.py ──► GDSC2 screen
                                        │
                                        └─ paper/ ──► manuscript generation
```

---

## Results Gallery

| | |
|---|---|
| ![ROC](results/roc_curves_comparison.png) | ![Calibration](results/calibration_comparison_curves.png) |
| ![KM](results/kaplan_meier_tcga_pan.png) | ![Drug](results/drug_sensitivity_top_drugs.png) |
| ![DCA](results/clinical_dca.png) | ![SHAP](results/shap_summary_xgboost.png) |

---

## Methodology

### Data Sources

| Dataset | Platform | Size | Role |
|---|---|---|---|
| GEO GSE39582 | Affymetrix GPL570 | n=585 | Primary training |
| GEO GSE17538 | Affymetrix GPL570 | n=238 | Expanded training |
| TCGA-COAD | Illumina RNA-seq | n=329 | External validation |
| TCGA-READ | Illumina RNA-seq | n=105 | Pan-cancer validation |
| CPTAC-COAD | Proteomics | n=105 | Cross-platform validation |
| GDSC2 | Drug screening | 295 drugs × 969 lines | Drug sensitivity |

### Models

- Logistic Regression
- Random Forest
- XGBoost
- MLP Neural Network

### Calibration Methods

- No Calibration (baseline)
- Platt Scaling
- Isotonic Regression
- QN+Platt
- QN-only

### Leakage Prevention

The 10 proliferation-defining genes (Whitfield et al., 2002) used to compute the binary target are removed from feature matrices prior to any data splitting or model training. Feature selection uses a bootstrap-stability-based selector (Meinshausen & Bühlmann, 2010).

---

## Reproduce

```bash
pip install -r requirements.txt
bash reproduce.sh
# ~10 minutes on a modern CPU
```

### Docker

```bash
docker compose up --build
```

### Tests

```bash
pytest tests/ -v
```

---

## Repository Structure

```
├── src/             16 Python modules — preprocessing, training, evaluation,
│                        calibration, survival, drug sensitivity, synthetic validation
├── notebooks/       Pipeline notebooks (EDA, preprocessing, training, evaluation)
├── results/         Figures (PNG/PDF) and metrics CSVs with bootstrap CIs
├── paper/           LaTeX manuscript source, bibliography, build scripts
├── models/          12 trained .joblib pipelines (4 models × 3 datasets)
├── scripts/         Utility scripts (inference demo, etc.)
├── tests/           Unit tests (data leakage, pipeline integrity)
├── poster/          Conference poster (HTML)
├── reproduce.sh     Full pipeline orchestrator
├── Dockerfile       Containerized reproducibility
├── docker-compose.yml
└── requirements.txt
```

---

## Citation

If you use this work, please cite:

```bibtex
@software{cologrowth_ml_2026,
  author = {Saindane, Ronit},
  title = {{ColoGrowth-ML}: Cross-Platform Calibration Benchmark for
           Colon Cancer Proliferation Classification},
  year = {2026},
  url = {https://github.com/Ronisnotasianfr/ColoGrowth-ML}
}
```

---

## License

MIT License. See [LICENSE](LICENSE).
