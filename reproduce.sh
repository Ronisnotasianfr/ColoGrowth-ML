#!/usr/bin/env bash
set -euo pipefail

echo "=== ColoGrowth-ML: Full Reproducibility Pipeline ==="
echo "Estimated runtime: 5-10 min on modern CPU"
echo ""

# 1. Install dependencies
pip install -r requirements.txt

# 2. Download & preprocess real data from GEO + TCGA/Xena
python -m src.preprocess --download

# 3. Merge GEO cohorts (GSE39582 + GSE17538 = geo_pan dataset)
python -m src.preprocess --geo-merged

# 4. Merge TCGA cohorts (COAD + READ = tcga_pan dataset)
python -m src.preprocess --tcga-pan

# 5. Train classifiers (nested CV, 4 models saved as .joblib)
python -m src.train --dataset geo_pan

# 6. Holdout evaluation (confusion matrices, ROC curves)
python -m src.evaluate --dataset geo_pan

# 7. Cross-platform external validation (GEO -> TCGA)
python -m src.external_validation --train-dataset geo_pan --test-dataset tcga

# 8. Calibration benchmark (4 methods x 4 models) using pan datasets
python -m src.calibration_benchmark --train-dataset geo_pan --test-dataset tcga_pan

# 9. Survival analysis (Kaplan-Meier, log-rank)
python -m src.survival

# 10. Ki-67 biological validation
python -m src.ki67_correlation

# 11. Advanced analyses (bootstrap CI, DCA, NNT, subgroups, Cox PH)
python -m src.complete_analysis

# 12. Power analysis (Schoenfeld formula)
python -m src.power_analysis

# 13. Drug sensitivity screen (GDSC2, 295 drugs, Bonferroni-corrected)
python src/drug_sensitivity.py --drugs 20

# 14. Manuscripts are pure LaTeX — compile from overleaf/ subfolders:
#     research_paper (flagship, natbib + references.bib), poster, one_page_summary,
#     synthica_submission (journal submission, IEEEtran refs)
#     (each folder is self-contained: main.tex + figures/ [+ references.bib])

# 15. Synthetic ground-truth validation
python -m src.synthetic_validation

# 16. LASSO minimal gene set (external review response)
python -m src.lasso_minimal_model --dataset geo_pan

echo ""
echo "=== ALL PIPELINE STEPS COMPLETE ==="
echo "Results in: results/"
echo "LaTeX in:   overleaf/ (research_paper, poster, one_page_summary, synthica_submission)"
echo "Models in:  models/"
echo ""
echo "Docker: docker compose up --build"
echo "Outputs are mounted at ./data, ./models, ./results, ./paper"
