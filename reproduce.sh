#!/usr/bin/env bash
set -euo pipefail

echo "=== ColoGrowth-ML: Full Reproducibility Pipeline ==="
echo "Estimated runtime: 5-10 min on modern CPU"
echo ""

# 1. Install dependencies
pip install -r requirements.txt

# 2. Download & preprocess real data from GEO + TCGA/Xena
python -m src.preprocess --download

# 3. Train classifiers (nested CV, 4 models saved as .joblib)
python -m src.train --dataset geo_pan

# 4. Holdout evaluation (confusion matrices, ROC curves)
python -m src.evaluate --dataset geo_pan

# 5. Cross-platform external validation (GEO -> TCGA)
python -m src.external_validation --train-dataset geo_pan --test-dataset tcga

# 6. Calibration benchmark (5 methods x 4 models)
python -m src.calibration_benchmark --train-dataset geo --test-dataset tcga

# 7. Survival analysis (Kaplan-Meier, log-rank)
python -m src.survival

# 8. Ki-67 biological validation
python -m src.ki67_correlation

# 9. Advanced analyses (bootstrap CI, DCA, NNT, subgroups, Cox PH)
python -m src.complete_analysis

# 10. Power analysis (Schoenfeld formula)
python -m src.power_analysis

# 11. Drug sensitivity screen (GDSC2, 295 drugs, Bonferroni-corrected)
python src/drug_sensitivity.py --drugs 20

# 12. Rebuild paper (Word + LaTeX + PDF)
python paper/build_paper.py --dataset geo_pan
python paper/build_pdf.py --dataset geo_pan

echo ""
echo "=== ALL PIPELINE STEPS COMPLETE ==="
echo "Results in: results/"
echo "Paper in:   paper/"
echo "Models in:  models/"
