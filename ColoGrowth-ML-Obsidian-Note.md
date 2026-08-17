---
title: ColoGrowth-ML — Full Project Note
tags:
  - project
  - machine-learning
  - bioinformatics
  - colon-cancer
  - transcriptomics
  - science-fair
date: 2026-08-17
source: README.md, NOTES.md, src/*, reproduce.sh
---

# ColoGrowth-ML: Cross-Platform Calibration Benchmark for Colon Cancer Proliferation Classification

> [!info] One-liner
> A leakage-free ML pipeline that **classifies colon cancer tumor proliferation status (High vs. Low)** from gene expression data, trained on GEO microarrays, validated across RNA-seq (TCGA) and proteomics (CPTAC) platforms, and tested for clinical relevance (survival, Ki-67, drug sensitivity).
>
> **Author:** Rohan Saindane (Ronisnotasianfr) — ScienceMontgomery 2025
> **Repo:** `github.com/Ronisnotasianfr/ColoGrowth-ML` · **License:** MIT · **Language:** Python 3.8–3.12 (4 models × 12 saved pipelines)

---

## Table of Contents

1. [What This Project Is](#what-this-project-is)
2. [The Biological Question](#the-biological-question)
3. [Pipeline Architecture](#pipeline-architecture)
4. [Data Sources](#data-sources)
5. [Target & Leakage Prevention](#target--leakage-prevention)
6. [Methodology](#methodology)
7. [Module-by-Module Code Map](#module-by-module-code-map)
8. [How to Run (Reproduction)](#how-to-run-reproduction)
9. [Key Results](#key-results)
10. [Output Artifacts](#output-artifacts)
11. [Lessons Learned & Known Weaknesses](#lessons-learned--known-weaknesses)
12. [Next Steps / Future Work](#next-steps--future-work)

---

## What This Project Is

A systematic benchmark evaluating **5 calibration methods** across **4 model classes** for leakage-free colon cancer proliferation classification:

- **Models:** Logistic Regression, Random Forest, XGBoost, MLP Neural Network
- **Calibration methods:** None (baseline), Platt Scaling, Isotonic Regression, QN+Platt, QN-only
- **Training data:** GEO microarray cohorts (merged GSE39582 + GSE17538, n = 823)
- **External validation:** TCGA-COAD RNA-seq (n = 329), TCGA-READ (n = 105), CPTAC-COAD proteomics (n = 105)
- **Clinical relevance tests:** Kaplan-Meier survival + log-rank, Ki-67 (MKI67) correlation, GDSC2 drug screen (295 drugs, Bonferroni-corrected)

It grew out of a science-fair project (ScienceMontgomery) and was hardened into a peer-review-able manuscript (LaTeX/PDF) with full reproducibility (Docker + `reproduce.sh`).

---

## The Biological Question

Tumor proliferation is a hallmark of cancer aggressiveness. It's normally measured with **Ki-67 immunohistochemistry (IHC staining)** — slow, subjective, pathologist-dependent. This project asks:

> Can we infer proliferation status **computationally** from gene expression (microarray / RNA-seq / proteomics), using a leakage-free pipeline that *generalizes across platforms*?

The proxy target uses the **10-gene cell-cycle signature** (Whitfield et al., 2002): the average expression of MKI67, PCNA, TOP2A, MCM2, MCM6, AURKA, BUB1, CCNB1, CDK1, BIRC5 — binarized at the median into High vs. Low proliferation.

---

## Pipeline Architecture

```mermaid
flowchart TD
    A[GEO GSE39582 n=585] --> M[Merge GEO cohorts]
    B[GEO GSE17538 n=238] --> M
    M --> P[preprocess.py<br/>target + feature matrices]
    P --> T[train.py — nested 5-fold CV<br/>GridSearch + StabilitySelector]
    T --> MOD[(12 models .joblib<br/>4 models x 3 datasets)]
    MOD --> EV[evaluate.py — holdout<br/>confusion, ROC, metrics]
    MOD --> EX[external_validation.py<br/>TCGA / CPTAC + QN + Platt]
    MOD --> CAL[calibration_benchmark.py<br/>5 methods x 4 models]
    MOD --> SURV[survival.py — KM + log-rank]
    MOD --> K67[ki67_correlation.py — MKI67 check]
    MOD --> CA[complete_analysis.py — DCA, Cox PH,<br/>subgroups, SHAP, enrichment]
    MOD --> SW[synthetic_validation.py —<br/>ground-truth recovery]
    MOD --> DV[drug_sensitivity.py — GDSC2 screen]
    MOD --> PA[power_analysis.py — Schoenfeld]
    MOD --> LS[lasso_minimal_model.py — 83-gene panel]
    EV --> OVERLEAF[overleaf/ — research_paper/main.tex<br/>+ references.bib (natbib) + figures/]
```

**Load-bearing idea:** everything preprocessing-related lives inside sklearn `Pipeline` objects refit per-CV-fold, so no validation information ever leaks into training. The 10 signature genes are dropped from features *before* any split (see [[#Target & Leakage Prevention]]).

---

## Data Sources

| Dataset | Platform | Size | Role |
|---|---|---|---|
| GEO GSE39582 | Affymetrix GPL570 microarray | n = 585 | Primary training |
| GEO GSE17538 | Affymetrix GPL570 microarray | n = 238 | Expanded training (merged = `geo_pan`) |
| TCGA-COAD | Illumina RNA-seq | n = 329 | External validation |
| TCGA-READ | Illumina RNA-seq | n = 105 | Pan-cancer validation |
| TCGA PanColor (**coad + read**) | RNA-seq | n = 434 | `tcga_pan` dataset |
| CPTAC-COAD | Mass-spec proteomics | n = 105 | Cross-platform validation (protein level!) |
| GDSC2 | Drug screening | 295 drugs × ~969 cell lines | Drug sensitivity |

**Raw data lives in `data/raw/`:** series matrix files, GPL570 probe annotation, Xena TCGA downloads, CPTAC files — all downloaded by `preprocess.py` (NOT committed to git; re-downloadable).

**Processed data in `data/processed/`** — per dataset `{name}_X_features.csv`, `{name}_y_target.csv`, `{name}_proliferation_scores.csv`, `{name}_clinical.csv`. Datasets: `geo`, `geo17538`, `geo_pan`, `tcga`, `tcga_read`, `tcga_pan`, `cptac`, `synthetic`.

---

## Target & Leakage Prevention

> [!warning] The story behind this section
> Originally the 10 proliferation genes stayed in the feature matrix → model hit **AUC 0.99+** and looked amazing. It was actually predicting the label from the very genes used to *create* the label. Removing them dropped AUC to ~0.78 (demoralizing), then proper training recovered 0.97. See `NOTES.md`.

### How the target is made
1. Compute proliferation score = mean expression of the 10 signature genes (Whitfield et al., 2002).
2. Binarize at the **median** → `1 = High proliferation`, `0 = Low`.

### How leakage is prevented (3 layers)
1. **Gene removal:** `remove_proliferation_genes()` strips all 10 genes (case-insensitive match) from the feature matrix before anything else.
2. **Runtime assertion:** `validate_no_leakage()` asserts zero overlap between features and `PROLIF_GENES` — raises `AssertionError` if contaminated (covered by `tests/test_leakage.py`).
3. **Split-then-bin:** `train.py` re-binarizes using the **training-only median threshold**, so even the threshold carries no test information.
4. **Pipelines inside CV:** scaler, variance filter, `StabilitySelector`, and classifier are all refit inside each fold.

> [!tip] The 10 signature genes
> `MKI67, PCNA, TOP2A, MCM2, MCM6, AURKA, BUB1, CCNB1, CDK1, BIRC5` — defined in `src/preprocess.py:PROLIF_GENES` and duplicated in `src/synthetic_data.py`.

---

## Methodology

### Feature pipeline (per model, shared)
```
VarianceThreshold(0.01) → StandardScaler → StabilitySelector(k=500, B=30 bootstraps, min_pct=0.5) → classifier
```

### StabilitySelector (`src/stability_selector.py`)
Custom sklearn transformer (Meinshausen & Bühlmann, 2010). Instead of one-shot ANOVA F-score ranking:
- Resamples training data **B times** (default 100, pipeline uses 30)
- Computes `f_classif` F-scores each round, keeps top-k per bootstrap
- Retains only features selected in **≥ min_pct of rounds** → robust to data perturbation
- Parallelized with `joblib`. Stores `selection_frequencies_` per feature.

### Hyperparameter tuning
`GridSearchCV` per model on the training pool — grids in `src/train.py:PARAM_GRIDS` (e.g. RF: n_estimators 100/200, max_depth 5/10/None; XGB: lr 0.01–0.1; MLP: two architectures).

### Model definitions (`src/model.py`)
| Model | Hyperparameters |
|---|---|
| Logistic Regression | C=0.1, l2, liblinear, 1000 iter |
| Random Forest | 100 trees, max_depth=10, min_samples_leaf=4 |
| XGBoost | 100 trees, depth 5, lr 0.05, logloss |
| MLP | (256, 128, 64), relu, adam, early stopping |

### Calibration methods (on external cohort)
1. **None** — raw `predict_proba` (baseline)
2. **Platt Scaling** — sigmoid fit via LogisticRegression on raw probabilities
3. **Isotonic Regression** — non-parametric monotone fit
4. **QN + Platt** — quantile normalization of the target platform to the training distribution, then Platt
5. **QN-only** — quantile normalization alone

Metric: **ECE** (Expected Calibration Error) with 1000-bootstrap CIs, Brier score, plus ROC-AUC.

### 12 trained pipelines (`models/`)
`{model}_{dataset}.joblib` for model ∈ {logistic_regression, random_forest, xgboost, neural_network_mlp} × dataset ∈ {geo, geo_pan, synthetic}.

---

## Module-by-Module Code Map

### `src/` — 16 modules

| Module | CLI / entrypoint | What it does |
|---|---|---|
| `preprocess.py` | `python -m src.preprocess --download` (+ `--geo-merged`, `--tcga-pan`, `--cptac`; default = synthetic) | Downloads GEO/TCGA/CPTAC, parses series matrices, probe→gene mapping (first gene before `///`), computes proliferation target, removes signature genes, saves processed CSVs. |
| `synthetic_data.py` | (imported) | Generates n=300 × 2000-gene synthetic expression where 10 known signature genes drive a latent proliferation factor → ground truth is *known*. |
| `train.py` | `python -m src.train --dataset geo_pan` | Stratified split, per-model nested CV + GridSearch, leakage-guarded binarization, saves pipelines + `all_models_{dataset}_leakage_fixed_metrics.csv`. |
| `evaluate.py` | `python -m src.evaluate --dataset geo_pan` | Holdout evaluation: accuracy/precision/recall/F1/AUC, confusion matrices, ROC curves. |
| `external_validation.py` | `python -m src.external_validation --train-dataset geo_pan --test-dataset tcga` | Cross-platform validation: feature alignment (zero-fill missing), column-wise **quantile normalization**, Platt calibration, soft-voting ensembles (All / Top-3). |
| `calibration_benchmark.py` | `python -m src.calibration_benchmark --train-dataset geo_pan --test-dataset tcga_pan` | 5×4 calibration benchmark with bootstrap ECE/Brier CIs. |
| `survival.py` | `python -m src.survival` | Kaplan-Meier curves + log-rank tests (lifelines); normalizes heterogeneous clinical column names across cohorts. |
| `ki67_correlation.py` | `python -m src.ki67_correlation` | Correlates *model predictions* with raw MKI67 expression (Pearson/Spearman) — biological validity despite the gene being removed from features. |
| `complete_analysis.py` | `python -m src.complete_analysis` | The "kitchen sink": bootstrap CIs, DCA (decision curve), NNT, subgroup analysis, SHAP, pathway enrichment (gseapy), Cox PH, sensitivity analysis, calibration comparison. |
| `drug_sensitivity.py` | `python src/drug_sensitivity.py --drugs 20` | Downloads GDSC2 whole dose-response, Mann-Whitney U colon vs. other lines per drug, Bonferroni correction (α/295). |
| `synthetic_validation.py` | `python -m src.synthetic_validation` | Provenance check: recover known signal genes, measure feature-recovery precision/recall, FPR, AUC. |
| `ablation_study.py` | runnable | Compares `SelectKBest` vs. `StabilitySelector` under identical pipelines (leakage-fixed). |
| `power_analysis.py` | `python -m src.power_analysis` | Schoenfeld-formula power curves (exposes underpowered cohorts like CPTAC). |
| `lasso_minimal_model.py` | runnable | LASSO logistic regression ($C=0.1$) inside the same leakage-free pipeline → **83-gene minimal panel** + bootstrap gene-stability output. |
| `stability_selector.py` | (imported) | The bootstrap-stability feature selector (see methodology). |
| `model.py` | (imported) | Centralized model builders — single source of truth for hyperparameters. |
| `calibration_comparison_test.py` / `drug_permutation_test.py` | quick scripts | Ad-hoc statistical checks (calibration comparison logic, permutation tests for drugs). |

### `scripts/`
- `demo_predict.py` — 5-sample inference demo on `random_forest_geo_pan.joblib` (built for science-fair judging, runs in <5 s).

### `tests/` (`pytest tests/ -v`)
- **Leakage suite:** `remove_proliferation_genes` removes all 10 genes (upper/lower case); `validate_no_leakage` passes on clean / raises on contaminated matrices.
- **StabilitySelector:** fit/transform correctness, selection-frequency output, and parallel (`n_jobs=-1`) vs. serial consistency.

### `notebooks/`
Pipeline story in 4 notebooks (each with an `_executed` twin): `01_eda` → `02_preprocessing` → `03_model_training` → `04_evaluation`.

### `paper/`
Citation source only (all build scripts removed Aug 2026):
- `references.bib` — 35 BibTeX entries (marisa2013, whitfield2002, smith2010, tcga2012, platt1999, meinshausen2010, …) — copied to `overleaf/research_paper/references.bib`
- `paper_metrics.py` — orphaned metric loader from the deleted build system (cleanup candidate)

### `overleaf/`
Definitive LaTeX deliverables, organized by artifact (each folder compiles standalone on Overleaf):
- `research_paper/` — **flagship manuscript** (`main.tex`, 12pt article, natbib citations via `references.bib`, ~10 figures incl. Ki-67 correlation × 2 + feature-selection stability, LASSO + ablation + power sections)
- `poster/` — 48×36in beamerposter (Problem, Pipeline, External Validation, Calibration Benchmark, Ablation, Survival, Drugs, Ki-67, References)
- `one_page_summary/` — ScienceMontgomery handout (revised Aug 2026)
- `synthica_submission/` — journal submission form rewrite (IEEEtran refs)
- Deleted Aug 2026: `biorxiv/`, `synthica/` (old July-era manuscript variants), `nair_summary/`

### `poster/`
`poster/main.tex` (beamerposter, current source of truth) + `poster.md` (legacy markdown). Overleaf poster uses `figures/` copies of key PNGs.

### `results/`
Every figure (PNG/PDF) + metric CSV from all analyses (60+ artifacts) — listed by type in [[#Output Artifacts]].

### Root config files
| File | Purpose |
|---|---|
| `reproduce.sh` | 16-step end-to-end orchestrator (~5–10 min on modern CPU); step 14 now points at `overleaf/` instead of building PDFs locally |
| `Dockerfile` | `python:3.11-slim`, installs deps, runs pytest, then `reproduce.sh` |
| `docker-compose.yml` | Builds image, mounts `./data ./models ./results ./paper` as volumes |
| `requirements.txt` | 18 packages (scikit-learn, xgboost, shap, lifelines, gseapy, GEOparse, python-docx, reportlab…) |
| `NOTES.md` | Personal log of struggles/lessons (summarized in [[#Lessons Learned & Known Weaknesses]]) |
| `files/` | Email drafts — mentor outreach (Dr. Nair reply, Ms. Yu) |
| `.backup_pre_competition/` | Pre-competition `.bak` snapshots of README, train.py, survival.py, paper scripts (gitignored, local only) |

---

## How to Run (Reproduction)

```bash
# Native
pip install -r requirements.txt
bash reproduce.sh                 # ~10 min on a modern CPU

# Docker (builds, runs pytest, then reproduce.sh)
docker compose up --build

# Tests only
pytest tests/ -v
```

`reproduce.sh` pipeline (16 steps):
1. `preprocess --download` — fetch GEO + TCGA/Xena + CPTAC, build all datasets
2. `preprocess --geo-merged` → `geo_pan`
3. `preprocess --tcga-pan` → `tcga_pan`
4. `train --dataset geo_pan` → 4 .joblib pipelines
5. `evaluate --dataset geo_pan` → holdout metrics + figures
6. `external_validation --train-dataset geo_pan --test-dataset tcga` → cross-platform ROC
7. `calibration_benchmark --train-dataset geo_pan --test-dataset tcga_pan` → 5×4 benchmark
8. `survival` → KM + log-rank
9. `ki67_correlation` → biological validation
10. `complete_analysis` → DCA, subgroups, Cox PH, SHAP…
11. `power_analysis` → Schoenfeld power
12. `drug_sensitivity --drugs 20` → Bonferroni-corrected GDSC2 screen
13. (comment) manuscripts are pure LaTeX → compile from `overleaf/` (research_paper, poster, one_page_summary); no local PDF build
14. `synthetic_validation` → ground-truth recovery
15. `lasso_minimal_model --dataset geo_pan` → 83-gene LASSO panel + gene stability CSVs
16. Final: `pytest tests/ -v` (run separately, not in script)

> [!warning] Known reproduction fragility
> GEO FTP / Xena / CPTAC download endpoints can be flaky or change — if a download fails mid-run, results silently degrade. Data is gitignored, so a fresh clone must re-download everything.

---

## Key Results

### Headline table (external validation, TCGA via geo_pan training)

| Model | Optimal Calibration | TCGA AUC | TCGA Accuracy | TCGA ECE |
|---|---|---|---|---|
| Random Forest | Platt Scaling | 0.973 | 0.921 | **0.043** |
| XGBoost | Platt Scaling | 0.968 | 0.903 | **0.038** |
| Logistic Regression | None | 0.936 | 0.855 | 0.082 |
| MLP Neural Network | Isotonic Regression | 0.935 | 0.848 | **0.029** |

### Holdout (leakage-fixed, geo_pan)
`results/all_models_geo_pan_leakage_fixed_metrics.csv`: holdout ROC-AUC ≈ **0.981–0.991** (XGBoost 0.991, RF 0.988); accuracy ≈ 0.897–0.939 (RF highest).

### Calibration findings
- **Platt Scaling cuts tree-model ECE ~3×** (RF: 0.115 → 0.043; non-overlapping 95% CIs, p<0.05)
- **Isotonic Regression is best for MLP** (ECE 0.029)
- QN+Platt was *worse* than Platt alone for RF/LR/MLP in this config (AUC collapse to 0.69–0.97) — a nuance captured in `calibration_benchmark.csv`

### Cross-platform validation (`external_validation_results.csv`)
- RF: raw AUC 0.970 → calibrated 0.963; accuracy 0.869 → 0.879; Brier 0.091 → 0.079
- Ensembles: **All-Models** AUC 0.978/0.971 (raw/cal), **Top-3** 0.980/0.973 — ensembles robustly improve calibration
- CPTAC proteomics: RF 0.949 AUC / 0.868 accuracy — works across *platforms AND data modalities*

### Biological / clinical validation
- **Survival:** high-proliferation predicts worse OS — TCGA PanCancer log-rank p = 0.009, GEO GSE39582 p = 0.037, geo_pan p = 0.036
  - BUT: Cox PH adjusted for stage → HR = 0.84, p = 0.31 (proliferation adds no *independent* survival signal beyond staging; CPTAC too underpowered, 7 events)
  - Schoenfeld power analysis (9 cohorts): univariate tests adequately powered (79–194 events); CPTAC (7) & TCGA-READ (21) nulls are underpowered → don't over-interpret
- **Ki-67:** model predictions vs. MKI67 expression — GEO r = 0.589 (p = 5.4e-56), TCGA r = 0.543 (p = 1.2e-26); proliferation *score* vs. MKI67: GEO r = 0.825
- **Drug screen:** top drug **Trametinib** (p = 1.8e-12), then PD0325901 (p = 5.9e-12)… **5/5 top hits are MEK inhibitors (MAPK/ERK pathway)** — all survive Bonferroni (α/295 = 1.69e-4). Colon lines are dramatically more sensitive to MEK blockade.
- **Subgroups:** no significant interaction effects (age/sex/stage) — performance is uniform across subgroups
- **Synthetic ground truth:** recovered 20/20 signal genes at 7.7× enrichment, low FPR
- **Ablation (StabilitySelector vs. SelectKBest):** hold-out AUC nearly identical for LR/RF/XGB (Δ ≤ 0.001), but SS **improved MLP by +0.009** (0.983 → 0.992) — SS ≥ SKB on all 4 models
- **LASSO minimal panel** (`lasso_minimal_model_geo_pan.csv`): C=0.1 → **83 genes**, hold-out AUC **0.9849** / acc 0.9152 vs. full 500-gene pipeline 0.9834 / 0.9030 — statistically indistinguishable; 30 genes stable ≥50% (KIF23 1.00, DNA2 0.97, MCM3 0.97, DDIAS 0.93, SNRPB 0.90)

### Honest caveats (from paper abstract)
- High AUC partly reflects an *easy problem* (Cohen's d = 2.3 between classes)
- External cohorts may share distribution characteristics with training via platform preprocessing

---

## Output Artifacts

| Folder | Contents |
|---|---|
| `results/` | ~60 files: ROC curves, calibration curves (per-model + ensemble), KM curves (geo, geo_pan, tcga, tcga_pan, tcga_read, cptac, synthetic, stage-stratified), Cox forest plot, DCA, SHAP summaries (LR/RF/XGB), confusion matrices, pathway enrichment, power analysis, sensitivity plots, drug sensitivity + p-value distribution, feature-selection stability, Ki-67 correlations, ablation study + all metric CSVs. **LASSO:** `lasso_minimal_model_geo_pan.csv`, `lasso_gene_stability_geo_pan.csv` (KIF23 1.0, DNA2 0.967, MCM3 0.967, DDIAS 0.933, SNRPB 0.9), `lasso_selected_genes_geo_pan.csv` |
| `models/` | 12 `.joblib` pipelines (4 models × geo, geo_pan, synthetic) |
| `overleaf/` | Research paper (natbib + references.bib), poster (beamerposter), one-page summary, synthica submission — each folder self-contained for Overleaf |
| `data/processed/` | 38 CSVs — features, targets, proliferation scores, clinical per 8 datasets |
| `data/raw/` | Downloaded GEO/TCGA/CPTAC raw inputs (gitignored) |
| `files/` | Mentor email drafts only (Dr. Nair reply w/ proposed call times, Ms. Yu intro) — competition prep docs removed Aug 2026 |

---

## Lessons Learned & Known Weaknesses

> [!bug] Target leakage (caught early)
> AUC 0.99+ was a lie — the model was reading the label out of its own recipe genes. Fixing the leak crashed AUC to ~0.78; proper training eventually recovered 0.97. **Lesson: audit the label→feature path before celebrating.**

> [!warning] Cross-platform calibration is hard
> Microarray ≠ RNA-seq distributions. Raw probabilities on TCGA were confident-but-wrong. Quantile normalization fixed the distribution mismatch but not calibration; **QN+Platt combos need careful validation** (in the benchmark, QN+Platt actually *hurt* most models vs. Platt alone).

> [!warning] StabilitySelector is the weakest link
> 100 bootstraps on ~20k features × 585 samples is slow (parallelized, but B may still be too small); selection frequency fluctuates between runs.

> [!warning] CPTAC survival analysis is basically useless
> 7 events / 105 samples — Schoenfeld power analysis (added *late*) would have said so immediately. Do power analysis *first* next time.

> [!warning] GEO merging is messy
> Cohorts use different clinical column names; probe→gene mapping takes the first gene before `///`; unmapped probes dropped; merge keeps only common genes.

> [!warning] Download fragility
> GEO FTP + Xena + CPTAC endpoints flake; a judge running `reproduce.sh` could fail on step 1. Should cache data / vendor archives.

> [!note] Poster TeX fight
> The beamerposter clipped content past the paper edge invisibly — "Overfull vbox" warnings were real clipping. Fixed by measuring column bottoms programmatically (PyMuPDF) and tuning figure widths; `\vspace*` between blocks, not `\vfill`.

> [!note] Local LaTeX is possible on Windows
> No MiKTeX, no tectonic wheels — but TinyTeX works (3 tries), plus `tlmgr install` for missing fonts (courier, times, helvetic, psnfss). Missing fonts and missing packages look identical on TinyTeX.

> [!note] MCM6 vs MCM10 typo
> Manuscript said MCM10 in one spot; code says MCM6. Diff the manuscript against the code, don't trust either alone. Same pass caught a name mismatch (Ronit vs Rohan) in an old citation.

---

## Next Steps / Future Work

- [x] **Respond to Dr. Nair's microbiome pushback** — LASSO minimal model (83 genes, AUC 0.985) + Ki-67 correlation evidence; reply drafted, call Mon 8/17 11am / Wed 8/19 2:30pm / Fri 8/21 10am ET (files/email_to_dr_nair_reply_draft.md)
- [x] **Definitive manuscript** — rewrite research_paper/main.tex with natbib + references.bib, add LASSO/ablation/power/Ki-67 sections; sync poster + one-page summary
- [x] **Local LaTeX toolchain** — TinyTeX installed; compile `overleaf/` locally instead of relying on Overleaf
- [ ] **Stratify drug screen by KRAS/BRAF status** — MEK hits may be explained by KRAS mutation, not proliferation (cheapest fix, most likely to flip a conclusion)
- [ ] Replace median-binarization with a **fixed threshold anchored to a reference cohort** (clinical claim shouldn't depend on whichever cohort computed the median)
- [ ] **Single-sample scoring** — QN needs the whole batch; fix before anyone can deploy
- [ ] Replace median-binarization with **continuous regression** (why throw away information?)
- [ ] **Prospective validation with matched Ki-67 IHC** + RNA-seq (transcriptomic MKI67 correlation is strong but is not IHC)
- [ ] Add a third-platform held-out cohort (nanostring or fresh proteomics)
- [ ] Wet-lab follow-up on top genes (MCM10, NCAPH, EXO1, CHEK1, KIF23, DNA2) — knock-down experiments
- [ ] Validate top drug hits (Trametinib etc.) in actual cell lines, not just GDSC2 correlation
- [ ] Recruit a collaborator who understands the biology (especially the Cox/stage interaction finding)
- [ ] Cache raw datasets to fix reproduction fragility
- [ ] Increase StabilitySelector bootstrap count / benchmark stability formally

---

## Quick Reference: Key Commands

```bash
python -m src.preprocess --download                 # fetch + build all datasets
python -m src.train --dataset geo_pan               # train 4 models
python -m src.evaluate --dataset geo_pan            # holdout metrics/figs
python -m src.external_validation --train-dataset geo_pan --test-dataset tcga
python -m src.calibration_benchmark --train-dataset geo_pan --test-dataset tcga_pan
python -m src.survival
python -m src.ki67_correlation
python -m src.complete_analysis
python -m src.power_analysis
python -m src.synthetic_validation
python src/drug_sensitivity.py --drugs 20
python scripts/demo_predict.py                       # judge demo
pytest tests/ -v                                     # leakage + selector tests
```

**Generated note is self-contained.** To split into multiple notes later, use the headings as [[wikilink]] anchors and copy the relevant tables/figures into per-topic notes.