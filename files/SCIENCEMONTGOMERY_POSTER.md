# ColoGrowth-ML: Colon Cancer Proliferation Classifier
## ScienceMontgomery Tri-Fold Poster — Layout Guide
**Title: Cross-Platform Colon Cancer Proliferation Classification via Leakage-Free ML**

```
┌──────────────────────────┬──────────────────────────┬──────────────────────────┐
│       LEFT PANEL         │       MIDDLE PANEL       │      RIGHT PANEL         │
│      (24in × 36in)      │      (24in × 36in)       │      (24in × 36in)       │
│                          │                          │                          │
│  ┌──────────────────┐    │  ┌──────────────────┐    │  ┌──────────────────┐    │
│  │  TITLE BLOCK     │    │  │  METHODS         │    │  │  RESULTS         │    │
│  │  (spans all 3)   │    │  │  FLOWCHART       │    │  │  TABLE 1         │    │
│  └──────────────────┘    │  └──────────────────┘    │  │  Internal         │    │
│                          │                          │  │  Validation       │    │
│  ┌──────────────────┐    │  ┌──────────────────┐    │  └──────────────────┘    │
│  │  PROBLEM         │    │  │  CALIBRATION     │    │                          │
│  │  Colon cancer:   │    │  │  5 methods       │    │  ┌──────────────────┐    │
│  │  2nd leading     │    │  │  Platt Scaling   │    │  │  TABLE 2         │    │
│  │  cause of death  │    │  │  Isotonic Reg    │    │  │  External         │    │
│  │                  │    │  │  QN+Platt        │    │  │  Validation       │    │
│  │  Ki-67 IHC is    │    │  │  QN Only         │    │  └──────────────────┘    │
│  │  subjective.     │    │  │  None (raw)      │    │                          │
│  │  Can ML do       │    │  │                  │    │  ┌──────────────────┐    │
│  │  better?         │    │  └──────────────────┘    │  │  DRUG SENS.       │    │
│  │                  │    │                          │  │  Trametinib       │    │
│  └──────────────────┘    │  ┌──────────────────┐    │  │  p=1.8e-12*       │    │
│                          │  │  DRUG SCREEN     │    │  │  5/6 MAPK/ERK     │    │
│  ┌──────────────────┐    │  │  GDSC2: 295×969  │    │  │  (*Bonferroni     │    │
│  │  KEY RESULT      │    │  │  Mann-Whitney U  │    │  │  survives)        │    │
│  │  AUC 0.97        │    │  └──────────────────┘    │  └──────────────────┘    │
│  │  cross-platform! │    │                          │                          │
│  │                  │    │  ┌──────────────────┐    │  ┌──────────────────┐    │
│  │  Ki-67 r=0.59    │    │  │  COLOUR         │    │  │  SURVIVAL        │    │
│  │  (gene removed)  │    │  │  SCHEME         │    │  │  TCGA log-rank   │    │
│  └──────────────────┘    │  │  #2B3A67 navy   │    │  │  p=0.009         │    │
│                          │  │  #E85D75 coral  │    │  └──────────────────┘    │
│  ┌──────────────────┐    │  │  #F4D35E gold   │    │                          │
│  │  MATH HANDOUT    │    │  └──────────────────┘    │  └──────────────────┘    │
│  │  (100 copies)    │    │                          │  ACKNOWLEDGEMENTS        │
│  └──────────────────┘    │  └──────────────────┘    │  TCGA, GEO, GDSC         │
│                          │                          │  Code: github.com/       │
│                          │                          │  Ronisnotasianfr/        │
│                          │                          │  ColoGrowth-ML           │
└──────────────────────────┴──────────────────────────┴──────────────────────────┘
```

## FONT SIZES (readable at 3ft)
- **Title**: 72pt bold, sans-serif (Arial/Helvetica)
- **Section headers**: 36pt bold
- **Body text**: 28pt
- **Figure labels**: 24pt
- **Footnotes**: 20pt

## PANEL LAYOUT SPECIFICATIONS

### LEFT PANEL — Problem + Motivation + Key Finding (24w × 36h inches)

**Title Block** (8in tall, spans all 3 panels at top)
```
COLON CANCER PROLIFERATION PREDICTOR
A Leakage-Free Machine Learning Pipeline with Cross-Platform Validation
```

**Box 1: The Problem** (28pt body)
```
Current clinical proliferation assessment uses Ki-67 IHC.
Problems: inter-observer variability, single-gene signal, no drug prediction.
Can transcriptomic ML do better?
```

**Box 2: Biological Foundation**
```
10-gene proliferation signature (Whitfield 2002):
MKI67, TOP2A, AURKA, CCNB1, CDK1, PLK1, KIF11, BUB1, KIF20A, CCNA2
→ Binarized at median: High vs Low proliferation class
```

**Box 3: Headline Result** (36pt, coral highlight box)
```
CROSS-PLATFORM AUC = 0.973
Ki-67 correlation r = 0.589 (gene held out)
```

### MIDDLE PANEL — Graphical Abstract + Pipeline (24w × 36h inches)

**GRAPHICAL ABSTRACT** (top half, occupies ~18in height, printed as large diagram)

```
┌──────────────────────────────────────────────────────────────────────┐
│                         PIPELINE OVERVIEW                            │
│                                                                      │
│  ┌──────────┐   ┌──────────────┐   ┌─────────────────────────────┐  │
│  │ GEO DATA  │   │ REMOVE 10    │   │ STABILITY SELECTOR          │  │
│  │ GSE39582  │──▶│ PROLIF GENES │──▶│ Bootstrap 100x              │  │
│  │ n=585     │   │ (leakage     │   │ Keep features in top K      │  │
│  │ GSE17538  │   │  prevention) │   │ in >=50% of resamples       │  │
│  │ n=232     │   │              │   │ (CUSTOM ALGORITHM)           │  │
│  └──────────┘   └──────────────┘   └──────────┬──────────────────┘  │
│                                                │                     │
│  ┌──────────┐   ┌──────────────┐   ┌──────────▼──────────────────┐  │
│  │ TCGA     │   │ QUANTILE     │   │ 4 MODELS TRAINED             │  │
│  │ RNA-seq  │──▶│ NORMALIZE    │──▶│ LR, RF, XGBoost, MLP         │  │
│  │ n=329    │   │ (align dist) │   │ Nested CV + Platt calib      │  │
│  └──────────┘   └──────────────┘   └──────────┬──────────────────┘  │
│                                                │                     │
│  ┌──────────┐   ┌──────────────┐   ┌──────────▼──────────────────┐  │
│  │ GDSC2    │   │ BONFERRONI   │   │ RESULTS                      │  │
│  │ 295 DRUGS│──▶│ CORRECTED    │──▶│ AUC 0.973 (RF)               │  │
│  │ 969 LINES│   │ α/295=1.69e-4│   │ Ki-67 r=0.59                 │  │
│  │          │   │ 5/5 MAPK/ERK │   │ Trametinib p=1.8e-12         │  │
│  └──────────┘   └──────────────┘   └──────────────────────────────┘  │
│                                                                      │
│  ★ CUSTOM ALGORITHM: StabilitySelector                              │
│  sklearn-compatible bootstrap feature selector. Novel contribution.  │
└──────────────────────────────────────────────────────────────────────┘
```

**Box: Calibration Benchmark (5 methods)**
```
Method              Description              Best For
─────────────────────────────────────────────────────────
None                Raw scores               RF, XGB (ECE < 0.04)
Platt Scaling       Sigmoid fit on held-out  General purpose
Isotonic            Non-parametric binning   Large calibration sets
QN+Platt            QN alignment → Platt     LR cross-platform
QN Only             Quantile normalization   Comparison baseline

Key Finding: Tree models need NO calibration (ECE 0.032-0.038).
LR needs QN+Platt for cross-platform (ECE drops from 0.12 to 0.04).
```

**Box: Drug Sensitivity Screen (Bonferroni-corrected)**
```
GDSC2: 295 drugs × 969 cell lines
Mann-Whitney U: colon vs other tissues
α_adjusted = 0.05 / 295 = 1.69 × 10⁻⁴
Top 5 drugs ALL survive: Trametinib (p=1.8e-12) → MAPK/ERK pathway
```

### RIGHT PANEL — Results + Conclusions (24w × 36h inches)

**Box 1: Internal Validation (GEO 80/20 split)**
```
Model          AUC     Acc
LR             0.983   0.909
RF             0.988   0.939
XGBoost        0.991   0.927
MLP            0.981   0.897
```

**Box 2: External Validation (GEO → TCGA)**
```
Model          Raw AUC   Cal AUC   Cal Acc
LR             0.974     0.963     0.873
RF             0.962     0.952     0.885
XGBoost        0.971     0.964     0.885
Ensemble       0.978     0.969     0.897
```

**Box 3: Drug Sensitivity (Top 5, Bonferroni survives)**
```
Drug             p-value           Target
Trametinib       1.8e-12           MEK
PD0325901        5.9e-12           MEK
SCH772984        1.1e-10           ERK
Refametinib      2.7e-10           MEK
Selumetinib      1.2e-09           MEK
```

**Box 4: Survival**
```
TCGA PanCancer: log-rank p = 0.009
GEO GSE39582:   log-rank p = 0.037
CPTAC:          underpowered (7 events)
```

### MATH HANDOUT SHEET (100 copies on table, 8.5×11in)

```
Logistic Regression:
P(high|X) = 1 / (1 + exp(-(β₀ + β₁x₁ + ... + β₁₀x₁₀)))

Platt Scaling:
P_cal(y=1|s) = 1 / (1 + exp(A·s + B))

ECE:
ECE = Σ|Bₘ|/N · |oₘ − eₘ|

Schoenfeld Sample Size:
N = (z_α/2 + z_β)² / (p(1−p) · (log HR)²)

Mann-Whitney U:
U = ΣᵢΣⱼ I(xᵢ > yⱼ)

Bonferroni Correction:
α_adjusted = 0.05 / 295 = 1.69 × 10⁻⁴
```

### PRINTING SPECS
- **Printer**: Staples/Bureau Veritas engineering prints
- **Paper**: Semi-gloss poster paper
- **Size**: 72in × 36in (standard tri-fold board)
- **Layout**: 3 columns × 36in each
- **Orientation**: Landscape
- **Margins**: 0.5in all sides
- **Mount**: Foam core board, spray adhesive

### COLOUR PALETTE
- Navy #2B3A67 — headers, borders
- Coral #E85D75 — key numbers, highlights
- Gold #F4D35E — callout boxes
- White #FFFFFF — background
- Dark grey #333333 — body text
