# Author Contributions and Competing Interests

## Author Contributions

> [!NOTE]
> This is a template for the author to confirm and edit before submission. Adapt as appropriate for the target journal's CRediT taxonomy format.

| Role | Contributor |
|:---|:---|
| **Conceptualization** | Rohan Saindane |
| **Data Curation** | Rohan Saindane (GEO GSE39582 download, TCGA-COAD download, preprocessing, probe mapping) |
| **Formal Analysis** | Rohan Saindane (model training, cross-validation, bootstrap confidence intervals, Cox PH modeling, DCA, subgroup analyses) |
| **Methodology** | Rohan Saindane (leakage-free pipeline design, three-way validation architecture, Platt calibration) |
| **Software** | Rohan Saindane (Python implementation: `src/preprocess.py`, `src/train.py`, `src/evaluate.py`, `src/external_validation.py`, `src/survival.py`, `src/complete_analysis.py`) |
| **Visualization** | Rohan Saindane (all figures generated via pipeline scripts) |
| **Writing – Original Draft** | Rohan Saindane |
| **Writing – Review & Editing** | Rohan Saindane |

---

## Data and Code Availability Statement

All data used in this study are derived from publicly available repositories:

- **GEO GSE39582**: Available at https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE39582
- **TCGA-COAD**: Available via the UCSC Xena Browser at https://xenabrowser.net/ and the NCI Genomic Data Commons

The complete code, trained model pipelines, and reproducibility instructions are available at:
**https://github.com/Ronisnotasianfr/colon-cancer-predictor**

---

## Competing Interests Statement

The author declares no competing financial or non-financial interests in relation to the work described.

---

## Funding Statement

This research received no external funding. All computational resources were provided by the author on personal hardware.

---

## Ethical Approval Statement

This study represents a secondary analysis of de-identified, publicly available cancer genomics datasets. No new patient data were collected. No institutional review board (IRB) or ethics committee approval was required for this type of secondary analysis.

---

## Acknowledgements

*(Author to complete if applicable)*

The author acknowledges the open-source scientific Python ecosystem (scikit-learn, XGBoost, lifelines, shap, gseapy, matplotlib, seaborn, reportlab, python-docx) and the public cancer genomics databases (GEO, TCGA) that made this analysis possible.
