"""
build_fpdf_paper.py — Generate a publication-quality PDF via fpdf2
"""

import json, sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fpdf import FPDF
from paper_metrics import (
    project_root, load_leakage_fixed_metrics, load_dataset_stats,
    load_logrank_results, load_cptac_external_results,
    build_abstract, build_introduction_p1, build_introduction_p2,
    build_methods_p1, build_methods_p2, build_methods_leakage_paragraph,
    build_methods_split_justification, build_system_architecture_text,
    build_reproducibility_text, build_synthetic_validation_text,
    build_results_opening, build_results_closing, build_discussion_paragraph,
    build_discussion_pathway_expansion, build_discussion_benchmarking_intro,
    build_interpretation_p1, build_interpretation_p2, build_interpretation_p3,
    build_clinical_validation_p1, build_clinical_validation_p2,
    build_clinical_validation_p3, build_cox_paragraph, build_sensitivity_p1,
    dataset_table_rows, metrics_table_rows, build_hyperparameters_table,
    build_top_genes_table, build_nnt_table, build_subgroups_table,
    build_cox_table, build_sensitivity_table, build_benchmarking_table,
    build_synthetic_validation_table,
)


class PaperPDF(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 7)
            self.set_text_color(100, 100, 100)
            self.cell(0, 4, "ColoGrowth-ML | Leakage-controlled cross-platform study", align="L")
            self.cell(0, 4, f"Page {self.page_no()}", align="R", new_x="LMARGIN", new_y="NEXT")

    def footer(self):
        pass

    def section_title(self, title):
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(20, 30, 50)
        self.cell(0, 7, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def subsection_title(self, title):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(40, 60, 80)
        self.cell(0, 6, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def body_text(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 5, text)
        self.ln(2)

    def bullet(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 5, "- " + text)
        self.ln(1)

    def add_image(self, path, caption, w=150):
        if os.path.exists(path):
            self.image(path, w=w)
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(100, 100, 100)
            self.multi_cell(0, 4, caption)
            self.ln(3)

    def add_table(self, headers, rows, caption, col_widths=None):
        self.set_font("Helvetica", "B", 8)
        self.set_fill_color(240, 243, 248)
        self.set_text_color(20, 30, 50)
        if col_widths is None:
            col_widths = [190 / len(headers)] * len(headers)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 6, h, border=1, fill=True, align="C")
        self.ln()
        self.set_font("Helvetica", "", 8)
        self.set_text_color(30, 30, 30)
        for row in rows:
            for i, cell in enumerate(row):
                self.cell(col_widths[i], 5, str(cell)[:30], border=1, align="C")
            self.ln()
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(100, 100, 100)
        self.multi_cell(0, 4, caption)
        self.ln(3)


def build():
    base = project_root()
    results_dir = base / "results"
    data_dir = base / "data" / "processed"
    metrics = load_leakage_fixed_metrics("geo", results_dir)
    stats = load_dataset_stats("geo", data_dir)

    pdf = PaperPDF("P", "mm", "A4")
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # ── Title ──
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(20, 30, 50)
    pdf.multi_cell(0, 8, "Leakage-controlled machine learning classifiers for cross-platform\ncolon cancer proliferation prediction from transcriptomic data", align="C")
    pdf.ln(3)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 6, "Rohan Saindane", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8)

    # ── Abstract ──
    pdf.section_title("Abstract")
    pdf.body_text(
        "Machine learning classifiers for cancer transcriptomics often suffer from data leakage, "
        "poor cross-platform generalization, and inflated performance claims. This study presents "
        "a leakage-controlled framework for predicting proliferation status in colon adenocarcinoma "
        "using four classifier classes trained on GEO microarray data (GSE39582, n = 585) after "
        "removing the ten cell-cycle genes that define the target. All preprocessing was wrapped "
        "in scikit-learn Pipelines with nested cross-validation to prevent information flow between "
        "training and test folds. External validation on independent cohorts - TCGA-COAD (RNA-seq, "
        "n = 322) and CPTAC-COAD (proteogenomics, n = 105) - gave ROC-AUCs above 0.97 after "
        "cross-platform calibration. A synthetic ground-truth experiment confirmed that the pipeline "
        "recovers all true signal genes at 7.7x enrichment. Despite high classification performance, "
        "the effect size is large (Cohen's d = 2.3), and Cox regression showed proliferation does "
        "not independently predict survival after stage adjustment. These results demonstrate that "
        "leakage-controlled pipelines can generalize across platforms but underscore the importance "
        "of reporting effect size and independent prognostic value alongside accuracy metrics."
    )
    pdf.body_text("Keywords: colon cancer; proliferation; machine learning; cross-platform validation; feature selection; transcriptomics; leakage control")
    pdf.ln(2)

    # ── Introduction ──
    pdf.section_title("1. Introduction")
    pdf.body_text(build_introduction_p1())
    pdf.body_text(build_introduction_p2())

    # ── Methods ──
    pdf.section_title("2. Materials and Methods")
    pdf.subsection_title("2.1 System Architecture")
    pdf.body_text(build_system_architecture_text())

    pdf.subsection_title("2.2 Datasets and Preprocessing")
    pdf.body_text(build_methods_p1(stats))
    d_rows = [["Dataset file", "Samples", "Features", "Class balance"]]
    d_rows.extend(list(dataset_table_rows(stats)))
    pdf.add_table(d_rows[0], d_rows[1:], "Table 1. Processed datasets.")
    pdf.body_text(build_methods_split_justification())

    pdf.subsection_title("2.3 Model Training and Evaluation")
    pdf.body_text(build_methods_p2())
    pdf.body_text(build_methods_leakage_paragraph())
    hp_rows = [["Model", "Hyperparameter", "Search Range", "Selected", "Notes"]]
    hp_rows.extend(list(build_hyperparameters_table()))
    pdf.add_table(hp_rows[0], hp_rows[1:], "Table 2. Hyperparameter search ranges.", [38, 38, 38, 30, 46])

    pdf.subsection_title("2.4 Software Environment")
    sw = [
        ["Python 3.14", "scikit-learn 1.7+", "XGBoost 2.1+"],
        ["NumPy 2.x", "pandas 2.x", "matplotlib 3.x"],
        ["SciPy 1.x", "lifelines 0.30+", "SHAP 0.46+"],
        ["ReportLab 4.x", "fpdf2", "Docker 27+"],
    ]
    pdf.add_table(["Core", "ML", "Stats/Viz"], sw, "Table 3. Software versions.")
    pdf.body_text(build_reproducibility_text())

    # ── Results ──
    pdf.section_title("3. Results")
    pdf.body_text(build_results_opening(metrics))
    perf_rows = [["Model", "CV ROC-AUC", "Holdout Acc", "Holdout AUC"]]
    perf_rows.extend([list(r) for r in metrics_table_rows(metrics)])
    pdf.add_table(perf_rows[0], perf_rows[1:], "Table 4. Model performance.", [47, 48, 48, 47])

    if os.path.exists(str(results_dir / "calibration_comparison_curves.png")):
        pdf.add_image(str(results_dir / "calibration_comparison_curves.png"),
                      "Figure 1. Calibration curves.", w=160)

    pdf.body_text(build_results_closing(metrics))

    # ── Interpretation ──
    pdf.section_title("4. Interpretation and Biological Readout")
    pdf.body_text(build_interpretation_p1())
    gene_rows = [["Rank", "Gene", "F-Score", "Sel. Freq."]]
    gene_rows.extend(list(build_top_genes_table()))
    pdf.add_table(gene_rows[0], gene_rows[1:], "Table 5. Top genes.", [25, 55, 55, 55])

    if os.path.exists(str(results_dir / "shap_summary_random_forest.png")):
        pdf.add_image(str(results_dir / "shap_summary_random_forest.png"),
                      "Figure 2. SHAP summary.", w=140)

    pdf.body_text(build_interpretation_p2())

    if os.path.exists(str(results_dir / "pathway_enrichment.png")):
        pdf.add_image(str(results_dir / "pathway_enrichment.png"),
                      "Figure 3. Enriched pathways (FDR < 0.05) linked to cell-cycle progression and division.", w=150)

    pdf.body_text(build_interpretation_p3())
    if os.path.exists(str(results_dir / "clinical_dca.png")):
        pdf.add_image(str(results_dir / "clinical_dca.png"),
                      "Figure 4. Decision curve analysis.", w=140)

    nnt_rows = [["Model", "Threshold", "Sens.", "Spec.", "PPV", "NPV", "NNT"]]
    nnt_rows.extend(list(build_nnt_table()))
    pdf.add_table(nnt_rows[0], nnt_rows[1:], "Table 6. Clinical performance.", [27, 27, 27, 27, 27, 27, 28])

    # ── Prognostic Validation ──
    pdf.section_title("5. Subgroup and Prognostic Validation")
    pdf.body_text(build_clinical_validation_p1())
    sub_rows = [["Subgroup", "N", "Accuracy", "ROC-AUC", "AUC 95% CI", "Int. p-value", "95% CI"]]
    sub_rows.extend(list(build_subgroups_table()))
    pdf.add_table(sub_rows[0], sub_rows[1:], "Table 7. Subgroup analysis.", [30, 15, 25, 25, 35, 25, 35])

    pdf.body_text(build_clinical_validation_p2())
    if os.path.exists(str(results_dir / "kaplan_meier_geo.png")):
        pdf.add_image(str(results_dir / "kaplan_meier_geo.png"),
                      "Figure 5. KM curves (GEO).", w=140)
    if os.path.exists(str(results_dir / "kaplan_meier_stage_stratified.png")):
        pdf.add_image(str(results_dir / "kaplan_meier_stage_stratified.png"),
                      "Figure 6. KM curves by stage.", w=140)

    pdf.body_text(build_clinical_validation_p3())
    cox_rows = [["Predictor", "Coef", "HR", "95% CI", "p"]]
    cox_rows.extend(list(build_cox_table()))
    pdf.add_table(cox_rows[0], cox_rows[1:], "Table 8. Cox model.", [50, 35, 35, 35, 35])

    # ── Sensitivity ──
    pdf.section_title("6. Sensitivity Analysis")
    pdf.body_text(build_sensitivity_p1())
    sens_rows = [["K", "AUC (k)", "VT", "Features", "AUC (VT)"]]
    sens_rows.extend(list(build_sensitivity_table()))
    pdf.add_table(sens_rows[0], sens_rows[1:], "Table 9. Sensitivity.", [38, 38, 38, 38, 38])

    # ── Synthetic Validation ──
    pdf.section_title("7. Synthetic Ground-Truth Validation")
    pdf.body_text(build_synthetic_validation_text())
    synth_rows = [["Model", "AUC", "Acc", "Signal", "Enrich."]]
    synth_rows.extend([list(r) for r in build_synthetic_validation_table()])
    pdf.add_table(synth_rows[0], synth_rows[1:], "Table 10. Synthetic validation.", [35, 28, 28, 28, 28, 25, 18])

    # ── Discussion ──
    pdf.section_title("8. Discussion")
    pdf.body_text(build_discussion_paragraph())
    pdf.body_text(build_discussion_benchmarking_intro())
    bench_rows = [["Study", "Year", "N", "AUC", "Leakage?", "Cross-plat?"]]
    bench_rows.extend([list(r) for r in build_benchmarking_table()])
    pdf.add_table(bench_rows[0], bench_rows[1:], "Table 11. Benchmarking.", [35, 22, 30, 22, 30, 25, 26])
    pdf.body_text(build_discussion_pathway_expansion())

    # ── Limitations ──
    pdf.section_title("9. Limitations and Future Directions")
    pdf.subsection_title("Limitations")
    for lim in [
        "Sample size: GEO GSE39582 (n=585) is moderate. CPTAC-COAD (n=105) has only 7 survival events.",
        "Target binarization at the median is standard but arbitrary. Continuous risk scores would preserve more information.",
        "Proliferation classes barely overlap (Cohen's d = 2.3), making binary separation easier than typical clinical ML tasks.",
        "Cox regression showed proliferation class was not an independent predictor after adjusting for stage.",
        "Microarray and RNA-seq platforms have different dynamic ranges, requiring post-hoc Platt calibration.",
        "Drug screen uses GDSC2 cell line data and does not account for tissue-specific expression differences.",
        "SHAP scores reflect correlation, not causation. Top features should not be interpreted as therapeutic targets.",
        "Only 20% of the data was held out. A larger holdout would give more stable performance estimates.",
    ]:
        pdf.bullet(lim)

    pdf.subsection_title("Future Work")
    for fut in [
        "Prospective validation on new COAD biopsy cohorts with matched RNA-seq and Ki-67 IHC.",
        "qPCR knockdown of top SHAP genes (RPS3, RPS11, MCM10) to distinguish correlation from causation.",
        "Continuous risk score modeling instead of binarized proliferation class.",
        "Integration of CNVs, somatic mutations, and methylation data into the feature space.",
        "Submission of trained pipelines as a web API for independent validation.",
    ]:
        pdf.bullet(fut)

    # ── Declarations ──
    pdf.section_title("Declarations")
    pdf.subsection_title("Data Availability")
    pdf.body_text("GEO GSE39582: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE39582. TCGA-COAD: https://xenabrowser.net/. CPTAC-COAD: https://cptac-data-portal.georgetown.edu/. Code: https://github.com/Ronisnotasianfr/ColoGrowth-ML.")
    pdf.subsection_title("Competing Interests")
    pdf.body_text("The author declares no competing interests.")
    pdf.subsection_title("Funding")
    pdf.body_text("This research was conducted independently with no external funding.")

    output = str(base / "paper" / "biorxiv_manuscript.pdf")
    pdf.output(output)
    print(f"PDF written to {output}")


if __name__ == "__main__":
    build()
