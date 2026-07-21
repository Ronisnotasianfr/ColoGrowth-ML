"""
build_pdf.py - Assemble the PDF research paper from leakage-fixed pipeline outputs.
Updated to include all Phase 3 polish improvements (benchmarking, subgroup interactions, split justifications).

Usage:
    python paper/build_pdf.py --dataset geo
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Table,
    TableStyle,
    PageBreak,
    KeepTogether,
)

from paper_metrics import (
    project_root,
    load_leakage_fixed_metrics,
    load_dataset_stats,
    build_abstract,
    build_methods_leakage_paragraph,
    build_results_opening,
    build_results_closing,
    build_discussion_paragraph,
    metrics_table_rows,
    dataset_table_rows,
    build_hyperparameters_table,
    build_top_genes_table,
    build_nnt_table,
    build_subgroups_table,
    build_cox_table,
    build_sensitivity_table,
    build_benchmarking_table,
    build_methods_split_justification,
    build_discussion_pathway_expansion,
    build_introduction_p1,
    build_introduction_p2,
    build_methods_p1,
    build_methods_p2,
    build_interpretation_p1,
    build_interpretation_p2,
    build_interpretation_p3,
    build_clinical_validation_p1,
    build_clinical_validation_p2,
    build_clinical_validation_p3,
    build_sensitivity_p1,
    build_discussion_benchmarking_intro,
)

TITLE = "Cross-Platform Colon Cancer Proliferation Classification via Leakage-Free ML"
SUBTITLE = "Independent computational biology research report prepared for peer-reviewed journal submission"
AUTHOR_LINE = "Rohan Saindane | Independent Research Project | June 2026"

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(
    name="PaperTitle", parent=styles["Title"], fontName="Times-Bold",
    fontSize=16, leading=19, alignment=TA_LEFT, spaceAfter=8,
    textColor=colors.HexColor("#141F2B"),
))
styles.add(ParagraphStyle(
    name="PaperSubtitle", parent=styles["Normal"], fontName="Times-Italic",
    fontSize=10.0, leading=13, textColor=colors.HexColor("#555555"), spaceAfter=10,
))
styles.add(ParagraphStyle(
    name="Meta", parent=styles["Normal"], fontName="Times-Roman",
    fontSize=9, leading=11, textColor=colors.HexColor("#555555"), spaceAfter=12,
))
styles.add(ParagraphStyle(
    name="H1", parent=styles["Heading1"], fontName="Times-Bold",
    fontSize=13.0, leading=16, textColor=colors.HexColor("#1F4E79"),
    spaceBefore=10, spaceAfter=5,
))
styles.add(ParagraphStyle(
    name="H2", parent=styles["Heading2"], fontName="Times-Bold",
    fontSize=11.2, leading=14, textColor=colors.HexColor("#2D4F6C"),
    spaceBefore=7, spaceAfter=4,
))
styles.add(ParagraphStyle(
    name="BodyJust", parent=styles["BodyText"], fontName="Times-Roman",
    fontSize=10.0, leading=13.0, alignment=TA_JUSTIFY, spaceAfter=6,
))
styles.add(ParagraphStyle(
    name="Caption", parent=styles["BodyText"], fontName="Times-Italic",
    fontSize=8.3, leading=10, alignment=TA_CENTER,
    textColor=colors.HexColor("#555555"), spaceBefore=2, spaceAfter=6,
))
styles.add(ParagraphStyle(
    name="Ref", parent=styles["BodyText"], fontName="Times-Roman",
    fontSize=9.0, leading=11.0, leftIndent=14, firstLineIndent=-14, spaceAfter=3,
))
styles.add(ParagraphStyle(
    name="PaperBullet", parent=styles["BodyText"], fontName="Times-Roman",
    fontSize=9.8, leading=12.0, leftIndent=13, bulletIndent=3, spaceAfter=3,
))
styles.add(ParagraphStyle(
    name="SchematicStyle", parent=styles["Normal"], fontName="Courier",
    fontSize=7.2, leading=8.5, textColor=colors.HexColor("#141F2B"),
    spaceBefore=4, spaceAfter=4,
))


def p(story, text, style="BodyJust"):
    story.append(Paragraph(text, styles[style]))


def h(story, text):
    story.append(Paragraph(text, styles["H1"]))


def h2(story, text):
    story.append(Paragraph(text, styles["H2"]))


def fig(story, path, caption, width=5.0 * inch):
    path = Path(path)
    if not path.exists():
        p(story, f"[Figure unavailable: {path}]", "BodyJust")
        story.append(Paragraph(caption, styles["Caption"]))
        return
    img = Image(str(path))
    scale = width / img.drawWidth
    img.drawWidth *= scale
    img.drawHeight *= scale
    story.append(KeepTogether([img, Paragraph(caption, styles["Caption"])]))


def schematic(story, text, caption):
    html_text = text.replace(' ', '&nbsp;').replace('\n', '<br/>')
    p_schem = Paragraph(html_text, styles["SchematicStyle"])
    story.append(KeepTogether([p_schem, Paragraph(caption, styles["Caption"])]))


def table(story, data, widths, caption, size=7.8):
    formatted_data = []
    cell_style = ParagraphStyle(name="Cell", fontName="Times-Roman", fontSize=size, leading=size + 2)
    hdr_style = ParagraphStyle(name="HdrCell", fontName="Times-Bold", fontSize=size + 0.2, leading=size + 2.2, textColor=colors.HexColor("#141F2B"))
    
    for row_idx, row in enumerate(data):
        formatted_row = []
        for col_idx, cell in enumerate(row):
            if isinstance(cell, str) and (len(cell) > 20 or '\n' in cell or '(' in cell):
                style = hdr_style if row_idx == 0 else cell_style
                formatted_row.append(Paragraph(cell, style))
            else:
                formatted_row.append(cell)
        formatted_data.append(formatted_row)

    t = Table(formatted_data, colWidths=widths, repeatRows=1, hAlign="CENTER")
    t.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, -1), "Times-Roman", size),
        ("FONT", (0, 0), (-1, 0), "Times-Bold", size + 0.2),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F4F8")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#141F2B")),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#C9D0DA")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(KeepTogether([t, Paragraph(caption, styles["Caption"])]))


def build_story(metrics, stats, results_dir):
    story = []

    story.append(Paragraph(TITLE, styles["PaperTitle"]))
    story.append(Paragraph(SUBTITLE, styles["PaperSubtitle"]))
    story.append(Paragraph(
        f"{AUTHOR_LINE} | Dataset: {stats['dataset'].upper()} | Leakage-corrected manuscript",
        styles["Meta"],
    ))

    h(story, "Abstract")
    p(story, build_abstract(metrics, stats))
    story.append(Spacer(1, 10))

    h(story, "1. Introduction")
    p(story, build_introduction_p1())
    p(story, build_introduction_p2())

    story.append(PageBreak())

    h(story, "2. Materials and Methods")
    p(story, build_methods_p1(stats))

    data_rows = [["Dataset file", "Samples", "Features/columns", "Class balance"]]
    data_rows.extend(list(dataset_table_rows(stats)))
    table(story, data_rows, [1.75 * inch, 0.85 * inch, 1.35 * inch, 1.75 * inch],
          "Table 1. Processed data files used for this leakage-corrected report.")

    # Task 3.3 - Three-way validation split justification
    p(story, build_methods_split_justification())
    
    schem_text = (
        "                     [ GEO Cohort (n=585) ]\n"
        "                               |\n"
        "                 +-------------+-------------+\n"
        "                 v                           v\n"
        "        GEO-Train (80%, n=468)      GEO-Holdout (20%, n=117)\n"
        "                 |                           |\n"
        "                 +-> [Feature Selection]     +-> [Evaluate Model]\n"
        "                 +-> [GridSearchCV Tuning]   |\n"
        "                 +-> [Fit Model Coefficients]|\n"
        "                                             |\n"
        "                     [ TCGA Cohort (n=322) ] <-+\n"
        "                               |\n"
        "                 +-------------+-------------+\n"
        "                 v                           v\n"
        "       TCGA-Calib (50%, n=161)      TCGA-Eval (50%, n=161)\n"
        "                 |                           |\n"
        "                 +-> [Platt Scaling Fit]     +-> [Evaluate Calibrated AUC/Acc]"
    )
    schematic(story, schem_text, "Figure 7. Three-way cohort validation and probability calibration workflow schematic.")

    p(story, build_methods_p2())
    
    p(story, build_methods_leakage_paragraph())
    
    hp_rows = [["Model", "Hyperparameter", "Search Range", "Selected", "Notes"]]
    hp_rows.extend(list(build_hyperparameters_table()))
    table(story, hp_rows, [1.3 * inch, 1.4 * inch, 1.3 * inch, 0.9 * inch, 1.85 * inch],
          "Table 2. Hyperparameter optimization search ranges and selected settings.", size=7.5)

    story.append(PageBreak())

    h(story, "3. Results")
    p(story, build_results_opening(metrics))

    perf_rows = [["Model", "CV ROC-AUC (mean +/- std)", "Holdout Accuracy (95% CI)", "Holdout ROC-AUC (95% CI)"]]
    perf_rows.extend([list(r) for r in metrics_table_rows(metrics)])
    table(story, perf_rows, [1.5 * inch, 1.6 * inch, 1.85 * inch, 1.8 * inch],
          "Table 3. Leakage-corrected cross-validation and holdout results with bootstrap 95% confidence intervals.")

    fig(story, results_dir / "calibration_comparison_curves.png",
        "Figure 1. Calibration curves showing observed positive fraction vs. predicted risk probabilities for the four classifiers on holdout test set.",
        width=4.3 * inch)
        
    p(story, build_results_closing(metrics))

    story.append(PageBreak())

    h(story, "4. Interpretation and Biological Readout")
    p(story, build_interpretation_p1())

    gene_rows = [["Rank", "Gene Symbol", "ANOVA F-Score", "CV Selection Frequency"]]
    gene_rows.extend(list(build_top_genes_table()))
    table(story, gene_rows, [0.7 * inch, 1.7 * inch, 1.8 * inch, 1.85 * inch],
          "Table 4. Top selected transcriptomic feature genes ranked by ANOVA F-score.")

    fig(story, results_dir / "shap_summary_random_forest.png",
        "Figure 2. Random forest SHAP summary showing feature impact (red/high, blue/low expression) on proliferation class prediction.",
        width=3.95 * inch)
        
    p(story, build_interpretation_p2())
      
    fig(story, results_dir / "pathway_enrichment.png",
        "Figure 3. Enriched biological pathways (FDR < 0.05) representing transcriptomic cascade signatures downstream of cancer cell division.",
        width=4.2 * inch)

    story.append(PageBreak())

    p(story, build_interpretation_p3())
      
    fig(story, results_dir / "clinical_dca.png",
        "Figure 4. Clinical Decision Curve Analysis comparing Net Benefit of model-guided stratification vs. default intervention strategies.",
        width=4.0 * inch)
        
    nnt_rows = [["Model", "Threshold", "Sensitivity", "Specificity", "PPV", "NPV", "NNT"]]
    nnt_rows.extend(list(build_nnt_table()))
    table(story, nnt_rows, [1.4 * inch, 0.7 * inch, 0.9 * inch, 0.9 * inch, 0.9 * inch, 0.9 * inch, 0.7 * inch],
          "Table 5. Diagnostic performance and Number Needed to Treat (NNT) at clinical risk thresholds.")

    story.append(PageBreak())

    h(story, "5. Demographic Subgroups & Prognostic Validation")
    p(story, build_clinical_validation_p1())

    # Table 6: updated to 7 columns!
    sub_rows = [["Subgroup", "N", "Accuracy", "ROC-AUC", "ROC-AUC 95% CI", "Interaction p-value", "Interaction 95% CI"]]
    sub_rows.extend(list(build_subgroups_table()))
    table(story, sub_rows, [1.1 * inch, 0.5 * inch, 0.8 * inch, 0.8 * inch, 1.2 * inch, 1.1 * inch, 1.25 * inch],
          "Table 6. Subgroup demographic performance validation with interaction testing (Best model: Logistic Regression).", size=7.2)

    p(story, build_clinical_validation_p2())
      
    fig(story, results_dir / "kaplan_meier_geo.png",
        "Figure 5. Kaplan-Meier overall survival curves comparing predicted high vs. low proliferation cohorts in GEO.",
        width=3.95 * inch)
        
    fig(story, results_dir / "kaplan_meier_stage_stratified.png",
        "Figure 6. Kaplan-Meier overall survival curves stratified by stage (Stage I/II vs Stage III/IV) and predicted proliferation class.",
        width=4.15 * inch)

    p(story, build_clinical_validation_p3())

    cox_rows = [["Predictor / Covariate", "Coefficient", "Hazard Ratio (HR)", "95% CI", "p-value"]]
    cox_rows.extend(list(build_cox_table()))
    table(story, cox_rows, [1.8 * inch, 1.0 * inch, 1.3 * inch, 1.3 * inch, 1.0 * inch],
          "Table 7. Multivariate Cox Proportional Hazards survival analysis summary.")

    story.append(PageBreak())

    h(story, "6. Pre-Processing Sensitivity & Robustness Analysis")
    p(story, build_sensitivity_p1())

    sens_rows = [["SelectKBest k", "Holdout ROC-AUC (k)", "Variance Threshold (VT)", "Features Passed VT", "Holdout ROC-AUC (VT)"]]
    sens_rows.extend(list(build_sensitivity_table()))
    table(story, sens_rows, [1.2 * inch, 1.45 * inch, 1.5 * inch, 1.35 * inch, 1.45 * inch],
          "Table 8. Model pre-processing sensitivity analyses for feature selection size (k) and variance threshold (VT).")

    h(story, "7. Discussion")
    p(story, build_discussion_paragraph())

    # Task 3.1 - Benchmarking Table
    p(story, build_discussion_benchmarking_intro())
    bench_rows = [["Study", "Year", "Cohort/Platform", "N", "AUC/Accuracy", "Leakage-controlled?", "Cross-platform validated?"]]
    bench_rows.extend(list(build_benchmarking_table()))
    table(story, bench_rows, [1.1 * inch, 0.5 * inch, 1.3 * inch, 0.6 * inch, 1.1 * inch, 1.25 * inch, 1.15 * inch],
          "Table 9. Performance and methodological comparisons with published signatures.", size=7.2)

    # Task 3.4 - Biological Mechanism Discussion Expansion
    p(story, build_discussion_pathway_expansion())

    h(story, "8. Limitations and Future Directions")
    h2(story, "Limitations")
    for item in [
        "GEO GSE39582 (n=585) is moderate. CPTAC-COAD (n=105) has only 7 survival events.",
        "Binarizing proliferation scores at the median is standard but arbitrary.",
        "Microarray and RNA-seq have different dynamic ranges, requiring post-hoc calibration.",
        "SHAP scores reflect correlation, not causation.",
    ]:
        story.append(Paragraph("- " + item, styles["PaperBullet"]))
    story.append(Spacer(1, 4))

    h2(story, "Future Work")
    for item in [
        "Prospective validation of the Top-3 Ensemble on new COAD biopsy cohorts.",
        "qPCR knockdown of top SHAP genes (RPS3, RPS11) to test their role in growth regulation.",
        "Integration of CNVs and somatic mutation data into the feature space.",
    ]:
        story.append(Paragraph("- " + item, styles["PaperBullet"]))
    story.append(Spacer(1, 6))

    h(story, "References")
    refs = [
        "Marisa, L. et al. Gene expression classification of colon cancer into molecular subtypes: characterization, validation, and prognostic value. PLoS Medicine, 2013. DOI: 10.1371/journal.pmed.1001453",
        "Whitfield, M. L. et al. Identification of genes periodically expressed in the human cell cycle and their expression in tumors. Molecular Biology of the Cell, 2002. DOI: 10.1091/mbc.02-02-0030",
        "Lundberg, S. M. and Lee, S.-I. A unified approach to interpreting model predictions. Advances in Neural Information Processing Systems, 2017.",
        "The Cancer Genome Atlas Research Network. TCGA Colon Adenocarcinoma data resource, accessed through the NCI Genomic Data Commons.",
        "National Center for Biotechnology Information Gene Expression Omnibus. GSE39582 dataset record.",
        "Zeng, D.-T. et al. Prognostic role of Ki-67 in colorectal carcinoma: Development and evaluation of machine learning prediction models. World Journal of Clinical Oncology, 2025. DOI: 10.5306/wjco.v16.i8.107306",
        "Agesen, T. H. et al. ColoGuideEx: a robust gene classifier specific for stage II colorectal cancer prognosis. Gut, 2012. DOI: 10.1136/gutjnl-2011-301179",
        "O'Connell, M. J. et al. Relationship between tumor gene expression and recurrence in four independent studies of patients with stage II/III colon cancer treated with surgery alone or surgery plus adjuvant fluorouracil plus leucovorin. Journal of Clinical Oncology, 2010. DOI: 10.1200/JCO.2010.28.9538",
        "Langston, L. D. et al. Mcm10 promotes rapid isomerization of CMG-DNA for replisome bypass of lagging strand DNA blocks. eLife, 2017. DOI: 10.7554/eLife.29118",
        "Bharadwaj, R., Qi, W., and Yu, H. Identification of two novel components of the human NDC80 kinetochore complex. Journal of Biological Chemistry, 2004. DOI: 10.1074/jbc.M310224200",
        "Seipold, S. et al. Non-SMC condensin I complex proteins control chromosome segregation and survival of proliferating cells in the zebrafish neural retina. BMC Developmental Biology, 2009. DOI: 10.1186/1471-213X-9-40",
        "Overmeer, R. M. et al. Replication factor C recruits DNA polymerase delta to sites of nucleotide excision repair but is not required for PCNA recruitment. Molecular and Cellular Biology, 2010. DOI: 10.1128/MCB.00285-10"
    ]
    for i, ref in enumerate(refs, 1):
        story.append(Paragraph(f"{i}. {ref}", styles["Ref"]))

    h(story, "Data and Code Availability")
    p(story,
      "GEO GSE39582 is available at: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE39582. "
      "TCGA-COAD is available at UCSC Xena: https://xenabrowser.net/. "
      "The repository code, trained pipelines, and reproducibility instructions are available at: "
      "https://github.com/Ronisnotasianfr/ColoGrowth-ML.")

    h(story, "Ethical Considerations and AI Disclosure")
    p(story,
      "Secondary analysis of de-identified public datasets did not require institutional review board (IRB) "
      "approval. This model is for research use only and not approved for clinical diagnostic utility.")
    p(story,
      "Claude (Anthropic) was used as a coding assistant during implementation. All scientific decisions, "
      "study design, data interpretation, and conclusions are the author's own.")

    return story


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Times-Roman", 8)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawString(0.72 * inch, 0.42 * inch,
                      "ColoGrowth-ML | Leakage-free cross-platform validation study")
    canvas.drawRightString(7.78 * inch, 0.42 * inch, f"Page {doc.page}")
    canvas.restoreState()


def main():
    parser = argparse.ArgumentParser(description="Build PDF research paper from pipeline metrics")
    parser.add_argument("--dataset", default="geo", choices=["geo", "tcga", "synthetic"])
    args = parser.parse_args()

    base = project_root()
    results_dir = base / "results"
    data_dir = base / "data" / "processed"
    pdf_path = base / "paper" / "colon_cancer_growth_prediction_research_paper.pdf"

    metrics = load_leakage_fixed_metrics(args.dataset, results_dir)
    stats = load_dataset_stats(args.dataset, data_dir)
    story = build_story(metrics, stats, results_dir)

    pdf = SimpleDocTemplate(
        str(pdf_path), pagesize=letter,
        rightMargin=0.72 * inch, leftMargin=0.72 * inch,
        topMargin=0.65 * inch, bottomMargin=0.65 * inch,
    )
    pdf.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"PDF paper compiled at {pdf_path}")


if __name__ == "__main__":
    main()
