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
    build_system_architecture_text,
    build_synthetic_validation_text,
    build_synthetic_validation_table,
)

TITLE = "Leakage-controlled machine learning classifiers for cross-platform colon cancer proliferation prediction from transcriptomic data"
SUBTITLE = ""
AUTHOR_LINE = "Rohan Saindane"

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
        AUTHOR_LINE,
        styles["Meta"],
    ))

    h(story, "Abstract")
    h2(story, "Background")
    p(story, "Proliferation rate is a key determinant of survival and chemotherapy response in colorectal cancer, but current clinical proxies (Ki-67 immunohistochemistry, tumor staging) have well-known limitations. This study evaluates whether machine learning classifiers can predict proliferation status from transcriptomic data in a methodologically rigorous, leakage-free framework.")
    h2(story, "Methods")
    p(story, "Four classifiers (Logistic Regression, Random Forest, XGBoost, MLP) were trained on GEO microarray data (GSE39582, n = 585) to predict high vs. low proliferation after removing the ten cell-cycle genes that define the target. All preprocessing was wrapped in scikit-learn Pipelines with nested 5-fold cross-validation to prevent data leakage. External validation was performed on TCGA-COAD (RNA-seq, n = 322) and CPTAC-COAD (proteogenomics, n = 105). Platt scaling corrected cross-platform probability shifts. A synthetic ground-truth experiment tested whether the pipeline recovers known signal genes.")
    h2(story, "Results")
    p(story, build_abstract(metrics, stats))
    h2(story, "Conclusions")
    p(story, "Leakage-controlled ML classifiers achieve high cross-platform accuracy for colon cancer proliferation classification, but the effect size is large (d = 2.3), and proliferation does not independently predict survival after stage adjustment. The framework prioritizes methodological rigor over inflated performance claims and provides a reproducible template for transcriptomic classifier development.")
    story.append(Spacer(1, 10))

    h2(story, "Keywords")
    p(story, "colon cancer; proliferation; machine learning; cross-platform validation; feature selection; transcriptomics; leakage control; reproducibility")

    story.append(Spacer(1, 6))

    h(story, "Introduction")
    p(story, build_introduction_p1())
    p(story, build_introduction_p2())

    story.append(PageBreak())

    h(story, "Materials and Methods")
    h2(story, "System Architecture")
    p(story, build_system_architecture_text())
    p(story, build_methods_p1(stats))

    data_rows = [["Dataset file", "Samples", "Features/columns", "Class balance"]]
    data_rows.extend(list(dataset_table_rows(stats)))
    table(story, data_rows, [1.75 * inch, 0.85 * inch, 1.35 * inch, 1.75 * inch],
          "Table 1. Processed data files used for this leakage-corrected report.")

    # Task 3.3 - Three-way validation split justification
    p(story, build_methods_split_justification())
    
    # Replace with clean table-based workflow description
    workflow_rows = [
        ["Stage", "Cohort", "Split / Purpose"],
        ["Training", "GEO (n=585)", "80% training pool (5-fold CV) + 20% holdout"],
        ["External validation", "TCGA-COAD (n=322)", "50% Platt calibration + 50% evaluation"],
        ["External validation", "CPTAC-COAD (n=105)", "50% Platt calibration + 50% evaluation"],
    ]
    table(story, workflow_rows, [1.2 * inch, 1.5 * inch, 2.8 * inch],
          "Figure 7. Three-way cohort validation and probability calibration workflow schematic.")

    p(story, build_methods_p2())
    
    p(story, build_methods_leakage_paragraph())
    
    hp_rows = [["Model", "Hyperparameter", "Search Range", "Selected", "Notes"]]
    hp_rows.extend(list(build_hyperparameters_table()))
    table(story, hp_rows, [1.3 * inch, 1.4 * inch, 1.3 * inch, 0.9 * inch, 1.85 * inch],
          "Table 2. Hyperparameter optimization search ranges and selected settings.", size=7.5)

    h2(story, "Software Environment")
    sw_rows = [
        ["Software", "Version", "Purpose"],
        ["Python", "3.14.5", "Core programming language"],
        ["scikit-learn", "1.7+", "ML models, Pipelines, cross-validation"],
        ["XGBoost", "2.1+", "Gradient-boosted tree classifier"],
        ["NumPy", "2.x", "Numerical computing"],
        ["pandas", "2.x", "Data manipulation"],
        ["matplotlib", "3.x", "Figure generation"],
        ["SciPy", "1.x", "Statistical tests"],
        ["lifelines", "0.30+", "Survival analysis (Kaplan-Meier, Cox PH)"],
        ["SHAP", "0.46+", "Feature importance interpretation"],
        ["ReportLab", "4.x", "PDF generation"],
        ["Docker", "27+", "Containerized reproducibility"],
    ]
    table(story, sw_rows, [1.4 * inch, 1.2 * inch, 3.2 * inch],
          "Table 3. Software environment and package versions used in this study.", size=7.5)

    story.append(PageBreak())

    h(story, "Results")
    p(story, build_results_opening(metrics))

    perf_rows = [["Model", "CV ROC-AUC (mean +/- std)", "Holdout Accuracy (95% CI)", "Holdout ROC-AUC (95% CI)"]]
    perf_rows.extend([list(r) for r in metrics_table_rows(metrics)])
    table(story, perf_rows, [1.5 * inch, 1.6 * inch, 1.85 * inch, 1.8 * inch],
          "Table 4. Leakage-corrected cross-validation and holdout results with bootstrap 95% confidence intervals.")

    fig(story, results_dir / "calibration_comparison_curves.png",
        "Figure 1. Calibration curves showing observed positive fraction vs. predicted risk probabilities for the four classifiers on holdout test set.",
        width=4.3 * inch)
        
    p(story, build_results_closing(metrics))

    story.append(PageBreak())

    h(story, "Interpretation and Biological Readout")
    p(story, build_interpretation_p1())

    gene_rows = [["Rank", "Gene Symbol", "ANOVA F-Score", "CV Selection Frequency"]]
    gene_rows.extend(list(build_top_genes_table()))
    table(story, gene_rows, [0.7 * inch, 1.7 * inch, 1.8 * inch, 1.85 * inch],
          "Table 5. Top selected transcriptomic feature genes ranked by ANOVA F-score.")

    fig(story, results_dir / "shap_summary_random_forest.png",
        "Figure 2. Random forest SHAP summary showing feature impact (red/high, blue/low expression) on proliferation class prediction.",
        width=3.95 * inch)
        
    p(story, build_interpretation_p2())
      
    fig(story, results_dir / "pathway_enrichment.png",
        "Figure 3. Enriched biological pathways (FDR < 0.05) linked to cell-cycle progression and division.",
        width=4.2 * inch)

    story.append(PageBreak())

    p(story, build_interpretation_p3())
      
    fig(story, results_dir / "clinical_dca.png",
        "Figure 4. Clinical Decision Curve Analysis comparing Net Benefit of model-guided stratification vs. default intervention strategies.",
        width=4.0 * inch)
        
    nnt_rows = [["Model", "Threshold", "Sensitivity", "Specificity", "PPV", "NPV", "NNT"]]
    nnt_rows.extend(list(build_nnt_table()))
    table(story, nnt_rows, [1.4 * inch, 0.7 * inch, 0.9 * inch, 0.9 * inch, 0.9 * inch, 0.9 * inch, 0.7 * inch],
          "Table 6. Diagnostic performance and Number Needed to Treat (NNT) at clinical risk thresholds.")

    story.append(PageBreak())

    h(story, "Subgroup and Prognostic Validation")
    p(story, build_clinical_validation_p1())

    # Table 6: updated to 7 columns!
    sub_rows = [["Subgroup", "N", "Accuracy", "ROC-AUC", "ROC-AUC 95% CI", "Interaction p-value", "Interaction 95% CI"]]
    sub_rows.extend(list(build_subgroups_table()))
    table(story, sub_rows, [1.1 * inch, 0.5 * inch, 0.8 * inch, 0.8 * inch, 1.2 * inch, 1.1 * inch, 1.25 * inch],
          "Table 7. Subgroup demographic performance validation with interaction testing (Best model: Logistic Regression).", size=7.2)

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
          "Table 8. Multivariate Cox Proportional Hazards survival analysis summary.")

    story.append(PageBreak())

    h(story, "Sensitivity and Robustness Analysis")
    p(story, build_sensitivity_p1())

    sens_rows = [["SelectKBest k", "Holdout ROC-AUC (k)", "Variance Threshold (VT)", "Features Passed VT", "Holdout ROC-AUC (VT)"]]
    sens_rows.extend(list(build_sensitivity_table()))
    table(story, sens_rows, [1.2 * inch, 1.45 * inch, 1.5 * inch, 1.35 * inch, 1.45 * inch],
          "Table 9. Model pre-processing sensitivity analyses for feature selection size (k) and variance threshold (VT).")

    h(story, "Synthetic Ground-Truth Validation")
    p(story, build_synthetic_validation_text())
    synth_rows = [["Model", "AUC", "Accuracy", "Signal Genes Selected", "Enrichment"]]
    synth_rows.extend([list(r) for r in build_synthetic_validation_table()])
    table(story, synth_rows, [1.5 * inch, 1.0 * inch, 1.0 * inch, 1.5 * inch, 1.5 * inch],
          "Table 9. Synthetic ground-truth validation: all four models recover 100% of the 20 true signal genes.")

    story.append(PageBreak())

    h(story, "Discussion")
    p(story, build_discussion_paragraph())

    # Task 3.1 - Benchmarking Table
    p(story, build_discussion_benchmarking_intro())
    bench_rows = [["Study", "Year", "Cohort/Platform", "N", "AUC/Accuracy", "Leakage-controlled?", "Cross-platform validated?"]]
    bench_rows.extend(list(build_benchmarking_table()))
    table(story, bench_rows, [1.1 * inch, 0.5 * inch, 1.3 * inch, 0.6 * inch, 1.1 * inch, 1.25 * inch, 1.15 * inch],
          "Table 10. Performance and methodological comparisons with published signatures.", size=7.2)

    # Task 3.4 - Biological Mechanism Discussion Expansion
    p(story, build_discussion_pathway_expansion())

    h(story, "Limitations and Future Directions")
    h2(story, "Limitations")
    for item in [
        "Sample size: GEO GSE39582 (n=585) is moderate. CPTAC-COAD (n=105) has only 7 survival events, substantially limiting statistical power for survival analysis on that cohort.",
        "Target binarization at the median is standard but arbitrary. Continuous risk scores would preserve more information.",
        "Proliferation classes barely overlap (Cohen's d = 2.3), making binary separation easier than typical clinical ML tasks. The model's AUC > 0.97 should be interpreted in this context.",
        "Cox regression showed proliferation class was not an independent predictor after adjusting for stage. This limits the clinical utility of the prognostic claim.",
        "Microarray and RNA-seq platforms have different dynamic ranges, requiring post-hoc Platt calibration for cross-platform probability calibration.",
        "The drug screen uses GDSC2 cell line data and does not account for tissue-specific expression differences across cell lines.",
        "SHAP scores reflect correlation, not causation. Top features should not be interpreted as therapeutic targets without experimental validation.",
        "Only 20% of the data was held out for final evaluation. A larger holdout would give more stable performance estimates.",
    ]:
        story.append(Paragraph("- " + item, styles["PaperBullet"]))
    story.append(Spacer(1, 4))

    h2(story, "Future Work")
    for item in [
        "Prospective validation on new COAD biopsy cohorts with matched RNA-seq and Ki-67 IHC.",
        "qPCR knockdown of top SHAP genes (RPS3, RPS11, MCM10) to distinguish correlation from causation.",
        "Continuous risk score modeling (instead of binarized proliferation class) for survival analysis.",
        "Integration of CNVs, somatic mutations, and methylation data into the feature space.",
        "Submission of the trained pipelines as a web API for independent validation by other labs.",
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
        "Overmeer, R. M. et al. Replication factor C recruits DNA polymerase delta to sites of nucleotide excision repair but is not required for PCNA recruitment. Molecular and Cellular Biology, 2010. DOI: 10.1128/MCB.00285-10",
        "Guinney, J. et al. The consensus molecular subtypes of colorectal cancer. Nature Medicine, 2015. DOI: 10.1038/nm.3967",
        "The Cancer Genome Atlas Network. Comprehensive molecular characterization of human colon and rectal cancer. Nature, 2012. DOI: 10.1038/nature11252",
        "Platt, J. Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. Advances in Large Margin Classifiers, 1999.",
        "Yang, W. et al. Genomics of Drug Sensitivity in Cancer (GDSC): a resource for therapeutic biomarker discovery in cancer cells. Nucleic Acids Research, 2013. DOI: 10.1093/nar/gks1111",
        "Meinshausen, N. and Buhlmann, P. Stability selection. Journal of the Royal Statistical Society: Series B, 2010. DOI: 10.1111/j.1467-9868.2010.00740.x",
        "Cohen, J. Statistical Power Analysis for the Behavioral Sciences. 2nd ed. Lawrence Erlbaum Associates, 1988.",
        "Vasaikar, S. et al. Proteogenomic analysis of human colon cancer reveals new therapeutic opportunities. Cell, 2019. DOI: 10.1016/j.cell.2019.03.030",
        "Pedregosa, F. et al. Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 2011.",
        "Chen, T. and Guestrin, C. XGBoost: A scalable tree boosting system. Proceedings of the 22nd ACM SIGKDD, 2016. DOI: 10.1145/2939672.2939785",
        "Varoquaux, G. et al. Scikit-learn: Getting started for machine learning. Frontiers in Neuroinformatics, 2015.",
        "Cox, D. R. Regression models and life-tables. Journal of the Royal Statistical Society: Series B, 1972.",
        "Kaplan, E. L. and Meier, P. Nonparametric estimation from incomplete observations. Journal of the American Statistical Association, 1958.",
        "Niculescu-Mizil, A. and Caruana, R. Predicting good probabilities with supervised learning. ICML, 2005.",
        "Zadrozny, B. and Elkan, C. Transforming classifier scores into accurate multiclass probability estimates. KDD, 2002.",
        "Vickers, A. J. and Elkin, E. B. Decision curve analysis: a novel method for evaluating prediction models. Medical Decision Making, 2006. DOI: 10.1177/0272989X06295361",
        "Subramanian, A. et al. Gene set enrichment analysis: a knowledge-based approach for interpreting genome-wide expression profiles. PNAS, 2005. DOI: 10.1073/pnas.0506580102",
        "Ashburner, M. et al. Gene Ontology: tool for the unification of biology. Nature Genetics, 2000. DOI: 10.1038/75556",
        "Kanehisa, M. and Goto, S. KEGG: Kyoto Encyclopedia of Genes and Genomes. Nucleic Acids Research, 2000.",
        "Davoli, T. et al. Cumulative haploinsufficiency and triplosensitivity drive aneuploidy patterns and shape the cancer genome. Cell, 2013. DOI: 10.1016/j.cell.2013.10.011",
        "Hanahan, D. and Weinberg, R. A. Hallmarks of cancer: the next generation. Cell, 2011. DOI: 10.1016/j.cell.2011.02.013",
        "Stratton, M. R. et al. The cancer genome. Nature, 2009. DOI: 10.1038/nature07943",
        "Sadanandam, A. et al. A colorectal cancer classification system that associates cellular phenotype and responses to therapy. Nature Medicine, 2013. DOI: 10.1038/nm.3175",
        "De Sousa E Melo, F. et al. Poor-prognosis colon cancer is defined by a molecularly distinct subtype and develops from serrated precursor lesions. Nature Medicine, 2013. DOI: 10.1038/nm.3174",
        "Isella, C. et al. Stromal contribution to the colorectal cancer transcriptome. Nature Genetics, 2015. DOI: 10.1038/ng.3224",
        "Calon, A. et al. Dependency of colorectal cancer on a TGF-beta-driven program in stromal cells. Nature Genetics, 2012. DOI: 10.1038/ng.2249",
        "Siegel, R. L. et al. Colorectal cancer statistics, 2025. CA: A Cancer Journal for Clinicians, 2025. DOI: 10.3322/caac.21873",
        "Bray, F. et al. Global cancer statistics 2022. CA: A Cancer Journal for Clinicians, 2024. DOI: 10.3322/caac.21834",
        "Jass, J. R. Classification of colorectal cancer based on correlation of clinical, morphological and molecular features. Histopathology, 2007. DOI: 10.1111/j.1365-2559.2007.02649.x",
    ]
    for i, ref in enumerate(refs, 1):
        story.append(Paragraph(f"{i}. {ref}", styles["Ref"]))

    h(story, "Declarations")

    h2(story, "Ethics Approval and Consent to Participate")
    p(story,
      "This study uses only de-identified public datasets (GEO, TCGA, CPTAC). "
      "Institutional review board (IRB) approval was not required for secondary analysis of "
      "de-identified data under 45 CFR 46.104.")

    h2(story, "Data Availability")
    p(story,
      "GEO GSE39582: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE39582. "
      "TCGA-COAD: https://xenabrowser.net/ (UCSC Xena). "
      "CPTAC-COAD: https://cptac-data-portal.georgetown.edu/. "
      "GDSC drug sensitivity data: https://www.cancerrxgene.org/. "
      "All processed datasets, trained pipeline objects, and code are available at the "
      "project repository: https://github.com/Ronisnotasianfr/ColoGrowth-ML.")

    h2(story, "Code Availability")
    p(story,
      "The full ColoGrowth-ML pipeline (preprocessing, feature selection, model training, "
      "evaluation, and paper generation) is available at "
      "https://github.com/Ronisnotasianfr/ColoGrowth-ML under the MIT license. "
      "A Docker image is provided for fully reproducible execution.")

    h2(story, "Competing Interests")
    p(story, "The author declares no competing interests.")

    h2(story, "Funding")
    p(story, "This research was conducted independently with no external funding.")

    h2(story, "Acknowledgments")
    p(story,
      "The author thanks the developers and maintainers of the GEO, TCGA, CPTAC, and GDSC "
      "public data resources for making their data freely available to the research community.")

    h2(story, "AI Disclosure")
    p(story,
      "Claude (Anthropic) was used as a coding assistant during implementation. All study design "
      "decisions, data interpretation, statistical analysis, and written conclusions are the author's "
      "own. The model architecture, leakage-control strategy, and validation framework were designed "
      "by the author. Claude assisted with debugging syntax errors, generating table formats, and "
      "optimizing parallel computation settings. Full prompt logs are available in the project repository.")

    h2(story, "Author Contributions")
    p(story,
      "RS conceived the study, designed the computational framework, implemented all software, "
      "performed the analyses, interpreted the results, and wrote the manuscript.")

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
    parser.add_argument("--dataset", default="geo", choices=["geo", "geo_pan", "tcga", "tcga_pan", "synthetic"])
    args = parser.parse_args()

    base = project_root()
    results_dir = base / "results"
    data_dir = base / "data" / "processed"
    pdf_path = base / "paper" / "biorxiv_manuscript.pdf"

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
