"""
build_pdf.py - Assemble the PDF research paper from leakage-fixed pipeline outputs.

Usage:
    python paper/build_pdf.py --dataset synthetic
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
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
)

TITLE = "Machine Learning Prediction of Colon Cancer Proliferation Class from Gene Expression and Clinical Features"
SUBTITLE = "Independent computational biology research report prepared for external faculty peer review"
AUTHOR_LINE = "Rohan Saindane | Independent Research Project | June 2026"

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(
    name="PaperTitle", parent=styles["Title"], fontName="Times-Bold",
    fontSize=18, leading=21, alignment=TA_JUSTIFY, spaceAfter=8,
    textColor=colors.HexColor("#141F2B"),
))
styles.add(ParagraphStyle(
    name="PaperSubtitle", parent=styles["Normal"], fontName="Times-Italic",
    fontSize=10.5, leading=13, textColor=colors.HexColor("#555555"), spaceAfter=12,
))
styles.add(ParagraphStyle(
    name="Meta", parent=styles["Normal"], fontName="Times-Roman",
    fontSize=9, leading=11, textColor=colors.HexColor("#555555"), spaceAfter=14,
))
styles.add(ParagraphStyle(
    name="H1", parent=styles["Heading1"], fontName="Times-Bold",
    fontSize=13.2, leading=16, textColor=colors.HexColor("#1F4E79"),
    spaceBefore=8, spaceAfter=5,
))
styles.add(ParagraphStyle(
    name="BodyJust", parent=styles["BodyText"], fontName="Times-Roman",
    fontSize=10.2, leading=13.1, alignment=TA_JUSTIFY, spaceAfter=7,
))
styles.add(ParagraphStyle(
    name="Caption", parent=styles["BodyText"], fontName="Times-Italic",
    fontSize=8.5, leading=10, alignment=TA_CENTER,
    textColor=colors.HexColor("#555555"), spaceBefore=2, spaceAfter=6,
))
styles.add(ParagraphStyle(
    name="Ref", parent=styles["BodyText"], fontName="Times-Roman",
    fontSize=9.2, leading=11.2, leftIndent=14, firstLineIndent=-14, spaceAfter=4,
))
styles.add(ParagraphStyle(
    name="PaperBullet", parent=styles["BodyText"], fontName="Times-Roman",
    fontSize=10, leading=12.2, leftIndent=13, bulletIndent=3, spaceAfter=4,
))


def p(story, text, style="BodyJust"):
    story.append(Paragraph(text, styles[style]))


def h(story, text):
    story.append(Paragraph(text, styles["H1"]))


def fig(story, path, caption, width=5.1 * inch):
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


def table(story, data, widths, caption):
    t = Table(data, colWidths=widths, repeatRows=1, hAlign="CENTER")
    t.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, -1), "Times-Roman", 8.2),
        ("FONT", (0, 0), (-1, 0), "Times-Bold", 8.3),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F4F8")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#141F2B")),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#C9D0DA")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t)
    story.append(Paragraph(caption, styles["Caption"]))


def build_story(metrics, stats, results_dir):
    story = []

    story.append(Paragraph(TITLE, styles["PaperTitle"]))
    story.append(Paragraph(SUBTITLE, styles["PaperSubtitle"]))
    story.append(Paragraph(
        f"{AUTHOR_LINE} | Dataset: {stats['dataset'].upper()} | Leakage-corrected draft",
        styles["Meta"],
    ))

    h(story, "Abstract")
    p(story, build_abstract(metrics, stats))

    h(story, "1. Introduction")
    p(story,
      "Cellular proliferation is a major biological hallmark of cancer progression and a primary indicator of "
      "tumor growth speed. In colon adenocarcinoma (COAD), assessing tumor cell division rates holds profound clinical "
      "value, directly correlating with patient survival outcomes, disease recurrence probability, and "
      "sensitivity to chemotherapeutic agents. In standard clinical practice, proliferation is estimated "
      "via histological staining techniques (e.g., Ki-67 immunohistochemistry) or staging systems. However, "
      "these manual methods can suffer from inter-observer variability and do not capture the broad, underlying "
      "transcriptomic changes associated with cell-cycle deregulation. A computational approach using machine learning "
      "applied to high-throughput gene expression datasets could provide an automated, objective method for tumor growth "
      "classification and reveal novel transcriptional correlates of aggressive tumor division.")
    p(story,
      "This study develops a highly rigorous, leakage-free machine learning framework to classify colon cancer "
      "samples into high vs. low proliferation states using gene expression profiles (microarray and RNA-seq) "
      "and clinical covariates. A key challenge addressed is the prevention of target leakage: the 10 hallmark "
      "proliferation genes utilized to build the target classification metric were completely removed from the feature space "
      "prior to training. The pipeline compares Logistic Regression, Random Forest, XGBoost, and Multilayer Perceptron "
      "models, validating internally via nested cross-validation and externally across platforms (microarray to RNA-seq).")

    story.append(PageBreak())

    h(story, "2. Materials and Methods")
    p(story,
      f"The {stats['dataset'].upper()} processed dataset comprises {stats['n_samples']} samples and "
      f"{stats['n_features']} features after removing target-defining genes. Clinical covariates include age, sex, and tumor stage. "
      f"The target label is balanced with {stats['class_balance']}. The dataset was split into an 80% training pool "
      f"({stats['train_n']} samples) and a 20% stratified holdout test split ({stats['test_n']} samples).")

    data_rows = [["Dataset file", "Samples", "Features/columns", "Class balance"]]
    data_rows.extend(list(dataset_table_rows(stats)))
    table(story, data_rows, [1.75 * inch, 0.85 * inch, 1.35 * inch, 1.75 * inch],
          "Table 1. Processed data files used for this leakage-corrected report.")

    p(story,
      "To predict proliferation classes, four classifiers were wrapped in scikit-learn Pipelines to guarantee strict "
      "data separation during cross-validation. Standard scaling, low-variance feature filtering (threshold = 0.01), "
      "and SelectKBest feature selection (based on the ANOVA F-value) were fit exclusively on the training folds. "
      "Hyperparameters were optimized using GridSearchCV on the training pool.")
    p(story, build_methods_leakage_paragraph())

    story.append(PageBreak())

    h(story, "3. Results")
    p(story, build_results_opening(metrics))

    perf_rows = [["Model", "CV ROC-AUC (mean +/- std)", "Holdout Accuracy", "Holdout ROC-AUC"]]
    perf_rows.extend([list(r) for r in metrics_table_rows(metrics)])
    table(story, perf_rows, [1.55 * inch, 1.55 * inch, 1.1 * inch, 1.1 * inch],
          "Table 2. Leakage-corrected cross-validation and holdout results.")

    fig(story, results_dir / "roc_curves_comparison.png",
        "Figure 1. Holdout ROC curves comparing the four trained classifiers.",
        width=4.85 * inch)
    p(story, build_results_closing(metrics))

    story.append(PageBreak())

    h(story, "4. Interpretation and Biological Readout")
    p(story,
      "To open the 'black box' of our machine learning models, we computed SHAP (SHapley Additive exPlanations) values "
      "for the pipeline-transformed features on the holdout split. These values reflect the marginal contribution of each "
      "gene feature to the model's final prediction score. Because the 10 direct cell-cycle signature genes were removed, "
      "the top SHAP features identify novel, indirect gene pathways associated with cancer cell proliferation rates.")
    fig(story, results_dir / "shap_summary_random_forest.png",
        "Figure 2. Random forest SHAP summary from the leakage-corrected evaluation run.",
        width=4.25 * inch)
    p(story,
      "To clinically validate our computationally derived proliferation classes, we conducted Kaplan-Meier overall survival "
      "analysis. Log-Rank tests were performed to compare the survival probabilities of the high vs. low proliferation cohorts.")
    fig(story, results_dir / f"kaplan_meier_{stats['dataset']}.png",
        f"Figure 3. Kaplan-Meier overall survival curves comparing predicted high vs. low proliferation cohorts ({stats['dataset'].upper()} cohort).",
        width=3.95 * inch)

    h(story, "5. Discussion")
    p(story,
      "By ensuring that feature selection and scaling are restricted to internal training folds, we avoided artificial "
      "inflation of model performance. The biological readouts and survival outcomes suggest that secondary transcriptional pathways "
      "can serve as strong surrogate markers for tumor growth rates.")
    p(story, build_discussion_paragraph())

    h(story, "6. Limitations and Next Steps")
    for item in [
        "Incorporate platform batch-correction algorithms (e.g., ComBat) to align microarray and RNA-seq feature distributions.",
        "Explore probability calibration techniques (e.g., Platt Scaling, Isotonic Regression) to improve cross-cohort accuracy.",
        "Integrate clinical features directly as model features to investigate potential synergistic effects on prediction.",
        "Perform wet-lab qPCR validation on top-performing SHAP gene targets.",
        "Conduct survival modeling (e.g., Cox Proportional Hazards) to evaluate proliferation as an independent prognostic factor.",
    ]:
        story.append(Paragraph("- " + item, styles["PaperBullet"]))
    story.append(Spacer(1, 6))

    h(story, "References")
    refs = [
        "Marisa, L. et al. Gene expression classification of colon cancer into molecular subtypes with distinct clinical and survival characteristics. PLoS Medicine, 2013.",
        "Whitfield, M. L. et al. Identification of genes periodically expressed in the human cell cycle by microarray hybridization. Molecular Biology of the Cell, 2002.",
        "Lundberg, S. M. and Lee, S.-I. A unified approach to interpreting model predictions. Advances in Neural Information Processing Systems, 2017.",
        "The Cancer Genome Atlas Research Network. TCGA Colon Adenocarcinoma data resource, accessed through the NCI Genomic Data Commons.",
        "National Center for Biotechnology Information Gene Expression Omnibus. GSE39582 dataset record.",
    ]
    for i, ref in enumerate(refs, 1):
        story.append(Paragraph(f"{i}. {ref}", styles["Ref"]))

    h(story, "Peer Review Contact Plan")
    p(story,
      "This manuscript is prepared to request peer feedback from academic reviewers specializing in cancer genomics or "
      "biomedical machine learning. Reviewers will be asked to critique the leakage-free pipeline design, the calibration shift "
      "under cross-platform validation, and the physiological relevance of SHAP-selected genes.")

    return story


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Times-Roman", 8)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawString(0.72 * inch, 0.42 * inch,
                      "Rohan Saindane | Colon Cancer ML Project | Leakage-corrected draft")
    canvas.drawRightString(7.78 * inch, 0.42 * inch, f"Page {doc.page}")
    canvas.restoreState()


def main():
    parser = argparse.ArgumentParser(description="Build PDF research paper from pipeline metrics")
    parser.add_argument("--dataset", default="synthetic", choices=["geo", "tcga", "synthetic"])
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
    print(pdf_path)


if __name__ == "__main__":
    main()
