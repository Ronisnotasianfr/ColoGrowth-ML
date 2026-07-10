"""
build_paper.py - Assemble the Word research paper from leakage-fixed pipeline outputs.

Usage:
    python paper/build_paper.py --dataset synthetic
"""

import argparse
from pathlib import Path

from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

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
AUTHOR = "Rohan Saindane"
DATE = "June 2026"

INK = RGBColor(20, 31, 43)
BLUE = RGBColor(31, 78, 121)
GRAY = RGBColor(90, 90, 90)
LIGHT_FILL = "F4F6F9"
BORDER = "D9DEE8"


def set_cell_shading(cell, fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = tcPr.find(qn('w:shd'))
    if shd is None:
        shd = OxmlElement('w:shd')
        tcPr.append(shd)
    shd.set(qn('w:fill'), fill)


def set_cell_text(cell, text, bold=False, size=9.0, color=None):
    cell.text = ""
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(0)
    run = p.add_run(str(text))
    run.font.name = "Calibri"
    run._element.rPr.rFonts.set(qn("w:ascii"), "Calibri")
    run._element.rPr.rFonts.set(qn("w:hAnsi"), "Calibri")
    run.font.size = Pt(size)
    run.bold = bold
    if color:
        run.font.color.rgb = color
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def set_table_borders(table, color=BORDER):
    tbl = table._tbl
    tblPr = tbl.tblPr
    borders = tblPr.first_child_found_in("w:tblBorders")
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        tblPr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        tag = "w:" + edge
        element = borders.find(qn(tag))
        if element is None:
            element = OxmlElement(tag)
            borders.append(element)
        element.set(qn("w:val"), "single")
        element.set(qn("w:sz"), "4")
        element.set(qn("w:space"), "0")
        element.set(qn("w:color"), color)


def set_repeat_table_header(row):
    trPr = row._tr.get_or_add_trPr()
    tblHeader = OxmlElement('w:tblHeader')
    tblHeader.set(qn('w:val'), "true")
    trPr.append(tblHeader)


def style_run(run, size=None, bold=None, italic=None, color=None):
    run.font.name = "Calibri"
    run._element.rPr.rFonts.set(qn("w:ascii"), "Calibri")
    run._element.rPr.rFonts.set(qn("w:hAnsi"), "Calibri")
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic
    if color is not None:
        run.font.color.rgb = color


def add_para(doc, text="", after=7, before=0, align=None, size=10.6):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(before)
    p.paragraph_format.space_after = Pt(after)
    p.paragraph_format.line_spacing = 1.18
    if align:
        p.alignment = align
    if text:
        r = p.add_run(text)
        style_run(r, size=size, color=INK)
    return p


def add_heading(doc, text, level=1):
    p = doc.add_paragraph(style=f"Heading {level}")
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.space_before = Pt(12 if level == 1 else 8)
    p.paragraph_format.space_after = Pt(5)
    return p


def add_caption(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(8)
    r = p.add_run(text)
    style_run(r, size=8.8, italic=True, color=GRAY)


def add_figure(doc, path, caption, width=5.45):
    if not Path(path).exists():
        add_para(doc, f"[Figure unavailable: {path}]", size=9.0, color=GRAY)
        add_caption(doc, caption)
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    run = p.add_run()
    run.add_picture(str(path), width=Inches(width))
    add_caption(doc, caption)


def add_page_break(doc):
    doc.add_page_break()


def apply_styles(doc):
    sec = doc.sections[0]
    sec.top_margin = Inches(0.75)
    sec.bottom_margin = Inches(0.75)
    sec.left_margin = Inches(0.82)
    sec.right_margin = Inches(0.82)

    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Calibri"
    normal._element.rPr.rFonts.set(qn("w:ascii"), "Calibri")
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), "Calibri")
    normal.font.size = Pt(10.6)
    normal.font.color.rgb = INK

    for name, size, color, before, after in [
        ("Heading 1", 15, BLUE, 12, 5),
        ("Heading 2", 12.5, BLUE, 8, 4),
        ("Heading 3", 11.2, RGBColor(45, 75, 105), 6, 3),
    ]:
        st = styles[name]
        st.font.name = "Calibri"
        st._element.rPr.rFonts.set(qn("w:ascii"), "Calibri")
        st._element.rPr.rFonts.set(qn("w:hAnsi"), "Calibri")
        st.font.size = Pt(size)
        st.font.bold = True
        st.font.color.rgb = color
        st.paragraph_format.space_before = Pt(before)
        st.paragraph_format.space_after = Pt(after)
        st.paragraph_format.keep_with_next = True

    footer = sec.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    r = footer.add_run("Colon Cancer ML Project | Leakage-corrected research report")
    style_run(r, size=8.5, color=GRAY)


def add_title_page(doc, metrics, stats):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(18)
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run(TITLE)
    style_run(r, size=21, bold=True, color=INK)

    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(12)
    r = p.add_run(SUBTITLE)
    style_run(r, size=11.5, italic=True, color=GRAY)

    meta = doc.add_table(rows=4, cols=2)
    meta.alignment = WD_TABLE_ALIGNMENT.LEFT
    labels = [
        ("Prepared by", AUTHOR),
        ("Date", DATE),
        ("Dataset", stats['dataset'].upper()),
        ("Status", "Leakage-corrected faculty peer-review draft"),
    ]
    for i, (k, v) in enumerate(labels):
        set_cell_text(meta.cell(i, 0), k, bold=True, size=8.8, color=GRAY)
        set_cell_text(meta.cell(i, 1), v, size=8.8, color=INK)
    set_table_borders(meta, "FFFFFF")

    add_heading(doc, "Abstract", 1)
    add_para(doc, build_abstract(metrics, stats), size=10.4)

    add_heading(doc, "1. Introduction", 1)
    add_para(
        doc,
        "Cell proliferation is one of the central behaviors that separates aggressive tumors "
        "from slower-growing disease. In colon cancer, proliferation-related markers are often "
        "discussed alongside stage, survival, and molecular subtype because growth rate can "
        "reflect both tumor biology and clinical risk. Direct clinical measurement usually "
        "depends on pathology or immunohistochemical staining. A computational model cannot "
        "replace those assays, but it can test whether expression patterns carry enough signal "
        "to approximate a proliferation class and point toward genes that deserve closer review.",
        size=10.4,
    )
    add_para(
        doc,
        "This project builds a reproducible machine learning pipeline that predicts a binary "
        "high-versus-low proliferation label from expression and clinical features. A central "
        "design requirement is the exclusion of target-defining signature genes from the "
        "feature matrix so that reported performance reflects genuine biological signal rather "
        "than label reconstruction.",
        size=10.4,
    )


def add_methods(doc, stats):
    add_heading(doc, "2. Materials and Methods", 1)
    add_para(
        doc,
        f"The {stats['dataset'].upper()} processed dataset contains {stats['n_samples']} samples "
        f"and {stats['n_features']} features after removing the ten proliferation signature "
        f"genes. Clinical covariates include age, sex, and stage. The binary target is balanced "
        f"with {stats['class_balance']}. Models were trained on an 80% stratified training pool "
        f"({stats['train_n']} samples) and evaluated on a held-out 20% test split "
        f"({stats['test_n']} samples).",
        size=10.4,
    )

    data_table = doc.add_table(rows=1, cols=4)
    data_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = data_table.rows[0]
    set_repeat_table_header(hdr)
    for cell, text in zip(hdr.cells, ["Dataset file", "Samples", "Features/columns", "Class balance"]):
        set_cell_shading(cell, LIGHT_FILL)
        set_cell_text(cell, text, bold=True, size=8.8, color=INK)
    for row in dataset_table_rows(stats):
        cells = data_table.add_row().cells
        for c, text in zip(cells, row):
            set_cell_text(c, text, size=8.6)
    set_table_borders(data_table)
    add_caption(doc, "Table 1. Processed data files used for this leakage-corrected report.")

    add_para(
        doc,
        "Four classifiers were compared: logistic regression, random forest, XGBoost, and a "
        "multilayer perceptron. Each model was wrapped in an sklearn Pipeline containing "
        "standardization, variance filtering, SelectKBest feature selection, and the classifier. "
        "Five-fold stratified cross-validation was performed on the training pool with "
        "fold-local preprocessing, followed by GridSearchCV hyperparameter tuning and final "
        "holdout evaluation.",
        size=10.4,
    )
    add_para(doc, build_methods_leakage_paragraph(), size=10.4)


def add_results(doc, metrics, results_dir):
    add_heading(doc, "3. Results", 1)
    add_para(doc, build_results_opening(metrics), size=10.4)

    perf = doc.add_table(rows=1, cols=4)
    perf.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers = ["Model", "CV ROC-AUC (mean +/- std)", "Holdout Accuracy", "Holdout ROC-AUC"]
    for cell, text in zip(perf.rows[0].cells, headers):
        set_cell_shading(cell, LIGHT_FILL)
        set_cell_text(cell, text, bold=True, size=8.2, color=INK)
    set_repeat_table_header(perf.rows[0])
    for row in metrics_table_rows(metrics):
        cells = perf.add_row().cells
        for c, text in zip(cells, row):
            set_cell_text(c, text, size=8.1)
    set_table_borders(perf)
    add_caption(doc, "Table 2. Leakage-corrected cross-validation and holdout results.")

    add_figure(
        doc,
        results_dir / "roc_curves_comparison.png",
        "Figure 1. Holdout ROC curves comparing the four trained classifiers.",
        width=5.2,
    )
    add_para(doc, build_results_closing(metrics), size=10.4)


def add_interpretation(doc, results_dir):
    add_heading(doc, "4. Interpretation and Biological Readout", 1)
    add_para(
        doc,
        "SHAP summaries were generated from pipeline-transformed, leakage-free features. "
        "Because the ten signature genes used to define the proliferation label were removed "
        "before training, top-ranked features should be interpreted as candidate biological or "
        "clinical correlates rather than label-construction artifacts.",
        size=10.4,
    )
    add_figure(
        doc,
        results_dir / "shap_summary_random_forest.png",
        "Figure 2. Random forest SHAP summary from the leakage-corrected evaluation run.",
        width=5.35,
    )
    add_para(
        doc,
        "Kaplan-Meier survival plots provide a secondary clinical visualization linking "
        "proliferation grouping to outcome-style endpoints where survival metadata are available.",
        size=10.4,
    )
    add_figure(
        doc,
        results_dir / "kaplan_meier_synthetic.png",
        "Figure 3. Kaplan-Meier curve (synthetic cohort workflow demonstration).",
        width=5.2,
    )


def add_discussion(doc):
    add_heading(doc, "5. Discussion", 1)
    add_para(
        doc,
        "The project demonstrates an end-to-end computational workflow for proliferation "
        "classification with explicit leakage controls at both the target-definition and "
        "cross-validation stages. Encapsulating preprocessing inside the model pipeline is "
        "the correct design choice for unbiased performance estimation.",
        size=10.4,
    )
    add_para(doc, build_discussion_paragraph(), size=10.4)

    add_heading(doc, "6. Limitations and Next Steps", 1)
    items = [
        "Validate generalization on independent external cohorts (GEO to TCGA) using aligned, leakage-free features.",
        "Report confidence intervals or repeated split results instead of relying on one holdout partition.",
        "Link survival analysis to model predictions rather than ground-truth proliferation labels alone.",
        "Regenerate all figures after each full pipeline rerun to keep the manuscript synchronized with code.",
        "Request faculty review of target definition, leakage controls, and external validation design.",
    ]
    for item in items:
        p = doc.add_paragraph(style="List Bullet")
        p.paragraph_format.space_after = Pt(3)
        r = p.add_run(item)
        style_run(r, size=10.2, color=INK)


def add_references(doc):
    add_heading(doc, "References", 1)
    refs = [
        "Marisa, L. et al. Gene expression classification of colon cancer into molecular subtypes with distinct clinical and survival characteristics. PLoS Medicine, 2013.",
        "Whitfield, M. L. et al. Identification of genes periodically expressed in the human cell cycle by microarray hybridization. Molecular Biology of the Cell, 2002.",
        "Lundberg, S. M. and Lee, S.-I. A unified approach to interpreting model predictions. Advances in Neural Information Processing Systems, 2017.",
        "The Cancer Genome Atlas Research Network. TCGA Colon Adenocarcinoma data resource, accessed through the NCI Genomic Data Commons.",
        "National Center for Biotechnology Information Gene Expression Omnibus. GSE39582 dataset record.",
    ]
    for ref in refs:
        p = doc.add_paragraph(style="List Number")
        p.paragraph_format.space_after = Pt(3)
        r = p.add_run(ref)
        style_run(r, size=9.8, color=INK)

    add_heading(doc, "Peer Review Contact Plan", 1)
    add_para(
        doc,
        "This report is suitable to send to a faculty reviewer in bioinformatics, computational "
        "biology, cancer genomics, or biomedical machine learning. The review request should "
        "ask specifically for feedback on target definition, leakage control, external validation, "
        "and whether the corrected conclusions are stated cautiously enough for academic submission.",
        size=10.2,
    )


def build_latex(metrics, stats, out_path):
    abstract_text = build_abstract(metrics, stats)
    methods_leakage = build_methods_leakage_paragraph()
    results_opening = build_results_opening(metrics)
    results_closing = build_results_closing(metrics)
    discussion = build_discussion_paragraph()

    t1_rows = []
    for row in dataset_table_rows(stats):
        t1_rows.append(f"  {row[0]} & {row[1]} & {row[2]} & {row[3]} \\\\")
    t1_content = "\n".join(t1_rows)

    t2_rows = []
    for row in metrics_table_rows(metrics):
        t2_rows.append(f"  {row[0]} & {row[1]} & {row[2]} & {row[3]} \\\\")
    t2_content = "\n".join(t2_rows)

    tex_content = f"""\\documentclass[11pt]{{article}}
\\usepackage[margin=0.8in]{{geometry}}
\\usepackage{{graphicx}}
\\usepackage{{booktabs}}
\\usepackage{{array}}
\\usepackage{{times}}
\\usepackage{{caption}}
\\usepackage{{float}}
\\usepackage[hidelinks]{{hyperref}}
\\graphicspath{{{{../}}{{./}}}}

\\title{{\\textbf{{Machine Learning Prediction of Colon Cancer Proliferation Class from Gene Expression and Clinical Features}}}}
\\author{{Rohan Saindane\\\\Independent Research Project}}
\\date{{June 2026}}

\\begin{{document}}
\\maketitle

\\begin{{abstract}}
{abstract_text}
\\end{{abstract}}

\\section{{Introduction}}
Cell proliferation is one of the central behaviors that separates aggressive tumors from slower-growing disease. In colon cancer, proliferation-related markers are often discussed alongside stage, survival, and molecular subtype because growth rate can reflect both tumor biology and clinical risk. Direct clinical measurement usually depends on pathology or immunohistochemical staining. A computational model cannot replace those assays, but it can test whether expression patterns carry enough signal to approximate a proliferation class and point toward genes that deserve closer review.

The goal of this project was to build a reproducible machine learning pipeline that predicts a binary high-versus-low proliferation label from expression and clinical features, then inspect model behavior with standard metrics and interpretation plots. The report is intended for external academic feedback from faculty or researchers in bioinformatics, computational biology, cancer genomics, or biomedical machine learning.

\\section{{Materials and Methods}}
The processed dataset contains {stats['n_samples']} samples and {stats['n_features']} features after signature-gene removal. Clinical covariates include age, sex, and stage. The binary target is balanced with {stats['class_balance']}. An 80/20 stratified split produced a training pool of {stats['train_n']} samples and a holdout test set of {stats['test_n']} samples.

\\begin{{table}}[H]
\\centering
\\caption{{Processed data files available in the project at the time of this report.}}
\\begin{{tabular}}{{lccc}}
\\toprule
\\textbf{{Dataset file}} & \\textbf{{Samples}} & \\textbf{{Features/columns}} & \\textbf{{Class balance}} \\\\
\\midrule
{t1_content}
\\bottomrule
\\end{{tabular}}
\\end{{table}}

Four classifiers were compared: logistic regression, random forest, XGBoost, and a multilayer perceptron. Each model was wrapped in an sklearn Pipeline containing standardization, variance filtering, SelectKBest feature selection, and the classifier. Five-fold stratified cross-validation was performed on the training pool with fold-local preprocessing, followed by GridSearchCV hyperparameter tuning and final holdout evaluation.

{methods_leakage}

\\section{{Results}}
{results_opening}

\\begin{{table}}[H]
\\centering
\\caption{{Cross-validation and holdout results from the evaluation run.}}
\\begin{{tabular}}{{lccc}}
\\toprule
\\textbf{{Model}} & \\textbf{{CV ROC-AUC (mean +/- std)}} & \\textbf{{Holdout Accuracy}} & \\textbf{{Holdout ROC-AUC}} \\\\
\\midrule
{t2_content}
\\bottomrule
\\end{{tabular}}
\\end{{table}}

\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.78\\linewidth]{{results/roc_curves_comparison.png}}
\\caption{{Holdout ROC curves comparing the four trained classifiers.}}
\\end{{figure}}

{results_closing}

\\section{{Interpretation and Biological Readout}}
SHAP summaries were generated from pipeline-transformed, leakage-free features. Because the ten signature genes used to define the proliferation label were removed before training, top-ranked features should be interpreted as candidate biological or clinical correlates rather than label-construction artifacts.

\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.78\\linewidth]{{results/shap_summary_random_forest.png}}
\\caption{{Random forest SHAP summary from the leakage-corrected evaluation run.}}
\\end{{figure}}

Kaplan-Meier survival plots provide a secondary clinical visualization linking proliferation grouping to outcome-style endpoints where survival metadata are available.

\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.72\\linewidth]{{results/kaplan_meier_{stats['dataset']}.png}}
\\caption{{Kaplan-Meier curve ({stats['dataset']} cohort workflow demonstration).}}
\\end{{figure}}

\\section{{Discussion}}
The project has a strong structure for an early-stage computational biology study. It includes modular preprocessing, several baseline and nonlinear classifiers, standard metrics, interpretation figures, and survival-oriented visualization. The strongest technical choice is the attempt to put scaling and feature selection inside the model pipeline, because that is the right direction for reducing data leakage during cross-validation.

{discussion}

\\section{{Limitations and Next Steps}}
\\begin{{itemize}}
    \\item Validate generalization on independent external cohorts (GEO to TCGA) using aligned, leakage-free features.
    \\item Report confidence intervals or repeated split results instead of relying on one holdout partition.
    \\item Link survival analysis to model predictions rather than ground-truth proliferation labels alone.
    \\item Regenerate all figures after each full pipeline rerun to keep the manuscript synchronized with code.
    \\item Request faculty review of target definition, leakage controls, and external validation design.
\\end{{itemize}}

\\section{{References}}
\\begin{{enumerate}}
    \\item Marisa, L. et al. Gene expression classification of colon cancer into molecular subtypes with distinct clinical and survival characteristics. \\textit{{PLoS Medicine}}, 2013.
    \\item Whitfield, M. L. et al. Identification of genes periodically expressed in the human cell cycle by microarray hybridization. \\textit{{Molecular Biology of the Cell}}, 2002.
    \\item Used, S. M. and Lee, S.-I. A unified approach to interpreting model predictions. \\textit{{Advances in Neural Information Processing Systems}}, 2017.
    \\item The Cancer Genome Atlas Research Network. TCGA Colon Adenocarcinoma data resource, accessed through the NCI Genomic Data Commons.
    \\item National Center for Biotechnology Information Gene Expression Omnibus. GSE39582 dataset record.
\\end{{enumerate}}

\\section*{{Peer Review Contact Plan}}
This report is suitable to send to a faculty reviewer in bioinformatics, computational biology, cancer genomics, or biomedical machine learning. The review request should ask specifically for feedback on target definition, leakage control, external validation, and whether the corrected conclusions are stated cautiously enough for academic submission.

\\end{{document}}
"""
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(tex_content)
    print(f"LaTeX paper written to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Build Word research paper from pipeline metrics")
    parser.add_argument("--dataset", default="synthetic", choices=["geo", "tcga", "synthetic"])
    args = parser.parse_args()

    base = project_root()
    results_dir = base / "results"
    data_dir = base / "data" / "processed"
    out_docx = base / "paper" / "colon_cancer_growth_prediction_research_paper.docx"
    out_tex = base / "paper" / "colon_cancer_growth_prediction_research_paper.tex"

    metrics = load_leakage_fixed_metrics(args.dataset, results_dir)
    stats = load_dataset_stats(args.dataset, data_dir)

    doc = Document()
    apply_styles(doc)
    add_title_page(doc, metrics, stats)
    add_page_break(doc)
    add_methods(doc, stats)
    add_page_break(doc)
    add_results(doc, metrics, results_dir)
    add_page_break(doc)
    add_interpretation(doc, results_dir)
    add_page_break(doc)
    add_discussion(doc)
    add_page_break(doc)
    add_references(doc)
    doc.save(out_docx)
    print(out_docx)

    build_latex(metrics, stats, out_tex)


if __name__ == "__main__":
    main()
