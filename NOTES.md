# Project Notes & Struggles

Things that went wrong, what I learned, and what I'd do differently.

## Target Leakage (caught early, thankfully)

Originally I left the 10 proliferation genes in the feature matrix. The model hit AUC 0.99+ and I thought I was a genius. Took me a while to realize the model was basically predicting the label from the same genes used to create it. Fixing this dropped AUC to ~0.78, which was demoralizing, but eventually got back to 0.97 after proper training. Lesson: always think about what information leaks from target to features.

## Cross-platform calibration is not trivial

The GEO data (microarray) and TCGA data (RNA-seq) have completely different distributions. When I first ran the trained model on TCGA, the probabilities were all over the place, the model was confident but wrong. Quantile normalization fixed the distribution mismatch but didn't fix the probability calibration. Combining QN+Platt was the real solution, and it took a lot of trial and error to figure that out. The final benchmark showed Platt alone beats QN+Platt for the tree models in most configs, which was the opposite of what I expected going in.

## StabilitySelector parallelization

Running 100 bootstrap iterations on ~20,000 features x 585 samples was painfully slow at first. Parallelized with joblib which helped, but I'm still not sure 100 iterations is enough. Meinshausen & Buhlmann (2010) suggest it is, but in practice the selection frequency sometimes fluctuates between runs. This is probably the weakest part of the pipeline.

## The CPTAC survival analysis is basically useless

Only 7 survival events out of 105 samples. I should have checked the power analysis before spending time on it. The Schoenfeld formula would have told me immediately it's underpowered. The power analysis was added late, should have done it first. The weird part is the Cox result: proliferation predicts survival before you add stage, then vanishes (HR 0.84, p=0.31). That's a genuine null, not an underpowered one, and it took the power analysis to prove it.

## GEO data merging headaches

GSE39582 and GSE17538 use different clinical column names and the probe-to-gene mapping is sloppy. The annotation file has genes separated by ' /// ' for multi-target probes, I just take the first one. Some probes map to no gene at all. The merge only keeps common genes, so I lose platform-specific ones.

## Reproduce.sh doesn't always work

The GEO FTP servers can be flaky. TCGA Xena URLs sometimes timeout. CPTAC AWS links could break any time. If a judge tries to run reproduce.sh at the fair and it fails on a download, that's a bad look. Should have cached the data.

## The beamerposter tried to kill me

The poster is 48x36 inches and I chased "Overfull vbox" errors for hours. The lesson that actually mattered: beamer draws overflowing columns past the paper edge and just makes them invisible, so the tex warning was real clipping I couldn't see in the PDF reader. I ended up measuring column bottoms programmatically with PyMuPDF and tuning figure widths until every column ended inside the page. Every content check (searching the extracted PDF text for the references block, the drug hits, the tables) said "missing", and every time it meant something was genuinely being cut. Fiddling with \vfill inside beamer columns made things worse; \vspace* between blocks was the fix.

## Local LaTeX is possible on Windows after all

User said no MiKTeX, and pip install tectonic has no wheels. TinyTeX (the portable TeX Live installer) worked after three tries with the right Windows build. I still had to tlmgr install courier, times, helvetic, and psnfss on top, because TinyTeX ships minimal fonts and any documentclass that wants Courier or Times dies with "TFM file not found". Lesson: on TinyTeX, missing fonts and missing packages look identical.

## One typo that mattered: MCM6 vs MCM10

The signature gene list in preprocess.py is MCM6, and the manuscript draft said MCM10 in one spot, which would have been wrong in the paper itself. Caught it while cross-checking text against code. Same pass caught the title author name mismatch (Ronit vs Rohan) in an earlier version of the citation. Moral: the manuscript and the code should be diffed against each other, not trusted separately.

## What I would do next

- Stratify the drug screen by KRAS/BRAF mutation status. The MEK hits (Trametinib etc.) are exactly what KRAS-mutant lines are known to respond to, so "proliferation predicts MEK sensitivity" might actually be "KRAS predicts both". GDSC2 carries the mutation data, this is the cheapest fix and the most likely to flip a conclusion.
- Replace binarization with a fixed threshold anchored to a reference cohort. Right now "high" vs "low" depends on whichever cohort computed the median, which is embarrassing for a clinical claim.
- Single-sample scoring. Quantile normalization needs the whole batch, so the model can't score one patient alone. This has to be solved before anyone can actually deploy it.
- Replace binarization with continuous regression (why throw away information?)
- Add a held-out validation cohort from a third platform (nanostring or proteomics)
- Test the drug predictions in actual cell lines instead of just GDSC2 correlation
- Get a collaborator who understands the biology better than I do