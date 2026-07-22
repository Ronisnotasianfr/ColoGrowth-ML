# Project Notes & Struggles

Things that went wrong, what I learned, and what I'd do differently.

## Target Leakage (caught early, thankfully)

Originally I left the 10 proliferation genes in the feature matrix. The model hit AUC 0.99+ and I thought I was a genius. Took me a while to realize the model was basically predicting the label from the same genes used to create it. Fixing this dropped AUC to ~0.78, which was demoralizing, but eventually got back to 0.97 after proper training. Lesson: always think about what information leaks from target to features.

## Cross-platform calibration is not trivial

The GEO data (microarray) and TCGA data (RNA-seq) have completely different distributions. When I first ran the trained model on TCGA, the probabilities were all over the place — the model was confident but wrong. Quantile normalization fixed the distribution mismatch but didn't fix the probability calibration. Combining QN+Platt was the real solution, and it took a lot of trial and error to figure that out.

## StabilitySelector parallelization

Running 100 bootstrap iterations on ~20,000 features × 585 samples was painfully slow at first. Parallelized with joblib which helped, but I'm still not sure 100 iterations is enough. Meinshausen & Buhlmann (2010) suggest it is, but in practice the selection frequency sometimes fluctuates between runs. This is probably the weakest part of the pipeline.

## The CPTAC survival analysis is basically useless

Only 7 survival events out of 105 samples. I should have checked the power analysis before spending time on it. The Schoenfeld formula would have told me immediately it's underpowered. The power analysis was added late — should have done it first.

## GEO data merging headaches

GSE39582 and GSE17538 use different clinical column names and the probe-to-gene mapping is sloppy. The annotation file has genes separated by ' /// ' for multi-target probes — I just take the first one. Some probes map to no gene at all. The merge only keeps common genes, so I lose platform-specific ones.

## Reproduce.sh doesn't always work

The GEO FTP servers can be flaky. TCGA Xena URLs sometimes timeout. CPTAC AWS links could break any time. If a judge tries to run reproduce.sh at the fair and it fails on a download, that's a bad look. Should have cached the data.

## What I would do next

- Replace binarization with continuous regression (why throw away information?)
- Add a held-out validation cohort from a third platform (nanostring or proteomics)
- Test the drug predictions in actual cell lines instead of just GDSC2 correlation
- Get a collaborator who understands the biology better than I do
