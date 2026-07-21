# ScienceMontgomery Judge Q&A Prep

## Likely Trap Questions and Good Answers

### Q: "Your internal accuracy is high but your external accuracy drops. Isn't that overfitting?"

**Good answer**: "We saw a drop when transferring from microarray (GEO) to RNA-seq (TCGA) because the expression distributions are fundamentally different between platforms. The raw probability thresholds don't transfer. But the *rank ordering* generalized well — external ROC-AUC of 0.97 (TCGA) and 0.95 (CPTAC), meaning patients the model scores as high-risk are indeed high-risk. We corrected the threshold shift with Platt scaling calibration. This cross-platform AUC > 0.95 on two independent validation cohorts is strong evidence the model learned real biology, not platform-specific artifacts."

### Q: "Your Cox PH model shows p = 0.092 for proliferation status. Why do you call it significant?"

**Good answer**: "We don't. The Cox PH multivariate result is p = 0.092, which we report as a trend — it's not statistically significant at the conventional α = 0.05 threshold. The hazard ratio direction (HR = 0.78) is consistent with the significant log-rank test (p = 0.037, p = 0.034) from the Kaplan-Meier analysis, suggesting the signal is real but the multivariate Cox model may be underpowered with only 585 samples and several covariates."

### Q: "How did you prevent data leakage?"

**Good answer**: "Three layers. First, we removed the 10 proliferation signature genes from features in preprocess.py before any splitting. Second, train.py strips them again as a safety check. Third, all preprocessing (scaling, variance filtering, feature selection) is inside an sklearn Pipeline, so parameters are learned on training folds only during cross-validation."

### Q: "Why didn't you use deep learning?"

**Good answer**: "With 585 samples and 22,000 features, deep learning would massively overfit. sklearn models with regularization (L2, tree depth limits) and SelectKBest are more appropriate for this sample size. The focus was on a rigorous, reproducible pipeline — not the most complex model."

### Q: "What would you do next with more time?"

**Good answer**: "I already added a third validation cohort (CPTAC-COAD, n=105) and demonstrated cross-platform AUC of 0.95. With more time I'd like to do the opposite direction — train on RNA-seq and validate on microarray — to test symmetry of generalizability. I'd also do an ablation analysis to quantify how much the clinical covariates (age, sex, stage) add beyond gene expression alone."

### Q: "What's the clinical significance?"

**Good answer**: "Proliferation rate is a key prognostic marker currently measured via Ki-67 staining, which has inter-observer variability. A transcriptomic classifier could provide an objective, reproducible alternative from standard RNA-seq data. The model identifies downstream pathways (DNA replication, mitochondrial translation) that could suggest novel therapeutic targets."

### Q: "Did you use any AI assistance?"

**Good answer**: "Yes. I used an AI coding assistant (Claude by Anthropic) for parts of the implementation and documentation. ScienceMontgomery allows this as long as it's disclosed. I've acknowledged this in my materials. The core experimental design, biological interpretation, and conclusions are my own work."
