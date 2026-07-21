# 🤖 AGENT PROMPT — Fix All Citation Errors
## ColoGrowth-ML Manuscript: Reference Correction Pass

**Paste this to your coding agent with repo access.**

---

## WHY THIS EXISTS

A prior AI-assisted editing pass added 7 new citations to the manuscript's Discussion and Table 9 (literature benchmarking + gene mechanism paragraphs). Manual verification found **7 out of 7 had errors** — wrong journal, wrong DOI, wrong title, or in one case a fabricated author. This prompt fixes those, using citations that have already been human-verified against PubMed/publisher sources, and re-verifies the remaining details that are still uncertain. It also re-checks the 5 original references that predate this whole editing process, since they were never independently verified either.

---

## HARD RULES (same as before — these are why the last pass failed)

- Never trust a citation because it "sounds plausible." Every title/journal/year/DOI/author must be confirmed against a real, checkable source (PubMed, the publisher's page, or a DOI resolver) before it goes in the manuscript.
- If you cannot fully verify all four fields (title, journal, year, DOI) for a citation, do not guess the missing piece. Mark it `[UNVERIFIED — needs manual check]` in your report and leave the old (flagged) version in place with a comment, rather than replacing it with another guess.
- A citation with a correct title/journal/author but wrong DOI is still broken — fix the whole entry, not just the parts that were already right.
- Do not add any new citations beyond what's listed below. This is a correction pass, not an expansion pass.

---

## PART 1 — DROP-IN REPLACEMENTS (already verified, safe to use as-is)

Find and replace these 4 reference list entries and their corresponding in-text citations/DOIs wherever they appear (References section, Discussion body text, and Table 9 if applicable):

**Replace the Langston entry with:**
```
Langston, L. D., Mayle, R., Schauer, G. D., Yurieva, O., Zhang, D., Yao, N. Y., 
Georgescu, R. E., & O'Donnell, M. E. (2017). Mcm10 promotes rapid isomerization of 
CMG-DNA for replisome bypass of lagging strand DNA blocks. eLife, 6, e29118. 
https://doi.org/10.7554/eLife.29118
```

**Replace the Bharadwaj entry with:**
```
Bharadwaj, R., Qi, W., & Yu, H. (2004). Identification of two novel components of the 
human NDC80 kinetochore complex. Journal of Biological Chemistry, 279(13), 13076–13085. 
https://doi.org/10.1074/jbc.M310224200
```

**Replace the Overmeer entry with:**
```
Overmeer, R. M., Gourdin, A. M., Giglia-Mari, A., Kool, H., Houtsmuller, A. B., Siegal, G., 
Fousteri, M. I., Mullenders, L. H., & Vermeulen, W. (2010). Replication factor C recruits 
DNA polymerase delta to sites of nucleotide excision repair but is not required for PCNA 
recruitment. Molecular and Cellular Biology, 30(20), 4828–4839. 
https://doi.org/10.1128/MCB.00285-10
```

**Replace the Agesen (ColoGuideEx) entry with:**
```
Ågesen, T. H., Sveen, A., Merok, M. A., Lind, G. E., Nesbakken, A., Skotheim, R. I., & 
Lothe, R. A. (2012). ColoGuideEx: a robust gene classifier specific for stage II 
colorectal cancer prognosis. Gut, 61(11), 1560–1567. 
https://doi.org/10.1136/gutjnl-2011-301179
```

**Replace the Zeng entry with (title/journal were already correct, only the DOI was wrong):**
```
Zeng, D.-T., Li, M.-J., Lin, R., Huang, W.-J., Li, S.-D., Huang, W.-Y., Li, B., Li, Q., 
Chen, G., & Jiang, J.-S. (2025). Prognostic role of Ki-67 in colorectal carcinoma: 
Development and evaluation of machine learning prediction models. World Journal of 
Clinical Oncology, 16(8), 107306. https://doi.org/10.5306/wjco.v16.i8.107306
```

**Replace the fabricated "Genomic Health" entry with a real, named-author citation:**
```
O'Connell, M. J., Lavery, I., Yothers, G., Paik, S., Clark-Langone, K. M., Lopatin, M., 
Watson, D., Baehner, F. L., Shak, S., Baker, J., Cowens, J. W., & Wolmark, N. (2010). 
Relationship between tumor gene expression and recurrence in four independent studies of 
patients with stage II/III colon cancer treated with surgery alone or surgery plus 
adjuvant fluorouracil plus leucovorin. Journal of Clinical Oncology, 28(25), 3937–3944.
```
⚠️ Before inserting this one: search PubMed for "O'Connell tumor gene expression recurrence four independent studies colon cancer 2010" and confirm the exact DOI yourself (it should resolve on pubmed.ncbi.nlm.nih.gov and match volume 28, issue 25, pages 3937–3944). Do not fabricate the DOI — leave it blank with `[DOI PENDING VERIFICATION]` in the manuscript if you cannot confirm it directly, rather than inventing one.

Update Table 9's "Genomic Health (OncoType DX)" row to cite this real author (O'Connell et al., 2010) instead of the organization name.

---

## PART 2 — STILL NEEDS VERIFICATION (do not guess these)

**Seipold et al. (NCAPH citation):** Confirmed wrong journal (real paper is in *BMC Developmental Biology*, 2009, not *EMBO Reports*), but the exact DOI was not confirmed. 
- Task: Search PubMed/BMC Developmental Biology for "Seipold Priller Goldsmith Harris Baier Abdelilah-Seyfried condensin zebrafish neural retina 2009" and confirm the exact DOI, volume, and page numbers.
- If found: replace the reference with the fully verified entry.
- If not confidently confirmed: mark `[CITATION NEEDS MANUAL VERIFICATION]` in the manuscript and flag it in your report — do not insert a DOI you can't confirm resolves correctly.

**All 5 original references** (Marisa et al. 2013, Whitfield et al. 2002, Lundberg & Lee 2017, TCGA Research Network, GEO GSE39582 record) — these predate this whole editing project and were never independently checked.
- Task: For each, verify title/journal/year/DOI against PubMed or the publisher's page.
- Specifically double-check Marisa et al.: the manuscript currently says the title is *"Gene expression Classification of Colon Cancer defines six molecular subtypes with distinct clinical, molecular and survival characteristics"* — a preliminary check suggests the real title may instead be closer to *"Gene expression classification of colon cancer into molecular subtypes: characterization, validation, and prognostic value"* (PLoS Medicine, 2013, doi: 10.1371/journal.pmed.1001453). Confirm which is correct and fix if needed.
- Report exact findings for all 5, corrected or confirmed-as-is.

---

## PART 3 — MECHANISM OF WORK: FIND, DON'T JUST REPLACE

The Discussion paragraph built around these citations makes specific mechanistic claims (e.g., "MCM10 is a critical replication initiation factor that coordinates pre-replication complex loading"). After fixing the citations:
- Re-read the surrounding sentence for each corrected citation and confirm the mechanistic claim is still consistent with the *actual* paper's abstract/findings (not the fabricated title that was there before).
- If a claim no longer matches what the real cited paper actually shows, rewrite that sentence to accurately reflect the real paper's findings rather than leaving a mismatched claim next to a corrected citation.

---

## PART 4 — WHERE TO MAKE THESE CHANGES

- Locate the manuscript source (`paper/build_paper.py`, `paper/paper_metrics.py`, or wherever the References/Discussion text is generated or stored — check the repo structure first).
- Make the same corrections in both the generation script/source-of-truth AND any already-rendered `.docx`/`.pdf`/`.tex` output, so future rebuilds stay consistent. Back up the current rendered files before editing.
- If citations are hardcoded as plain strings in a Python file, update them there so `python paper/build_paper.py` regenerates a correct document.

---

## REPORT FORMAT (use exactly this)

```
## Part 1 — Drop-in Replacements
[ ] Langston — replaced, DOI 10.7554/eLife.29118 confirmed resolves correctly
[ ] Bharadwaj — replaced, DOI 10.1074/jbc.M310224200 confirmed resolves correctly
[ ] Overmeer — replaced, DOI 10.1128/MCB.00285-10 confirmed resolves correctly
[ ] Agesen — replaced, DOI 10.1136/gutjnl-2011-301179 confirmed resolves correctly
[ ] Zeng — DOI corrected to 10.5306/wjco.v16.i8.107306, confirmed resolves correctly
[ ] O'Connell (replacing "Genomic Health") — replaced with real authors; DOI status: [confirmed / pending]

## Part 2 — Newly Verified or Flagged
Seipold: [outcome + DOI if found, or flag if not]
Marisa et al.: [confirmed correct as-is / corrected to: ...]
Whitfield et al.: [confirmed correct as-is / corrected to: ...]
Lundberg & Lee: [confirmed correct as-is / corrected to: ...]
TCGA reference: [confirmed correct as-is / corrected to: ...]
GEO GSE39582 reference: [confirmed correct as-is / corrected to: ...]

## Part 3 — Mechanistic Claim Consistency Check
[Any sentence rewritten because the real paper didn't support the original claim, with before/after]

## Files Changed
[list]

## Anything Still Requiring Manual Author Verification
[list — be honest, this should not be empty if anything remains uncertain]
```

---

## DO NOT

- Do not mark anything "confirmed" without actually having found it via search/PubMed in this session.
- Do not invent a DOI to fill a gap, ever — an empty `[PENDING]` marker is always better than a fabricated identifier.
- Do not touch any other part of the manuscript (statistics, figures, other sections) — this is a citations-only pass.
