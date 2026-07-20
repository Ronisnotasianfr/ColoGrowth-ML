"""Generate drafts CSV for research-outreach-bot with all 17 mentor contacts."""

import csv
import hashlib
import json
import os
from pathlib import Path

BOT_DIR = Path(r"C:\Users\ronsa\Dev Stuff\research-outreach-bot")
PROFILE_PATH = BOT_DIR / "student_profile.json"
OUTPUT = BOT_DIR / "mentor_drafts.csv"

profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))

def fingerprint_email(email):
    return hashlib.sha256((email or "").strip().lower().encode("utf-8")).hexdigest()

contacts = [
    # NIH / NCI
    {"professor_name": "S. Cenk Sahinalp", "email": "cenk.sahinalp@nih.gov", "institution": "NIH/NCI Cancer Data Science Lab", "topic": "algorithms for cancer genomics"},
    {"professor_name": "Peng Jiang", "email": "peng.jiang@nih.gov", "institution": "NIH/NCI Cancer Data Science Lab", "topic": "AI frameworks for cancer transcriptomics"},
    {"professor_name": "Sridhar Hannenhalli", "email": "sridhar.hannenhalli@nih.gov", "institution": "NIH/NCI Cancer Data Science Lab", "topic": "gene regulation in cancer"},
    {"professor_name": "Nishanth Nair", "email": "nishanth.nair@nih.gov", "institution": "NIH/NCI Cancer Data Science Lab", "topic": "ML for cancer drug response"},
    {"professor_name": "Mikhail Kolmogorov", "email": "mikhail.kolmogorov@nih.gov", "institution": "NCI", "topic": "computational cancer genomics"},
    {"professor_name": "Yingdong Zhao", "email": "yingdong.zhao@nih.gov", "institution": "NIH Computational & Systems Biology", "topic": "survival modeling and ML for biomarkers"},
    {"professor_name": "Teresa Przytycka", "email": "przytyck@ncbi.nlm.nih.gov", "institution": "NCBI/NLM", "topic": "systems biology of cancer"},
    # UMD Baltimore
    {"professor_name": "Elana Fertig", "email": "ejfertig@som.umaryland.edu", "institution": "UMB Institute for Genome Sciences", "topic": "multi-omics cancer biology"},
    {"professor_name": "Yuji Zhang", "email": "yuzhang@som.umaryland.edu", "institution": "UMB Biostatistics & Bioinformatics", "topic": "multi-omics integration in cancer"},
    {"professor_name": "Wei Li", "email": "wli2@som.umaryland.edu", "institution": "UMB Institute for Genome Sciences", "topic": "AI for genomics"},
    # UMD College Park
    {"professor_name": "Alexander Xu", "email": "alexmxu@umd.edu", "institution": "UMD Bioengineering", "topic": "cancer biomarkers and ML"},
    {"professor_name": "Lan Ma", "email": "lanma@umd.edu", "institution": "UMD Bioengineering", "topic": "computational cancer biology"},
    # Johns Hopkins
    {"professor_name": "Rachel Karchin", "email": "karchin@jhu.edu", "institution": "Johns Hopkins BME", "topic": "computational cancer genomics"},
    {"professor_name": "Michael Schatz", "email": "mschatz@jhu.edu", "institution": "Johns Hopkins", "topic": "cancer genomics tools"},
    {"professor_name": "Joel Bader", "email": "joel.bader@jhu.edu", "institution": "Johns Hopkins BME", "topic": "cancer genomics"},
    {"professor_name": "Andrew Feinberg", "email": "afeinberg@jhu.edu", "institution": "Johns Hopkins", "topic": "cancer epigenetics"},
    {"professor_name": "Robert Scharpf", "email": "robscharpf@jhmi.edu", "institution": "Johns Hopkins Oncology", "topic": "ML for cancer detection"},
]

def build_subject(prof):
    return f"High school researcher — colon cancer ML classifier, cross-platform validation"

def build_body(prof, profile):
    last = prof["professor_name"].split()[-1]
    return f"""Dear Professor {last},

My name is {profile['name']}, a high school student at {profile['school']} in {profile['location']}. I built a leakage-free ML pipeline that classifies colon cancer proliferation from transcriptomic data. The classifier trains on GEO microarray (n=585) and generalizes to TCGA (n=322) and CPTAC-COAD (n=105) RNA-seq cohorts, achieving cross-platform AUC > 0.95 through Platt scaling calibration.

Your work on {prof['topic']} is directly relevant to the generalization and biological signal challenges I'm addressing. I've attached a 1-page summary and would be grateful for any feedback — even a quick thought would mean a lot.

Thanks,
{profile['name']}
{profile['school']}
{profile['location']}
{profile['email']}
"""

rows = []
for c in contacts:
    rows.append({
        "professor_name": c["professor_name"],
        "email": c["email"],
        "institution": c["institution"],
        "topic": c["topic"],
        "status": "draft",
        "subject": build_subject(c),
        "body": build_body(c, profile),
        "email_hash": fingerprint_email(c["email"]),
    })

fieldnames = ["professor_name", "email", "institution", "topic", "status", "subject", "body", "email_hash"]
with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {len(rows)} drafts to {OUTPUT}")
