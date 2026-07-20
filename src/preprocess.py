"""
preprocess.py - Data loading, target computation, and feature engineering.

Notes:
- Target is computed from a 10-gene cell cycle signature.
- Signature genes are removed from features to prevent leakage.
- Variance filtering is handled in train.py via pipeline.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import requests
import gzip


PROLIF_GENES = [
    'MKI67', 'PCNA', 'TOP2A', 'MCM2', 'MCM6',
    'AURKA', 'BUB1', 'CCNB1', 'CDK1', 'BIRC5'
]


def remove_proliferation_genes(X, prolif_genes=None):
    """
    Remove proliferation signature genes from the feature matrix.
    Prevents predicting labels from the same genes used to create them.
    """
    if prolif_genes is None:
        prolif_genes = PROLIF_GENES

    cols_upper = {c.upper(): c for c in X.columns}
    to_drop = [cols_upper[g] for g in prolif_genes if g in cols_upper]

    if to_drop:
        print(f"Removing {len(to_drop)} proliferation genes from features to prevent leakage.")
        X = X.drop(columns=to_drop)
    else:
        print("No proliferation genes found in features.")

    return X


def validate_no_leakage(X, prolif_genes=None):
    """
    Assert that no proliferation signature gene remains in the feature matrix.
    Raises AssertionError if leakage is detected.
    """
    if prolif_genes is None:
        prolif_genes = PROLIF_GENES

    cols_upper = set(c.upper() for c in X.columns)
    overlap = cols_upper.intersection(set(prolif_genes))

    assert len(overlap) == 0, f"Target leakage detected: {overlap} are in feature matrix."
    print(f"Leakage check passed. Features: {X.shape[1]} columns.")


def generate_synthetic_data(n_samples=300, n_genes=2000):
    """
    Generate synthetic gene expression and clinical data for pipeline testing.
    Includes proliferation genes, survival times, and clinical metadata.
    """
    print(f"Generating {n_samples} synthetic samples with {n_genes} genes...")
    np.random.seed(42)

    other_genes = [f"GENE_{i}" for i in range(n_genes - len(PROLIF_GENES))]
    all_genes = PROLIF_GENES + other_genes

    # Simulate expression matrix
    expression = np.random.normal(loc=6.0, scale=1.5, size=(n_samples, n_genes))

    # Latent proliferation signal drives proliferation genes
    latent = np.random.normal(0, 1, size=(n_samples, 1))
    for i, gene in enumerate(PROLIF_GENES):
        idx = all_genes.index(gene)
        expression[:, idx] += latent.flatten() * 1.2

    # Some non-prolif genes also correlate with proliferation (biological reality)
    for j in range(10, 30):
        expression[:, j] += latent.flatten() * np.random.uniform(0.3, 0.7)

    df_expr = pd.DataFrame(expression, columns=all_genes,
                           index=[f"Sample_{i}" for i in range(n_samples)])

    # Clinical data including simulated survival
    stages = np.random.choice([1, 2, 3, 4], p=[0.2, 0.4, 0.3, 0.1], size=n_samples)
    os_time = np.random.exponential(scale=60, size=n_samples).clip(1, 200)
    # Higher proliferation and stage → shorter survival (simulated)
    os_time -= latent.flatten() * 8 + stages * 3
    os_time = os_time.clip(1, 200)
    os_event = np.random.binomial(1, 0.6, size=n_samples)

    clinical = pd.DataFrame({
        'age': np.random.randint(40, 85, size=n_samples),
        'gender': np.random.choice(['MALE', 'FEMALE'], size=n_samples),
        'stage': [f"Stage {'I' * s if s <= 3 else 'IV'}" for s in stages],
        'os_time': os_time,
        'os_event': os_event,
    }, index=[f"Sample_{i}" for i in range(n_samples)])

    return df_expr, clinical


def download_geo_dataset(geo_id="GSE39582", dest_dir="data/raw", gpl_id=None):
    """Download GEO series matrix file from NCBI. Works for any GSE ID.
    
    Some datasets use {GSE}_series_matrix.txt.gz, others use {GSE}-{GPL}_series_matrix.txt.gz.
    If gpl_id is provided, tries the qualified name first, then falls back to unqualified.
    """
    os.makedirs(dest_dir, exist_ok=True)
    filepath_plain = os.path.join(dest_dir, f"{geo_id}_series_matrix.txt.gz")
    filepath_qual = os.path.join(dest_dir, f"{geo_id}-{gpl_id}_series_matrix.txt.gz") if gpl_id else None

    if os.path.exists(filepath_plain):
        print(f"[GEO] {geo_id} already cached at {filepath_plain}")
        return filepath_plain
    if filepath_qual and os.path.exists(filepath_qual):
        print(f"[GEO] {geo_id} already cached at {filepath_qual}")
        return filepath_qual

    # Extract numeric part and compute directory prefix (e.g. 39582 -> GSE39nnn)
    num_part = geo_id.replace("GSE", "")
    prefix = f"GSE{num_part[:2]}nnn"
    
    urls_to_try = []
    if gpl_id:
        urls_to_try.append(
            f"https://ftp.ncbi.nlm.nih.gov/geo/series/{prefix}/{geo_id}/matrix/{geo_id}-{gpl_id}_series_matrix.txt.gz"
        )
    urls_to_try.append(
        f"https://ftp.ncbi.nlm.nih.gov/geo/series/{prefix}/{geo_id}/matrix/{geo_id}_series_matrix.txt.gz"
    )
    
    for url in urls_to_try:
        target_path = filepath_qual if gpl_id and f"-{gpl_id}" in url else filepath_plain
        print(f"[GEO] Trying {url} ...")
        try:
            resp = requests.get(url, stream=True, timeout=180)
            resp.raise_for_status()
            with open(target_path, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    f.write(chunk)
            print(f"[GEO] Download complete: {target_path}")
            return target_path
        except Exception as e:
            print(f"[GEO] URL failed: {e}")
    
    print(f"[GEO] All download attempts failed for {geo_id}.")
    return None


def download_gpl_annotation(gpl_id="GPL570", dest_dir="data/raw"):
    """Download Affymetrix GPL570 probe-to-gene annotation from NCBI."""
    os.makedirs(dest_dir, exist_ok=True)
    filepath = os.path.join(dest_dir, f"{gpl_id}.annot.gz")

    if os.path.exists(filepath):
        print(f"[GPL] Annotation cached at {filepath}")
        return filepath

    # Calculate GEO platform directory (e.g. GPL570 -> GPLnnn, GPL1053 -> GPL1nnn)
    num_part = ''.join(c for c in gpl_id if c.isdigit())
    if len(num_part) <= 3:
        range_str = "GPLnnn"
    else:
        range_str = f"GPL{num_part[:-3]}nnn"
    url = f"https://ftp.ncbi.nlm.nih.gov/geo/platforms/{range_str}/{gpl_id}/annot/{gpl_id}.annot.gz"
    print(f"[GPL] Downloading {gpl_id} annotation from {url}...")
    try:
        resp = requests.get(url, stream=True, timeout=120)
        resp.raise_for_status()
        with open(filepath, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)
        print(f"[GPL] Annotation downloaded: {filepath}")
        return filepath
    except Exception as e:
        print(f"[GPL] Download failed: {e}")
        return None


def parse_gpl_annotation(filepath):
    """
    Parse GPL570 annotation file to extract probe_id → gene_symbol mapping.
    The annotation file contains a table between !platform_table_begin and !platform_table_end.
    """
    print(f"[GPL] Parsing annotation: {filepath}")
    rows = []
    header = None
    in_table = False
    with gzip.open(filepath, 'rt', encoding='utf-8', errors='replace') as f:
        for line in f:
            line_str = line.strip()
            if not in_table:
                if line_str.startswith('!platform_table_begin'):
                    in_table = True
                continue
            if line_str.startswith('!platform_table_end'):
                break
            if line_str == '':
                continue
            parts = line_str.split('\t')
            if header is None:
                header = parts
                continue
            rows.append(parts)

    df = pd.DataFrame(rows, columns=header)

    # Find the probe ID and gene symbol columns
    id_col = None
    gene_col = None
    for col in df.columns:
        col_lower = col.lower().strip()
        if col_lower in ('id', 'probe set id', 'probe_id'):
            id_col = col
        if col_lower in ('gene symbol', 'gene_symbol'):
            gene_col = col
            
    # Fallback to substring matching if exact match not found
    if gene_col is None:
        for col in df.columns:
            col_lower = col.lower().strip()
            if 'gene symbol' in col_lower or 'gene_symbol' in col_lower:
                gene_col = col
                break

    if id_col is None or gene_col is None:
        print(f"[GPL] Could not find ID or Gene Symbol columns. Columns: {list(df.columns[:10])}")
        return pd.DataFrame(columns=['probe_id', 'gene_symbol'])

    mapping = df[[id_col, gene_col]].rename(columns={id_col: 'probe_id', gene_col: 'gene_symbol'})
    mapping = mapping[mapping['gene_symbol'].notna() & (mapping['gene_symbol'] != '')]
    # Some entries have multiple gene symbols separated by ' /// '
    mapping['gene_symbol'] = mapping['gene_symbol'].str.split(' /// ').str[0]
    print(f"[GPL] Mapped {len(mapping)} probes to gene symbols.")
    return mapping


def parse_geo_matrix(filepath):
    """Parse GEO series matrix file into expression DataFrame and metadata."""
    print(f"[GEO] Parsing series matrix: {filepath}")
    metadata_lines = {}
    expression_rows = []
    probe_ids = []
    sample_ids = None

    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
        in_table = False
        for line in f:
            if line.startswith('!Sample_characteristics_ch1') or line.startswith('!Sample_geo_accession'):
                parts = line.strip().split('\t')
                key = parts[0]
                values = [v.strip('"') for v in parts[1:]]
                if key not in metadata_lines:
                    metadata_lines[key] = []
                metadata_lines[key].append(values)

            elif line.startswith('"ID_REF"'):
                in_table = True
                header = line.strip().replace('"', '').split('\t')
                sample_ids = header[1:]
                continue

            if in_table:
                if line.startswith('!series_matrix_table_end'):
                    break
                parts = line.strip().replace('"', '').split('\t')
                if len(parts) > 1:
                    probe_ids.append(parts[0])
                    vals = []
                    for v in parts[1:]:
                        try:
                            vals.append(float(v))
                        except ValueError:
                            vals.append(np.nan)
                    expression_rows.append(vals)

    expr_df = pd.DataFrame(expression_rows, index=probe_ids, columns=sample_ids).T
    expr_df = expr_df.fillna(expr_df.mean())

    # Build clinical metadata from sample characteristics
    clinical = pd.DataFrame(index=sample_ids)
    if '!Sample_characteristics_ch1' in metadata_lines:
        for i, char_values in enumerate(metadata_lines['!Sample_characteristics_ch1']):
            if len(char_values) == len(sample_ids):
                # Extract key:value from characteristic strings like "age: 65"
                first_val = char_values[0]
                if ':' in first_val:
                    col_name = first_val.split(':')[0].strip().lower().replace(' ', '_')
                    clinical[col_name] = [v.split(':', 1)[-1].strip() if ':' in v else v for v in char_values]

    print(f"[GEO] Parsed expression: {expr_df.shape}, clinical: {clinical.shape}")
    return expr_df, clinical


def map_probes_to_genes(expr_df, probe_mapping):
    """
    Convert probe-level expression to gene-level by mapping probe IDs to
    gene symbols and averaging expression across probes for the same gene.
    """
    probe_to_gene = dict(zip(probe_mapping['probe_id'], probe_mapping['gene_symbol']))

    # Map column names (probe IDs) to gene symbols
    gene_names = [probe_to_gene.get(pid, None) for pid in expr_df.columns]
    valid_mask = [g is not None and g != '' and g != '---' for g in gene_names]

    expr_mapped = expr_df.loc[:, valid_mask].copy()
    expr_mapped.columns = [g for g, v in zip(gene_names, valid_mask) if v]

    # Collapse duplicate genes by taking the mean
    expr_gene = expr_mapped.T.groupby(expr_mapped.columns[valid_mask.count(True) - sum(valid_mask):]).mean().T
    # Simpler approach: transpose, groupby columns, mean, transpose back
    expr_gene = expr_mapped.groupby(expr_mapped.columns, axis=1).mean()

    print(f"[PROBE MAP] {expr_df.shape[1]} probes -> {expr_gene.shape[1]} unique genes")
    return expr_gene


def load_and_process_geo(data_dir="data", geo_id="GSE39582", gpl_id="GPL570"):
    """
    Download, parse, and process a GEO series into gene-level expression + clinical data.
    Supports GSE39582 (default) and GSE17538 (same GPL570 platform).
    Returns (expression_df, clinical_df) with gene symbols as column names.
    """
    raw_dir = os.path.join(data_dir, "raw")

    # Download series matrix (pass gpl_id for platform-qualified filenames)
    matrix_path = download_geo_dataset(geo_id, raw_dir, gpl_id=gpl_id)
    if matrix_path is None:
        return None, None

    # Parse series matrix (probe-level expression + clinical)
    expr_probes, clinical = parse_geo_matrix(matrix_path)

    # Download and parse GPL570 annotation for probe-to-gene mapping
    gpl_path = download_gpl_annotation("GPL570", raw_dir)
    if gpl_path is not None:
        probe_map = parse_gpl_annotation(gpl_path)
        if len(probe_map) > 0:
            expr_genes = map_probes_to_genes(expr_probes, probe_map)
            return expr_genes, clinical

    print("[GEO] WARNING: Could not map probes to genes. Returning probe-level data.")
    return expr_probes, clinical


def load_and_process_geo_merged(data_dir="data"):
    """
    Download and merge GSE39582 + GSE17538 (both GPL570, colon cancer).
    Returns merged expression + clinical DataFrames with samples row-concatenated
    and gene columns intersected.
    """
    print("=" * 60)
    print("LOADING GSE39582 (discovery cohort, ~585 samples)")
    exp1, clin1 = load_and_process_geo(data_dir, "GSE39582")

    print("=" * 60)
    print("LOADING GSE17538 (validation cohort, ~238 samples)")
    exp2, clin2 = load_and_process_geo(data_dir, "GSE17538")

    if exp1 is None or exp2 is None:
        print("[ERROR] Could not load one or both GEO datasets.")
        return None, None

    # Align on common gene columns
    common_genes = exp1.columns.intersection(exp2.columns)
    print(f"Merging on {len(common_genes)} common genes.")
    exp_merged = pd.concat([exp1[common_genes], exp2[common_genes]], axis=0)
    clin_merged = pd.concat([clin1, clin2], axis=0)

    print(f"Merged expression: {exp_merged.shape}")
    return exp_merged, clin_merged


def download_tcga_read(dest_dir="data/raw"):
    """Download TCGA-READ expression and clinical data from UCSC Xena."""
    os.makedirs(dest_dir, exist_ok=True)
    expr_path = os.path.join(dest_dir, "tcga_read_expression.tsv.gz")
    clinical_path = os.path.join(dest_dir, "tcga_read_clinical.tsv")

    expr_url = "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.READ.sampleMap%2FHiSeqV2.gz"
    clinical_url = "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.READ.sampleMap%2FREAD_clinicalMatrix"

    for path, url, name in [(expr_path, expr_url, "expression"), (clinical_path, clinical_url, "clinical")]:
        if not os.path.exists(path):
            print(f"[TCGA-READ] Downloading {name} from Xena...")
            try:
                r = requests.get(url, timeout=180)
                r.raise_for_status()
                with open(path, 'wb') as f:
                    f.write(r.content)
                print(f"[TCGA-READ] {name} downloaded: {path}")
            except Exception as e:
                print(f"[TCGA-READ] Failed to download {name}: {e}")
                return None, None

    return expr_path, clinical_path


def load_and_process_tcga_read(data_dir="data"):
    """
    Download, parse, and process TCGA-READ into expression + clinical DataFrames.
    Same structure as TCGA-COAD.
    """
    raw_dir = os.path.join(data_dir, "raw")
    expr_path, clinical_path = download_tcga_read(raw_dir)

    if expr_path is None:
        return None, None

    print(f"[TCGA-READ] Loading expression...")
    expr = pd.read_csv(expr_path, sep='\t', index_col=0, compression='gzip')
    expr = expr.T
    print(f"[TCGA-READ] Expression shape: {expr.shape}")

    print(f"[TCGA-READ] Loading clinical...")
    clinical = pd.read_csv(clinical_path, sep='\t', index_col=0)

    col_map = {}
    for c in clinical.columns:
        cl = c.lower()
        if 'age' in cl and 'diagnos' in cl:
            col_map[c] = 'age'
        elif cl in ('gender', 'gender.demographic'):
            col_map[c] = 'gender'
        elif 'pathologic_stage' in cl or 'pathologicstage' in cl.replace('_', ''):
            col_map[c] = 'stage'
        elif cl in ('_os', 'os.time', 'days_to_death'):
            col_map[c] = 'os_time'
        elif cl in ('_os_ind', 'os.indicator', 'vital_status'):
            col_map[c] = 'os_event'

    if col_map:
        clinical = clinical.rename(columns=col_map)

    common = expr.index.intersection(clinical.index)
    if len(common) > 0:
        expr = expr.loc[common]
        clinical = clinical.loc[common]

    print(f"[TCGA-READ] Matched {len(common)} samples.")
    return expr, clinical


def load_and_process_tcga_pan(data_dir="data"):
    """
    Merge TCGA-COAD + TCGA-READ into a single pan-colorectal expression + clinical DataFrame.
    """
    print("=" * 60)
    print("LOADING TCGA-COAD (colon adenocarcinoma, ~450 samples)")
    exp1, clin1 = load_and_process_tcga(data_dir)

    print("=" * 60)
    print("LOADING TCGA-READ (rectum adenocarcinoma, ~160 samples)")
    exp2, clin2 = load_and_process_tcga_read(data_dir)

    if exp1 is None or exp2 is None:
        print("[ERROR] Could not load one or both TCGA datasets.")
        return None, None

    common_genes = exp1.columns.intersection(exp2.columns)
    print(f"Merging on {len(common_genes)} common genes.")
    exp_merged = pd.concat([exp1[common_genes], exp2[common_genes]], axis=0)
    clin_merged = pd.concat([clin1, clin2], axis=0)

    print(f"Merged expression: {exp_merged.shape}")
    return exp_merged, clin_merged


def download_tcga_coad(dest_dir="data/raw"):
    """Download TCGA-COAD expression and clinical data from UCSC Xena."""
    os.makedirs(dest_dir, exist_ok=True)
    expr_path = os.path.join(dest_dir, "tcga_coad_expression.tsv.gz")
    clinical_path = os.path.join(dest_dir, "tcga_coad_clinical.tsv")

    expr_url = "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.COAD.sampleMap%2FHiSeqV2.gz"
    clinical_url = "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.COAD.sampleMap%2FCOAD_clinicalMatrix"

    for path, url, name in [(expr_path, expr_url, "expression"), (clinical_path, clinical_url, "clinical")]:
        if not os.path.exists(path):
            print(f"[TCGA] Downloading {name} from Xena...")
            try:
                r = requests.get(url, timeout=180)
                r.raise_for_status()
                with open(path, 'wb') as f:
                    f.write(r.content)
                print(f"[TCGA] {name} downloaded: {path}")
            except Exception as e:
                print(f"[TCGA] Failed to download {name}: {e}")
                return None, None

    return expr_path, clinical_path


def load_and_process_tcga(data_dir="data"):
    """
    Download, parse, and process TCGA-COAD into expression + clinical DataFrames.
    Xena expression is genes × samples (log2(norm_count+1)), already gene-symbol indexed.
    """
    raw_dir = os.path.join(data_dir, "raw")
    expr_path, clinical_path = download_tcga_coad(raw_dir)

    if expr_path is None:
        return None, None

    # Parse expression: genes × samples, transpose to samples × genes
    print(f"[TCGA] Loading expression from {expr_path}...")
    expr = pd.read_csv(expr_path, sep='\t', index_col=0, compression='gzip')
    expr = expr.T  # Now samples × genes
    print(f"[TCGA] Expression shape: {expr.shape}")

    # Parse clinical
    print(f"[TCGA] Loading clinical from {clinical_path}...")
    clinical = pd.read_csv(clinical_path, sep='\t', index_col=0)

    # Standardize clinical column names
    col_map = {}
    for c in clinical.columns:
        cl = c.lower()
        if 'age' in cl and 'diagnos' in cl:
            col_map[c] = 'age'
        elif cl in ('gender', 'gender.demographic'):
            col_map[c] = 'gender'
        elif 'pathologic_stage' in cl or 'pathologicstage' in cl.replace('_', ''):
            col_map[c] = 'stage'
        elif cl in ('_os', 'os.time', 'days_to_death'):
            col_map[c] = 'os_time'
        elif cl in ('_os_ind', 'os.indicator', 'vital_status'):
            col_map[c] = 'os_event'

    if col_map:
        clinical = clinical.rename(columns=col_map)

    # Match samples between expression and clinical
    common = expr.index.intersection(clinical.index)
    if len(common) > 0:
        expr = expr.loc[common]
        clinical = clinical.loc[common]

    print(f"[TCGA] Matched {len(common)} samples between expression and clinical.")
    return expr, clinical


def download_cptac_coad(dest_dir="data/raw"):
    """Download CPTAC COAD RNA-seq, clinical, and survival data from AWS S3."""
    os.makedirs(dest_dir, exist_ok=True)

    files = {
        "cptac_coad_rnaseq.txt": "https://cptac-pancancer-data.s3.us-west-2.amazonaws.com/data_freeze_v1.2_reorganized/COAD/COAD_RNAseq_gene_RSEM_coding_UQ_1500_log2_Tumor.txt",
        "cptac_coad_meta.txt": "https://cptac-pancancer-data.s3.us-west-2.amazonaws.com/data_freeze_v1.2_reorganized/COAD/COAD_meta.txt",
        "cptac_coad_survival.txt": "https://cptac-pancancer-data.s3.us-west-2.amazonaws.com/data_freeze_v1.2_reorganized/COAD/COAD_survival.txt",
    }

    paths = {}
    for fname, url in files.items():
        path = os.path.join(dest_dir, fname)
        paths[fname] = path
        if not os.path.exists(path):
            print(f"[CPTAC] Downloading {fname} ...")
            try:
                r = requests.get(url, timeout=180)
                r.raise_for_status()
                with open(path, 'wb') as f:
                    f.write(r.content)
                print(f"[CPTAC] Saved {path}")
            except Exception as e:
                print(f"[CPTAC] Failed to download {fname}: {e}")
                paths[fname] = None
        else:
            print(f"[CPTAC] {fname} already exists.")
    return paths


def ensembl_to_symbol_batch(ensembl_ids, chunk_size=1000):
    """
    Convert Ensembl gene IDs (ENSG...) to HUGO gene symbols using mygene.info API.
    Strips version suffixes (e.g., '.15') before lookup.
    """
    import time
    clean_ids = [eid.split('.')[0] for eid in ensembl_ids]
    id_to_symbol = {}

    for i in range(0, len(clean_ids), chunk_size):
        chunk = clean_ids[i:i + chunk_size]
        try:
            r = requests.post(
                "https://mygene.info/v3/gene",
                json={'ids': chunk, 'fields': 'symbol', 'species': 'human'},
                timeout=120,
            )
            if r.status_code == 200:
                for hit in r.json():
                    if isinstance(hit, dict) and hit.get('symbol'):
                        id_to_symbol[hit['query']] = hit['symbol']
                    elif isinstance(hit, str):
                        pass  # Not found entry is just the ID string
        except Exception as e:
            print(f"[CPTAC] mygene API error at batch {i}: {e}")
        time.sleep(0.3)

    return id_to_symbol


def load_and_process_cptac(data_dir="data"):
    """
    Download, parse, and process CPTAC COAD into expression + clinical DataFrames.
    Returns (expression_df, clinical_df) with gene symbols as column names.
    """
    raw_dir = os.path.join(data_dir, "raw")
    paths = download_cptac_coad(raw_dir)

    if any(paths[f] is None for f in paths):
        print("[CPTAC] One or more downloads failed. Skipping.")
        return None, None

    # Parse RNA-seq: genes × samples, transpose to samples × genes
    print(f"[CPTAC] Loading RNA-seq from {paths['cptac_coad_rnaseq.txt']}...")
    expr = pd.read_csv(paths['cptac_coad_rnaseq.txt'], sep='\t', index_col=0)
    print(f"[CPTAC] Raw expression: {expr.shape[0]} genes × {expr.shape[1]} samples")

    # Transpose to samples × genes
    expr = expr.T

    # Map Ensembl IDs to gene symbols
    ensembl_ids = list(expr.columns)
    print(f"[CPTAC] Mapping {len(ensembl_ids)} Ensembl IDs to gene symbols...")
    symbol_map = ensembl_to_symbol_batch(ensembl_ids)
    mapped_cols = {eid: symbol_map.get(eid.split('.')[0], eid) for eid in expr.columns}
    expr = expr.rename(columns=mapped_cols)

    # Drop unmapped columns (keep Ensembl IDs if no symbol found) but flag them
    # Actually drop columns that are still Ensembl IDs (unmapped)
    unmapped = [c for c in expr.columns if c.startswith('ENSG')]
    if unmapped:
        print(f"[CPTAC] Dropping {len(unmapped)} unmapped Ensembl IDs.")
        expr = expr.drop(columns=unmapped)

    # Deduplicate gene symbols by taking mean
    expr = expr.groupby(expr.columns, axis=1).mean()
    print(f"[CPTAC] Expression after symbol mapping: {expr.shape}")

    # Parse clinical meta file (skip second row which is data_type)
    print(f"[CPTAC] Loading clinical meta from {paths['cptac_coad_meta.txt']}...")
    meta = pd.read_csv(paths['cptac_coad_meta.txt'], sep='\t', skiprows=[1], index_col=0)
    print(f"[CPTAC] Meta samples: {meta.shape[0]}")

    # Parse survival
    print(f"[CPTAC] Loading survival from {paths['cptac_coad_survival.txt']}...")
    surv = pd.read_csv(paths['cptac_coad_survival.txt'], sep='\t', index_col=0)
    print(f"[CPTAC] Survival samples: {surv.shape[0]}")

    # Merge clinical data
    clinical = meta.join(surv, how='inner')
    print(f"[CPTAC] Merged clinical: {clinical.shape[0]} samples")

    # Rename columns to standard names expected by encode_clinical and survival analysis
    col_map = {}
    for c in clinical.columns:
        cl = c.lower()
        if cl == 'age':
            col_map[c] = 'age'
        elif cl == 'sex':
            col_map[c] = 'gender'
        elif cl == 'stage':
            col_map[c] = 'stage'
        elif cl == 'os_days':
            col_map[c] = 'os_time'
        elif cl == 'os_event':
            col_map[c] = 'os_event'
    if col_map:
        clinical = clinical.rename(columns=col_map)

    # Match samples between expression and clinical
    common = expr.index.intersection(clinical.index)
    if len(common) > 0:
        expr = expr.loc[common]
        clinical = clinical.loc[common]

    print(f"[CPTAC] Matched {len(common)} samples between expression and clinical.")
    return expr, clinical


def compute_proliferation_score(expr_df):
    """
    Compute proliferation score as the mean z-score of the 10-gene
    cell cycle proliferation signature.
    """
    cols_upper = {c.upper(): c for c in expr_df.columns}
    available = [cols_upper[g] for g in PROLIF_GENES if g in cols_upper]

    if len(available) == 0:
        print("Warning: No proliferation genes found, using mean of first 10 genes instead.")
        available = list(expr_df.columns[:10])

    print(f"Computing score from {len(available)}/{len(PROLIF_GENES)} available genes.")

    sub = expr_df[available].values
    scaler = StandardScaler()
    z = scaler.fit_transform(sub)
    scores = np.mean(z, axis=1)

    return pd.Series(scores, index=expr_df.index, name="proliferation_score")


def binarize_target(scores, threshold=None):
    """Binarize continuous scores at the median into Low (0) and High (1)."""
    if threshold is None:
        threshold = scores.median()
    print(f"Binarizing target at median: {threshold:.4f}")
    return (scores >= threshold).astype(int)


def encode_clinical(clinical_df):
    """Extract and encode clinical features: age, gender, stage."""
    result = pd.DataFrame(index=clinical_df.index)

    # Age
    age_cols = [c for c in clinical_df.columns if 'age' in c.lower()]
    if age_cols:
        result['clinical_age'] = pd.to_numeric(clinical_df[age_cols[0]], errors='coerce')
        result['clinical_age'] = result['clinical_age'].fillna(result['clinical_age'].median())
    else:
        result['clinical_age'] = 60.0

    # Gender
    gender_cols = [c for c in clinical_df.columns if 'gender' in c.lower() or 'sex' in c.lower()]
    if gender_cols:
        result['clinical_is_male'] = (
            clinical_df[gender_cols[0]].astype(str).str.upper().str.startswith('M').astype(int)
        )
    else:
        result['clinical_is_male'] = 0

    # Stage
    stage_cols = [c for c in clinical_df.columns if 'stage' in c.lower()]
    stage_map = {'I': 1, 'II': 2, 'III': 3, 'IV': 4, '1': 1, '2': 2, '3': 3, '4': 4}
    if stage_cols:
        mapped = []
        for val in clinical_df[stage_cols[0]].astype(str).str.upper():
            v = 0
            for k, num in stage_map.items():
                if k in val:
                    v = num
            mapped.append(v)
        result['clinical_stage'] = mapped
    else:
        result['clinical_stage'] = 2

    return result


def preprocess_and_save_data(expr_df, clinical_df, output_dir="data/processed", dataset_name="dataset"):
    """
    Build features and target from expression + clinical data.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Compute target from proliferation genes
    scores = compute_proliferation_score(expr_df)
    y = binarize_target(scores)

    # 2. Match samples
    common = expr_df.index.intersection(clinical_df.index).intersection(scores.index)
    print(f"Matched {len(common)} samples.")
    expr_df = expr_df.loc[common]
    clinical_df = clinical_df.loc[common]
    y = y.loc[common]
    scores = scores.loc[common]

    # 3. Remove proliferation genes
    X_expr = remove_proliferation_genes(expr_df)

    # 4. Encode clinical features
    X_clin = encode_clinical(clinical_df)

    # 5. Combine
    X = pd.concat([X_clin, X_expr], axis=1)

    # 6. Validate no leakage
    validate_no_leakage(X)

    # 7. Save
    prefix = f"{dataset_name}_" if dataset_name != "dataset" else ""
    X.to_csv(os.path.join(output_dir, f"{prefix}X_features.csv"))
    y.to_frame("target").to_csv(os.path.join(output_dir, f"{prefix}y_target.csv"))
    scores.to_frame("score").to_csv(os.path.join(output_dir, f"{prefix}proliferation_scores.csv"))

    # Save clinical data with survival columns for survival.py
    clinical_df.loc[common].to_csv(os.path.join(output_dir, f"{prefix}clinical.csv"))

    print(f"Saved {dataset_name}: X={X.shape}, y={y.shape}")
    return X, y



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess datasets for ColoGrowth-ML")
    parser.add_argument("--download", action="store_true", help="Download real GEO + TCGA datasets")
    parser.add_argument("--cptac", action="store_true", help="Download and process CPTAC-COAD dataset")
    parser.add_argument("--geo-merged", action="store_true", help="Download and merge GSE39582 + GSE17538")
    parser.add_argument("--tcga-pan", action="store_true", help="Download and merge TCGA-COAD + TCGA-READ")
    parser.add_argument("--synthetic", action="store_true", help="Generate and process synthetic data")
    parser.add_argument("--data-dir", default="data", help="Root data directory")
    args = parser.parse_args()

    proc_dir = os.path.join(args.data_dir, "processed")

    if args.download:
        print("=" * 60)
        print("DOWNLOADING AND PROCESSING GSE39582 (GEO)")
        print("=" * 60)
        geo_expr, geo_clin = load_and_process_geo(args.data_dir, "GSE39582")
        if geo_expr is not None:
            preprocess_and_save_data(geo_expr, geo_clin, proc_dir, dataset_name="geo")
        else:
            print("[ERROR] Could not load GEO data.")

        print("\n" + "=" * 60)
        print("DOWNLOADING AND PROCESSING GSE17538 (GEO)")
        print("=" * 60)
        geo17538_expr, geo17538_clin = load_and_process_geo(args.data_dir, "GSE17538")
        if geo17538_expr is not None:
            preprocess_and_save_data(geo17538_expr, geo17538_clin, proc_dir, dataset_name="geo17538")
        else:
            print("[ERROR] Could not load GSE17538 data.")

        print("\n" + "=" * 60)
        print("DOWNLOADING AND PROCESSING TCGA-COAD")
        print("=" * 60)
        tcga_expr, tcga_clin = load_and_process_tcga(args.data_dir)
        if tcga_expr is not None:
            preprocess_and_save_data(tcga_expr, tcga_clin, proc_dir, dataset_name="tcga")
        else:
            print("[ERROR] Could not load TCGA data.")

        print("\n" + "=" * 60)
        print("DOWNLOADING AND PROCESSING TCGA-READ")
        print("=" * 60)
        tcga_read_expr, tcga_read_clin = load_and_process_tcga_read(args.data_dir)
        if tcga_read_expr is not None:
            preprocess_and_save_data(tcga_read_expr, tcga_read_clin, proc_dir, dataset_name="tcga_read")
        else:
            print("[ERROR] Could not load TCGA-READ data.")

        print("\n" + "=" * 60)
        print("DOWNLOADING AND PROCESSING CPTAC-COAD")
        print("=" * 60)
        cptac_expr, cptac_clin = load_and_process_cptac(args.data_dir)
        if cptac_expr is not None:
            preprocess_and_save_data(cptac_expr, cptac_clin, proc_dir, dataset_name="cptac")
        else:
            print("[ERROR] Could not load CPTAC data.")

    elif args.geo_merged:
        print("=" * 60)
        print("PROCESSING MERGED GEO (GSE39582 + GSE17538)")
        print("=" * 60)
        merged_expr, merged_clin = load_and_process_geo_merged(args.data_dir)
        if merged_expr is not None:
            preprocess_and_save_data(merged_expr, merged_clin, proc_dir, dataset_name="geo_pan")
        else:
            print("[ERROR] Could not load merged GEO data.")

    elif args.tcga_pan:
        print("=" * 60)
        print("PROCESSING TCGA PANCOLORECTAL (COAD + READ)")
        print("=" * 60)
        merged_expr, merged_clin = load_and_process_tcga_pan(args.data_dir)
        if merged_expr is not None:
            preprocess_and_save_data(merged_expr, merged_clin, proc_dir, dataset_name="tcga_pan")
        else:
            print("[ERROR] Could not load TCGA pan data.")

    elif args.cptac:
        print("=" * 60)
        print("PROCESSING CPTAC-COAD COHORT")
        print("=" * 60)
        cptac_expr, cptac_clin = load_and_process_cptac(args.data_dir)
        if cptac_expr is not None:
            preprocess_and_save_data(cptac_expr, cptac_clin, proc_dir, dataset_name="cptac")
        else:
            print("[ERROR] Could not load CPTAC data.")
    else:
        # Default: generate synthetic data
        print("Generating synthetic data for pipeline testing...")
        expr, clin = generate_synthetic_data()
        preprocess_and_save_data(expr, clin, proc_dir, dataset_name="synthetic")
