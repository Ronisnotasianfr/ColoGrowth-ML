import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import requests
import io
import gzip

def generate_synthetic_data(n_samples=200, n_genes=1000):
    """
    Generates synthetic gene expression and clinical data for testing purposes.
    """
    print(f"Generating {n_samples} synthetic samples with {n_genes} genes...")
    np.random.seed(42)
    
    # Proliferation-related genes
    prolif_genes = ['MKI67', 'PCNA', 'TOP2A', 'MCM2', 'MCM6', 'AURKA', 'BUB1', 'CCNB1', 'CDK1', 'BIRC5']
    other_genes = [f"GENE_{i}" for i in range(n_genes - len(prolif_genes))]
    all_genes = prolif_genes + other_genes
    
    # Simulate expression matrix (samples x genes)
    # Proliferation genes will have higher covariance
    expression = np.random.normal(loc=6.0, scale=1.5, size=(n_samples, n_genes))
    
    # Make proliferation genes correlated to simulate cell growth rate
    latent_proliferation = np.random.normal(loc=0.0, scale=1.0, size=(n_samples, 1))
    prolif_indices = [all_genes.index(g) for g in prolif_genes]
    for idx in prolif_indices:
        expression[:, idx] += latent_proliferation.flatten() * 1.2
        
    df_expr = pd.DataFrame(expression, columns=all_genes)
    df_expr.index = [f"Sample_{i}" for i in range(n_samples)]
    
    # Simulate clinical data
    clinical = {
        'sample_id': [f"Sample_{i}" for i in range(n_samples)],
        'age': np.random.randint(40, 85, size=n_samples),
        'gender': np.random.choice(['MALE', 'FEMALE'], size=n_samples),
        'stage': np.random.choice(['Stage I', 'Stage II', 'Stage III', 'Stage IV'], p=[0.2, 0.4, 0.3, 0.1], size=n_samples),
        'tumor_size_mm': np.random.normal(loc=45.0, scale=15.0, size=n_samples).clip(10, 120)
    }
    df_clinical = pd.DataFrame(clinical).set_index('sample_id')
    
    return df_expr, df_clinical

def download_geo_dataset(geo_id="GSE39582", dest_dir="data/raw"):
    """
    Downloads GSE39582 dataset series matrix file.
    """
    os.makedirs(dest_dir, exist_ok=True)
    filepath = os.path.join(dest_dir, f"{geo_id}_series_matrix.txt.gz")
    
    if os.path.exists(filepath):
        print(f"Dataset {geo_id} already exists at {filepath}")
        return filepath
        
    url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/GSE39nnn/{geo_id}/matrix/{geo_id}_series_matrix.txt.gz"
    print(f"Downloading {geo_id} from {url}...")
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download completed.")
        return filepath
    except Exception as e:
        print(f"Failed to download GEO dataset: {e}")
        return None

def download_tcga_coad(dest_dir="data/raw"):
    """
    Downloads TCGA-COAD gene expression and clinical metadata from UCSC Xena.
    """
    os.makedirs(dest_dir, exist_ok=True)
    expr_path = os.path.join(dest_dir, "tcga_coad_expression.tsv.gz")
    clinical_path = os.path.join(dest_dir, "tcga_coad_clinical.tsv")
    
    # URLs for UCSC Xena TCGA-COAD cohort (HiSeqV2 expression and phenotype data)
    expr_url = "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.COAD.sampleMap%2FHiSeqV2.gz"
    clinical_url = "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.COAD.sampleMap%2FCOAD_clinicalMatrix"
    
    success = True
    
    if not os.path.exists(expr_path):
        print(f"Downloading TCGA-COAD Expression from {expr_url}...")
        try:
            r = requests.get(expr_url, timeout=120)
            r.raise_for_status()
            with open(expr_path, 'wb') as f:
                f.write(r.content)
            print("Expression downloaded.")
        except Exception as e:
            print(f"Failed to download TCGA-COAD expression: {e}")
            success = False
            
    if not os.path.exists(clinical_path):
        print(f"Downloading TCGA-COAD Clinical Matrix from {clinical_url}...")
        try:
            r = requests.get(clinical_url, timeout=60)
            r.raise_for_status()
            with open(clinical_path, 'wb') as f:
                f.write(r.content)
            print("Clinical data downloaded.")
        except Exception as e:
            print(f"Failed to download TCGA-COAD clinical matrix: {e}")
            success = False
            
    return (expr_path, clinical_path) if success else (None, None)

def parse_geo_matrix(filepath):
    """
    Parses a GEO series matrix file manually to avoid memory issues and external package dependencies if needed.
    """
    print(f"Parsing GEO Series Matrix: {filepath}")
    metadata = {}
    expression_data = []
    gene_ids = []
    
    # Read series matrix gzipped file
    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
        in_matrix = False
        for line in f:
            if line.startswith("!Series_") or line.startswith("!Sample_"):
                # Parse metadata line
                parts = line.strip().split("\t")
                key = parts[0]
                values = parts[1:]
                if key not in metadata:
                    metadata[key] = []
                metadata[key].append(values)
            elif line.startswith("\"ID_REF\""):
                in_matrix = True
                header = line.strip().replace('"', '').split("\t")
                sample_ids = header[1:]
                continue
            
            if in_matrix:
                if line.startswith("!series_matrix_table_end"):
                    break
                parts = line.strip().replace('"', '').split("\t")
                if len(parts) > 1:
                    gene_ids.append(parts[0])
                    # Convert to floats, handle missing values
                    vals = []
                    for v in parts[1:]:
                        try:
                            vals.append(float(v))
                        except ValueError:
                            vals.append(np.nan)
                    expression_data.append(vals)
                    
    expr_df = pd.DataFrame(expression_data, index=gene_ids, columns=sample_ids).T
    # Interpolate missing values
    expr_df = expr_df.fillna(expr_df.mean())
    
    # Create metadata DataFrame
    meta_df = pd.DataFrame()
    if "!Sample_title" in metadata:
        meta_df['title'] = metadata["!Sample_title"][0]
    if "!Sample_geo_accession" in metadata:
        meta_df['geo_accession'] = metadata["!Sample_geo_accession"][0]
        meta_df = meta_df.set_index('geo_accession')
    
    # Extract clinical keys if available
    for key, vals in metadata.items():
        if key.startswith("!Sample_characteristics_ch1"):
            for i, val_list in enumerate(vals):
                col_name = f"characteristic_{i}"
                if len(val_list) == len(meta_df):
                    meta_df[col_name] = val_list
                    
    return expr_df, meta_df

def compute_proliferation_score(expr_df):
    """
    Computes a proliferation score as the mean z-score of a set of proliferation genes.
    """
    prolif_genes = ['MKI67', 'PCNA', 'TOP2A', 'MCM2', 'MCM6', 'AURKA', 'BUB1', 'CCNB1', 'CDK1', 'BIRC5']
    
    # Find matching columns (case-insensitive)
    columns_map = {col.upper(): col for col in expr_df.columns}
    available_genes = [columns_map[g] for g in prolif_genes if g in columns_map]
    
    if not available_genes:
        print("Warning: No proliferation genes found! Using raw average of first 5 genes as fallback.")
        available_genes = list(expr_df.columns[:5])
        
    print(f"Computing proliferation score based on {len(available_genes)} genes: {available_genes}")
    
    # Z-score normalize the selected genes across samples
    sub_df = expr_df[available_genes]
    scaler = StandardScaler()
    z_scores = scaler.fit_transform(sub_df)
    
    # Proliferation score is the mean z-score across these genes for each sample
    scores = np.mean(z_scores, axis=1)
    
    return pd.Series(scores, index=expr_df.index, name="proliferation_score")

def binarize_target(scores, threshold=None):
    """
    Binarizes continuous scores into 0 (Low Proliferation) and 1 (High Proliferation) at the median.
    """
    if threshold is None:
        threshold = scores.median()
    print(f"Binarizing proliferation score at median threshold: {threshold:.4f}")
    return (scores >= threshold).astype(int)

def preprocess_and_save_data(expr_df, clinical_df, output_dir="data/processed"):
    """
    Preprocesses, normalizes, extracts features, computes target, and saves train/test sets.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Compute target (Proliferation score)
    scores = compute_proliferation_score(expr_df)
    y = binarize_target(scores)
    
    # 2. Match samples between expression and clinical data
    common_samples = expr_df.index.intersection(clinical_df.index)
    print(f"Matched {len(common_samples)} samples between expression and clinical metadata.")
    
    X_expr = expr_df.loc[common_samples]
    X_clin = clinical_df.loc[common_samples]
    y = y.loc[common_samples]
    
    # 3. Clean clinical features
    # Process age
    age_col = [c for c in X_clin.columns if 'age' in c.lower()]
    if age_col:
        X_clin['age_clean'] = pd.to_numeric(X_clin[age_col[0]], errors='coerce')
        X_clin['age_clean'] = X_clin['age_clean'].fillna(X_clin['age_clean'].median())
    else:
        X_clin['age_clean'] = 60.0
        
    # Process gender/sex
    gender_col = [c for c in X_clin.columns if 'gender' in c.lower() or 'sex' in c.lower()]
    if gender_col:
        X_clin['is_male'] = X_clin[gender_col[0]].astype(str).str.upper().str.startswith('M').astype(int)
    else:
        X_clin['is_male'] = np.random.choice([0, 1], size=len(X_clin))
        
    # Process stage
    stage_col = [c for c in X_clin.columns if 'stage' in c.lower() or 'pathology_t_stage' in c.lower()]
    if stage_col:
        # Simple label encoder for stage
        stage_map = {'STAGE I': 1, 'STAGE II': 2, 'STAGE III': 3, 'STAGE IV': 4,
                     'I': 1, 'II': 2, 'III': 3, 'IV': 4,
                     '1': 1, '2': 2, '3': 3, '4': 4}
        stages_mapped = []
        for val in X_clin[stage_col[0]].astype(str).str.upper():
            mapped_val = 0
            for k, v in stage_map.items():
                if k in val:
                    mapped_val = v
                    break
            stages_mapped.append(mapped_val)
        X_clin['stage_clean'] = stages_mapped
    else:
        X_clin['stage_clean'] = np.random.randint(1, 5, size=len(X_clin))
        
    # Build feature matrix
    X_clin_clean = X_clin[['age_clean', 'is_male', 'stage_clean']].rename(
        columns={'age_clean': 'clinical_age', 'is_male': 'clinical_is_male', 'stage_clean': 'clinical_stage'}
    )
    
    # 4. Filter expression features (variance threshold)
    variances = X_expr.var(axis=0)
    # Select top 500 high-variance genes
    top_genes = variances.nlargest(500).index
    X_expr_filtered = X_expr[top_genes]
    
    # 5. Concatenate clinical and gene expression features
    X_all = pd.concat([X_clin_clean, X_expr_filtered], axis=1)
    
    # Save processed files
    X_all.to_csv(os.path.join(output_dir, "X_features.csv"))
    y.to_frame(name="target").to_csv(os.path.join(output_dir, "y_target.csv"))
    scores.to_frame(name="score").to_csv(os.path.join(output_dir, "proliferation_scores.csv"))
    
    print(f"Processed dataset saved successfully. X shape: {X_all.shape}, y shape: {y.shape}")
    return X_all, y

if __name__ == "__main__":
    # Test preprocess function
    expr, clin = generate_synthetic_data()
    X, y = preprocess_and_save_data(expr, clin)
