"""Synthetic gene expression data generation for pipeline testing."""

import numpy as np
import pandas as pd

from src.preprocess import PROLIF_GENES


def generate_synthetic_data(n_samples=300, n_genes=2000):
    print(f"Generating {n_samples} synthetic samples with {n_genes} genes...")
    np.random.seed(42)

    other_genes = [f"GENE_{i}" for i in range(n_genes - len(PROLIF_GENES))]
    all_genes = PROLIF_GENES + other_genes

    expression = np.random.normal(loc=6.0, scale=1.5, size=(n_samples, n_genes))

    latent = np.random.normal(0, 1, size=(n_samples, 1))
    for i, gene in enumerate(PROLIF_GENES):
        idx = all_genes.index(gene)
        expression[:, idx] += latent.flatten() * 1.2

    for j in range(10, 30):
        expression[:, j] += latent.flatten() * np.random.uniform(0.3, 0.7)

    df_expr = pd.DataFrame(expression, columns=all_genes,
                           index=[f"Sample_{i}" for i in range(n_samples)])

    stages = np.random.choice([1, 2, 3, 4], p=[0.2, 0.4, 0.3, 0.1], size=n_samples)
    os_time = np.random.exponential(scale=60, size=n_samples).clip(1, 200)
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
