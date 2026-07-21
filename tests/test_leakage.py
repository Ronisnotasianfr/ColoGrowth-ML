import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import pandas as pd
from src.preprocess import PROLIF_GENES, remove_proliferation_genes, validate_no_leakage


def make_demo_features(n_samples=100, n_genes=500):
    gene_names = PROLIF_GENES + [f'GENE_{i}' for i in range(500)]
    data = np.random.normal(6, 1.5, (n_samples, len(gene_names)))
    return pd.DataFrame(data, columns=gene_names,
                        index=[f'Sample_{i}' for i in range(n_samples)])


def test_remove_proliferation_genes_removes_all():
    X = make_demo_features()
    X_clean = remove_proliferation_genes(X)
    for gene in PROLIF_GENES:
        upper_variants = [gene, gene.upper(), gene.lower()]
        assert not any(g in X_clean.columns for g in upper_variants), \
            f'{gene} still in feature matrix after removal'


def test_validate_no_leakage_passes_on_clean():
    X = make_demo_features()
    X_clean = remove_proliferation_genes(X)
    try:
        validate_no_leakage(X_clean)
    except AssertionError:
        pytest.fail('validate_no_leakage raised on clean matrix')


def test_validate_no_leakage_raises_on_contaminated():
    X = make_demo_features()
    try:
        validate_no_leakage(X)
    except AssertionError:
        pass
    else:
        raise AssertionError('validate_no_leakage did not detect contaminated matrix')


def test_stability_selector_fit_transform():
    from src.stability_selector import StabilitySelector
    n, p = 200, 50
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (n, p))
    y = (X[:, 0] + X[:, 1] + rng.normal(0, 0.5, n)) > 0
    sel = StabilitySelector(k=10, n_bootstrap=30, min_pct=0.3, random_state=42)
    sel.fit(X, y)
    Xr = sel.transform(X)
    assert Xr.shape[1] <= 10
    assert Xr.shape[1] > 0
    freqs = sel.get_selection_frequencies()
    assert len(freqs) == p


def test_stability_selector_parallel_consistency():
    from src.stability_selector import StabilitySelector
    n, p = 100, 30
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (n, p))
    y = (X[:, 0] + rng.normal(0, 0.3, n)) > 0
    sel1 = StabilitySelector(k=5, n_bootstrap=20, min_pct=0.3, n_jobs=1, random_state=42)
    sel2 = StabilitySelector(k=5, n_bootstrap=20, min_pct=0.3, n_jobs=-1, random_state=42)
    sel1.fit(X, y)
    sel2.fit(X, y)
    assert np.array_equal(sel1.get_support(), sel2.get_support())
