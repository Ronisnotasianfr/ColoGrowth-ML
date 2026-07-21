import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection._base import SelectorMixin
from sklearn.feature_selection import f_classif
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from joblib import Parallel, delayed


class StabilitySelector(BaseEstimator, SelectorMixin):
    """
    Bootstrap-stability-based feature selector.

    Instead of ranking features by a single ANOVA F-score computation,
    resamples the training data B times, computes F-scores each time,
    and selects features that appear in the top K in at least `min_pct`
    of iterations. Features with high selection frequency are more
    robust to data perturbations — a known problem in high-dimensional
    bioinformatics (Meinshausen & Bühlmann, 2010).

    Parameters
    ----------
    k : int, default=500
        Number of top features to keep per bootstrap iteration.

    n_bootstrap : int, default=100
        Number of bootstrap resamples.

    min_pct : float, default=0.5
        Minimum proportion of bootstrap iterations a feature must be
        selected in to be retained.

    score_func : callable, default=f_classif
        Function taking (X, y) and returning (F-scores, p-values).

    n_jobs : int, default=-1
        Number of parallel jobs for bootstrap iterations.

    random_state : int, default=42
        Seed for reproducible bootstrap resampling.
    """
    def __init__(self, k=500, n_bootstrap=100, min_pct=0.5,
                 score_func=f_classif, n_jobs=-1, random_state=42):
        self.k = k
        self.n_bootstrap = n_bootstrap
        self.min_pct = min_pct
        self.score_func = score_func
        self.n_jobs = n_jobs
        self.random_state = random_state

    def _bootstrap_fscores(self, X, y, seed):
        rng = np.random.default_rng(seed)
        n = X.shape[0]
        idx = rng.integers(0, n, n)
        X_boot = X[idx]
        y_boot = y[idx]
        f_scores, _ = self.score_func(X_boot, y_boot)
        top_idx = np.argsort(f_scores)[-self.k:]
        return set(top_idx)

    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=True, dtype=None)
        n_features = X.shape[1]
        effective_k = min(self.k, n_features)
        seeds = [self.random_state + i for i in range(self.n_bootstrap)]
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._bootstrap_fscores)(X, y, s) for s in seeds
        )
        selection_counts = np.zeros(n_features, dtype=int)
        for selected_set in results:
            selection_counts[list(selected_set)] += 1
        self.selection_frequencies_ = selection_counts / self.n_bootstrap
        n_boot_effective = self.n_bootstrap
        threshold_count = max(1, int(np.ceil(self.min_pct * n_boot_effective)))
        self.support_ = selection_counts >= threshold_count
        if not self.support_.any():
            self.support_[np.argsort(selection_counts)[-int(effective_k * 0.1):]] = True
        self.n_features_in_ = X.shape[1]
        return self

    def _get_support_mask(self):
        check_is_fitted(self, 'support_')
        return self.support_

    def get_support(self, indices=False):
        mask = self._get_support_mask()
        return np.where(mask)[0] if indices else mask

    def transform(self, X):
        check_is_fitted(self, 'support_')
        X = check_array(X, accept_sparse=True, dtype=None)
        mask = self._get_support_mask()
        if X.shape[1] != len(mask):
            raise ValueError(f"X has {X.shape[1]} features but StabilitySelector was fitted on {len(mask)}")
        return X[:, mask]

    def get_selection_frequencies(self):
        check_is_fitted(self, 'selection_frequencies_')
        return self.selection_frequencies_
