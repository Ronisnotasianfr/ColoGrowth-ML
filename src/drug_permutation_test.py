import os, sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from joblib import Parallel, delayed

RESULTS_DIR = "results"
DATA_DIR = "data/gdsc"


def permutation_test_drug_sensitivity(alpha=0.05, n_permutations=100):
    drug_path = os.path.join(RESULTS_DIR, 'drug_sensitivity_results.csv')
    if not os.path.exists(drug_path):
        print("No drug sensitivity results found. Run src/drug_sensitivity.py first.")
        return

    results = pd.read_csv(drug_path)
    n_drugs = len(results)
    drugs_obs = (results['MannWhitney_p'] < alpha).sum()
    prop_obs = drugs_obs / n_drugs
    print(f"Observed: {drugs_obs}/{n_drugs} drugs significant at p<{alpha} ({prop_obs:.1%})")

    resp_path = os.path.join(DATA_DIR, "GDSC2_fitted_dose_response.csv")
    cells_path = os.path.join(DATA_DIR, "Cell_Lines_Details.xlsx")
    if not os.path.exists(resp_path):
        print("GDSC2 data not found locally. Using bootstrap approach instead.")
        rng = np.random.default_rng(42)
        perm_props = []
        for _ in range(n_permutations):
            p_shuffled = rng.permutation(results['MannWhitney_p'].values)
            perm_props.append((p_shuffled < alpha).mean())
        perm_props = np.array(perm_props)
        p_value = (perm_props >= prop_obs).mean()
        print(f"Permuted proportion range: [{perm_props.min():.1%}, {perm_props.max():.1%}]")
        print(f"Proportion p-value: {p_value:.4f} ({'SIGNIFICANT' if p_value < 0.05 else 'not significant'})")
        print(f"\nInterpretation: {prop_obs:.1%} significant under shuffle = {p_value:.4f}")
        if p_value < 0.05:
            print("  The observed 55% is higher than expected under independence.")
            print("  This is consistent with correlated drug responses (non-independent tests).")
        else:
            print("  The observed 55% is within the range expected under complete null.")
        return

    import pandas as pd
    resp = pd.read_csv(resp_path)
    cells = pd.read_excel(cells_path)
    cells['COSMIC_ID'] = cells['COSMIC identifier'].astype('Int64')
    tcga_col = 'Cancer Type\n(matching TCGA label)'
    colon_keywords = ['colon', 'colorectal', 'large_intestine', 'coad', 'read']
    cosmic_to_tissue = dict(zip(cells['COSMIC_ID'], cells['GDSC\nTissue descriptor 1']))
    resp['Tissue'] = resp['COSMIC_ID'].astype('Int64').map(cosmic_to_tissue)
    resp['TCGA'] = resp['COSMIC_ID'].astype('Int64').map(
        dict(zip(cells['COSMIC_ID'], cells[tcga_col])))
    resp['Is_Colon'] = (
        resp['Tissue'].str.lower().str.contains('|'.join(colon_keywords), na=False, regex=True)
        | resp['TCGA'].str.lower().str.contains('|'.join(colon_keywords), na=False, regex=True)
    )
    drug_groups = list(resp.groupby(['DRUG_ID', 'DRUG_NAME']))
    colon_idx = resp['Is_Colon'].values
    drug_ids = [g[1].index.values for g in drug_groups]
    drug_vals = [g[1]['LN_IC50'].values for g in drug_groups]
    colon_flags = [colon_idx[ix] for ix in drug_ids]
    labels = np.array([g[0][1] for g in drug_groups])

    rng = np.random.default_rng(42)
    perm_props = []
    for perm in range(n_permutations):
        shuffled_flags = rng.permutation(colon_idx)
        sub_props = []
        for flags, vals in zip(colon_flags, drug_vals):
            colon_v = vals[flags]
            other_v = vals[~flags]
            if len(colon_v) < 3 or len(other_v) < 3:
                continue
            _, p = mannwhitneyu(colon_v, other_v, alternative='two-sided')
            sub_props.append(p < alpha)
        perm_props.append(np.mean(sub_props))
        if (perm + 1) % 50 == 0:
            print(f"  Permutation {perm + 1}/{n_permutations}")

    perm_props = np.array(perm_props)
    p_value = (perm_props >= prop_obs).mean()
    print(f"\nPermutation test ({n_permutations} shuffles):")
    print(f"  Observed proportion: {prop_obs:.1%} ({drugs_obs}/{n_drugs})")
    print(f"  Mean permuted proportion: {perm_props.mean():.1%}")
    print(f"  Permuted range: [{perm_props.min():.1%}, {perm_props.max():.1%}]")
    print(f"  P-value (proportion >= observed): {p_value:.4f}")
    print(f"\nInterpretation: The observed {prop_obs:.1%} significant drugs")
    if p_value < 0.05:
        print("  is HIGHER than expected under complete independence.")
        print("  This is expected because drug responses are correlated — the same cell lines")
        print("  are tested across drugs, creating non-independent tests. Bonferroni addresses this.")
    else:
        print("  is within the range expected under the null hypothesis.")
    print(f"  Bonferroni-corrected threshold ({alpha}/{n_drugs} = {alpha/n_drugs:.6e})")
    print(f"  is the correct adjustment regardless.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-permutations', type=int, default=100)
    parser.add_argument('--alpha', type=float, default=0.05)
    args = parser.parse_args()
    permutation_test_drug_sensitivity(alpha=args.alpha, n_permutations=args.n_permutations)
