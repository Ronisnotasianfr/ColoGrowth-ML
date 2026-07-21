import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

RESULTS_DIR = "results"

def required_events(hr, alpha=0.05, power=0.80):
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    return (z_alpha + z_beta) ** 2 / (np.log(hr)) ** 2

def main():
    hr_observed = 0.78
    hr_geo_pan = 1.09
    event_rate = 0.33

    # Power at different levels for HR=0.78
    power_levels = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
    print(f"Required events at HR={hr_observed}, alpha=0.05:")
    for pwr in power_levels:
        n_events = required_events(hr_observed, power=pwr)
        n_total = n_events / event_rate
        print(f"  Power={pwr:.0%}: {n_events:.0f} events ({n_total:.0f} samples)")

    # Panel 1: Power curves across HR values
    hr_range = np.linspace(0.50, 1.0, 50)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sample_sizes = [400, 585, 823, 1200, 2000]
    for n in sample_sizes:
        n_events = n * event_rate
        powers = []
        for hr in hr_range:
            if hr == 1.0:
                powers.append(0.05)
                continue
            z_alpha = stats.norm.ppf(1 - 0.05 / 2)
            z = np.sqrt(n_events) * np.log(hr) - z_alpha
            powers.append(stats.norm.cdf(z))
        label = f"n={n}" + (" (ours)" if n in [585, 823] else "")
        axes[0].plot(hr_range, powers, label=label, linewidth=2)

    axes[0].axhline(0.80, color='gray', linestyle='--', alpha=0.5, label="80% power")
    axes[0].axvline(hr_observed, color='red', linestyle=':', alpha=0.7, label=f"HR={hr_observed}")
    axes[0].set_xlabel("Hazard Ratio")
    axes[0].set_ylabel("Statistical Power")
    axes[0].set_title("Power vs Effect Size at alpha=0.05")
    axes[0].legend(loc="lower left", fontsize=8)
    axes[0].grid(True, linestyle=':', alpha=0.5)

    # Panel 2: Required N vs HR
    hr_plot = np.linspace(0.50, 0.95, 45)
    req_events = [required_events(h) for h in hr_plot]
    req_n = [e / event_rate for e in req_events]
    axes[1].plot(hr_plot, req_n, 'b-', linewidth=2)
    axes[1].axhline(585, color='green', linestyle='--', alpha=0.6, label="GSE39582 (n=585)")
    axes[1].axhline(823, color='orange', linestyle='--', alpha=0.6, label="Merged GEO (n=823)")
    axes[1].axvline(hr_observed, color='red', linestyle=':', alpha=0.7, label=f"HR={hr_observed}")
    axes[1].set_xlabel("Hazard Ratio")
    axes[1].set_ylabel("Required Sample Size (80% power)")
    axes[1].set_title("Sample Size Required for 80% Power at alpha=0.05")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "power_analysis.png"), dpi=300)
    print(f"\nPlot saved to {RESULTS_DIR}/power_analysis.png")

    cohorts = [
        ("GSE39582 (GEO)", 585, 194, 0.33, 0.037),
        ("GSE17538 (GEO)", 238, 87, 0.37, 0.885),
        ("Merged GEO", 823, 281, 0.34, 0.938),
        ("TCGA-COAD", 329, 79, 0.24, 0.034),
        ("TCGA-READ", 105, 21, 0.20, 0.271),
        ("TCGA PanCancer", 434, 100, 0.23, 0.009),
        ("CPTAC-COAD", 105, 7, 0.07, 0.356),
    ]
    summary = pd.DataFrame(cohorts, columns=["Cohort", "N", "Events", "Event Rate", "Log-Rank p"])
    print(f"\n{summary.to_string(index=False)}")
    summary.to_csv(os.path.join(RESULTS_DIR, "survival_power_summary.csv"), index=False)

    print(f"\nKey takeaways:")
    print(f"  Detecting HR={hr_observed} at 80% power needs ~{int(required_events(hr_observed, power=0.80) / event_rate)} samples")
    print(f"  TCGA PanCancer (p=0.009) is the primary survival evidence")
    print(f"  CPTAC (7 events) is severely underpowered")
    print(f"  Merged GEO (HR={hr_geo_pan}) has divergent survival signals between cohorts")

if __name__ == "__main__":
    main()
