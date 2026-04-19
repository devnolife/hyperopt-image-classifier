"""Analisis hasil HPO: tabel komparasi + grafik."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.utils import ensure_dir


METHOD_LABEL = {
    "grid_search": "Grid Search",
    "random_search": "Random Search",
    "bayesian_tpe": "Bayesian (TPE)",
    "hyperband_asha": "Hyperband/ASHA",
    "genetic": "Genetic Algorithm",
}


def main():
    results_path = ROOT / "results" / "all_hpo_results.json"
    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    fig_dir = ensure_dir(ROOT / "results" / "figures")
    tab_dir = ensure_dir(ROOT / "results" / "tables")

    # === 1. Tabel komparasi ===
    rows = []
    for name, r in results.items():
        bp = r["best"]["params"]
        rows.append({
            "Metode": METHOD_LABEL.get(name, name),
            "Jumlah Trial": r["n_trials"],
            "Best Val Acc": round(r["best"]["val_acc"], 4),
            "Best Epoch": r["best"]["best_epoch"],
            "Total Waktu (s)": round(r["total_time"], 1),
            "Learning Rate": f"{bp['learning_rate']:.2e}",
            "Batch Size": bp["batch_size"],
            "Optimizer": bp["optimizer"],
            "Dropout": round(bp["dropout"], 3),
            "Base Filters": bp["base_filters"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(tab_dir / "hpo_comparison.csv", index=False)
    print(df.to_string(index=False))

    # === 2. Konvergensi best-so-far ===
    plt.figure(figsize=(9, 5.5))
    for name, r in results.items():
        accs = [t["val_acc"] for t in r["trials"]]
        best_so_far = np.maximum.accumulate(accs) if accs else []
        plt.plot(range(1, len(best_so_far) + 1), best_so_far,
                 marker="o", markersize=4, label=METHOD_LABEL.get(name, name))
    plt.xlabel("Trial ke-")
    plt.ylabel("Best Validation Accuracy (so far)")
    plt.title("Konvergensi Best-so-far Tiap Metode HPO")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "convergence_best_so_far.png", dpi=140)
    plt.close()

    # === 3. Total waktu per metode ===
    plt.figure(figsize=(8, 5))
    names = [METHOD_LABEL.get(n, n) for n in results.keys()]
    times = [r["total_time"] for r in results.values()]
    bars = plt.bar(names, times, color=["#4C72B0", "#DD8452", "#55A467", "#C44E52", "#8172B2"])
    plt.ylabel("Total Waktu (detik)")
    plt.title("Total Waktu Eksekusi per Metode HPO")
    plt.xticks(rotation=15)
    for b, t in zip(bars, times):
        plt.text(b.get_x() + b.get_width()/2, b.get_height(), f"{t:.0f}s",
                 ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(fig_dir / "total_time_per_method.png", dpi=140)
    plt.close()

    # === 4. Best val acc per metode ===
    plt.figure(figsize=(8, 5))
    accs = [r["best"]["val_acc"] for r in results.values()]
    bars = plt.bar(names, accs, color=["#4C72B0", "#DD8452", "#55A467", "#C44E52", "#8172B2"])
    plt.ylabel("Best Validation Accuracy")
    plt.title("Best Validation Accuracy per Metode HPO")
    plt.ylim(0, 1)
    plt.xticks(rotation=15)
    for b, a in zip(bars, accs):
        plt.text(b.get_x() + b.get_width()/2, b.get_height(), f"{a:.4f}",
                 ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(fig_dir / "best_val_acc_per_method.png", dpi=140)
    plt.close()

    # === 5. Scatter: val_acc vs learning_rate ===
    plt.figure(figsize=(9, 5.5))
    for name, r in results.items():
        lrs = [t["params"]["learning_rate"] for t in r["trials"]]
        accs_ = [t["val_acc"] for t in r["trials"]]
        plt.scatter(lrs, accs_, label=METHOD_LABEL.get(name, name), alpha=0.75, s=35)
    plt.xscale("log")
    plt.xlabel("Learning Rate (log)")
    plt.ylabel("Validation Accuracy")
    plt.title("Sebaran Validation Accuracy vs Learning Rate")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "scatter_lr_vs_acc.png", dpi=140)
    plt.close()

    # === 6. Learning curve best trial tiap metode ===
    plt.figure(figsize=(9, 5.5))
    for name, r in results.items():
        hist = r["best"]["history"]
        plt.plot(range(1, len(hist["val_acc"]) + 1), hist["val_acc"],
                 marker="o", markersize=3, label=METHOD_LABEL.get(name, name))
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("Learning Curve Best Trial per Metode")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "learning_curve_best_trials.png", dpi=140)
    plt.close()

    print("\nFigures saved ke results/figures/")
    print("Tabel saved ke results/tables/hpo_comparison.csv")


if __name__ == "__main__":
    main()
