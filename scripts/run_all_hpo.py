"""Jalankan 5 metode HPO secara berurutan dan simpan hasilnya."""
from __future__ import annotations
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import mlflow

from src.utils import load_config, ensure_dir, save_json, set_seed
from src.hpo import grid_search, random_search, bayesian_optuna, hyperband_asha, genetic


METHODS = [
    ("grid_search", grid_search),
    ("random_search", random_search),
    ("bayesian_tpe", bayesian_optuna),
    ("hyperband_asha", hyperband_asha),
    ("genetic", genetic),
]


def _strip(result: dict) -> dict:
    """Keluarkan objek study (tidak JSON-serializable) dan kurangi history besar."""
    out = {k: v for k, v in result.items() if k != "study"}
    out["trials"] = [
        {
            "params": t["params"],
            "val_acc": t["val_acc"],
            "val_loss": t["val_loss"],
            "best_epoch": t["best_epoch"],
            "wall_time": t["wall_time"],
            "run_id": t.get("run_id"),
        }
        for t in out["trials"]
    ]
    out["best"] = {
        "params": result["best"]["params"],
        "val_acc": result["best"]["val_acc"],
        "val_loss": result["best"]["val_loss"],
        "best_epoch": result["best"]["best_epoch"],
        "wall_time": result["best"]["wall_time"],
        "run_id": result["best"].get("run_id"),
        "history": result["best"]["history"],
    }
    return out


def main():
    config = load_config(ROOT / "config.yaml")
    set_seed(config.get("seed", 42))

    mlflow.set_tracking_uri(str((ROOT / "mlruns").resolve().as_uri()))
    ensure_dir(ROOT / "results")

    all_results = {}
    grand_t0 = time.time()

    for name, module in METHODS:
        print(f"\n========== {name.upper()} ==========")
        exp_name = f"{config['mlflow']['experiment_prefix']}_{name}"
        t0 = time.time()
        result = module.run(config, exp_name)
        print(f"[{name}] done in {time.time()-t0:.1f}s | best_val_acc={result['best']['val_acc']:.4f}")
        all_results[name] = _strip(result)
        save_json(all_results, ROOT / "results" / "all_hpo_results.json")

    total = time.time() - grand_t0
    print(f"\n=== Semua HPO selesai dalam {total/60:.1f} menit ===")

    # Simpan best configs ringkas
    best_configs = {
        name: {
            "params": r["best"]["params"],
            "val_acc": r["best"]["val_acc"],
            "wall_time": r["best"]["wall_time"],
            "total_time_method": r["total_time"],
            "n_trials": r["n_trials"],
        }
        for name, r in all_results.items()
    }
    save_json(best_configs, ROOT / "results" / "best_configs.json")
    print("Best configs -> results/best_configs.json")


if __name__ == "__main__":
    main()
