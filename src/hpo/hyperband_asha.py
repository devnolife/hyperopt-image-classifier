"""Hyperband / ASHA via Optuna SuccessiveHalvingPruner."""
from __future__ import annotations
import time
from typing import Dict, Any

import optuna

from ..train import train_one_trial
from .random_search import _suggest


def run(config: Dict[str, Any], experiment_name: str) -> Dict[str, Any]:
    n_trials = config["hpo_budgets"]["hyperband_asha"]
    ss = config["search_space"]
    data_cache: dict = {}
    trials_info = []

    def objective(trial: optuna.Trial) -> float:
        params = _suggest(trial, ss)
        idx = trial.number + 1
        print(f"[Hyperband/ASHA] trial {idx}/{n_trials} -> {params}")

        def cb(epoch: int, val_acc: float) -> bool:
            trial.report(val_acc, step=epoch)
            if trial.should_prune():
                print(f"                pruned at epoch {epoch}")
                return True
            return False

        res = train_one_trial(
            params=params,
            config=config,
            trial_name=f"asha_{idx:03d}",
            experiment_name=experiment_name,
            epoch_callback=cb,
            data_cache=data_cache,
            tags={"hpo_method": "hyperband_asha", "trial_number": str(idx)},
        )
        trials_info.append(res)
        print(f"                val_acc={res['val_acc']:.4f} best_ep={res['best_epoch']}")
        # jika pruned via callback, raise agar study menandainya
        return res["val_acc"]

    pruner = optuna.pruners.SuccessiveHalvingPruner(
        min_resource=2, reduction_factor=3, min_early_stopping_rate=0
    )
    sampler = optuna.samplers.TPESampler(seed=config.get("seed", 42), n_startup_trials=5)
    study = optuna.create_study(
        direction="maximize", sampler=sampler, pruner=pruner, study_name="hyperband_asha"
    )
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    total_time = time.time() - t0

    best = max(trials_info, key=lambda r: r["val_acc"])
    return {
        "method": "hyperband_asha",
        "n_trials": len(trials_info),
        "total_time": total_time,
        "best": best,
        "trials": trials_info,
        "study": study,
    }
