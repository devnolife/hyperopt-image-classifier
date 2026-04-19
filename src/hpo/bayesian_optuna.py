"""Bayesian Optimization via Optuna TPE sampler."""
from __future__ import annotations
import time
from typing import Dict, Any

import optuna

from ..train import train_one_trial
from .random_search import _suggest


def run(config: Dict[str, Any], experiment_name: str) -> Dict[str, Any]:
    n_trials = config["hpo_budgets"]["bayesian_tpe"]
    ss = config["search_space"]
    data_cache: dict = {}
    trials_info = []

    def objective(trial: optuna.Trial) -> float:
        params = _suggest(trial, ss)
        idx = trial.number + 1
        print(f"[BayesianTPE] trial {idx}/{n_trials} -> {params}")
        res = train_one_trial(
            params=params,
            config=config,
            trial_name=f"bayes_{idx:03d}",
            experiment_name=experiment_name,
            data_cache=data_cache,
            tags={"hpo_method": "bayesian_tpe", "trial_number": str(idx)},
        )
        trials_info.append(res)
        print(f"             val_acc={res['val_acc']:.4f}")
        return res["val_acc"]

    sampler = optuna.samplers.TPESampler(seed=config.get("seed", 42), n_startup_trials=5)
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name="bayesian_tpe")
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    total_time = time.time() - t0

    best = max(trials_info, key=lambda r: r["val_acc"])
    return {
        "method": "bayesian_tpe",
        "n_trials": len(trials_info),
        "total_time": total_time,
        "best": best,
        "trials": trials_info,
        "study": study,
    }
