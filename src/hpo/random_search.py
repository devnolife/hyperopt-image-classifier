"""Random Search via Optuna RandomSampler."""
from __future__ import annotations
import time
from typing import Dict, Any

import optuna

from ..train import train_one_trial


def _suggest(trial: optuna.Trial, ss: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "learning_rate": trial.suggest_float("learning_rate", ss["learning_rate"]["low"], ss["learning_rate"]["high"], log=True),
        "batch_size": trial.suggest_categorical("batch_size", ss["batch_size"]["choices"]),
        "optimizer": trial.suggest_categorical("optimizer", ss["optimizer"]["choices"]),
        "dropout": trial.suggest_float("dropout", ss["dropout"]["low"], ss["dropout"]["high"]),
        "base_filters": trial.suggest_categorical("base_filters", ss["base_filters"]["choices"]),
    }


def run(config: Dict[str, Any], experiment_name: str) -> Dict[str, Any]:
    n_trials = config["hpo_budgets"]["random_search"]
    ss = config["search_space"]
    data_cache: dict = {}
    trials_info = []

    def objective(trial: optuna.Trial) -> float:
        params = _suggest(trial, ss)
        idx = trial.number + 1
        print(f"[RandomSearch] trial {idx}/{n_trials} -> {params}")
        res = train_one_trial(
            params=params,
            config=config,
            trial_name=f"random_{idx:03d}",
            experiment_name=experiment_name,
            data_cache=data_cache,
            tags={"hpo_method": "random_search", "trial_number": str(idx)},
        )
        trials_info.append(res)
        print(f"              val_acc={res['val_acc']:.4f}")
        return res["val_acc"]

    sampler = optuna.samplers.RandomSampler(seed=config.get("seed", 42))
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name="random_search")
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    total_time = time.time() - t0

    best = max(trials_info, key=lambda r: r["val_acc"])
    return {
        "method": "random_search",
        "n_trials": len(trials_info),
        "total_time": total_time,
        "best": best,
        "trials": trials_info,
        "study": study,
    }
