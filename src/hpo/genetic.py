"""Genetic Algorithm HPO menggunakan DEAP."""
from __future__ import annotations
import random
import time
from typing import Dict, Any, List

import numpy as np
from deap import base, creator, tools

from ..train import train_one_trial


# Hindari error duplicate creator saat dijalankan berulang
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMax)


def _decode(ind: List[float], ss: Dict[str, Any]) -> Dict[str, Any]:
    """ind = [lr_log, bs_idx, opt_idx, dropout, bf_idx] in [0,1] each for continuity."""
    lr_low = np.log10(ss["learning_rate"]["low"])
    lr_high = np.log10(ss["learning_rate"]["high"])
    lr = 10 ** (lr_low + ind[0] * (lr_high - lr_low))
    bs_choices = ss["batch_size"]["choices"]
    bs = bs_choices[int(ind[1] * len(bs_choices)) % len(bs_choices)]
    opt_choices = ss["optimizer"]["choices"]
    opt = opt_choices[int(ind[2] * len(opt_choices)) % len(opt_choices)]
    dropout = ss["dropout"]["low"] + ind[3] * (ss["dropout"]["high"] - ss["dropout"]["low"])
    bf_choices = ss["base_filters"]["choices"]
    bf = bf_choices[int(ind[4] * len(bf_choices)) % len(bf_choices)]
    return {
        "learning_rate": float(lr),
        "batch_size": int(bs),
        "optimizer": opt,
        "dropout": float(dropout),
        "base_filters": int(bf),
    }


def run(config: Dict[str, Any], experiment_name: str) -> Dict[str, Any]:
    ss = config["search_space"]
    ga_cfg = config["hpo_budgets"]["genetic"]
    pop_size = ga_cfg["population"]
    n_gen = ga_cfg["generations"]
    seed = config.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)

    data_cache: dict = {}
    trials_info: List[dict] = []
    counter = {"n": 0}

    def evaluate(ind):
        params = _decode(ind, ss)
        counter["n"] += 1
        idx = counter["n"]
        print(f"[GA] eval {idx} -> {params}")
        res = train_one_trial(
            params=params,
            config=config,
            trial_name=f"ga_{idx:03d}",
            experiment_name=experiment_name,
            data_cache=data_cache,
            tags={"hpo_method": "genetic", "trial_number": str(idx)},
        )
        trials_info.append(res)
        print(f"       val_acc={res['val_acc']:.4f}")
        return (res["val_acc"],)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.random)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=5)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxBlend, alpha=0.3)
    toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.2, indpb=0.3)
    toolbox.register("select", tools.selTournament, tournsize=3)

    t0 = time.time()
    pop = toolbox.population(n=pop_size)

    # Evaluasi populasi awal
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)

    for gen in range(n_gen):
        print(f"[GA] === Generation {gen+1}/{n_gen} ===")
        offspring = toolbox.select(pop, len(pop))
        offspring = [toolbox.clone(o) for o in offspring]

        # crossover
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.7:
                toolbox.mate(c1, c2)
                del c1.fitness.values
                del c2.fitness.values

        # mutation + clamp to [0,1]
        for m in offspring:
            if random.random() < 0.3:
                toolbox.mutate(m)
                for i in range(len(m)):
                    m[i] = min(1.0, max(0.0, m[i]))
                del m.fitness.values

        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid:
            ind.fitness.values = toolbox.evaluate(ind)

        pop[:] = offspring

    total_time = time.time() - t0
    best = max(trials_info, key=lambda r: r["val_acc"])
    return {
        "method": "genetic",
        "n_trials": len(trials_info),
        "total_time": total_time,
        "best": best,
        "trials": trials_info,
    }
