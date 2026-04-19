"""Grid Search HPO. Menggunakan subset 'grid' dari config search_space."""
from __future__ import annotations
import itertools
import time
from typing import Dict, Any, List

from ..train import train_one_trial


def run(config: Dict[str, Any], experiment_name: str) -> Dict[str, Any]:
    ss = config["search_space"]
    grid = {k: v["grid"] for k, v in ss.items()}
    keys = list(grid.keys())
    combos: List[dict] = []
    for combo in itertools.product(*[grid[k] for k in keys]):
        combos.append({k: v for k, v in zip(keys, combo)})

    print(f"[GridSearch] total {len(combos)} kombinasi")
    trials = []
    data_cache: dict = {}
    t0 = time.time()
    for i, params in enumerate(combos, 1):
        print(f"[GridSearch] trial {i}/{len(combos)} -> {params}")
        res = train_one_trial(
            params=params,
            config=config,
            trial_name=f"grid_{i:03d}",
            experiment_name=experiment_name,
            data_cache=data_cache,
            tags={"hpo_method": "grid_search", "trial_number": str(i)},
        )
        trials.append(res)
        print(f"           val_acc={res['val_acc']:.4f} time={res['wall_time']:.1f}s")

    best = max(trials, key=lambda r: r["val_acc"])
    return {
        "method": "grid_search",
        "n_trials": len(trials),
        "total_time": time.time() - t0,
        "best": best,
        "trials": trials,
    }
