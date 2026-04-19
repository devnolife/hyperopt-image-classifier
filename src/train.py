"""Training / evaluation loop with MLflow logging.

Mendukung callback per-epoch untuk Optuna pruner.
"""
from __future__ import annotations
import time
import math
from typing import Callable, Optional, Dict, Any

import mlflow
import torch
import torch.nn as nn

from .dataset import get_dataloaders
from .model import CustomCNN, build_optimizer
from .utils import get_device, set_seed


@torch.no_grad()
def evaluate(model: nn.Module, loader, device, criterion) -> tuple[float, float]:
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out = model(x)
        loss = criterion(out, y)
        loss_sum += loss.item() * x.size(0)
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    return loss_sum / total, correct / total


def train_one_trial(
    params: Dict[str, Any],
    config: Dict[str, Any],
    trial_name: str,
    experiment_name: str,
    epoch_callback: Optional[Callable[[int, float], bool]] = None,
    data_cache: Optional[dict] = None,
    epochs: Optional[int] = None,
    tags: Optional[Dict[str, str]] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Train satu konfigurasi hyperparameter, log ke MLflow.

    Returns dict: {val_acc, val_loss, best_epoch, wall_time, history, params}.
    ``epoch_callback(epoch, val_acc)`` -> True berarti minta prune/stop lebih awal.
    """
    set_seed(config.get("seed", 42))
    device = get_device()

    epochs = epochs or config["training"]["epochs_per_trial"]
    patience = config["training"].get("early_stopping_patience", 5)

    # Data loader (cached per batch size untuk efisiensi)
    bs = int(params["batch_size"])
    if data_cache is not None and bs in data_cache:
        train_loader, val_loader, _ = data_cache[bs]
    else:
        train_loader, val_loader, test_loader, _classes = get_dataloaders(
            root=config["data"]["root"],
            batch_size=bs,
            val_size=config["data"]["val_size"],
            num_workers=config["data"].get("num_workers", 0),
            seed=config.get("seed", 42),
            train_subset=config["data"].get("train_subset"),
        )
        if data_cache is not None:
            data_cache[bs] = (train_loader, val_loader, test_loader)

    model = CustomCNN(
        num_classes=config["data"]["num_classes"],
        base_filters=int(params["base_filters"]),
        dropout=float(params["dropout"]),
    ).to(device)

    optimizer = build_optimizer(params["optimizer"], model.parameters(), lr=float(params["learning_rate"]))
    criterion = nn.CrossEntropyLoss()

    mlflow.set_experiment(experiment_name)
    tags = tags or {}
    with mlflow.start_run(run_name=trial_name) as run:
        mlflow.log_params({k: v for k, v in params.items()})
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("device", str(device))
        for k, v in tags.items():
            mlflow.set_tag(k, v)

        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        best_val_acc = -math.inf
        best_epoch = -1
        epochs_no_improve = 0
        t0 = time.time()

        for epoch in range(1, epochs + 1):
            model.train()
            total, correct, loss_sum = 0, 0, 0.0
            for x, y in train_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item() * x.size(0)
                pred = out.argmax(1)
                correct += (pred == y).sum().item()
                total += x.size(0)
            train_loss = loss_sum / total
            train_acc = correct / total

            val_loss, val_acc = evaluate(model, val_loader, device, criterion)

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                },
                step=epoch,
            )

            if verbose:
                print(f"  [ep {epoch}/{epochs}] train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

            improved = val_acc > best_val_acc
            if improved:
                best_val_acc = val_acc
                best_epoch = epoch
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            # pruner callback
            if epoch_callback is not None:
                prune = epoch_callback(epoch, val_acc)
                if prune:
                    mlflow.set_tag("pruned", "true")
                    break

            if epochs_no_improve >= patience:
                mlflow.set_tag("early_stopped", "true")
                break

        wall_time = time.time() - t0
        mlflow.log_metric("best_val_acc", best_val_acc)
        mlflow.log_metric("wall_time_sec", wall_time)
        mlflow.log_metric("best_epoch", best_epoch)

        result = {
            "val_acc": best_val_acc,
            "val_loss": history["val_loss"][best_epoch - 1] if best_epoch > 0 else float("inf"),
            "best_epoch": best_epoch,
            "wall_time": wall_time,
            "history": history,
            "params": dict(params),
            "run_id": run.info.run_id,
        }
    return result
