"""Final training menggunakan best config dari metode HPO terbaik."""
from __future__ import annotations
import sys
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import mlflow
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

from src.utils import load_config, set_seed, get_device, ensure_dir, save_json
from src.dataset import get_dataloaders
from src.model import CustomCNN, build_optimizer
from src.train import evaluate


def main():
    config = load_config(ROOT / "config.yaml")
    set_seed(config.get("seed", 42))
    device = get_device()

    best_path = ROOT / "results" / "best_configs.json"
    with open(best_path, "r", encoding="utf-8") as f:
        best_configs = json.load(f)

    # pilih metode dengan val_acc tertinggi
    winner = max(best_configs.items(), key=lambda kv: kv[1]["val_acc"])
    method_name, info = winner
    params = info["params"]
    print(f"Winner: {method_name} | val_acc={info['val_acc']:.4f}")
    print(f"Params: {params}")

    epochs = config["training"]["final_epochs"]
    train_loader, val_loader, test_loader, classes = get_dataloaders(
        root=config["data"]["root"],
        batch_size=int(params["batch_size"]),
        val_size=config["data"]["val_size"],
        num_workers=config["data"].get("num_workers", 2),
        seed=config.get("seed", 42),
    )

    model = CustomCNN(
        num_classes=config["data"]["num_classes"],
        base_filters=int(params["base_filters"]),
        dropout=float(params["dropout"]),
    ).to(device)
    optimizer = build_optimizer(params["optimizer"], model.parameters(), lr=float(params["learning_rate"]))
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    mlflow.set_tracking_uri(str((ROOT / "mlruns").resolve().as_uri()))
    mlflow.set_experiment("HPO_final_training")
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    with mlflow.start_run(run_name=f"final_from_{method_name}") as run:
        mlflow.log_params(params)
        mlflow.log_param("epochs", epochs)
        mlflow.set_tag("winner_method", method_name)

        best_val, best_state = -1.0, None
        t0 = time.time()
        for epoch in range(1, epochs + 1):
            model.train()
            total, correct, loss_sum = 0, 0, 0.0
            for x, y in train_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item() * x.size(0)
                correct += (out.argmax(1) == y).sum().item()
                total += x.size(0)
            tl, ta = loss_sum / total, correct / total
            vl, va = evaluate(model, val_loader, device, criterion)
            scheduler.step()

            history["train_loss"].append(tl); history["train_acc"].append(ta)
            history["val_loss"].append(vl); history["val_acc"].append(va)
            mlflow.log_metrics({"train_loss": tl, "train_acc": ta, "val_loss": vl, "val_acc": va}, step=epoch)
            print(f"[ep {epoch:02d}/{epochs}] train_acc={ta:.4f} val_acc={va:.4f}")

            if va > best_val:
                best_val = va
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        # load best
        if best_state is not None:
            model.load_state_dict(best_state)

        # Test set
        test_loss, test_acc = evaluate(model, test_loader, device, criterion)
        print(f"\nTest accuracy = {test_acc:.4f}")
        mlflow.log_metric("test_acc", test_acc)
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("wall_time_sec", time.time() - t0)

        # Confusion matrix
        model.eval()
        all_pred, all_true = [], []
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device, non_blocking=True)
                pred = model(x).argmax(1).cpu().numpy()
                all_pred.append(pred); all_true.append(y.numpy())
        y_pred = np.concatenate(all_pred); y_true = np.concatenate(all_true)
        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)

        fig_dir = ensure_dir(ROOT / "results" / "figures")
        plt.figure(figsize=(8, 6.5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
        plt.xlabel("Prediksi"); plt.ylabel("Aktual")
        plt.title(f"Confusion Matrix — Test Acc={test_acc:.4f}")
        plt.tight_layout()
        cm_path = fig_dir / "final_confusion_matrix.png"
        plt.savefig(cm_path, dpi=140); plt.close()
        mlflow.log_artifact(str(cm_path))

        # Learning curves
        ep_axis = range(1, len(history["train_acc"]) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        axes[0].plot(ep_axis, history["train_loss"], label="train")
        axes[0].plot(ep_axis, history["val_loss"], label="val")
        axes[0].set_title("Loss"); axes[0].set_xlabel("epoch"); axes[0].legend(); axes[0].grid(alpha=0.3)
        axes[1].plot(ep_axis, history["train_acc"], label="train")
        axes[1].plot(ep_axis, history["val_acc"], label="val")
        axes[1].set_title("Accuracy"); axes[1].set_xlabel("epoch"); axes[1].legend(); axes[1].grid(alpha=0.3)
        plt.tight_layout()
        curves_path = fig_dir / "final_learning_curves.png"
        plt.savefig(curves_path, dpi=140); plt.close()
        mlflow.log_artifact(str(curves_path))

        # Save
        final_info = {
            "winner_method": method_name,
            "params": params,
            "test_acc": float(test_acc),
            "test_loss": float(test_loss),
            "best_val_acc": float(best_val),
            "classification_report": report,
            "epochs": epochs,
            "run_id": run.info.run_id,
            "history": history,
        }
        save_json(final_info, ROOT / "results" / "final_training.json")
        torch.save(model.state_dict(), ROOT / "results" / "final_model.pth")
        print("Saved final_model.pth & final_training.json")


if __name__ == "__main__":
    main()
