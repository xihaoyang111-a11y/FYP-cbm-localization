"""Member 4: train a 312-concept -> 200-species linear CBM classifier.

Example (run from project root):
    python evaluation/train_classifier.py --data-dir concept_scores --epochs 100
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def load_split(path):
    data = torch.load(path, map_location="cpu", weights_only=True)
    required = {"concept_probs", "class_labels"}
    if not required.issubset(data):
        raise KeyError(f"Missing keys in {path}: {required - data.keys()}")
    x = torch.as_tensor(data["concept_probs"], dtype=torch.float32)
    y = torch.as_tensor(data["class_labels"], dtype=torch.long)
    if x.ndim != 2 or x.shape[1] != 312 or y.ndim != 1 or len(x) != len(y):
        raise ValueError(f"Unexpected shapes at {path}: {x.shape}, {y.shape}")
    if not torch.isfinite(x).all() or not ((y >= 0) & (y < 200)).all():
        raise ValueError(f"Nonfinite concepts or invalid class labels in {path}")
    return x, y


def accuracy(model, x, y, device):
    model.eval()
    with torch.no_grad():
        logits = model(x.to(device))
        return (logits.argmax(dim=1) == y.to(device)).float().mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("concept_scores"))
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/results"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        parser.error("epochs, batch-size and patience must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    x_train, y_train = load_split(args.data_dir / "concept_predictions_train.pt")
    x_val, y_val = load_split(args.data_dir / "concept_predictions_val.pt")
    print(f"Device: {device} | Train: {tuple(x_train.shape)} | Val: {tuple(x_val.shape)}")

    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True)
    model = nn.Linear(312, 200).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_acc, best_epoch, wait = -1.0, 0, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_total = 0.0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(bx), by)
            loss.backward()
            optimizer.step()
            loss_total += loss.item() * len(bx)
        val_acc = accuracy(model, x_val, y_val, device)
        train_loss = loss_total / len(x_train)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_accuracy": val_acc})
        print(f"Epoch {epoch:03d} | train loss {train_loss:.4f} | val accuracy {val_acc:.4%}")
        if val_acc > best_acc + 1e-9:
            best_acc, best_epoch, wait = val_acc, epoch, 0
            torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "num_concepts": 312, "num_classes": 200,
                        "best_epoch": best_epoch, "best_val_accuracy": best_acc,
                        "input_key": "concept_probs", "seed": args.seed},
                       args.output_dir / "classifier_best.pt")
        else:
            wait += 1
            if wait >= args.patience:
                print("Early stopping")
                break
    (args.output_dir / "training_history.json").write_text(json.dumps(history, indent=2))
    print(f"Best epoch: {best_epoch} | best val accuracy: {best_acc:.4%}")
    print(f"Saved: {args.output_dir / 'classifier_best.pt'}")


if __name__ == "__main__":
    main()
