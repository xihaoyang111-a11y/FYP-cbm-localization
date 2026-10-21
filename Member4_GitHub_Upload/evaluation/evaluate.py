import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


def compute_f1(y_true, y_pred, num_classes):
    scores = []
    for i in range(num_classes):
        tp = np.sum((y_true == i) & (y_pred == i))
        fp = np.sum((y_true != i) & (y_pred == i))
        fn = np.sum((y_true == i) & (y_pred != i))
        denom = 2 * tp + fp + fn
        scores.append(2 * tp / denom if denom else 0.0)
    return float(np.mean(scores))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="concept_scores/concept_predictions_test.pt"
    )
    parser.add_argument(
        "--checkpoint",
        default="evaluation/results/classifier_best.pt"
    )
    args = parser.parse_args()

    # Load test data
    data = torch.load(
        args.data, map_location="cpu", weights_only=True
    )

    x = torch.as_tensor(
        data["concept_probs"], dtype=torch.float32
    )
    y = torch.as_tensor(
        data["class_labels"], dtype=torch.long
    )

    # Load trained classifier
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=True
    )

    model = nn.Linear(
        checkpoint["num_concepts"],
        checkpoint["num_classes"]
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # Species predictions
    with torch.no_grad():
        logits = model(x)
        predictions = logits.argmax(dim=1)
        top5 = logits.topk(5, dim=1).indices

    top1_accuracy = (predictions == y).float().mean().item()
    top5_accuracy = (top5 == y[:, None]).any(dim=1).float().mean().item()

    class_macro_f1 = compute_f1(
        y.numpy(), predictions.numpy(), 200
    )

    # Concept prediction evaluation
    concept_true = torch.as_tensor(
        data["concept_targets"]
    ).numpy().astype(bool)

    if "concept_predictions" in data:
        concept_pred = torch.as_tensor(
            data["concept_predictions"]
        ).numpy().astype(bool)
    else:
        threshold = float(data["threshold"])
        concept_pred = (x.numpy() >= threshold)

    if concept_true.shape != concept_pred.shape:
        raise ValueError("Concept prediction/target shapes do not match")

    tp = np.logical_and(concept_pred, concept_true).sum()
    fp = np.logical_and(concept_pred, ~concept_true).sum()
    fn = np.logical_and(~concept_pred, concept_true).sum()

    concept_micro_f1 = (
        2 * tp / (2 * tp + fp + fn)
        if (2 * tp + fp + fn) else 0.0
    )

    concept_f1_each = []
    for j in range(concept_true.shape[1]):
        a = concept_true[:, j]
        b = concept_pred[:, j]
        t = np.logical_and(a, b).sum()
        f = np.logical_and(~a, b).sum()
        n = np.logical_and(a, ~b).sum()
        d = 2 * t + f + n
        concept_f1_each.append(2 * t / d if d else 0.0)

    concept_macro_f1 = float(np.mean(concept_f1_each))

    # Confusion matrix: rows = actual, columns = predicted
    confusion = np.zeros((200, 200), dtype=int)
    np.add.at(
        confusion, (y.numpy(), predictions.numpy()), 1
    )

    results = {
        "best_epoch": int(checkpoint["best_epoch"]),
        "best_val_accuracy": float(checkpoint["best_val_accuracy"]),
        "test_samples": len(y),
        "top1_accuracy": top1_accuracy,
        "top5_accuracy": top5_accuracy,
        "classification_macro_f1": class_macro_f1,
        "concept_micro_f1": float(concept_micro_f1),
        "concept_macro_f1": concept_macro_f1
    }

    output_dir = Path("evaluation/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "test_metrics.json").write_text(
        json.dumps(results, indent=2)
    )
    np.save(output_dir / "confusion_matrix.npy", confusion)

    print("\n===== Member 4: Test Evaluation =====")
    print(f"Test Samples: {len(y)}")
    print(f"Top-1 Accuracy: {top1_accuracy:.2%}")
    print(f"Top-5 Accuracy: {top5_accuracy:.2%}")
    print(f"Classification Macro-F1: {class_macro_f1:.4f}")
    print(f"Concept Micro-F1: {concept_micro_f1:.4f}")
    print(f"Concept Macro-F1: {concept_macro_f1:.4f}")
    print("Results saved to evaluation/results/")


if __name__ == "__main__":
    main()
