
from pathlib import Path
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# ==================================================
# Reproducibility
# ==================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ==================================================
# Paths
# ==================================================
FEATURE_DIR = Path(
    "/content/drive/MyDrive/FYP_Member2_Outputs/FULL_CUB_CLEAN/clip_features"
)

OUTPUT_DIR = Path(
    "/content/drive/MyDrive/FYP_Member2_Outputs/FULL_CUB_CLEAN/cbl"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==================================================
# Settings
# ==================================================
INPUT_DIM = 512
NUM_CONCEPTS = 312

BATCH_SIZE = 256
EPOCHS = 200
LR = 1e-3
WEIGHT_DECAY = 1e-5
PATIENCE = 20

print("========================================")
print("FULL CUB CLEAN CBL")
print("TRAIN + VAL ONLY")
print("========================================")
print("Device:", DEVICE)

# ==================================================
# Load Train / Val only
# ==================================================
def load_split(split):
    data = torch.load(
        FEATURE_DIR / f"clip_features_{split}.pt",
        map_location="cpu",
        weights_only=False,
    )

    data["features"] = F.normalize(
        data["features"].float(),
        dim=1,
    )

    data["concept_targets"] = (
        data["concept_targets"].float()
    )

    return data


train_data = load_split("train")
val_data = load_split("val")

X_train = train_data["features"]
y_train = train_data["concept_targets"]

X_val = val_data["features"]
y_val = val_data["concept_targets"]

assert X_train.shape == (5394, 512)
assert y_train.shape == (5394, 312)
assert X_val.shape == (600, 512)
assert y_val.shape == (600, 312)

print("Train:", X_train.shape, y_train.shape)
print("Val:", X_val.shape, y_val.shape)

# ==================================================
# Mild imbalance weighting
# ==================================================
positive = y_train.sum(dim=0)
negative = len(y_train) - positive

ratio = negative / positive

# Much milder than neg/pos
pos_weight = torch.sqrt(ratio)

# Avoid huge weights for extremely rare concepts
pos_weight = torch.clamp(
    pos_weight,
    min=1.0,
    max=5.0,
)

print(
    "pos_weight range:",
    pos_weight.min().item(),
    pos_weight.max().item(),
)

# ==================================================
# Data loaders
# ==================================================
train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=BATCH_SIZE,
    shuffle=True,
)

val_loader = DataLoader(
    TensorDataset(X_val, y_val),
    batch_size=BATCH_SIZE,
    shuffle=False,
)

# ==================================================
# Baseline Concept Prediction Layer
# 512 -> 312
# ==================================================
model = nn.Linear(
    INPUT_DIM,
    NUM_CONCEPTS,
).to(DEVICE)

criterion = nn.BCEWithLogitsLoss(
    pos_weight=pos_weight.to(DEVICE)
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY,
)

# ==================================================
# Validation loss
# ==================================================
def evaluate_loss(loader):

    model.eval()
    total_loss = 0

    with torch.no_grad():

        for features, targets in loader:

            features = features.to(DEVICE)
            targets = targets.to(DEVICE)

            logits = model(features)

            loss = criterion(
                logits,
                targets,
            )

            total_loss += (
                loss.item()
                * len(features)
            )

    return (
        total_loss
        / len(loader.dataset)
    )

# ==================================================
# Train
# ==================================================
best_val_loss = float("inf")
best_epoch = 0
best_state = None
no_improvement = 0

print("\nStarting training...\n")

for epoch in range(1, EPOCHS + 1):

    model.train()

    total_loss = 0

    for features, targets in train_loader:

        features = features.to(DEVICE)
        targets = targets.to(DEVICE)

        optimizer.zero_grad()

        logits = model(features)

        loss = criterion(
            logits,
            targets,
        )

        loss.backward()
        optimizer.step()

        total_loss += (
            loss.item()
            * len(features)
        )

    train_loss = (
        total_loss
        / len(train_loader.dataset)
    )

    val_loss = evaluate_loss(
        val_loader
    )

    print(
        f"Epoch {epoch:03d} | "
        f"Train Loss={train_loss:.4f} | "
        f"Val Loss={val_loss:.4f}"
    )

    if val_loss < best_val_loss:

        best_val_loss = val_loss
        best_epoch = epoch

        best_state = copy.deepcopy(
            model.state_dict()
        )

        no_improvement = 0

    else:

        no_improvement += 1

    if no_improvement >= PATIENCE:

        print("\nEarly stopping.")
        break

# ==================================================
# Restore best model
# ==================================================
model.load_state_dict(
    best_state
)

print("\nBest epoch:", best_epoch)
print("Best Val Loss:", best_val_loss)

# ==================================================
# Generate Val logits
# ==================================================
model.eval()

with torch.no_grad():

    val_logits = model(
        X_val.to(DEVICE)
    ).cpu()

val_probs = torch.sigmoid(
    val_logits
)

# ==================================================
# Metrics
# ==================================================
def calculate_metrics(
    probs,
    targets,
    threshold,
):

    preds = (
        probs >= threshold
    ).float()

    tp = (
        preds * targets
    ).sum()

    fp = (
        preds * (1 - targets)
    ).sum()

    fn = (
        (1 - preds) * targets
    ).sum()

    precision = (
        tp / (tp + fp + 1e-8)
    ).item()

    recall = (
        tp / (tp + fn + 1e-8)
    ).item()

    micro_f1 = (
        2 * tp
        / (2 * tp + fp + fn + 1e-8)
    ).item()

    # Per-concept F1
    tp_c = (
        preds * targets
    ).sum(dim=0)

    fp_c = (
        preds * (1 - targets)
    ).sum(dim=0)

    fn_c = (
        (1 - preds) * targets
    ).sum(dim=0)

    concept_f1 = (
        2 * tp_c
        / (
            2 * tp_c
            + fp_c
            + fn_c
            + 1e-8
        )
    )

    macro_all = (
        concept_f1.mean().item()
    )

    # Only concepts with positives in Val
    support = (
        targets.sum(dim=0) > 0
    )

    macro_supported = (
        concept_f1[support]
        .mean()
        .item()
    )

    return (
        precision,
        recall,
        micro_f1,
        macro_all,
        macro_supported,
    )

# ==================================================
# Tune threshold on Validation only
# ==================================================
best_threshold = None
best_micro = -1

print("\nValidation threshold search:")

for threshold in np.arange(
    0.10,
    0.71,
    0.01,
):

    (
        precision,
        recall,
        micro,
        macro_all,
        macro_supported,
    ) = calculate_metrics(
        val_probs,
        y_val,
        float(threshold),
    )

    if micro > best_micro:

        best_micro = micro
        best_threshold = float(
            threshold
        )

print(
    "\nBest threshold:",
    round(best_threshold, 2)
)

# Final validation metrics
(
    precision,
    recall,
    micro,
    macro_all,
    macro_supported,
) = calculate_metrics(
    val_probs,
    y_val,
    best_threshold,
)

print("\n========================================")
print("FINAL VALIDATION RESULTS")
print("========================================")

print(
    f"Precision: {precision:.4f}"
)
print(
    f"Recall: {recall:.4f}"
)
print(
    f"Micro-F1: {micro:.4f}"
)
print(
    f"Macro-F1 (all 312): "
    f"{macro_all:.4f}"
)
print(
    f"Macro-F1 "
    f"(concepts with Val support): "
    f"{macro_supported:.4f}"
)

# ==================================================
# Save model
# ==================================================
save_path = (
    OUTPUT_DIR
    / "concept_layer_best.pt"
)

torch.save(
    {
        "model_state_dict":
            model.state_dict(),

        "input_dim":
            INPUT_DIM,

        "num_concepts":
            NUM_CONCEPTS,

        "best_epoch":
            best_epoch,

        "best_val_loss":
            best_val_loss,

        "best_threshold":
            best_threshold,

        "val_precision":
            precision,

        "val_recall":
            recall,

        "val_micro_f1":
            micro,

        "val_macro_f1_all":
            macro_all,

        "val_macro_f1_supported":
            macro_supported,

        "pos_weight":
            pos_weight,

        "clip_model":
            "ViT-B/32",

        "feature_normalization":
            "L2",

        "seed":
            SEED,

        "dataset":
            "CUB-200-2011",
    },

    save_path,
)

print("\nSaved:")
print(save_path)

print("\n========================================")
print("TRAINING COMPLETE")
print("TEST SET HAS NOT BEEN USED")
print("========================================")
