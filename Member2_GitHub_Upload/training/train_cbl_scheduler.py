
from pathlib import Path
import copy
import random
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


# ============================================================
# Settings
# ============================================================

SEED = 42

FEATURE_DIR = Path(
    "/content/drive/MyDrive/FYP_Member2_Outputs/FULL_CUB_CLEAN/clip_features"
)

OUTPUT_DIR = Path(
    "/content/drive/MyDrive/FYP_Member2_Outputs/FULL_CUB_CLEAN/cbl_scheduler"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

INPUT_DIM = 512
NUM_CONCEPTS = 312

BATCH_SIZE = 256

INITIAL_LR = 1e-3
WEIGHT_DECAY = 1e-5

MAX_EPOCHS = 400
EARLY_STOP_PATIENCE = 40


# ============================================================
# Reproducibility
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# Load data
# ============================================================

def load_split(split):

    data = torch.load(
        FEATURE_DIR / f"clip_features_{split}.pt",
        map_location="cpu",
        weights_only=False
    )

    features = F.normalize(
        data["features"].float(),
        dim=1
    )

    targets = data["concept_targets"].float()

    return features, targets


X_train, y_train = load_split("train")
X_val, y_val = load_split("val")

assert X_train.shape == (5394, 512)
assert y_train.shape == (5394, 312)

assert X_val.shape == (600, 512)
assert y_val.shape == (600, 312)


print("=" * 65)
print("FULL CUB - LINEAR CBL + LR SCHEDULER")
print("=" * 65)

print("Device:", DEVICE)
print("Train:", X_train.shape)
print("Val:", X_val.shape)
print("Initial LR:", INITIAL_LR)
print("Max epochs:", MAX_EPOCHS)
print("Model: Linear 512 -> 312")
print("Weighting: sqrt")
print("TEST SET NOT USED")
print("=" * 65)


# ============================================================
# sqrt pos_weight
# ============================================================

positive = y_train.sum(dim=0)
negative = len(y_train) - positive

ratio = negative / positive

pos_weight = torch.sqrt(ratio)

pos_weight = torch.clamp(
    pos_weight,
    min=1.0,
    max=5.0
)


criterion = nn.BCEWithLogitsLoss(
    pos_weight=pos_weight.to(DEVICE)
)


# ============================================================
# DataLoader
# ============================================================

generator = torch.Generator()
generator.manual_seed(SEED)

train_loader = DataLoader(
    TensorDataset(
        X_train,
        y_train
    ),
    batch_size=BATCH_SIZE,
    shuffle=True,
    generator=generator
)


# ============================================================
# Model
# ============================================================

model = nn.Linear(
    INPUT_DIM,
    NUM_CONCEPTS
).to(DEVICE)


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=INITIAL_LR,
    weight_decay=WEIGHT_DECAY
)


# Reduce LR when Val Loss stops improving
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=8,
    min_lr=1e-5
)


# ============================================================
# Training
# ============================================================

best_val_loss = float("inf")
best_epoch = 0
best_state = None

no_improvement = 0


for epoch in range(
    1,
    MAX_EPOCHS + 1
):

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model.train()

    total_train_loss = 0


    for features, targets in train_loader:

        features = features.to(DEVICE)
        targets = targets.to(DEVICE)

        optimizer.zero_grad()

        logits = model(features)

        loss = criterion(
            logits,
            targets
        )

        loss.backward()

        optimizer.step()


        total_train_loss += (
            loss.item()
            * len(features)
        )


    train_loss = (
        total_train_loss
        / len(X_train)
    )


    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    model.eval()

    with torch.no_grad():

        val_logits = model(
            X_val.to(DEVICE)
        )

        val_loss = criterion(
            val_logits,
            y_val.to(DEVICE)
        ).item()


    # Scheduler looks only at validation loss
    scheduler.step(val_loss)


    current_lr = (
        optimizer.param_groups[0]["lr"]
    )


    # --------------------------------------------------------
    # Save best epoch
    # --------------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss
        best_epoch = epoch

        best_state = copy.deepcopy(
            model.state_dict()
        )

        no_improvement = 0

    else:

        no_improvement += 1


    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    if (
        epoch == 1
        or epoch % 10 == 0
    ):

        print(
            f"Epoch {epoch:03d} | "
            f"Train={train_loss:.4f} | "
            f"Val={val_loss:.4f} | "
            f"LR={current_lr:.6f}"
        )


    # --------------------------------------------------------
    # Early stopping
    # --------------------------------------------------------

    if (
        no_improvement
        >= EARLY_STOP_PATIENCE
    ):

        print(
            f"\nEarly stopping "
            f"at epoch {epoch}"
        )

        break


# ============================================================
# Restore best model
# ============================================================

model.load_state_dict(
    best_state
)

model.eval()


with torch.no_grad():

    val_logits = model(
        X_val.to(DEVICE)
    ).cpu()


val_probs = torch.sigmoid(
    val_logits
)


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(
    probs,
    targets,
    threshold
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
        / (
            2 * tp
            + fp
            + fn
            + 1e-8
        )
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


    supported = (
        targets.sum(dim=0) > 0
    )


    macro_supported = (
        concept_f1[supported]
        .mean()
        .item()
    )


    return {
        "precision": precision,
        "recall": recall,
        "micro_f1": micro_f1,
        "macro_f1_all": macro_all,
        "macro_f1_supported": macro_supported
    }


# ============================================================
# Threshold search on Validation
# ============================================================

best_threshold = None
best_metrics = None
best_micro = -1


for threshold in np.arange(
    0.10,
    0.71,
    0.01
):

    threshold = float(threshold)

    metrics = calculate_metrics(
        val_probs,
        y_val,
        threshold
    )


    if (
        metrics["micro_f1"]
        > best_micro
    ):

        best_micro = (
            metrics["micro_f1"]
        )

        best_threshold = threshold

        best_metrics = metrics


# ============================================================
# Results
# ============================================================

print("\n")
print("=" * 65)
print("FINAL VALIDATION RESULT")
print("=" * 65)

print("Best epoch:", best_epoch)

print(
    "Best Val Loss:",
    best_val_loss
)

print(
    "Best threshold:",
    round(best_threshold, 2)
)

print(
    "Precision:",
    round(
        best_metrics["precision"],
        4
    )
)

print(
    "Recall:",
    round(
        best_metrics["recall"],
        4
    )
)

print(
    "Micro-F1:",
    round(
        best_metrics["micro_f1"],
        4
    )
)

print(
    "Macro-F1 all:",
    round(
        best_metrics["macro_f1_all"],
        4
    )
)

print(
    "Macro-F1 supported:",
    round(
        best_metrics[
            "macro_f1_supported"
        ],
        4
    )
)


# ============================================================
# Save
# ============================================================

save_path = (
    OUTPUT_DIR
    / "concept_layer_scheduler.pt"
)


torch.save(
    {
        "model_state_dict":
            best_state,

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

        "metrics":
            best_metrics,

        "initial_lr":
            INITIAL_LR,

        "batch_size":
            BATCH_SIZE,

        "weighting":
            "sqrt",

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

        "test_used":
            False
    },

    save_path
)


result_json = {
    "best_epoch":
        best_epoch,

    "best_val_loss":
        best_val_loss,

    "best_threshold":
        best_threshold,

    **best_metrics
}


with open(
    OUTPUT_DIR / "scheduler_result.json",
    "w"
) as f:

    json.dump(
        result_json,
        f,
        indent=4
    )


print("\nSaved:")
print(save_path)

print("\n")
print("=" * 65)
print("FINISHED")
print("TEST SET HAS NOT BEEN USED")
print("=" * 65)
