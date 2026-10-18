
from pathlib import Path
import copy
import json
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


# ============================================================
# 1. Basic settings
# ============================================================

SEED = 42

FEATURE_DIR = Path(
    "/content/drive/MyDrive/FYP_Member2_Outputs/FULL_CUB_CLEAN/clip_features"
)

OUTPUT_DIR = Path(
    "/content/drive/MyDrive/FYP_Member2_Outputs/FULL_CUB_CLEAN/cbl_tuning"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

INPUT_DIM = 512
NUM_CONCEPTS = 312

BATCH_SIZE = 256
MAX_EPOCHS = 200
PATIENCE = 20
WEIGHT_DECAY = 1e-5


# ============================================================
# 2. Hyperparameters to compare
# ============================================================

LEARNING_RATES = [
    1e-4,
    5e-4,
    1e-3,
]

WEIGHTING_MODES = [
    "none",
    "sqrt",
]


# ============================================================
# 3. Random seed
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 4. Load CLIP features
# ============================================================

def load_split(split):

    path = FEATURE_DIR / f"clip_features_{split}.pt"

    data = torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )

    features = data["features"].float()

    # L2 normalize CLIP features
    features = F.normalize(
        features,
        dim=1
    )

    targets = data["concept_targets"].float()

    return features, targets


X_train, y_train = load_split("train")
X_val, y_val = load_split("val")


# Safety checks
assert X_train.shape == (5394, 512)
assert y_train.shape == (5394, 312)

assert X_val.shape == (600, 512)
assert y_val.shape == (600, 312)


print("=" * 65)
print("FULL CUB CBL HYPERPARAMETER TUNING")
print("=" * 65)

print("Device:", DEVICE)

print(
    "Train:",
    X_train.shape,
    y_train.shape
)

print(
    "Val:",
    X_val.shape,
    y_val.shape
)

print("Model: Linear 512 -> 312")

print(
    "Learning rates:",
    LEARNING_RATES
)

print(
    "Weighting:",
    WEIGHTING_MODES
)

print("TEST SET IS NOT USED DURING TUNING")

print("=" * 65)


# ============================================================
# 5. Metrics
# ============================================================

def calculate_metrics(
    probs,
    targets,
    threshold
):

    preds = (
        probs >= threshold
    ).float()


    # -------------------------
    # Micro metrics
    # -------------------------

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
        tp
        / (tp + fp + 1e-8)
    ).item()


    recall = (
        tp
        / (tp + fn + 1e-8)
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


    # -------------------------
    # Per-concept F1
    # -------------------------

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
        concept_f1
        .mean()
        .item()
    )


    # Some concepts have zero positives in Val.
    supported = (
        targets.sum(dim=0) > 0
    )


    macro_supported = (
        concept_f1[supported]
        .mean()
        .item()
    )


    return {

        "precision":
            precision,

        "recall":
            recall,

        "micro_f1":
            micro_f1,

        "macro_f1_all":
            macro_all,

        "macro_f1_supported":
            macro_supported,
    }


# ============================================================
# 6. Train one configuration
# ============================================================

def train_one_config(
    learning_rate,
    weighting_mode
):

    set_seed(SEED)


    print("\n")
    print("=" * 65)

    print(
        f"Learning rate = {learning_rate}"
    )

    print(
        f"Weighting = {weighting_mode}"
    )

    print("=" * 65)


    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = nn.Linear(
        INPUT_DIM,
        NUM_CONCEPTS
    ).to(DEVICE)


    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    pos_weight = None


    if weighting_mode == "none":

        criterion = (
            nn.BCEWithLogitsLoss()
        )


    elif weighting_mode == "sqrt":

        positive = (
            y_train.sum(dim=0)
        )

        negative = (
            len(y_train)
            - positive
        )

        ratio = (
            negative
            / positive
        )


        pos_weight = torch.sqrt(
            ratio
        )


        # Prevent very rare concepts
        # from receiving huge weights.
        pos_weight = torch.clamp(
            pos_weight,
            min=1.0,
            max=5.0
        )


        criterion = (
            nn.BCEWithLogitsLoss(
                pos_weight=
                pos_weight.to(DEVICE)
            )
        )


    else:

        raise ValueError(
            "Unknown weighting mode"
        )


    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=WEIGHT_DECAY
    )


    # --------------------------------------------------------
    # Training DataLoader
    # --------------------------------------------------------

    generator = torch.Generator()

    generator.manual_seed(
        SEED
    )


    train_loader = DataLoader(

        TensorDataset(
            X_train,
            y_train
        ),

        batch_size=
            BATCH_SIZE,

        shuffle=True,

        generator=
            generator
    )


    # --------------------------------------------------------
    # Early stopping
    # --------------------------------------------------------

    best_val_loss = float("inf")

    best_epoch = 0

    best_state = None

    no_improvement = 0


    # ========================================================
    # Training loop
    # ========================================================

    for epoch in range(
        1,
        MAX_EPOCHS + 1
    ):


        # ---------------------
        # TRAIN
        # ---------------------

        model.train()

        total_train_loss = 0


        for (
            features,
            targets
        ) in train_loader:


            features = (
                features.to(DEVICE)
            )

            targets = (
                targets.to(DEVICE)
            )


            optimizer.zero_grad()


            logits = model(
                features
            )


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


        # ---------------------
        # VALIDATION
        # ---------------------

        model.eval()


        with torch.no_grad():

            val_logits = model(
                X_val.to(DEVICE)
            )


            val_loss = criterion(
                val_logits,
                y_val.to(DEVICE)
            ).item()


        # ---------------------
        # Save best epoch
        # ---------------------

        if (
            val_loss
            < best_val_loss
        ):

            best_val_loss = (
                val_loss
            )

            best_epoch = epoch


            best_state = (
                copy.deepcopy(
                    model.state_dict()
                )
            )


            no_improvement = 0


        else:

            no_improvement += 1


        # ---------------------
        # Print progress
        # ---------------------

        if (
            epoch == 1
            or epoch % 20 == 0
        ):

            print(

                f"Epoch {epoch:03d} | "

                f"Train Loss="
                f"{train_loss:.4f} | "

                f"Val Loss="
                f"{val_loss:.4f}"
            )


        # ---------------------
        # Early stopping
        # ---------------------

        if (
            no_improvement
            >= PATIENCE
        ):

            print(
                f"Early stopping "
                f"at epoch {epoch}"
            )

            break


    # ========================================================
    # Restore best epoch
    # ========================================================

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


    # ========================================================
    # Validation threshold search
    # ========================================================

    best_threshold = None

    best_metrics = None

    best_micro_f1 = -1


    for threshold in np.arange(
        0.10,
        0.71,
        0.01
    ):


        threshold = float(
            threshold
        )


        metrics = calculate_metrics(
            val_probs,
            y_val,
            threshold
        )


        if (
            metrics["micro_f1"]
            > best_micro_f1
        ):

            best_micro_f1 = (
                metrics["micro_f1"]
            )

            best_threshold = (
                threshold
            )

            best_metrics = (
                metrics
            )


    # ========================================================
    # Print this configuration
    # ========================================================

    print()

    print(
        "Best epoch:",
        best_epoch
    )

    print(
        "Best Val Loss:",
        round(
            best_val_loss,
            4
        )
    )

    print(
        "Best threshold:",
        round(
            best_threshold,
            2
        )
    )

    print(
        "Precision:",
        round(
            best_metrics[
                "precision"
            ],
            4
        )
    )

    print(
        "Recall:",
        round(
            best_metrics[
                "recall"
            ],
            4
        )
    )

    print(
        "Micro-F1:",
        round(
            best_metrics[
                "micro_f1"
            ],
            4
        )
    )

    print(
        "Macro-F1 all:",
        round(
            best_metrics[
                "macro_f1_all"
            ],
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


    # ========================================================
    # Return result
    # ========================================================

    result = {

        "learning_rate":
            float(learning_rate),

        "weighting":
            weighting_mode,

        "batch_size":
            BATCH_SIZE,

        "best_epoch":
            int(best_epoch),

        "best_val_loss":
            float(best_val_loss),

        "best_threshold":
            float(best_threshold),

        "precision":
            float(
                best_metrics[
                    "precision"
                ]
            ),

        "recall":
            float(
                best_metrics[
                    "recall"
                ]
            ),

        "micro_f1":
            float(
                best_metrics[
                    "micro_f1"
                ]
            ),

        "macro_f1_all":
            float(
                best_metrics[
                    "macro_f1_all"
                ]
            ),

        "macro_f1_supported":
            float(
                best_metrics[
                    "macro_f1_supported"
                ]
            ),
    }


    return (
        result,
        best_state,
        pos_weight
    )


# ============================================================
# 7. Run all configurations
# ============================================================

results = []


overall_best_score = -1

overall_best_result = None

overall_best_state = None

overall_best_pos_weight = None


for weighting_mode in WEIGHTING_MODES:

    for learning_rate in LEARNING_RATES:


        (
            result,
            state,
            pos_weight
        ) = train_one_config(

            learning_rate,
            weighting_mode
        )


        results.append(
            result
        )


        # Choose final setup based on
        # Validation Micro-F1
        if (
            result["micro_f1"]
            > overall_best_score
        ):

            overall_best_score = (
                result["micro_f1"]
            )

            overall_best_result = (
                result
            )

            overall_best_state = (
                copy.deepcopy(
                    state
                )
            )

            overall_best_pos_weight = (
                pos_weight
            )


# ============================================================
# 8. Save comparison table
# ============================================================

results_df = pd.DataFrame(
    results
)


results_df = results_df.sort_values(
    by="micro_f1",
    ascending=False
)


csv_path = (
    OUTPUT_DIR
    / "tuning_results.csv"
)


results_df.to_csv(
    csv_path,
    index=False
)


# ============================================================
# 9. Print comparison
# ============================================================

print("\n")
print("=" * 90)
print("ALL TUNING RESULTS")
print("=" * 90)


print(

    results_df[
        [
            "learning_rate",
            "weighting",
            "best_epoch",
            "best_threshold",
            "precision",
            "recall",
            "micro_f1",
            "macro_f1_all",
            "macro_f1_supported",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# 10. Save final best tuned model
# ============================================================

best_model_path = (
    OUTPUT_DIR
    / "concept_layer_best_tuned.pt"
)


torch.save(

    {

        "model_state_dict":
            overall_best_state,

        "input_dim":
            INPUT_DIM,

        "num_concepts":
            NUM_CONCEPTS,

        "config":
            overall_best_result,

        "pos_weight":
            overall_best_pos_weight,

        "clip_model":
            "ViT-B/32",

        "feature_normalization":
            "L2",

        "seed":
            SEED,

        "dataset":
            "CUB-200-2011",

        "train_size":
            5394,

        "val_size":
            600,

        "test_size":
            5794,

        "test_used_for_tuning":
            False,
    },

    best_model_path
)


# ============================================================
# 11. Save best config as JSON
# ============================================================

json_path = (
    OUTPUT_DIR
    / "best_config.json"
)


with open(
    json_path,
    "w"
) as f:

    json.dump(
        overall_best_result,
        f,
        indent=4
    )


# ============================================================
# 12. Final summary
# ============================================================

print("\n")
print("=" * 90)
print("BEST CONFIGURATION")
print("=" * 90)


for key, value in (
    overall_best_result.items()
):

    print(
        f"{key}: {value}"
    )


print("\nSaved comparison table:")
print(csv_path)


print("\nSaved best tuned CBL:")
print(best_model_path)


print("\nSaved best configuration:")
print(json_path)


print("\n")
print("=" * 90)
print("TUNING COMPLETE")
print("TEST SET HAS NOT BEEN USED FOR PARAMETER SELECTION")
print("=" * 90)
