
from pathlib import Path
import json

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Paths
# ============================================================

FEATURE_DIR = Path(
    "/content/drive/MyDrive/FYP_Member2_Outputs/FULL_CUB_CLEAN/clip_features"
)

MODEL_PATH = Path(
    "/content/drive/MyDrive/FYP_Member2_Outputs/FULL_CUB_CLEAN/cbl/concept_layer_best.pt"
)

OUTPUT_DIR = Path(
    "/content/drive/MyDrive/FYP_Member2_Outputs/FULL_CUB_CLEAN/final_predictions"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# Load final frozen CBL
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=False
)


INPUT_DIM = checkpoint[
    "input_dim"
]

NUM_CONCEPTS = checkpoint[
    "num_concepts"
]

THRESHOLD = checkpoint[
    "best_threshold"
]


model = nn.Linear(
    INPUT_DIM,
    NUM_CONCEPTS
).to(DEVICE)


model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)


model.eval()


print("=" * 70)
print("FINAL MEMBER 2 CONCEPT SCORE GENERATION")
print("=" * 70)

print("Device:", DEVICE)
print("Model:", MODEL_PATH)

print(
    "Fixed threshold:",
    THRESHOLD
)

print(
    "No further tuning is performed."
)


# ============================================================
# Metric function
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
        (1 - preds)
        * targets
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


    # Per-concept
    tp_c = (
        preds * targets
    ).sum(dim=0)

    fp_c = (
        preds * (1 - targets)
    ).sum(dim=0)

    fn_c = (
        (1 - preds)
        * targets
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
        concept_f1.mean()
        .item()
    )


    supported = (
        targets.sum(dim=0) > 0
    )


    macro_supported = (
        concept_f1[
            supported
        ]
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
# Generate one split
# ============================================================

def process_split(split):

    print("\n" + "=" * 70)

    print(
        f"Processing {split.upper()}"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # Load frozen CLIP features
    # --------------------------------------------------------

    data = torch.load(
        FEATURE_DIR
        / f"clip_features_{split}.pt",

        map_location="cpu",
        weights_only=False
    )


    features = (
        data["features"]
        .float()
    )


    # IMPORTANT:
    # Training used L2-normalized CLIP features
    features = F.normalize(
        features,
        dim=1
    )


    targets = (
        data["concept_targets"]
        .float()
    )


    # --------------------------------------------------------
    # Predict concept logits
    # --------------------------------------------------------

    with torch.no_grad():

        logits = model(
            features.to(DEVICE)
        ).cpu()


    probs = torch.sigmoid(
        logits
    )


    binary_predictions = (
        probs >= THRESHOLD
    ).float()


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics = calculate_metrics(
        probs,
        targets,
        THRESHOLD
    )


    print(
        "Precision:",
        round(
            metrics["precision"],
            4
        )
    )

    print(
        "Recall:",
        round(
            metrics["recall"],
            4
        )
    )

    print(
        "Micro-F1:",
        round(
            metrics["micro_f1"],
            4
        )
    )

    print(
        "Macro-F1 all:",
        round(
            metrics["macro_f1_all"],
            4
        )
    )

    print(
        "Macro-F1 supported:",
        round(
            metrics[
                "macro_f1_supported"
            ],
            4
        )
    )


    # --------------------------------------------------------
    # Save predictions for Member 4
    # --------------------------------------------------------

    save_path = (
        OUTPUT_DIR
        / f"concept_predictions_{split}.pt"
    )


    torch.save(
        {

            "image_ids":
                data["image_ids"],

            "class_labels":
                data["class_labels"],

            "concept_targets":
                targets,

            # Raw linear output
            "concept_logits":
                logits,

            # Continuous concept scores
            "concept_probs":
                probs,

            # Thresholded 0/1 concepts
            "concept_predictions":
                binary_predictions,

            "threshold":
                THRESHOLD,

            "clip_model":
                "ViT-B/32",

            "feature_normalization":
                "L2",

            "num_concepts":
                312,

            "num_classes":
                200,

            "split":
                split,

        },

        save_path
    )


    print(
        "Saved:",
        save_path
    )


    return metrics


# ============================================================
# Generate ALL final outputs
# ============================================================

all_metrics = {}


for split in [
    "train",
    "val",
    "test"
]:

    all_metrics[
        split
    ] = process_split(
        split
    )


# ============================================================
# Save metrics
# ============================================================

metrics_path = (
    OUTPUT_DIR
    / "final_metrics.json"
)


with open(
    metrics_path,
    "w"
) as f:

    json.dump(
        all_metrics,
        f,
        indent=4
    )


print("\n")
print("=" * 70)
print("MEMBER 2 FINAL OUTPUT COMPLETE")
print("=" * 70)

print(
    "Saved metrics:",
    metrics_path
)

print("\nFiles for Member 4:")

print(
    OUTPUT_DIR
    / "concept_predictions_train.pt"
)

print(
    OUTPUT_DIR
    / "concept_predictions_val.pt"
)

print(
    OUTPUT_DIR
    / "concept_predictions_test.pt"
)

print("\nTrained CBL:")

print(
    MODEL_PATH
)
