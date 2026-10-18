
from pathlib import Path
import sys
import torch
from torch.utils.data import DataLoader
import clip

# ============================================================
# Paths
# ============================================================

ROOT = Path(
    "/content/drive/MyDrive/CUB_Project_Data"
)

OUTPUT = Path(
    "/content/drive/MyDrive/FYP_Member2_Outputs/FULL_CUB_CLEAN/clip_features"
)

OUTPUT.mkdir(
    parents=True,
    exist_ok=True
)

sys.path.insert(
    0,
    str(ROOT)
)

from fyp_data.dataset import (
    load_cub_tables,
    build_concept_matrix,
    select_split,
    CUBDataset,
)


# ============================================================
# Settings
# ============================================================

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

CLIP_MODEL = "ViT-B/32"
BATCH_SIZE = 64


print("=" * 60)
print("FULL CUB - FINAL TEST FEATURE EXTRACTION")
print("=" * 60)

print("Device:", DEVICE)
print("CLIP:", CLIP_MODEL)


# ============================================================
# Load frozen CLIP
# ============================================================

model, preprocess = clip.load(
    CLIP_MODEL,
    device=DEVICE
)

model.eval()


# ============================================================
# Load canonical Full CUB
# ============================================================

images, classes, attributes = (
    load_cub_tables(ROOT)
)

concept_matrix = (
    build_concept_matrix(
        attributes,
        image_ids=images["image_id"]
    )
)

test_df = select_split(
    images,
    ROOT,
    policy="experiment",
    split="test"
)


assert len(test_df) == 5794
assert concept_matrix.shape == (11788, 312)


print("Test images:", len(test_df))
print(
    "Concept Matrix:",
    concept_matrix.shape
)


# ============================================================
# Dataset
# ============================================================

test_dataset = CUBDataset(
    test_df,
    root=ROOT,
    transform=preprocess,
    concept_matrix=concept_matrix,
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=(DEVICE == "cuda"),
)


# ============================================================
# Extract frozen CLIP features
# ============================================================

all_features = []
all_classes = []
all_concepts = []


with torch.no_grad():

    for batch_idx, batch in enumerate(
        test_loader
    ):

        (
            image_batch,
            class_labels,
            concept_labels
        ) = batch

        image_batch = image_batch.to(
            DEVICE,
            non_blocking=True
        )


        features = model.encode_image(
            image_batch
        )


        all_features.append(
            features.float().cpu()
        )

        all_classes.append(
            class_labels.cpu()
        )

        all_concepts.append(
            concept_labels.cpu()
        )


        if (
            batch_idx == 0
            or (batch_idx + 1) % 10 == 0
            or batch_idx + 1 == len(test_loader)
        ):

            print(
                f"test: batch "
                f"{batch_idx + 1}/"
                f"{len(test_loader)}"
            )


# ============================================================
# Combine
# ============================================================

features = torch.cat(
    all_features
)

class_labels = torch.cat(
    all_classes
)

concept_targets = torch.cat(
    all_concepts
)


image_ids = torch.tensor(
    test_df["image_id"].to_numpy(),
    dtype=torch.long
)


# ============================================================
# Safety checks
# ============================================================

assert features.shape == (5794, 512)

assert concept_targets.shape == (
    5794,
    312
)

assert class_labels.shape == (
    5794,
)

assert image_ids.shape == (
    5794,
)

assert torch.isfinite(
    features
).all()

assert class_labels.min() >= 0
assert class_labels.max() < 200


# ============================================================
# Save
# ============================================================

save_path = (
    OUTPUT
    / "clip_features_test.pt"
)


torch.save(
    {

        "image_ids":
            image_ids,

        "features":
            features,

        "class_labels":
            class_labels,

        "concept_targets":
            concept_targets,

        "split":
            "test",

        "clip_model":
            CLIP_MODEL,

        "feature_dimension":
            512,

        "num_concepts":
            312,

        "num_classes":
            200,

        "dataset":
            "CUB-200-2011",

    },

    save_path
)


print("\n" + "=" * 60)
print("TEST FEATURE EXTRACTION COMPLETE")
print("=" * 60)

print(
    "Features:",
    features.shape
)

print(
    "Concept targets:",
    concept_targets.shape
)

print(
    "Saved:",
    save_path
)
