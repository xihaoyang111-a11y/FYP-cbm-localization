
from pathlib import Path
import sys
import torch
from torch.utils.data import DataLoader
import clip

# =========================
# Paths
# =========================
ROOT = Path("/content/drive/MyDrive/CUB_Project_Data")

OUTPUT = Path(
    "/content/drive/MyDrive/FYP_Member2_Outputs/FULL_CUB_CLEAN/clip_features"
)
OUTPUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))

from fyp_data.dataset import (
    load_cub_tables,
    build_concept_matrix,
    select_split,
    CUBDataset,
)

# =========================
# Settings
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_MODEL = "ViT-B/32"
BATCH_SIZE = 64

print("====================================")
print("FULL CUB - CLIP Feature Extraction")
print("TRAIN + VAL ONLY")
print("====================================")
print("Device:", DEVICE)
print("CLIP:", CLIP_MODEL)

# =========================
# Load CLIP
# =========================
model, preprocess = clip.load(
    CLIP_MODEL,
    device=DEVICE
)
model.eval()

# =========================
# Load FULL CUB
# =========================
images, classes, attributes = load_cub_tables(ROOT)

concept_matrix = build_concept_matrix(
    attributes,
    image_ids=images["image_id"]
)

train_df = select_split(
    images, ROOT,
    policy="experiment",
    split="train"
)

val_df = select_split(
    images, ROOT,
    policy="experiment",
    split="val"
)

assert len(train_df) == 5394
assert len(val_df) == 600
assert concept_matrix.shape == (11788, 312)

print("\nTrain:", len(train_df))
print("Val:", len(val_df))
print("Concept Matrix:", concept_matrix.shape)


def extract_split(split_name, split_df):

    print("\n====================================")
    print(f"Processing {split_name}: {len(split_df)} images")
    print("====================================")

    dataset = CUBDataset(
        split_df,
        root=ROOT,
        transform=preprocess,
        concept_matrix=concept_matrix,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=(DEVICE == "cuda"),
    )

    all_features = []
    all_classes = []
    all_concepts = []

    with torch.no_grad():

        for batch_idx, batch in enumerate(loader):

            image_batch, class_labels, concept_labels = batch

            image_batch = image_batch.to(
                DEVICE,
                non_blocking=True
            )

            features = model.encode_image(image_batch)

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
                or batch_idx + 1 == len(loader)
            ):
                print(
                    f"{split_name}: "
                    f"batch {batch_idx + 1}/{len(loader)}"
                )

    features = torch.cat(all_features)
    class_labels = torch.cat(all_classes)
    concept_targets = torch.cat(all_concepts)

    image_ids = torch.tensor(
        split_df["image_id"].to_numpy(),
        dtype=torch.long
    )

    assert features.shape == (len(split_df), 512)
    assert concept_targets.shape == (len(split_df), 312)
    assert class_labels.shape == (len(split_df),)
    assert image_ids.shape == (len(split_df),)

    assert torch.isfinite(features).all()
    assert class_labels.min() >= 0
    assert class_labels.max() < 200

    save_path = (
        OUTPUT /
        f"clip_features_{split_name}.pt"
    )

    torch.save(
        {
            "image_ids": image_ids,
            "features": features,
            "class_labels": class_labels,
            "concept_targets": concept_targets,
            "split": split_name,
            "clip_model": CLIP_MODEL,
            "feature_dimension": 512,
            "num_concepts": 312,
            "num_classes": 200,
            "dataset": "CUB-200-2011",
        },
        save_path
    )

    print("\nFinished:", split_name)
    print("Features:", features.shape)
    print("Concept targets:", concept_targets.shape)
    print("Saved:", save_path)


extract_split("train", train_df)
extract_split("val", val_df)

print("\n====================================")
print("TRAIN + VAL FEATURE EXTRACTION COMPLETE")
print("TEST HAS NOT BEEN USED")
print("====================================")
