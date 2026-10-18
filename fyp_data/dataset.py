from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


DEFAULT_ROOT = Path("/content/drive/MyDrive/CUB_Project_Data")


def load_cub_tables(root=DEFAULT_ROOT):
    """Load image index, class index, and compressed attribute annotations."""
    root = Path(root)
    processed = root / "processed"

    images = pd.read_csv(processed / "images.csv")
    classes = pd.read_csv(processed / "classes.csv")
    attributes = pd.read_csv(processed / "attributes.csv.gz")

    required_image_cols = {
        "image_id", "relative_path", "class_id",
        "is_official_train", "official_split"
    }
    missing = required_image_cols - set(images.columns)
    if missing:
        raise ValueError(f"images.csv missing columns: {sorted(missing)}")

    if images["image_id"].duplicated().any():
        raise ValueError("Duplicate image_id values in images.csv")

    return images, classes, attributes


def build_concept_matrix(attributes, image_ids=None):
    """
    Convert long-format attribute annotations to a wide concept matrix.

    Expected semantic fields:
      image_id, attribute_id, is_present

    Returns:
      DataFrame indexed by image_id, with one column per attribute_id.

    Important:
      This function does not infer uncertain-label policy.
      Check the source column meanings before using labels for training.
    """
    df = attributes.copy()

    # Match common names without assuming capitalization.
    normalized = {str(c).strip().lower(): c for c in df.columns}

    aliases = {
        "image_id": ["image_id", "img_id"],
        "attribute_id": ["attribute_id", "attr_id"],
        "is_present": ["is_present", "present", "label", "attribute_label"],
    }

    found = {}
    for semantic_name, candidates in aliases.items():
        for candidate in candidates:
            if candidate in normalized:
                found[semantic_name] = normalized[candidate]
                break

    missing = set(aliases) - set(found)
    if missing:
        raise ValueError(
            "Cannot identify required attribute fields: "
            f"{sorted(missing)}. Actual columns: {list(df.columns)}. "
            "Inspect attributes.csv.gz and update the aliases."
        )

    df = df.rename(columns={
        found["image_id"]: "image_id",
        found["attribute_id"]: "attribute_id",
        found["is_present"]: "is_present",
    })

    # The annotation table should have one row per image-attribute pair.
    if df.duplicated(["image_id", "attribute_id"]).any():
        raise ValueError(
            "Duplicate image_id/attribute_id pairs found. "
            "Resolve duplicates before creating concept labels."
        )

    # Validate labels before pivoting.
    values = set(df["is_present"].dropna().unique().tolist())
    if not values.issubset({0, 1, 0.0, 1.0, True, False}):
        raise ValueError(
            f"is_present contains unexpected values: {values}. "
            "Inspect the annotation encoding before training."
        )

    matrix = df.pivot(
        index="image_id",
        columns="attribute_id",
        values="is_present"
    ).sort_index(axis=1)

    matrix.columns = [f"attribute_{int(c)}" for c in matrix.columns]
    matrix.index.name = "image_id"

    if image_ids is not None:
        matrix = matrix.reindex(image_ids)

    if list(matrix.columns) != [f"attribute_{i}" for i in range(1, 313)]:
        raise ValueError("Expected attribute IDs exactly 1..312")
    if matrix.isna().any().any():
        raise ValueError("Missing concept labels; no implicit zero filling")
    return matrix


class CUBDataset(Dataset):
    """
    CUB image dataset.

    Returns:
      image: FloatTensor [3, H, W]
      class_label: LongTensor scalar, labels 0..199
      concept_label: FloatTensor [num_concepts], if concept matrix supplied
    """

    def __init__(
        self,
        dataframe,
        root=DEFAULT_ROOT,
        transform=None,
        concept_matrix=None,
        *, grounding=False, joint_transform=None,
    ):
        self.root = Path(root)
        self.images_dir = self.root / "CUB_200_2011" / "images"
        self.df = dataframe.reset_index(drop=True).copy()
        if self.df.image_id.duplicated().any() or not self.df.class_id.isin(range(1, 201)).all():
            raise ValueError("Expected unique image IDs and class IDs 1..200")
        for value in self.df.relative_path:
            path = Path(value)
            if path.is_absolute() or ".." in path.parts or "\\" in value or ":" in value:
                raise ValueError("relative_path must be a safe path relative to CUB_200_2011/images")
        self.transform = transform
        self.concept_matrix = concept_matrix
        self.grounding = grounding
        self.joint_transform = joint_transform
        if grounding and transform is not None:
            raise ValueError("Grounding requires joint_transform, not image-only transform")
        if not grounding and joint_transform is not None:
            raise ValueError("joint_transform requires grounding=True")
        if grounding:
            self.boxes = pd.read_csv(self.root / "processed/bounding_boxes.csv").set_index("image_id", verify_integrity=True)
            parts = pd.read_csv(self.root / "processed/parts.csv")
            if parts.duplicated(["image_id", "part_id"]).any():
                raise ValueError("Duplicate image/part pairs")
            self.parts = {int(k): v.sort_values("part_id") for k, v in parts.groupby("image_id")}
            for image_id in self.df.image_id:
                if image_id not in self.boxes.index or image_id not in self.parts or self.parts[image_id].part_id.tolist() != list(range(1, 16)):
                    raise ValueError(f"Incomplete grounding labels for {image_id}")

        if concept_matrix is not None:
            if concept_matrix.index.duplicated().any() or list(concept_matrix.columns) != [f"attribute_{i}" for i in range(1,313)]:
                raise ValueError("Concept matrix must have unique IDs and ordered attribute_1..attribute_312")
            if concept_matrix.isna().any().any() or not concept_matrix.isin([0,1]).all().all():
                raise ValueError("Concept matrix must contain finite binary labels")
            self.concept_matrix = concept_matrix

            missing_ids = set(self.df["image_id"]) - set(
                concept_matrix.index
            )
            if missing_ids:
                raise ValueError(
                    f"{len(missing_ids)} images have no concept row"
                )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = self.images_dir / row["relative_path"]

        with Image.open(image_path) as im:
            image = im.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        # CUB class IDs are 1..200; PyTorch targets are 0..199.
        class_label = torch.tensor(
            int(row["class_id"]) - 1, dtype=torch.long
        )

        if self.concept_matrix is None:
            if self.grounding:
                raise ValueError("grounding=True requires concept_matrix")
            return image, class_label

        concept_values = self.concept_matrix.loc[
            int(row["image_id"])
        ].to_numpy(dtype="float32", na_value=float("nan"))

        concept_label = torch.tensor(
            concept_values, dtype=torch.float32
        )

        if not self.grounding:
            return image, class_label, concept_label
        image_id = int(row["image_id"])
        p = self.parts[image_id]
        sample = dict(image=image, label=class_label, attributes=concept_label,
                      bbox=torch.tensor(self.boxes.loc[image_id, ["x", "y", "width", "height"]].to_numpy(dtype="float32")),
                      parts=torch.tensor(p[["x", "y"]].to_numpy(dtype="float32")),
                      parts_visible=torch.tensor(p.visible.to_numpy() == 1),
                      image_id=image_id, original_size=image.size,
                      split=row.get("experiment_split", row["official_split"]),
                      coordinate_space="original_pixels")
        return self.joint_transform(sample) if self.joint_transform else sample


def select_split(images, root=DEFAULT_ROOT, *, policy, split):
    """Explicit policy: official train/test, or frozen experiment train/val/test."""
    if policy == "official":
        if split not in {"train", "test"}:
            raise ValueError("Official split has no validation subset")
        return images.loc[images.official_split == split].copy()
    if policy != "experiment" or split not in {"train", "val", "test"}:
        raise ValueError("Specify policy='official' or 'experiment' and valid split")
    assignments = pd.read_csv(Path(root) / "processed/image_splits.csv")
    merged = images.merge(assignments[["image_id", "class_id", "official_split", "experiment_split"]],
                          on="image_id", how="outer", validate="one_to_one", indicator=True,
                          suffixes=("", "_assignment"))
    if not merged._merge.eq("both").all():
        raise ValueError("Split assignments must cover exactly all image IDs")
    if not merged.class_id.eq(merged.class_id_assignment).all() or not merged.official_split.eq(merged.official_split_assignment).all():
        raise ValueError("Split metadata disagrees with images.csv")
    if not merged.experiment_split.isin(["train", "val", "test"]).all():
        raise ValueError("Unknown experiment split")
    if not merged.experiment_split.eq("test").eq(merged.official_split.eq("test")).all():
        raise ValueError("Experiment split leaks official test set")
    return merged.loc[merged.experiment_split == split].drop(columns=["_merge", "class_id_assignment", "official_split_assignment"]).sort_values("image_id").reset_index(drop=True)


def make_default_transform(image_size=224):
    """Basic deterministic transform for initial data-loading tests."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
