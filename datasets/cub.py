
import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class CUBDataset(Dataset):
    """
    CUB-200-2011 dataset loader.

    Returns:
        image: RGB PIL image, or transformed image
        label: zero-based class label
        attributes: float tensor [312]
        bbox: float tensor [4], (x, y, width, height)
        parts: float tensor [15, 2], (x, y)
        parts_visible: bool tensor [15]
        image_id: integer image ID
        split: train, val, or test
    """

    def __init__(
        self,
        root="/content/cub_data/CUB_200_2011",
        processed_dir="/content/cub_data/processed",
        split="train",
        transform=None,
    ):
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")

        self.root = root
        self.processed_dir = processed_dir
        self.transform = transform

        # Load image metadata and the experiment split.
        images = pd.read_csv(
            os.path.join(processed_dir, "images.csv")
        )
        splits = pd.read_csv(
            os.path.join(processed_dir, "image_splits.csv")
        )

        metadata = images.merge(
            splits[["image_id", "experiment_split"]],
            on="image_id",
            how="inner",
            validate="one_to_one",
        )

        self.metadata = (
            metadata[metadata["experiment_split"] == split]
            .sort_values("image_id")
            .reset_index(drop=True)
        )

        if self.metadata.empty:
            raise ValueError(f"No images found for split={split}")

        # Load bounding boxes.
        boxes = pd.read_csv(
            os.path.join(processed_dir, "bounding_boxes.csv")
        )
        self.boxes = boxes.set_index("image_id")

        # Load parts and keep the original pixel coordinates.
        parts = pd.read_csv(
            os.path.join(processed_dir, "parts.csv")
        )
        self.parts = {
            int(image_id): group.sort_values("part_id")
            for image_id, group in parts.groupby("image_id")
        }

        # Load attribute annotations and create a 312-value vector
        # for each image. Check that every image has 312 entries.
        attributes = pd.read_csv(
            os.path.join(processed_dir, "attributes.csv.gz")
        )

        counts = attributes.groupby("image_id")[
            "attribute_id"
        ].nunique()

        expected_ids = set(self.metadata["image_id"].astype(int))
        actual_ids = set(counts.index.astype(int))

        missing_images = expected_ids - actual_ids
        incomplete_images = counts[
            counts.index.isin(expected_ids) & (counts != 312)
        ]

        if missing_images or not incomplete_images.empty:
            raise ValueError(
                "Attribute annotations are incomplete. "
                f"Missing images: {len(missing_images)}, "
                f"incomplete images: {len(incomplete_images)}"
            )

        self.attribute_matrix = (
            attributes.pivot(
                index="image_id",
                columns="attribute_id",
                values="is_present",
            )
            .reindex(columns=range(1, 313))
            .sort_index()
        )

        if self.attribute_matrix.isna().any().any():
            raise ValueError(
                "Attribute matrix contains missing values."
            )

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        row = self.metadata.iloc[index]
        image_id = int(row["image_id"])

        image_path = row["image_path"]
        if not isinstance(image_path, str) or not os.path.isabs(image_path):
            image_path = os.path.join(self.root, row["relative_path"])

        if not os.path.isfile(image_path):
            raise FileNotFoundError(image_path)

        image = Image.open(image_path).convert("RGB")

        # Note: bbox and part coordinates remain in original-image
        # pixels. If the transform resizes/crops the image, transform
        # these coordinates consistently before using them.
        if self.transform is not None:
            image = self.transform(image)

        bbox_row = self.boxes.loc[image_id]
        bbox = torch.tensor(
            bbox_row[["x", "y", "width", "height"]].to_numpy(
                dtype="float32"
            ),
            dtype=torch.float32,
        )

        part_df = self.parts[image_id]
        parts = torch.tensor(
            part_df[["x", "y"]].to_numpy(dtype="float32"),
            dtype=torch.float32,
        )
        parts_visible = torch.tensor(
            part_df["visible"].to_numpy() == 1,
            dtype=torch.bool,
        )

        attributes = torch.tensor(
            self.attribute_matrix.loc[image_id].to_numpy(
                dtype="float32"
            ),
            dtype=torch.float32,
        )

        return {
            "image": image,
            "label": int(row["class_id"]) - 1,
            "attributes": attributes,
            "bbox": bbox,
            "parts": parts,
            "parts_visible": parts_visible,
            "image_id": image_id,
            "split": row["experiment_split"],
        }
