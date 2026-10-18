# CUB-200-2011 Dataset Handoff

## Purpose
Prepared metadata and a PyTorch dataset loader for the team project.

## Contents
- images.csv: image metadata and paths
- classes.csv: 200 bird classes
- attribute_definitions.csv: 312 attribute definitions
- attributes.csv.gz: image-level attribute annotations
- bounding_boxes.csv: image bounding boxes (x, y, width, height)
- parts.csv: 15 body-part annotations per image
- image_splits.csv: train/validation/test assignments
- dataset.py: CUBDataset PyTorch loader

## Experiment split counts
- train: 5,394 images
- val: 600 images
- test: 5,794 images

## Important
1. The original CUB-200-2011 image dataset is not included in this package.
   Each user must obtain the images through the team's agreed dataset source
   and configure the image paths in images.csv or pass the correct root path.
2. The loader returns bounding boxes and part coordinates in original-image
   pixel coordinates. If images are resized or cropped, the coordinates must
   be transformed consistently.
3. In parts.csv, invisible parts (visible == 0) may have placeholder
   coordinates (0, 0). Do not treat those as real visible keypoints.
4. The attribute file contains 606 records with an unusual six-field source
   row format. These records were preserved during preprocessing, but the
   extra-field semantics still need verification. Review before relying on
   those records for final experiments.

## Basic usage in Colab
Place dataset.py and the processed CSV files in the same directory, then:

    import sys
    sys.path.insert(0, "/path/to/processed")
    from dataset import CUBDataset

    train_ds = CUBDataset(
        root="/path/to/CUB_200_2011",
        processed_dir="/path/to/processed",
        split="train"
    )
    sample = train_ds[0]
    print(sample.keys())
    print(sample["attributes"].shape)

## Tested checks
The loader smoke test passed:
- train: 5,394 images
- val: 600 images
- test: 5,794 images
- attributes shape: [312]
- bbox shape: [4]
- parts shape: [15, 2]
- visible-parts shape: [15]

These are basic loading checks, not a claim that all annotation semantics
have been fully audited.
