# FYP: Concept Bottlenecks and Visual Grounding on CUB-200-2011

Code for bird classification, concept prediction, and visual grounding.
New experiments use the canonical Drive loader synchronized into `fyp_data/`.
Training implementations and baseline results are not included.

## Member 2: clone and install

The repository is private. The owner must invite Member 2 as a collaborator,
and Member 2 must accept before cloning with their own GitHub account.

```sh
git clone https://github.com/xihaoyang111-a11y/FYP-cbm-localization.git
cd FYP-cbm-localization
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
```

Use Python 3.11 or 3.12. For GPU use install a compatible torch/torchvision
pair for your hardware first. Dependency ranges are not a tested lock.
See [Member 2 quickstart](MEMBER2_QUICKSTART.md) for Colab and loading examples.

## Drive data access

Request access to the team's
[CUB_Project_Data folder](https://drive.google.com/drive/folders/1qyg9zJWy9OYnRotP3FZag_grlQJn5AX5).
Drive access is separate from GitHub access; neither teammate's access has been verified.
In Colab, add a shortcut in My Drive if needed, mount Drive, and verify the path.
Locally, download/sync the images and processed tables while preserving this layout:

```text
CUB_Project_Data/                 ROOT passed to the canonical loader
  CUB_200_2011/
    images/
  processed/
    images.csv
    classes.csv
    attributes.csv.gz
    image_splits.csv
    attribute_definitions.csv
    bounding_boxes.csv
    parts.csv
```

Keep data outside the Git checkout. Do not upload images, processed CSVs,
archives, checkpoints or training outputs to GitHub.

## Canonical API and split policy

Run from the cloned repository root so code comes from this checkout:

```python
from pathlib import Path
from fyp_data.dataset import CUBDataset, load_cub_tables, build_concept_matrix, select_split

ROOT = Path('/content/drive/MyDrive/CUB_Project_Data')  # or your local project data folder
images, classes, attributes = load_cub_tables(ROOT)
concepts = build_concept_matrix(attributes, images.image_id)
train = select_split(images, ROOT, policy='experiment', split='train')
dataset = CUBDataset(train, ROOT, concept_matrix=concepts)
image, label, concept = dataset[0]  # RGB PIL, torch.long scalar, float32 [312]
```

ROOT is the project data folder containing both `processed/` and
`CUB_200_2011/`, not the image directory. Stored absolute image_path values
are ignored; images resolve from relative_path.

Use `policy='experiment'` explicitly for new work: frozen train 5,394 /
val 600 / test 5,794. Validation comes only from official training images;
official test membership is unchanged. Use validation for model selection
and test for final evaluation. Do not recreate the split or silently train on validation.
The original sampling seed/procedure has not been reconstructed.

For legacy reproduction, `policy='official'` gives train 5,994 / test 5,794;
requesting val fails. Record the policy in each experiment.
Concepts use raw binary is_present, attribute IDs 1..312 in order, without
certainty filtering or imputation. Class targets are 0..199.

Pass your chosen CLIP checkpoint's own preprocessing as `transform` for
classification/concept training. The basic make_default_transform is only
a loading check and does not perform CLIP normalization.
For grounding use `grounding=True` with a concept matrix; samples are dictionaries.
Use a joint_transform to update image, boxes, parts and visibility together.
Image-only transforms are rejected for grounding. Coordinates otherwise remain
original pixels; invisible parts must be masked.

## Existing files and migration

`datasets/cub.py`, `scripts/smoke_dataset.py` and
`docs/dataset_handoff.md` remain unchanged for existing notebooks.
Their API and root semantics differ from `fyp_data.dataset`; do not mix them.
The previous root README is preserved in
[docs/README_before_canonical_sync.md](docs/README_before_canonical_sync.md).
Models, training, evaluation, configs and notebooks remain in their existing directories.

The inspected Drive root README is retained in
[docs/DRIVE_CANONICAL_README.md](docs/DRIVE_CANONICAL_README.md).
Its historical verification statements describe the earlier Drive handoff,
not tests performed during this repository synchronization. Its Drive layout
is not the Git repository layout. The historical ZIP/root-loader entrypoints
are not the canonical API for new experiments.

## Validation and scientific limits

Run the canonical smoke check after installing dependencies and obtaining data:

```sh
python -m scripts.smoke_canonical --root /path/to/CUB_Project_Data
```

The check verifies frozen split counts, concept shape, image loading, tuple and
grounding sample shapes. Real data loading and training were not run during this
sync preparation. The 606 unusual annotation records require further semantic
audit; do not treat missing no_anno as zero or as visibility/confidence.
See the Drive README for coordinate and annotation caveats.

The upload bundle was prepared from the supplied repository ZIP and inspected
Drive sources. It is not evidence of a remote GitHub commit.

