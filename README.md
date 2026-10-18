# FYP: Concept Bottlenecks and Visual Grounding on CUB-200-2011

Code repository for bird classification, concept prediction, and visual grounding.
This initial setup includes the team's existing dataset loader; model training
and evaluation implementations are still to be added. No baseline results are claimed.

## Setup

Use Python 3.11 or 3.12 in a virtual environment:

```sh
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
```

For GPU use, install a compatible torch/torchvision pair for your hardware first.
The requirements provide version ranges, not a tested dependency lock.

## Local data

Obtain CUB images separately from the team's agreed source. Extract the CSVs
from `CUB_dataset_handoff.zip` into `data/processed/` locally:

```text
data/processed/
  images.csv
  image_splits.csv
  classes.csv
  attribute_definitions.csv
  attributes.csv.gz
  bounding_boxes.csv
  parts.csv
```

`--root` points at the CUB_200_2011 directory containing `images/`. The loader
joins this root to `relative_path` for relative image paths. An absolute
`image_path` in images.csv takes precedence: update stale absolute paths before
moving to another machine.

Run from the repository root:

```sh
python -m scripts.smoke_dataset --root /path/to/CUB_200_2011
```

```python
from datasets.cub import CUBDataset

train = CUBDataset(root='/path/to/CUB_200_2011',
                   processed_dir='data/processed', split='train')
sample = train[0]
```

The handoff defines train/validation/test as 5,394 / 600 / 5,794 images.
Use these fixed assignments; do not re-split test data into training.
Each sample includes a PIL RGB image, zero-based class label, 312 attribute
values, an xywh bounding box, 15 xy part coordinates, visibility flags, image ID,
and split. Apply a tensor transform before batching images with a DataLoader.

**Grounding caveats:** boxes and parts remain in original-image pixel coordinates.
Image resizing/cropping/flipping must transform coordinates consistently.
Invisible parts must be masked. The supplied attributes contain 606 unusual
six-field source records whose extra-field semantics need verification before
final experiments. See [the original handoff](docs/dataset_handoff.md).

## Layout

- `datasets/`: existing handoff loader (`cub.py`, preserved unchanged).
- `models/`: baseline and concept bottleneck models.
- `training/`: optimization, losses, and training loops.
- `evaluation/`: classification, concepts, and grounding metrics.
- `configs/`: reproducible experiment settings.
- `scripts/`: runnable checks and future experiment entry points.
- `notebooks/`: exploratory analysis.
- `docs/`: dataset handoff and project notes.

Only source code and small documentation belong in Git. Dataset directories,
CSVs, image files, archives, weights, and experiment outputs are ignored.
The local processed tables are not part of the repository. Do not force-add them.

## Verification limits

Setup checks cover Python syntax, handoff file integrity, metadata split counts,
and Git exclusions. Real-image loading and training require the image dataset
and installed PyTorch/pandas; they have not been run in this setup environment.
