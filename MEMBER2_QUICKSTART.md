# Member 2: CLIP + CBM quickstart

Canonical code: `fyp_data/dataset.py` in the GitHub checkout. Data stays in Drive.
First follow the clone/install steps in README.md. In Colab, clone the private
repository with your own authorized account, then change to the checkout and
install requirements.txt. Never save credentials in shared notebooks.
Canonical import: `from fyp_data.dataset import CUBDataset, load_cub_tables, build_concept_matrix, select_split`.
Do not use the dataset.py inside the recovered ZIP for new experiments.

Mount Drive in Colab and verify your path (shared folders may require a shortcut in My Drive). Access to this folder is required independently of GitHub membership.

```python
from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
import sys
ROOT = Path('/content/drive/MyDrive/CUB_Project_Data')
REPO = Path('/content/FYP-cbm-localization')  # adjust to your clone location
assert (REPO / 'fyp_data/dataset.py').is_file()
sys.path.insert(0, str(REPO))
import fyp_data.dataset as cub
print(cub.__file__)
assert Path(cub.__file__).resolve() == (REPO / 'fyp_data/dataset.py').resolve()
images, classes, attributes = cub.load_cub_tables(ROOT)
concepts = cub.build_concept_matrix(attributes, images.image_id)
train = cub.select_split(images, ROOT, policy='experiment', split='train')
val = cub.select_split(images, ROOT, policy='experiment', split='val')
test = cub.select_split(images, ROOT, policy='experiment', split='test')
assert (len(train), len(val), len(test)) == (5394, 600, 5794)
assert concepts.shape == (11788, 312)

# Dependency-light image/concept smoke test in the torch Dataset:
ds = cub.CUBDataset(train, ROOT, cub.make_default_transform(), concepts)
image, label, concept = ds[0]
assert tuple(image.shape) == (3,224,224)
assert tuple(concept.shape) == (312,)
assert 0 <= label.item() < 200
assert concept.isfinite().all() and ((concept == 0) | (concept == 1)).all()
print(image.shape, label.item(), concept.shape)
```

For actual CLIP use the preprocessing callable returned by your chosen CLIP implementation/checkpoint, replacing make_default_transform(). The basic transform above only resizes and converts to a tensor; it is not CLIP normalization. Record the checkpoint and preprocessing in each experiment. Batch size, workers and device belong in the training notebook. The initial concept matrix uses raw binary is_present without certainty masking.

For grounding:

```python
grounding_ds = cub.CUBDataset(train, ROOT, concept_matrix=concepts, grounding=True)
sample = grounding_ds[0]
assert tuple(sample['bbox'].shape) == (4,)
assert tuple(sample['parts'].shape) == (15, 2)
assert tuple(sample['parts_visible'].shape) == (15,)
assert sample['coordinate_space'] == 'original_pixels'
# image is PIL here. For training, implement joint_transform(sample) that
# changes image AND annotations consistently; see README coordinate caveats.
```

Legacy reproduction: use policy='official' explicitly (5,994 train, 5,794 test), with no val. The old root dataset.py is preserved; it does not forward to this module. Restart the Colab runtime after replacing imports to avoid Python's cached legacy modules. The new API deliberately rejects incomplete concept matrices and image-only transforms in grounding mode. Review the 606 unusual annotation records and scientific policy before final experiments.

Historical Drive checks are described in docs/DRIVE_CANONICAL_README.md.
This upload preparation did not repeat those data checks. Run this smoke test before training.

