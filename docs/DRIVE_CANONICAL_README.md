# CUB FYP canonical dataset layout — 2026-10-18

Start with MEMBER2_QUICKSTART.md. Canonical import: `from fyp_data.dataset import CUBDataset, load_cub_tables, build_concept_matrix, select_split`.

```text
CUB_Project_Data/
  README.md                       this index
  MEMBER2_QUICKSTART.md            Colab setup and split policy
  VERIFICATION.json               measured schemas/counts and test limits
  fyp_data/
    __init__.py
    dataset.py                    canonical unified implementation
  CUB_200_2011/                   preserved original images/annotations
  processed/                      preserved CSVs; shared source for all loaders
  dataset.py                      legacy tuple API; preserved for existing notebooks
  CUB_FYP_delivery_2026-10-18/      recovered ZIP and previous delivery notes
    loader_snapshots_2026-10-18/   both inspected original loaders
```

This is an additive organization: existing directories and legacy import paths are preserved. New work uses `fyp_data.dataset`; `from dataset import ...` still selects the old root loader. The ZIP is an immutable historical handoff, not the current code entrypoint. Earlier README files are historical and are superseded by this root index. No original data, notes, or loaders were deleted or overwritten.

## API and intentional differences

The canonical implementation extends the root loader's API: `load_cub_tables(root)` returns `(images, classes, attributes)`; `build_concept_matrix(attributes, image_ids=None)` returns a DataFrame indexed by image_id with columns attribute_1 through attribute_312; `CUBDataset(dataframe, root, transform, concept_matrix)` returns `(image, class_label, concept_label)` when a matrix is supplied, otherwise `(image, class_label)`. Class labels are torch.long 0..199; concepts are torch.float32 [312]. Without a transform the image is RGB PIL, despite the old root docstring's tensor wording. Pass CLIP's own preprocessing for CLIP training.

New keyword `grounding=True` changes the sample explicitly to a dictionary: image, label, attributes, bbox [4], parts [15,2], parts_visible [15], image_id, split, original_size (width,height), coordinate_space. It requires a concept matrix. Do not unpack it as the tuple API. It rejects image-only transforms; optional `joint_transform(sample)` must update image, coordinates, visibility and coordinate_space together. Geometry is not transformed automatically.

Unlike the historical ZIP, image_path absolute strings are ignored: all images resolve as root/CUB_200_2011/images/relative_path. `root` means the project folder, not CUB_200_2011. The ZIP's old root default and inner-merge split behavior are not retained. Missing split assignments, mismatched IDs/classes/official split, duplicate concept pairs, missing concepts, nonbinary values and wrong concept ID sets fail explicitly. No automatic label imputation or random splitting occurs.

## Split policy

Every new experiment calls `select_split(images, root, policy=..., split=...)` explicitly. Recommended Member 2 policy is `experiment`: frozen image_splits.csv, train 5,394 / val 600 / test 5,794. Validation is drawn only from the official training set; official test membership is unchanged. CSV assignments are authoritative; the original sampling seed/procedure has not been reconstructed. Keep the validation set for model selection; evaluate final results on test. Report the policy in each run and never merge validation into training silently.

For reproducing older root-loader results use `official`: train 5,994 / test 5,794. There is no official validation subset; requesting val fails. Existing root notebooks continue to follow their own explicit filtering and have not been migrated.

## Annotation caveats

The 3,677,856 records cover every image/attribute pair with binary is_present and IDs 1..312. Concepts use the stored is_present values directly, without certainty filtering, no_anno filtering, thresholding or class-level aggregation. Definitions are ordered by attribute_id, never alphabetically. certainty contains values 1..4 and is retained in the raw table; a certainty mask would be a separately documented experimental choice.

606 processed records have no_anno=0; the other 3,677,250 have no_anno missing. The recovered README reports unusual six-field source rows. This work confirms their processed representation only; the extra-field semantics and preprocessing correctness against original raw rows remain unverified. Do not interpret missing no_anno as zero or use this field as a confidence/visibility label. Retain source data for audit before final scientific claims.

Bounding boxes are original pixel (x,y,width,height); parts are original pixel (x,y), sorted part_id 1..15. Invisible parts may have placeholder (0,0); mask them. No one-based/zero-based coordinate-origin conversion is applied; confirm the source convention before pixel-sensitive evaluation. CLIP resize/center-crop and augmentations invalidate original coordinates unless the identical geometry is applied to boxes/parts. Crops require offsets/clipping and visibility updates; horizontal flips require coordinate and left/right part-ID handling. CLIP image normalization changes intensities, not geometry. Do not treat original coordinates as coordinates on a 224x224 tensor.

## Verification and limits

Both loaders were read. All recovered CSV schemas and counts were inspected. Drive images.csv and attributes.csv.gz were downloaded and SHA256 matched the ZIP byte-for-byte; other Drive CSVs were listed but not byte-compared. Full concept completeness, binary labels, duplicate pairs, split coverage/classes/test isolation, box uniqueness and 15 part IDs per image were checked on recovered metadata. See VERIFICATION.json for exact schemas and measured results.

One actual Drive JPEG (image_id 259, class label 4) decoded as RGB 271x404 and matched a float32 312-vector with 25 positives. This was a PIL/pandas check, not execution of the torch Dataset. Canonical source compilation and the actual pure matrix/split functions were tested separately. PyTorch is not installed in this local runtime, so __getitem__, DataLoader, CLIP preprocessing, model training and joint geometry transforms were not run. All 11,788 image files were not scanned. No claim is made about Member 2's access or private GitHub permissions. Run the quickstart smoke test in Colab before training.

