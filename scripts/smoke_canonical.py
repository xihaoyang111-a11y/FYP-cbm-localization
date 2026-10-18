"""Check canonical loading using real data; run from the repository root."""
import argparse
from fyp_data.dataset import CUBDataset, load_cub_tables, build_concept_matrix, select_split, make_default_transform


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Project data folder containing processed and CUB_200_2011")
    args = parser.parse_args()
    images, _, attributes = load_cub_tables(args.root)
    concepts = build_concept_matrix(attributes, images.image_id)
    assert concepts.shape == (11788, 312)
    seen = set()
    for split, count in (("train", 5394), ("val", 600), ("test", 5794)):
        frame = select_split(images, args.root, policy="experiment", split=split)
        assert len(frame) == count
        assert not seen.intersection(frame.image_id)
        seen.update(frame.image_id)
        ds = CUBDataset(frame, args.root, make_default_transform(), concepts)
        image, label, concept = ds[0]
        assert tuple(image.shape) == (3, 224, 224)
        assert 0 <= label.item() < 200
        assert tuple(concept.shape) == (312,)
        assert concept.isfinite().all() and ((concept == 0) | (concept == 1)).all()
        sample = CUBDataset(frame, args.root, concept_matrix=concepts, grounding=True)[0]
        assert tuple(sample["bbox"].shape) == (4,)
        assert tuple(sample["parts"].shape) == (15, 2)
        assert tuple(sample["parts_visible"].shape) == (15,)
        assert sample["coordinate_space"] == "original_pixels"
        print(f"{split}: {len(ds)} images; tuple and grounding samples OK")


if __name__ == "__main__":
    main()
