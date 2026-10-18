"""Load a real image from each split without resizing its annotations."""
import argparse

from datasets.cub import CUBDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, help='CUB_200_2011 image root')
    parser.add_argument('--processed-dir', default='data/processed')
    args = parser.parse_args()
    for split in ('train', 'val', 'test'):
        dataset = CUBDataset(root=args.root, processed_dir=args.processed_dir, split=split)
        sample = dataset[0]
        assert sample['attributes'].shape == (312,)
        assert sample['bbox'].shape == (4,)
        assert sample['parts'].shape == (15, 2)
        assert sample['parts_visible'].shape == (15,)
        assert 0 <= sample['label'] < 200
        print(f'{split}: {len(dataset)} images; sample {sample["image_id"]} OK')


if __name__ == '__main__':
    main()
