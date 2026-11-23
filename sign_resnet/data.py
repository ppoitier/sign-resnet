"""Isolated sign datasets: each clip of poses is turned into an image for the ResNet
(see sign_resnet/transforms.py). Random augmentations are only applied to the training set.
"""

from collections import Counter

import torch
from sldl import SignLanguageDataset
from sldl.utils.weighting import compute_class_weights
from torch.utils.data import DataLoader

from sign_resnet.config import DataConfig, SplitConfig
from sign_resnet.transforms import build_pose_transform

SPLITS = ("training", "validation", "testing")


def collate(samples: list[dict]) -> dict:
    """images (N, 3, L, n_frames) float, labels (N,), ids (list of N str)."""
    return {
        "images": torch.stack([torch.as_tensor(s["poses"]) for s in samples]).float(),
        "labels": torch.tensor([s["label_id"] for s in samples], dtype=torch.long),
        "ids": [s["id"] for s in samples],
    }


def build_dataset(split: SplitConfig, data: DataConfig, augment: bool = False) -> SignLanguageDataset:
    return SignLanguageDataset(
        shards_url=split.shards_url,
        isolated=True,
        body_parts=tuple(data.body_parts),
        pose_transform=build_pose_transform(data, augment=augment),
    )


def load_datasets_and_loaders(
    config: DataConfig,
) -> tuple[dict[str, SignLanguageDataset], dict[str, DataLoader]]:
    """Load the training, validation and testing sets."""
    datasets, loaders = {}, {}
    for name in SPLITS:
        split: SplitConfig = getattr(config, name)
        datasets[name] = build_dataset(split, config, augment=name == "training")
        loaders[name] = DataLoader(
            datasets[name],
            batch_size=split.batch_size,
            shuffle=split.shuffle,
            num_workers=split.n_workers,
            pin_memory=True,
            collate_fn=collate,
        )
    return datasets, loaders


def compute_label_weights(dataset: SignLanguageDataset, n_classes: int, strategy: str) -> torch.Tensor:
    """Per-class weights (n_classes,) from the label frequencies of a dataset."""
    occurrences = Counter(sample["label_id"] for sample in dataset.samples)
    weights = compute_class_weights(occurrences, strategy=strategy)
    return torch.tensor([weights[c] for c in range(n_classes)], dtype=torch.float32)
