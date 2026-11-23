"""Isolated sign datasets: each clip of poses is turned into an image for the ResNet.

    poses {body_part: (T, L_i, 3)}  ->  concatenate  ->  (T, L, 3)
                                    ->  keep the first n_frames, resample to n_frames
                                    ->  image (3, L, n_frames): landmarks as rows, frames as
                                        columns, x/y/z as RGB channels (scaled to [0, 255])
"""

from collections import Counter

import torch
from sldl import SignLanguageDataset
from sldl.utils.weighting import compute_class_weights
from sign_language_tools.common.transforms import Compose
from sign_language_tools.pose.transforms import (
    Concatenate,
    DropCoordinates,
    Resample,
    TemporalCrop,
    ToRGBImage,
)
from torch.utils.data import DataLoader

from sign_resnet.config import DataConfig, SplitConfig

SPLITS = ("training", "validation", "testing")


def build_pose_to_image_transform(body_parts: list[str], n_frames: int = 64, drop_z: bool = False):
    transforms = [Concatenate(body_parts)]
    if drop_z:
        transforms.append(DropCoordinates("z"))
    transforms += [TemporalCrop(n_frames), Resample(n_frames, "nearest"), ToRGBImage()]
    return Compose(transforms)


def collate(samples: list[dict]) -> dict:
    """images (N, 3, L, n_frames) float in [0, 255], labels (N,), ids (list of N str)."""
    return {
        "images": torch.stack([torch.as_tensor(s["poses"]) for s in samples]).float(),
        "labels": torch.tensor([s["label_id"] for s in samples], dtype=torch.long),
        "ids": [s["id"] for s in samples],
    }


def build_dataset(split: SplitConfig, data: DataConfig) -> SignLanguageDataset:
    return SignLanguageDataset(
        shards_url=split.shards_url,
        isolated=True,
        body_parts=tuple(data.body_parts),
        pose_transform=build_pose_to_image_transform(data.body_parts, data.n_frames, data.drop_z),
    )


def load_datasets_and_loaders(
    config: DataConfig,
) -> tuple[dict[str, SignLanguageDataset], dict[str, DataLoader]]:
    """Load the training, validation and testing sets."""
    datasets, loaders = {}, {}
    for name in SPLITS:
        split: SplitConfig = getattr(config, name)
        datasets[name] = build_dataset(split, config)
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
