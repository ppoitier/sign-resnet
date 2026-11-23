"""Pose transforms: from the body parts of a clip to the image given to the ResNet.

    poses {body_part: (T, L_i, 3)}
      -> [random horizontal flip]                     (training only)
      -> concatenate the body parts                   -> (T, L, 3)
      -> normalization (see below)
      -> [random rotation, scale, translation, noise] (training only)
      -> keep n_frames frames (first ones, or a random window during training),
         resample to n_frames
      -> image (3, L, n_frames): landmarks as rows, frames as columns, x/y/z as channels

Normalization:
    "shoulders":        poses are scaled by the mean shoulder width and centered on the
                        shoulders, frame by frame. Coordinates are in shoulder widths.
    "per-frame-minmax": each coordinate is rescaled to [0, 255] with its min/max over the
                        landmarks of each frame (as in slp). This cancels any translation
                        or scaling, so these two augmentations cannot be used with it.
"""

import random

import numpy as np
from sign_language_tools.common.transforms import Compose
from sign_language_tools.pose.transforms import (
    CenterOnLandmarks,
    Concatenate,
    DropCoordinates,
    GaussianNoise,
    NormalizeByReferenceEdge,
    RandomRotation2D,
    RandomScale,
    RandomTranslation,
    Resample,
    TemporalCrop,
    TemporalRandomCrop,
    ToRGBImage,
)

from sign_resnet.config import AugmentationConfig, DataConfig

# Shoulders in the concatenated landmarks (MediaPipe pose indices, "upper_pose" first).
SHOULDERS = (11, 12)

# MediaPipe pose landmarks 0-22 ("upper_pose") that are swapped by a horizontal flip:
# eyes, ears, mouth corners, shoulders, elbows, wrists and fingers.
_UPPER_POSE_MIRROR_PAIRS = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12),
                            (13, 14), (15, 16), (17, 18), (19, 20), (21, 22)]
_UPPER_POSE_MIRROR = list(range(23))
for _a, _b in _UPPER_POSE_MIRROR_PAIRS:
    _UPPER_POSE_MIRROR[_a], _UPPER_POSE_MIRROR[_b] = _b, _a


def _mirrored_name(body_part: str) -> str:
    if "left" in body_part:
        return body_part.replace("left", "right")
    if "right" in body_part:
        return body_part.replace("right", "left")
    return body_part


class RandomHorizontalFlip:
    """Mirrors the signer with probability ``p``: x -> 1 - x, and left/right are swapped
    (left_* <-> right_* body parts, and the left/right landmarks of upper_pose), so that
    the left hand of the flipped signer is still in the "left_hand" rows.

    Other body parts (e.g. lips) are mirrored but their landmarks are not reordered.
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, poses: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if random.random() >= self.p:
            return poses
        flipped = {}
        for body_part, landmarks in poses.items():
            landmarks = landmarks.copy()
            landmarks[..., 0] = 1.0 - landmarks[..., 0]
            if body_part == "upper_pose":
                landmarks = landmarks[:, _UPPER_POSE_MIRROR]
            target = _mirrored_name(body_part)
            flipped[target if target in poses else body_part] = landmarks
        return flipped


def build_pose_transform(config: DataConfig, augment: bool = False):
    """Poses (dict of body parts) -> image (3, n_landmarks, n_frames).

    Random augmentations (``config.augmentation``) are only used if ``augment`` is True.
    """
    augmentation: AugmentationConfig | None = config.augmentation if augment else None
    shoulders_normalization = config.normalization == "shoulders"
    if shoulders_normalization and config.body_parts[0] != "upper_pose":
        raise ValueError("The 'shoulders' normalization requires 'upper_pose' as first body part.")

    transforms = []
    if augmentation is not None and augmentation.flip > 0:
        transforms.append(RandomHorizontalFlip(augmentation.flip))

    transforms.append(Concatenate(config.body_parts))
    if config.drop_z:
        transforms.append(DropCoordinates("z"))
    if shoulders_normalization:
        transforms += [NormalizeByReferenceEdge(ref_edge=SHOULDERS), CenterOnLandmarks(SHOULDERS)]

    if augmentation is not None:
        # The poses are centered on the shoulders: rotations and scaling are around them.
        if augmentation.rotation > 0:
            angle = augmentation.rotation
            transforms.append(RandomRotation2D(angle_range=(-angle, angle), center=(0.0, 0.0)))
        if augmentation.scale > 0:
            transforms.append(RandomScale(1 - augmentation.scale, 1 + augmentation.scale, center=(0.0, 0.0)))
        if augmentation.translation > 0:
            offset = (-augmentation.translation, augmentation.translation)
            transforms.append(RandomTranslation(dx_range=offset, dy_range=offset))
        if augmentation.noise > 0:
            transforms.append(GaussianNoise(augmentation.noise))

    random_crop = augmentation is not None and augmentation.random_crop
    transforms += [
        TemporalRandomCrop(config.n_frames) if random_crop else TemporalCrop(config.n_frames),
        Resample(config.n_frames, "nearest"),
        ToRGBImage(normalize=not shoulders_normalization),
    ]
    return Compose(transforms)
