"""Experiment configuration: pydantic schemas and YAML loading.

A YAML file can inherit from another one with a top-level ``from: <relative path>``
key. Values of the child file override the values of the parent (deep merge).
"""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, model_validator


class Schema(BaseModel):
    # Unknown keys raise an error: typos in a YAML file are caught early.
    model_config = ConfigDict(extra="forbid")


class ExperimentConfig(Schema):
    id: str
    variant: str = "default"
    output_dir: str
    seed: int | Literal["random"] = "random"
    compile: bool = False
    debug: bool = False
    show_progress_bar: bool = False
    mlflow_uri: str | None = None


class SplitConfig(Schema):
    shards_url: str
    batch_size: int = 256
    shuffle: bool = False
    n_workers: int = 0


class AugmentationConfig(Schema):
    """Random augmentations of the training clips. 0 disables an augmentation."""

    # Probability of mirroring the signer (left and right are swapped).
    flip: float = 0.0
    # Rotation by an angle drawn in [-rotation, rotation] (radians), around the shoulders.
    rotation: float = 0.0
    # Scaling by a factor drawn in [1 - scale, 1 + scale], around the shoulders.
    scale: float = 0.0
    # Translation by offsets drawn in [-translation, translation] (shoulder widths).
    translation: float = 0.0
    # Standard deviation of the Gaussian noise added to each coordinate (shoulder widths).
    noise: float = 0.0
    # Crop a random window of n_frames frames (instead of the first ones) in long clips.
    random_crop: bool = False


class DataConfig(Schema):
    training: SplitConfig
    validation: SplitConfig
    testing: SplitConfig

    body_parts: list[str] = ["upper_pose", "left_hand", "right_hand"]
    # Each clip becomes a (3, n_landmarks, n_frames) image.
    n_frames: int = 64
    # Drop the z coordinate (the third image channel is then filled with zeros).
    drop_z: bool = False
    # "shoulders" or "per-frame-minmax" (as in slp), see sign_resnet/transforms.py.
    normalization: Literal["shoulders", "per-frame-minmax"] = "shoulders"
    # Random augmentations of the training set (None = no augmentation).
    augmentation: AugmentationConfig | None = None

    @model_validator(mode="after")
    def check_augmentation(self):
        aug = self.augmentation
        if self.normalization == "per-frame-minmax" and aug and (aug.scale > 0 or aug.translation > 0):
            raise ValueError(
                "Scale and translation augmentations have no effect with the 'per-frame-minmax' "
                "normalization. Use the 'shoulders' normalization or set them to 0."
            )
        return self


class ModelConfig(Schema):
    n_layers: Literal[18, 34, 50, 101, 152] = 18
    # Initialize with ImageNet weights.
    pretrained: bool = False


class OptimizerConfig(Schema):
    """Optimizer (the learning rate is training.learning_rate)."""

    name: Literal["sgd", "adam", "adamw"] = "adamw"
    # None = PyTorch default (0 for SGD and Adam, 0.01 for AdamW).
    weight_decay: float | None = None
    # SGD only.
    momentum: float = 0.9
    nesterov: bool = False
    # Adam and AdamW only.
    betas: tuple[float, float] = (0.9, 0.999)

    @model_validator(mode="after")
    def check_options(self):
        if self.nesterov and self.name != "sgd":
            raise ValueError("'nesterov' is only available with the 'sgd' optimizer.")
        if self.nesterov and self.momentum <= 0:
            raise ValueError("'nesterov' requires a momentum > 0.")
        return self


class SchedulerConfig(Schema):
    """Linear warmup (start_lr -> learning_rate), plateau, then cosine annealing to end_lr.
    The learning rate is updated every epoch."""

    warmup_epochs: int = 10
    plateau_epochs: int = 0
    start_lr: float = 1e-6
    end_lr: float = 1e-6


class TrainingConfig(Schema):
    n_classes: int
    learning_rate: float = 1e-4
    max_epochs: int = 150
    early_stopping_patience: int = 10
    gradient_clipping: float | None = None
    optimizer: OptimizerConfig = OptimizerConfig()
    lr_scheduler: SchedulerConfig | None = None
    # Class weighting strategy (see sldl.utils.weighting), e.g. "inverse". None = no weights.
    class_weights: str | None = None

    # Initialize the model with the weights of a checkpoint (e.g. for fine-tuning).
    checkpoint_path: str | None = None
    # Only evaluate the model (requires checkpoint_path).
    skip_training: bool = False


class Config(Schema):
    experiment: ExperimentConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig


def _deep_merge(source: dict, destination: dict) -> dict:
    """Recursively merge ``source`` into ``destination`` (source wins)."""
    for key, value in source.items():
        if isinstance(value, dict):
            _deep_merge(value, destination.setdefault(key, {}))
        else:
            destination[key] = value
    return destination


def _load_yaml(path: Path) -> dict:
    with open(path, "r") as file:
        config = yaml.safe_load(file)
    if "from" in config:
        parent = _load_yaml((path.parent / config.pop("from")).resolve())
        config = _deep_merge(config, parent)
    return config


def load_config(path: str | Path) -> Config:
    return Config.model_validate(_load_yaml(Path(path)))
