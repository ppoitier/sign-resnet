"""Experiment configuration: pydantic schemas and YAML loading.

A YAML file can inherit from another one with a top-level ``from: <relative path>``
key. Values of the child file override the values of the parent (deep merge).
"""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict


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


class DataConfig(Schema):
    training: SplitConfig
    validation: SplitConfig
    testing: SplitConfig

    body_parts: list[str] = ["upper_pose", "left_hand", "right_hand"]
    # Each clip becomes a (3, n_landmarks, n_frames) image.
    n_frames: int = 64
    # Drop the z coordinate (the third image channel is then filled with zeros).
    drop_z: bool = False


class ModelConfig(Schema):
    n_layers: Literal[18, 34, 50, 101, 152] = 18
    # Initialize with ImageNet weights.
    pretrained: bool = False


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
