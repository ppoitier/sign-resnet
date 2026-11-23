import os
import random
from typing import Literal

import numpy as np
import torch
from lightning.pytorch.loggers import CSVLogger, Logger, MLFlowLogger, TensorBoardLogger


def set_seed(seed: int | Literal["random"]) -> int:
    """Seed Python, NumPy and PyTorch. Returns the seed (drawn at random if "random")."""
    if seed == "random":
        seed = random.randint(0, 1_000_000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def build_loggers(logs_dir: str, experiment_id: str, mlflow_uri: str | None = None) -> list[Logger]:
    os.makedirs(logs_dir, exist_ok=True)
    loggers: list[Logger] = [
        TensorBoardLogger(name="tb", save_dir=logs_dir),
        CSVLogger(name="csv", save_dir=logs_dir),
    ]
    if mlflow_uri is not None:
        loggers.append(MLFlowLogger(experiment_name=experiment_id, tracking_uri=mlflow_uri))
    return loggers


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
