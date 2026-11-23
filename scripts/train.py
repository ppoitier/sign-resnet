"""Train and test a ResNet for isolated sign language recognition.

Usage:
    python scripts/train.py -c configs/lsfb/resnet18.yaml
"""

from pathlib import Path
from pprint import pprint

import click
import lightning as pl
import torch
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

from sign_resnet.config import load_config
from sign_resnet.data import compute_label_weights, load_datasets_and_loaders
from sign_resnet.model import build_resnet
from sign_resnet.trainer import RecognitionModule, load_model_weights
from sign_resnet.utils import build_loggers, count_parameters, set_seed

torch.set_float32_matmul_precision("medium")


@click.command()
@click.option("--config-path", "-c", required=True, type=click.Path(exists=True, dir_okay=False))
def main(config_path: str):
    config = load_config(config_path)
    pprint(config.model_dump())
    experiment, training = config.experiment, config.training

    seed = set_seed(experiment.seed)
    print("Seed:", seed)

    # --- Data
    datasets, loaders = load_datasets_and_loaders(config.data)
    class_weights = None
    if training.class_weights is not None:
        print(f"Computing class weights ({training.class_weights})...")
        class_weights = compute_label_weights(datasets["training"], training.n_classes, training.class_weights)

    # --- Model
    model = build_resnet(config.model.n_layers, training.n_classes, config.model.pretrained)
    print(f"ResNet-{config.model.n_layers}, number of parameters: {count_parameters(model):,}")
    if training.checkpoint_path is not None:
        print("Loading weights from:", training.checkpoint_path)
        load_model_weights(model, training.checkpoint_path)
    if experiment.compile:
        model = torch.compile(model)

    module = RecognitionModule(
        model=model,
        n_classes=training.n_classes,
        learning_rate=training.learning_rate,
        class_weights=class_weights,
        lr_scheduler=training.lr_scheduler,
        max_epochs=training.max_epochs,
    )

    run_path = f"{experiment.id}/{experiment.variant}/{seed}"
    output_dir = Path(experiment.output_dir)
    loggers = build_loggers(str(output_dir / "logs" / run_path), experiment.id, experiment.mlflow_uri)

    # --- Training
    best_checkpoint_path = None
    if not training.skip_training:
        checkpoint = ModelCheckpoint(
            dirpath=output_dir / "checkpoints" / run_path,
            monitor="validation/loss",
            save_top_k=1,
            save_last=True,
        )
        trainer = pl.Trainer(
            max_epochs=training.max_epochs,
            gradient_clip_val=training.gradient_clipping,
            logger=loggers,
            callbacks=[
                checkpoint,
                EarlyStopping(monitor="validation/loss", patience=training.early_stopping_patience),
                LearningRateMonitor(logging_interval="epoch"),
            ],
            enable_progress_bar=experiment.show_progress_bar,
            fast_dev_run=experiment.debug,
        )
        trainer.fit(module, train_dataloaders=loaders["training"], val_dataloaders=loaders["validation"])
        best_checkpoint_path = None if experiment.debug else checkpoint.best_model_path

    # --- Testing (with the best checkpoint, or the current weights if training was skipped)
    tester = pl.Trainer(
        logger=loggers,
        enable_progress_bar=experiment.show_progress_bar,
        fast_dev_run=experiment.debug,
    )
    tester.test(module, dataloaders=loaders["testing"], ckpt_path=best_checkpoint_path, weights_only=False)

    logits_dir = output_dir / "logits" / run_path
    logits_dir.mkdir(parents=True, exist_ok=True)
    torch.save(module.test_logits, logits_dir / "logits.pt")


if __name__ == "__main__":
    main()
