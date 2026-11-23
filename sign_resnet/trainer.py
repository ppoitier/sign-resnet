import lightning as pl
import torch
from torch import nn, Tensor
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR, LinearLR, SequentialLR
from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy, F1Score

from sign_resnet.config import OptimizerConfig, SchedulerConfig


def build_optimizer(parameters, learning_rate: float, config: OptimizerConfig) -> torch.optim.Optimizer:
    kwargs = {"lr": learning_rate}
    if config.weight_decay is not None:
        kwargs["weight_decay"] = config.weight_decay
    if config.name == "sgd":
        return torch.optim.SGD(parameters, momentum=config.momentum, nesterov=config.nesterov, **kwargs)
    if config.name == "adam":
        return torch.optim.Adam(parameters, betas=config.betas, **kwargs)
    if config.name == "adamw":
        return torch.optim.AdamW(parameters, betas=config.betas, **kwargs)
    raise ValueError(f"Unknown optimizer: {config.name}")


class ClassificationMetrics(MetricCollection):
    def __init__(self, n_classes: int, **kwargs):
        args = dict(task="multiclass", num_classes=n_classes, ignore_index=-1)
        super().__init__(
            {
                "macro_accuracy": Accuracy(average="macro", **args),
                "micro_accuracy": Accuracy(average="micro", **args),
                "macro_accuracy/top5": Accuracy(top_k=5, average="macro", **args),
                "macro_accuracy/top10": Accuracy(top_k=10, average="macro", **args),
                "micro_accuracy/top5": Accuracy(top_k=5, average="micro", **args),
                "micro_accuracy/top10": Accuracy(top_k=10, average="micro", **args),
                "f1": F1Score(average="macro", **args),
            },
            **kwargs,
        )


class RecognitionModule(pl.LightningModule):
    """Trains and evaluates a classifier for isolated sign language recognition.

    Logs the cross-entropy loss and classification metrics for every split. During
    testing, the logits of every instance are kept in ``self.test_logits``.

    Expected batch (see data.collate): images (N, 3, H, W), labels (N,), ids (list).
    """

    def __init__(
        self,
        model: nn.Module,
        n_classes: int,
        learning_rate: float,
        class_weights: Tensor | None = None,
        optimizer: OptimizerConfig | None = None,
        lr_scheduler: SchedulerConfig | None = None,
        max_epochs: int | None = None,
    ):
        super().__init__()
        self.model = model
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.learning_rate = learning_rate
        self.optimizer_config = optimizer or OptimizerConfig()
        self.lr_scheduler = lr_scheduler
        self.max_epochs = max_epochs

        # ("training" cannot be a ModuleDict key: it clashes with nn.Module.training)
        self.metrics = nn.ModuleDict(
            {
                f"{split}_metrics": ClassificationMetrics(n_classes, prefix=f"{split}/")
                for split in ("training", "validation", "testing")
            }
        )
        self.test_logits: dict[str, object] = {}

    def forward(self, images: Tensor) -> Tensor:
        return self.model(images)

    def shared_step(self, batch: dict, split: str) -> tuple[Tensor, Tensor]:
        batch_size = batch["labels"].size(0)
        logits = self(batch["images"])

        loss = self.criterion(logits, batch["labels"])
        self.log(f"{split}/loss", loss, on_step=True, on_epoch=True, batch_size=batch_size)

        metrics = self.metrics[f"{split}_metrics"](logits.detach().softmax(dim=1), batch["labels"])
        self.log_dict(metrics, on_step=False, on_epoch=True, batch_size=batch_size)
        return loss, logits.detach()

    def training_step(self, batch, batch_idx):
        loss, _ = self.shared_step(batch, "training")
        return loss

    def validation_step(self, batch, batch_idx):
        self.shared_step(batch, "validation")

    def test_step(self, batch, batch_idx):
        _, logits = self.shared_step(batch, "testing")
        for instance_id, instance_logits in zip(batch["ids"], logits):
            self.test_logits[instance_id] = instance_logits.float().cpu().numpy().astype("float16")

    def configure_optimizers(self):
        optimizer = build_optimizer(self.model.parameters(), self.learning_rate, self.optimizer_config)
        if self.lr_scheduler is None:
            return optimizer
        config = self.lr_scheduler
        cosine_epochs = self.max_epochs - config.warmup_epochs - config.plateau_epochs
        scheduler = SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(optimizer, start_factor=config.start_lr / self.learning_rate, total_iters=config.warmup_epochs),
                ConstantLR(optimizer, factor=1.0, total_iters=config.plateau_epochs),
                CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=config.end_lr),
            ],
            milestones=[config.warmup_epochs, config.warmup_epochs + config.plateau_epochs],
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}


def load_model_weights(model: nn.Module, checkpoint_path: str):
    """Load the model weights from a Lightning checkpoint of a RecognitionModule."""
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)["state_dict"]
    model_state_dict = {
        key.removeprefix("model.").replace("_orig_mod.", ""): value  # "_orig_mod" comes from torch.compile
        for key, value in state_dict.items()
        if key.startswith("model.")
    }
    model.load_state_dict(model_state_dict)
