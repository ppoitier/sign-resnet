# Sign ResNet

A ResNet for **isolated sign language recognition** (ISLR), trained on pose images.

Each sign clip of poses is turned into an image: landmarks are rows, frames are
columns, and the x/y/z coordinates are the RGB channels. A standard 2D ResNet then
classifies this image.

```
poses (T, 65 landmarks, xyz) ──► first 64 frames, resampled to 64 ──► image (3, 65, 64) ──► ResNet ──► sign class
```

## Installation

Python >= 3.10. Install `sign-language-tools` and
`sign-language-data-loading` (`sldl`), then:

```bash
pip install -e .
```

## Usage

Update the dataset paths and `output_dir` in `configs/lsfb/data.yaml`, then:

```bash
python scripts/train.py -c configs/lsfb/resnet18.yaml
```

The script trains the model (early stopping on the validation loss), tests the best
checkpoint and writes:

```
<output_dir>/checkpoints/<id>/<variant>/<seed>/       best and last checkpoints
<output_dir>/logs/<id>/<variant>/<seed>/              TensorBoard and CSV logs (+ MLflow if mlflow_uri is set)
<output_dir>/logits/<id>/<variant>/<seed>/logits.pt   test logits of every clip, by clip id
```

## Configurations

A config file can inherit from another with `from: <path>`; its values override the
parent ones. Example (`configs/lsfb/resnet50_pretrained.yaml`):

```yaml
from: "data.yaml"          # experiment + data settings

experiment:
  variant: "resnet50_pretrained"

model:
  n_layers: 50             # 18, 34, 50, 101 or 152
  pretrained: true         # ImageNet weights

training:
  n_classes: 500
  learning_rate: 1.0e-4
  max_epochs: 150
  lr_scheduler:            # optional: linear warmup, plateau, cosine annealing
    warmup_epochs: 10
```

All the options and their default values are in [sign_resnet/config.py](sign_resnet/config.py).

## Metrics

Macro and micro accuracy (top-1, top-5, top-10) and macro F1, for every split.

## Code structure

```
scripts/train.py      entry point: data -> model -> training -> testing
sign_resnet/
├── config.py         configuration schema and YAML loading
├── data.py           datasets, pose-to-image transform, dataloaders, class weights
├── model.py          ResNet classifier
├── trainer.py        Lightning module, metrics and learning rate schedule
└── utils.py          seeding, loggers
```
