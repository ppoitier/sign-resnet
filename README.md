# Sign ResNet

A ResNet for **isolated sign language recognition** (ISLR), trained on pose images.

Each sign clip of poses is turned into an image: landmarks are rows, frames are
columns, and the x/y/z coordinates are the RGB channels. A standard 2D ResNet then
classifies this image.

```
poses (T, 65 landmarks, xyz) ──► normalization ──► [augmentations] ──► 64 frames ──► image (3, 65, 64) ──► ResNet ──► sign class
```

**Normalization** (`data.normalization`):
- `shoulders` (default): poses are scaled by the shoulder width and centered on the
  shoulders. The image contains these coordinates directly.
- `per-frame-minmax`: each coordinate is rescaled to [0, 255] with its min/max over the
  landmarks of each frame, as in slp (see `configs/lsfb/resnet18_slp.yaml`). Translation
  and scaling have no effect on these images.

**Augmentations** (`data.augmentation`, training set only):

| Option        | Effect                                                                         |
|---------------|--------------------------------------------------------------------------------|
| `flip`        | probability of mirroring the signer (left/right hands and body sides swapped) |
| `rotation`    | rotation around the shoulders, angle in [-rotation, rotation] (radians)        |
| `scale`       | scaling around the shoulders, factor in [1 - scale, 1 + scale]                 |
| `translation` | offset in [-translation, translation] shoulder widths, along x and y          |
| `noise`       | std of the Gaussian noise added to each coordinate (shoulder widths)           |
| `random_crop` | random window of 64 frames instead of the first ones (clips > 64 frames)       |

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
  optimizer:               # optional, default: AdamW with PyTorch defaults
    name: "adamw"          # "sgd", "adam" or "adamw"
    weight_decay: 0.01     # omit for the PyTorch default
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
├── data.py           datasets, dataloaders, class weights
├── transforms.py     pose-to-image transform, normalization and augmentations
├── model.py          ResNet classifier
├── trainer.py        Lightning module, metrics and learning rate schedule
└── utils.py          seeding, loggers
```
