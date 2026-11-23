from torch import nn
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    ResNet152_Weights,
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
)

RESNETS = {
    18: (resnet18, ResNet18_Weights.IMAGENET1K_V1),
    34: (resnet34, ResNet34_Weights.IMAGENET1K_V1),
    50: (resnet50, ResNet50_Weights.IMAGENET1K_V2),
    101: (resnet101, ResNet101_Weights.IMAGENET1K_V2),
    152: (resnet152, ResNet152_Weights.IMAGENET1K_V2),
}


def build_resnet(n_layers: int, n_classes: int, pretrained: bool = False) -> nn.Module:
    """torchvision ResNet whose last layer is replaced by a linear classifier.

    Input:  images (N, 3, H, W) (here H = landmarks, W = frames).
    Output: logits (N, n_classes).
    """
    factory, weights = RESNETS[n_layers]
    model = factory(weights=weights if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model
