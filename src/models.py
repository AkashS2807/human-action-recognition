import torch.nn as nn
from torchvision import models
import timm


class SimpleCNN(nn.Module):
    """
    Custom CNN for human action classification.
    """

    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def create_cnn(num_classes):
    return SimpleCNN(num_classes)


def create_resnet(num_classes, pretrained=True, freeze_backbone=True):
    """
    ResNet18 with optional pretrained weights and backbone freezing.
    """

    weights = (
        models.ResNet18_Weights.DEFAULT
        if pretrained
        else None
    )

    model = models.resnet18(weights=weights)

    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False

    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes
    )

    return model


def create_vit(
    num_classes,
    pretrained=True,
    freeze_backbone=True
):
    """
    Lightweight Vision Transformer for CPU-friendly training.

    Uses ViT-Tiny instead of ViT-Base to significantly reduce
    training time while retaining the Vision Transformer approach.
    """

    model = timm.create_model(
        "vit_tiny_patch16_224",
        pretrained=pretrained,
        num_classes=num_classes
    )

    if freeze_backbone:

        for parameter in model.parameters():
            parameter.requires_grad = False

        # Train only the classification head.
        if hasattr(model, "head"):
            for parameter in model.head.parameters():
                parameter.requires_grad = True

        if hasattr(model, "fc_norm"):
            for parameter in model.fc_norm.parameters():
                parameter.requires_grad = False

    return model


def create_model(
    model_name,
    num_classes,
    pretrained=True,
    freeze_backbone=True
):
    """
    Create one of the supported models.

    Supported:
        cnn
        resnet
        vit
    """

    model_name = model_name.lower()

    if model_name == "cnn":

        return create_cnn(num_classes)

    elif model_name == "resnet":

        return create_resnet(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone
        )

    elif model_name == "vit":

        return create_vit(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone
        )

    else:

        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Choose from: cnn, resnet, vit."
        )