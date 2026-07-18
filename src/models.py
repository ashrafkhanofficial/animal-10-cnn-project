import torch.nn as nn
from src.constants import NUM_CLASSES
from torchvision.models import (
    resnet18,
    ResNet18_Weights,

    efficientnet_b0,
    EfficientNet_B0_Weights,
)

def get_activation(activation_name):
        """
        Return the activation function specified in the configuration.
        """

        activations = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
        }

        if activation_name not in activations:
            raise ValueError(f"Unsupported activation: {activation_name}")

        return activations[activation_name]


class ANNClassifier(nn.Module):
    """
    Fully Connected Artificial Neural Network for image classification.
    """

    def __init__(self, config):
        super().__init__()

        input_dim = 3 * config["image_size"] * config["image_size"]

        layers = []
        in_features = input_dim

        activation = get_activation(config["activation"])
        for hidden_dim in config["hidden_dims"]:
            layers.append(nn.Linear(in_features, hidden_dim))

            if config["use_batchnorm"]:
                layers.append(nn.BatchNorm1d(hidden_dim))

            layers.append(activation)

            if config["dropout"] > 0:
                layers.append(nn.Dropout(config["dropout"]))

            in_features = hidden_dim

        self.features = nn.Sequential(*layers)

        self.classifier = nn.Linear(in_features, NUM_CLASSES)

        self.flatten = nn.Flatten()


    def forward(self, x):
        x = self.flatten(x)
        x = self.features(x)
        x = self.classifier(x)
        return x
    

class CNNClassifier(nn.Module):
    """
    Convolutional Neural Network for image classification.
    """

    def __init__(self, config):
        super().__init__()

        channels = config["conv_channels"]

        kernel_size = config["kernel_size"]

        padding = kernel_size // 2

        activation = get_activation(config["activation"])

        layers = []

        in_channels = 3

        for out_channels in channels:

            layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                )
            )

            if config["use_batchnorm"]:
                layers.append(
                    nn.BatchNorm2d(out_channels)
                )

            layers.append(activation)

            layers.append(
                nn.MaxPool2d(
                    kernel_size=2,
                    stride=2,
                )
            )

            in_channels = out_channels

            self.features = nn.Sequential(*layers)

            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Dropout(config["dropout"]),
                nn.Linear(
                    in_features=channels[-1],
                    out_features=NUM_CLASSES,
                ),
            )


    def forward(self, x):
        """
        Forward pass.
        """

        x = self.features(x)
        x = self.classifier(x)

        return x


class ResNet18Classifier(nn.Module):
    """
    ResNet18 model for image classification using transfer learning.
    """

    def __init__(self, config):
        super().__init__()

        self.model = resnet18(
        weights=(
            ResNet18_Weights.DEFAULT
            if config["pretrained"]
            else None
        )
        )

        if config["freeze_backbone"]:
            for param in self.model.parameters():
                param.requires_grad = False

        self.model.fc = nn.Linear(
        in_features=self.model.fc.in_features,
        out_features=NUM_CLASSES,
        )

    def forward(self, x):
        """
        Forward pass.
        """

        return self.model(x)
    

class EfficientNetB0Classifier(nn.Module):
    """
    EfficientNet-B0 model for image classification using transfer learning.
    """

    def __init__(self, config):
        super().__init__()

        self.model = efficientnet_b0(
            weights=(
                EfficientNet_B0_Weights.DEFAULT
                if config["pretrained"]
                else None
            )
        )

        if config["freeze_backbone"]:
            for param in self.model.parameters():
                param.requires_grad = False