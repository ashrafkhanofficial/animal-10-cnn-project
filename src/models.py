import torch.nn as nn

from src.constants import NUM_CLASSES


class ANNClassifier(nn.Module):
    """
    Fully Connected Artificial Neural Network for image classification.
    """

    def __init__(self, config):
        super().__init__()

        input_dim = 3 * config["image_size"] * config["image_size"]

        layers = []
        in_features = input_dim

        activation = self._get_activation(config["activation"])

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

    def _get_activation(self, activation_name):
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

    def forward(self, x):
        x = self.flatten(x)
        x = self.features(x)
        x = self.classifier(x)
        return x