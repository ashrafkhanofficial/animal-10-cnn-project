import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import os

import pandas as pd

from src.constants import (
    PLOTS_DIR,
    EXPERIMENTS_CSV,
)


def plot_history(history):
    """
    Plot training and validation loss and accuracy.

    Args:
        history (dict): Training history returned by train().

    Returns:
        None
    """

    # Loss
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_accuracy"], label="Train Accuracy")
    plt.plot(history["val_accuracy"], label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_confusion_matrix(cm, class_names):
    """
    Plot a confusion matrix.

    Args:
        cm (numpy.ndarray): Confusion matrix.
        class_names (list): Dataset class names.

    Returns:
        None
    """

    plt.figure(figsize=(8, 8))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names,
    )

    display.plot(
        cmap="Blues",
        values_format="d",
    )

    plt.title("Confusion Matrix")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_training_history(
    history,
    model_name,
):
    """
    Plot training and validation loss and accuracy curves.

    Args:
        history (dict): History returned by train().
        model_name (str): Name of the model.

    Returns:
        None
    """

    os.makedirs(
        PLOTS_DIR,
        exist_ok=True,
    )

    plt.figure(figsize=(12, 5))

    # ==========================
    # Loss
    # ==========================

    plt.subplot(1, 2, 1)

    plt.plot(
        history["train_loss"],
        label="Train Loss",
    )

    plt.plot(
        history["val_loss"],
        label="Validation Loss",
    )

    plt.title(f"{model_name} Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # ==========================
    # Accuracy
    # ==========================

    plt.subplot(1, 2, 2)

    plt.plot(
        history["train_accuracy"],
        label="Train Accuracy",
    )

    plt.plot(
        history["val_accuracy"],
        label="Validation Accuracy",
    )

    plt.title(f"{model_name} Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            f"{model_name.lower()}_history.png",
        )
    )

    plt.show()

    plt.close()

def plot_metric_comparison(
    csv_path=EXPERIMENTS_CSV,
):
    """
    Plot comparison of evaluation metrics for all models.

    Args:
        csv_path (str): Path to experiments CSV.

    Returns:
        None
    """

    os.makedirs(
        PLOTS_DIR,
        exist_ok=True,
    )

    df = pd.read_csv(csv_path)

    metrics = [
        "test_accuracy",
        "precision",
        "recall",
        "f1_score",
    ]

    titles = [
        "Test Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
    ]

    plt.figure(figsize=(14, 10))

    for i, (metric, title) in enumerate(
        zip(metrics, titles),
        start=1,
    ):

        plt.subplot(2, 2, i)

        plt.bar(
            df["model"],
            df[metric],
        )

        plt.title(title)
        plt.xlabel("Model")
        plt.ylabel(metric.replace("_", " ").title())
        plt.ylim(0, 1)
        plt.grid(axis="y")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            "metric_comparison.png",
        )
    )

    plt.show()

    plt.close()


def plot_training_time(
    csv_path=EXPERIMENTS_CSV,
):
    """
    Plot training time comparison for all models.

    Args:
        csv_path (str): Path to experiments CSV.

    Returns:
        None
    """

    os.makedirs(
        PLOTS_DIR,
        exist_ok=True,
    )

    df = pd.read_csv(csv_path)

    plt.figure(figsize=(8, 5))

    plt.bar(
        df["model"],
        df["training_time"],
    )

    plt.title("Training Time Comparison")
    plt.xlabel("Model")
    plt.ylabel("Training Time (seconds)")
    plt.grid(axis="y")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            "training_time_comparison.png",
        )
    )

    plt.show()

    plt.close()


def plot_parameter_comparison(
    csv_path=EXPERIMENTS_CSV,
):
    """
    Plot total and trainable parameter comparison for all models.

    Args:
        csv_path (str): Path to experiments CSV.

    Returns:
        None
    """

    os.makedirs(
        PLOTS_DIR,
        exist_ok=True,
    )

    df = pd.read_csv(csv_path)

    x = range(len(df))
    width = 0.35

    plt.figure(figsize=(10, 5))

    plt.bar(
        [i - width / 2 for i in x],
        df["total_params"],
        width=width,
        label="Total Parameters",
    )

    plt.bar(
        [i + width / 2 for i in x],
        df["trainable_params"],
        width=width,
        label="Trainable Parameters",
    )

    plt.xticks(
        x,
        df["model"],
    )

    plt.title("Model Parameter Comparison")
    plt.xlabel("Model")
    plt.ylabel("Number of Parameters")
    plt.legend()
    plt.grid(axis="y")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            "parameter_comparison.png",
        )
    )

    plt.show()

    plt.close()