import torch
import os
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)


def predict(
    model,
    dataloader,
    device,
):
    """
    Generate predictions for a dataset.

    Args:
        model (nn.Module): Trained model.
        dataloader (DataLoader): DataLoader for evaluation.
        device: CPU or GPU device.

    Returns:
        tuple:
            (
                y_true,
                y_pred,
            )
    """

    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():

        for images, labels in dataloader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            predictions = torch.argmax(
                outputs,
                dim=1,
            )

            y_true.extend(
                labels.cpu().numpy()
            )

            y_pred.extend(
                predictions.cpu().numpy()
            )

    return y_true, y_pred


def calculate_metrics(model, dataloader, device):
    """
    Calculate classification metrics for a trained model.

    Args:
        model (nn.Module): Trained model.
        dataloader (DataLoader): Evaluation dataloader.
        device: CPU or GPU.

    Returns:
        dict: Dictionary containing accuracy, precision,
        recall, and F1-score.
    """

    true_labels, predicted_labels = predict(
        model,
        dataloader,
        device,
    )

    metrics = {
        "accuracy": accuracy_score(true_labels, predicted_labels),
        "precision": precision_score(
            true_labels,
            predicted_labels,
            average="weighted",
            zero_division=0,
        ),
        "recall": recall_score(
            true_labels,
            predicted_labels,
            average="weighted",
            zero_division=0,
        ),
        "f1_score": f1_score(
            true_labels,
            predicted_labels,
            average="weighted",
            zero_division=0,
        ),
    }

    return metrics


def get_confusion_matrix(model, dataloader, device):
    """
    Compute the confusion matrix for a trained model.

    Args:
        model (nn.Module): Trained model.
        dataloader (DataLoader): Evaluation dataloader.
        device: CPU or GPU.

    Returns:
        numpy.ndarray: Confusion matrix.
    """

    true_labels, predicted_labels = predict(
        model,
        dataloader,
        device,
    )

    cm = confusion_matrix(
        true_labels,
        predicted_labels,
    )

    return cm


def plot_confusion_matrix(
    cm,
    class_names,
    save_path,
):
    """
    Plot and save the confusion matrix.

    Args:
        cm (numpy.ndarray): Confusion matrix.
        class_names (list): List of class names.
        save_path (str): Path to save the confusion matrix image.

    Returns:
        None
    """

    os.makedirs(
        os.path.dirname(save_path),
        exist_ok=True,
    )

    fig, ax = plt.subplots(
        figsize=(8, 8),
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names,
    )

    display.plot(
        ax=ax,
        cmap="Blues",
        colorbar=False,
    )

    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def evaluate_model(
    model,
    dataloader,
    class_names,
    device,
    save_path,
):
    """
    Evaluate a trained model and save the confusion matrix.

    Args:
        model (nn.Module): Trained model.
        dataloader (DataLoader): Evaluation dataloader.
        class_names (list): List of class names.
        device: CPU or GPU.
        save_path (str): Path to save the confusion matrix.

    Returns:
        dict: Dictionary containing evaluation metrics.
    """

    metrics = calculate_metrics(
        model,
        dataloader,
        device,
    )

    cm = get_confusion_matrix(
        model,
        dataloader,
        device,
    )

    plot_confusion_matrix(
        cm,
        class_names,
        save_path,
    )

    return metrics