import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def predict(model, dataloader, device):
    """
    Generate predictions for a dataset.

    Args:
        model (nn.Module): Trained model.
        dataloader (DataLoader): DataLoader for inference.
        device: CPU or GPU.

    Returns:
        tuple: (true_labels, predicted_labels)
    """

    model.eval()

    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predictions = torch.max(outputs, dim=1)

            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predictions.cpu().numpy())

    return true_labels, predicted_labels


def evaluate(model, dataloader, device):
    """
    Evaluate a trained model using classification metrics.

    Args:
        model (nn.Module): Trained model.
        dataloader (DataLoader): Evaluation dataloader.
        device: CPU or GPU.

    Returns:
        dict: Evaluation metrics.
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