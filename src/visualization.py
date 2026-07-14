import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


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