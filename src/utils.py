import yaml
import torch


def load_config(config_path):
    """
    Load a YAML configuration file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: Configuration dictionary.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def save_checkpoint(
    model,
    optimizer,
    epoch,
    history,
    best_val_accuracy,
    filepath,
):
    """
    Save a training checkpoint.

    Args:
        model (nn.Module): Model to save.
        optimizer (Optimizer): Optimizer associated with the model.
        epoch (int): Next epoch to resume from.
        history (dict): Training history.
        best_val_accuracy (float): Best validation accuracy achieved.
        filepath (str): Destination checkpoint path.

    Returns:
        None
    """

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
        "best_val_accuracy": best_val_accuracy,
    }

    torch.save(checkpoint, filepath)


def load_checkpoint(
    model,
    optimizer,
    filepath,
    device,
):
    """
    Load a training checkpoint.

    Args:
        model (nn.Module): Model architecture.
        optimizer (Optimizer): Optimizer associated with the model.
        filepath (str): Checkpoint path.
        device: CPU or GPU device.

    Returns:
        tuple:
            (
                model,
                optimizer,
                start_epoch,
                history,
                best_val_accuracy,
            )
    """

    checkpoint = torch.load(
        filepath,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    start_epoch = checkpoint["epoch"]
    history = checkpoint["history"]
    best_val_accuracy = checkpoint["best_val_accuracy"]

    return (
        model,
        optimizer,
        start_epoch,
        history,
        best_val_accuracy,
    )