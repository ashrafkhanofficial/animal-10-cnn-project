import torch
from tqdm.notebook import tqdm
from src.utils import save_checkpoint
import os



def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train the model for one epoch.

    Args:
        model (nn.Module): Model to train.
        dataloader (DataLoader): Training dataloader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: CPU or GPU.

    Returns:
        tuple: (average_loss, accuracy)
    """

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(
    dataloader,
    desc="Training",
    leave=False
    )

    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)
        

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, predicted = torch.max(outputs, dim=1)

        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        progress_bar.set_postfix(
        loss=f"{loss.item():.4f}",
        acc=f"{100 * correct / total:.2f}%"
        )

    average_loss = running_loss / total
    accuracy = correct / total
   

    return average_loss, accuracy


def validate_one_epoch(model, dataloader, criterion, device):
    """
    Evaluate the model for one epoch.

    Args:
        model (nn.Module): Model to evaluate.
        dataloader (DataLoader): Validation dataloader.
        criterion: Loss function.
        device: CPU or GPU.

    Returns:
        tuple: (average_loss, accuracy)
    """

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        progress_bar = tqdm(
            dataloader,
            desc="Validation",
            leave=False
        )

        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, dim=1)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            progress_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{100 * correct / total:.2f}%"
            )

    average_loss = running_loss / total
    accuracy = correct / total

    return average_loss, accuracy


def train(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    config,
    device,
    checkpoint_dir,
    start_epoch=0,
    history=None,
    best_val_accuracy=0.0,
):
    """
    Train a model with checkpoint support.

    Args:
        model (nn.Module): Model to train.
        train_loader (DataLoader): Training dataloader.
        val_loader (DataLoader): Validation dataloader.
        criterion: Loss function.
        optimizer: Optimizer.
        config (dict): Configuration dictionary.
        device: CPU or GPU.
        checkpoint_dir (str): Directory to save checkpoints.
        start_epoch (int): Epoch to resume training from.
        history (dict): Existing training history.
        best_val_accuracy (float): Best validation accuracy so far.

    Returns:
        tuple:
            model,
            history,
            best_val_accuracy
    """

    if history is None:
        history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }

    os.makedirs(checkpoint_dir, exist_ok=True)

    epochs = config["epochs"]

    for epoch in range(start_epoch, epochs):

        print(f"\nEpoch {epoch + 1}/{epochs}")

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        val_loss, val_accuracy = validate_one_epoch(
            model,
            val_loader,
            criterion,
            device,
        )

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)

        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)


        # Save latest checkpoint after every epoch

        latest_checkpoint = os.path.join(
            checkpoint_dir,
            "latest_checkpoint.pth",
        )

        save_checkpoint(
            model,
            optimizer,
            epoch + 1,
            history,
            best_val_accuracy,
            latest_checkpoint,
        )


        # Save best model

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy

            best_model_path = os.path.join(
                checkpoint_dir,
                "best_model.pth",
            )

            save_checkpoint(
                model,
                optimizer,
                epoch + 1,
                history,
                best_val_accuracy,
                best_model_path,
            )


        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.2f}%"
        )


    return model, history, best_val_accuracy