import torch


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

    for images, labels in dataloader:
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

    average_loss = running_loss / total
    accuracy = correct / total

    return average_loss, accuracy