from torchvision import transforms
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.model_selection import train_test_split

from src.constants import DATASET_DIR, RANDOM_SEED


def get_transforms(config):
    """
    Create training and evaluation transforms.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        tuple: (train_transforms, test_transforms)
    """

    train_transforms = transforms.Compose([
        transforms.Resize((config["image_size"], config["image_size"])),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop(config["image_size"], scale=(0.8, 1.0)),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config["mean"],
            std=config["std"]
        ),
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((config["image_size"], config["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config["mean"],
            std=config["std"]
        ),
    ])

    return train_transforms, test_transforms


def split_dataset(dataset, config):
    """
    Split a dataset into train, validation, and test indices using
    stratified sampling.

    Args:
        dataset (ImageFolder): Dataset to split.
        config (dict): Configuration dictionary.

    Returns:
        tuple: (train_indices, val_indices, test_indices)
    """

    labels = np.array(dataset.targets)
    indices = np.arange(len(dataset))

    train_indices, temp_indices = train_test_split(
        indices,
        train_size=config["train_split"],
        stratify=labels,
        random_state=RANDOM_SEED,
    )

    temp_labels = labels[temp_indices]

    val_ratio = config["val_split"] / (
        config["val_split"] + config["test_split"]
    )

    val_indices, test_indices = train_test_split(
        temp_indices,
        train_size=val_ratio,
        stratify=temp_labels,
        random_state=RANDOM_SEED,
    )

    return train_indices, val_indices, test_indices


def get_dataloaders(config):
    """
    Create train, validation, and test dataloaders.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        tuple: (train_loader, val_loader, test_loader, class_names)
    """

    train_transforms, test_transforms = get_transforms(config)

    train_dataset = ImageFolder(
        root=DATASET_DIR,
        transform=train_transforms
    )

    eval_dataset = ImageFolder(
        root=DATASET_DIR,
        transform=test_transforms
    )

    train_indices, val_indices, test_indices = split_dataset(
        train_dataset,
        config
    )

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(eval_dataset, val_indices)
    test_subset = Subset(eval_dataset, test_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"]
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"]
    )

    test_loader = DataLoader(
        test_subset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"]
    )

    class_names = train_dataset.classes

    return train_loader, val_loader, test_loader, class_names