from torchvision import transforms

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