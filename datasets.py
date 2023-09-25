from typing import Optional, Literal, Tuple, Callable
import os
import torch
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST, CIFAR10, CIFAR100, ImageFolder, ImageNet
from torchvision.datasets.utils import download_and_extract_archive
import torchvision.transforms as transforms


DatasetOptions = Literal[
    "mnist",
    "cifar10",
    "cifar100",
    "tiny-imagenet",
    "imagenet",
]

num_classes_by_dataset = {
    "mnist": 10,
    "cifar10": 10,
    "cifar100": 100,
    "tiny-imagenet": 200,
    "imagenet": 1000,
}

# Image size after transforms are applied
image_size_by_dataset = {
    "mnist": 28,
    "cifar10": 32,
    "cifar100": 32,
    "tiny-imagenet": 64,
    "imagenet": 224,
}


# Based on https://github.com/ganguli-lab/Synaptic-Flow/blob/master/Utils/custom_datasets.py
# and https://towardsdatascience.com/pytorch-ignite-classifying-tiny-imagenet-with-efficientnet-e5b1768e5e8f
class TinyImageNet(ImageFolder):
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    filename = "tiny-imagenet-200.zip"
    base_dir = "tiny-imagenet-200"

    def __init__(
        self,
        root: str,
        train: Optional[bool] = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: Optional[bool] = False,
    ):
        if download and not self.exists(root, self.filename):
            download_and_extract_archive(
                url=self.url,
                download_root=root,
                extract_root=root,
                filename=self.filename,
            )
            self.setup(root, self.base_dir)

        elif not self.exists(root, self.filename):
            raise FileNotFoundError(
                f"The dataset is not found at {root}. Consider downloading it by setting download=True."
            )

        split_dir = "train" if train else "val"
        data_dir = os.path.join(root, self.base_dir, split_dir)
        super().__init__(data_dir, transform, target_transform)

    def exists(self, root, filename):
        return os.path.exists(os.path.join(root, filename))

    def setup(self, root, base_dir):
        val_dir = os.path.join(root, base_dir, "val")
        val_img_dir = os.path.join(val_dir, "images")
        annotation_path = os.path.join(val_dir, "val_annotations.txt")

        val_img_to_class = {}
        with open(annotation_path, "r") as f:
            for line in f.readlines():
                filename, classname, *_ = line.split("\t")
                val_img_to_class[filename] = classname

        # Create subfolders (if not present) for validation images based on label,
        # and move images into the respective folders
        for filename, classname in val_img_to_class.items():
            dest_dir = os.path.join(val_dir, classname)
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)

            src_path = os.path.join(val_img_dir, filename)
            dist_path = os.path.join(dest_dir, filename)
            if os.path.exists(src_path):
                os.rename(src_path, dist_path)

        os.remove(annotation_path)
        os.rmdir(val_img_dir)


def get_transforms(name: DatasetOptions) -> Tuple[Callable, Callable]:
    """Returns the train and test transformations for the given dataset.

    The dataset statistics (mean, std) for mnist and cifar were taken from: https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d8151

    """

    if name == "mnist":
        mean = torch.tensor([0.1307])
        std = torch.tensor([0.3081])

        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        test_transform = train_transform

    elif name == "cifar10":
        mean = torch.tensor([0.4914, 0.4822, 0.4465])
        std = torch.tensor([0.2470, 0.2435, 0.2616])

        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    elif name == "cifar100":
        mean = torch.tensor([0.5071, 0.4865, 0.4409])
        std = torch.tensor([0.2673, 0.2564, 0.2762])

        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    elif name == "tiny-imagenet":
        mean = torch.tensor([0.4802, 0.4481, 0.3975])
        std = torch.tensor([0.2770, 0.2691, 0.2821])

        train_transform = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    elif name == "imagenet":
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])

        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    else:
        raise NotImplementedError(
            f"Provided dataset '{name}' is not supported.")

    return train_transform, test_transform


def get_dataset(
    name: DatasetOptions,
    root: str,
    download: bool = False,
) -> Tuple[Dataset, Dataset]:
    """Returns two instances of a normalized pytorch dataset, one for training and one for testing.

    The training data includes augmentation.
    """

    train_transform, test_transform = get_transforms(name)

    if name == "mnist":
        train_data = MNIST(
            root, train=True, transform=train_transform, download=download)
        test_data = MNIST(
            root, train=False, transform=test_transform, download=download)

    elif name == "cifar10":
        train_data = CIFAR10(
            root, train=True, transform=train_transform, download=download)
        test_data = CIFAR10(
            root, train=False, transform=test_transform, download=download)

    elif name == "cifar100":
        train_data = CIFAR100(
            root, train=True, transform=train_transform, download=download)
        test_data = CIFAR100(
            root, train=False, transform=test_transform, download=download)

    elif name == "tiny-imagenet":
        train_data = TinyImageNet(
            root, train=True, transform=train_transform, download=download)
        test_data = TinyImageNet(
            root, train=False, transform=test_transform, download=download)

    elif name == "imagenet":
        root = os.path.join(root, "imagenet")
        train_data = ImageNet(root, split="train", transform=train_transform)
        test_data = ImageNet(root, split="val", transform=test_transform)

    else:
        raise NotImplementedError(
            f"Provided dataset '{name}' is not supported.")

    return train_data, test_data


def get_dataloader(
    name: DatasetOptions,
    root: str,
    train_batch_size: int,
    eval_batch_size: int,
    num_workers: int = 0,
    download: bool = False,
    device: Optional[torch.device] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Returns a train and test data loader."""

    train_data, test_data = get_dataset(name, root, download)

    if device is None:
        pin_memory = False
    else:
        pin_memory = device.type == "cuda"

    # TODO: persisting workers improves efficiency 
    # but cannot have multiple iterators open at ones
    # persist_workers = num_workers > 0
    persist_workers = False

    train_loader = DataLoader(
        train_data,
        batch_size=train_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persist_workers,
    )

    test_loader = DataLoader(
        test_data,
        batch_size=eval_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persist_workers,
    )

    return train_loader, test_loader
