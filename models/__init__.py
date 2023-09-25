# Models taken from: https://github.com/weiaicunzai/pytorch-cifar100

import math
import torch.nn as nn
import torchvision.models
from typing import Literal

from models.vgg_cifar import vgg11_bn_small, vgg13_bn_small, vgg16_bn_small, vgg19_bn_small
from models.resnet_cifar import (
    resnet20,
    resnet32,
    resnet44,
    resnet56,
    resnet110,
    resnet1202,
)


ModelOptions = Literal[
    "mlp",
    "resnet20",
    "resnet32",
    "resnet44",
    "resnet56",
    "resnet110",
    "resnet1202",
    "vgg11-small",
    "vgg13-small",
    "vgg16-small",
    "vgg19-small",
    "vit-small",
    "vit-tiny",
    "simple-vit",
    "simple-vit-tiny",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "vgg11",
    "vgg13",
    "vgg16",
    "vgg19",
]


def get_num_parameters(model: nn.Module) -> int:
    total_params = 0

    for param in filter(lambda p: p.requires_grad, model.parameters()):
        total_params += param.numel()

    return total_params


def get_model(name: ModelOptions, image_size: int, num_classes: int, width: float = 1) -> nn.Module:
    if name == "mlp":
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(image_size * image_size * 3, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )

    # CIFAR/TinyImagenet models

    if name == "resnet20":
        return resnet20(num_classes, width)
    elif name == "resnet32":
        return resnet32(num_classes, width)
    elif name == "resnet44":
        return resnet44(num_classes, width)
    elif name == "resnet56":
        return resnet56(num_classes, width)
    elif name == "resnet110":
        return resnet110(num_classes, width)
    elif name == "resnet1202":
        return resnet1202(num_classes, width)

    elif name == "vgg11-small":
        return vgg11_bn_small(num_classes, width)
    elif name == "vgg13-small":
        return vgg13_bn_small(num_classes, width)
    elif name == "vgg16-small":
        return vgg16_bn_small(num_classes, width)
    elif name == "vgg18-small":
        return vgg19_bn_small(num_classes, width)

    elif name == "vit-small":
        from models.vit_cifar import ViT
        return ViT(
            image_size=image_size,
            patch_size=4,
            num_classes=num_classes,
            dim=512,
            depth=6,
            heads=8,
            mlp_dim=512,
            dropout=0.1,
            emb_dropout=0.1
        )
    elif name == "vit-tiny":
        from models.vit_cifar import ViT
        return ViT(
            image_size=image_size,
            patch_size=4,
            num_classes=num_classes,
            dim=512,
            depth=4,
            heads=6,
            mlp_dim=256,
            dropout=0.1,
            emb_dropout=0.1
        )

    elif name == "simple-vit-tiny":
        from models.simple_vit import SimpleViT
        return SimpleViT(
            image_size=image_size,
            patch_size=4,
            num_classes=num_classes,
            dim=int(512 * width),
            depth=4,
            heads=int(6 * width),
            mlp_dim=int(256 * width),
        )

    # ImageNet models

    elif name == "simple-vit":
        from models.simple_vit import SimpleViT
        return SimpleViT(
            image_size=image_size,
            patch_size = 16,
            num_classes=num_classes,
            dim = int(1024 * width),
            depth = 6,
            heads = int(16 * width),
            mlp_dim = int(1024 * width)
        )

    elif name == "resnet18":
        return torchvision.models.resnet18(weights=None, num_classes=num_classes)
    elif name == "resnet34":
        return torchvision.models.resnet34(weights=None, num_classes=num_classes)
    elif name == "resnet50":
        return torchvision.models.resnet50(weights=None, num_classes=num_classes)
    elif name == "resnet101":
        return torchvision.models.resnet101(weights=None, num_classes=num_classes)
    elif name == "resnet152":
        return torchvision.models.resnet152(weights=None, num_classes=num_classes)

    elif name == "vgg11":
        return torchvision.models.vgg11_bn(weights=None, num_classes=num_classes)
    elif name == "vgg13":
        return torchvision.models.vgg13_bn(weights=None, num_classes=num_classes)
    elif name == "vgg16":
        return torchvision.models.vgg16_bn(weights=None, num_classes=num_classes)
    elif name == "vgg19":
        return torchvision.models.vgg19_bn(weights=None, num_classes=num_classes)

    else:
        raise NotImplementedError(
            f"Specified model '{name}' is not supported.")
