"""vgg in pytorch


[1] Karen Simonyan, Andrew Zisserman

    Very Deep Convolutional Networks for Large-Scale Image Recognition.
    https://arxiv.org/abs/1409.1556v6
"""
'''VGG11/13/16/19 in Pytorch.'''


# divide the convolution layer width by 4
import torch
import torch.nn as nn
cfg = {
    'A': [16,     'M', 32,     'M', 64, 64,         'M', 128, 128,           'M', 128, 128,           'M'],
    'B': [16, 16, 'M', 32, 32, 'M', 64, 64,         'M', 128, 128,           'M', 128, 128,           'M'],
    'D': [16, 16, 'M', 32, 32, 'M', 64, 64, 64,     'M', 128, 128, 128,      'M', 128, 128, 128,      'M'],
    'E': [16, 16, 'M', 32, 32, 'M', 64, 64, 64, 64, 'M', 128, 128, 128, 128, 'M', 128, 128, 128, 128, 'M']
}


class VGG(nn.Module):

    def __init__(self, features, num_classes=100, width=1):
        super().__init__()
        self.features = features

        # divide the linear layer width by 8
        self.classifier = nn.Sequential(
            nn.Linear(int(128 * width), int(512 * width)),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(int(512 * width), int(512 * width)),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(int(512 * width), num_classes)
        )

    def forward(self, x):
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)

        return output


def make_layers(cfg, batch_norm=False, width=1):
    layers = []

    input_channel = 3
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        layers += [nn.Conv2d(input_channel, int(l * width),
                             kernel_size=3, padding=1)]

        if batch_norm:
            layers += [nn.BatchNorm2d(int(l * width))]

        layers += [nn.ReLU()]
        input_channel = int(l * width)

    return nn.Sequential(*layers)


def vgg11_bn_small(num_classes: int, width: float = 1):
    return VGG(make_layers(cfg['A'], batch_norm=True, width=width), num_classes, width)


def vgg13_bn_small(num_classes: int, width: float = 1):
    return VGG(make_layers(cfg['B'], batch_norm=True, width=width), num_classes, width)


def vgg16_bn_small(num_classes: int, width: float = 1):
    return VGG(make_layers(cfg['D'], batch_norm=True, width=width), num_classes, width)


def vgg19_bn_small(num_classes: int, width: float = 1):
    return VGG(make_layers(cfg['E'], batch_norm=True, width=width), num_classes, width)
