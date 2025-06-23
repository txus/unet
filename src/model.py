import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import center_crop


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3)),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3)),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Model(nn.Module):
    def __init__(self, n_classes, dropout=0.0):
        super().__init__()

        self.contraction = nn.ModuleList(
            [
                Conv(1, 64),
                Conv(64, 128),
                Conv(128, 256),
                Conv(256, 512),
                Conv(512, 1024),
            ]
        )

        self.dropout = nn.Dropout(p=dropout)

        self.up_convs = nn.ModuleList(
            [
                nn.ConvTranspose2d(1024, 512, kernel_size=(2, 2), stride=2),
                nn.ConvTranspose2d(512, 256, kernel_size=(2, 2), stride=2),
                nn.ConvTranspose2d(256, 128, kernel_size=(2, 2), stride=2),
                nn.ConvTranspose2d(128, 64, kernel_size=(2, 2), stride=2),
            ]
        )

        self.expansion = nn.ModuleList(
            [
                Conv(1024, 512),
                Conv(512, 256),
                Conv(256, 128),
                Conv(128, 64),
            ]
        )

        self.max_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.out_conv = nn.Conv2d(64, n_classes, kernel_size=(1, 1))

        for mod in self.modules():
            if isinstance(mod, nn.Conv2d) or isinstance(mod, nn.ConvTranspose2d):
                n = mod.kernel_size[0] * mod.kernel_size[1] * mod.in_channels
                nn.init.normal_(mod.weight, mean=0, std=math.sqrt(2.0 / n))

    def forward(self, x):
        residuals = []

        for idx, contraction in enumerate(self.contraction):
            x = contraction(x)

            if idx < len(self.contraction) - 1:
                residuals.insert(0, x)  # residuals will be reversed
                x = self.max_pool(x)

        x = self.dropout(x)

        for idx, (up_conv, expansion) in enumerate(zip(self.up_convs, self.expansion)):
            x = up_conv(x)

            residual = center_crop(residuals[idx], (x.shape[-2], x.shape[-1]))

            x = torch.concat(
                [residual, x], dim=1
            )  # concat the residual along the channel axis

            x = expansion(x)

        return self.out_conv(x)
