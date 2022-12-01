import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms


VGG_TYPES = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "VGG19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


class VGG_Net(nn.Module):
    def __init__(self, in_channels=1, num_classes=10, architecture="VGG16", height=64, width=2584):
        super(VGG_Net, self).__init__()
        self.in_channels = in_channels
        self.conv_layers = self.create_conv_layers(VGG_TYPES[architecture])
        self.architecture = architecture

        self.num_max_pools = self.get_num_of_max_pools()
        self.height_conv = self.height_after_conv(height)
        self.width_conv = self.width_after_conv(width)

        self.fcs = nn.Sequential(
            nn.Linear(512 * self.height_conv *
                      self.width_conv, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fcs(x)
        return x

    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == int:
                out_channels = x

                layers += [
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        padding=(1, 1),
                    ),
                    nn.BatchNorm2d(x),
                    nn.ReLU(),
                ]
                in_channels = x
            elif x == "M":
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]
        return nn.Sequential(*layers)

    def get_num_of_max_pools(self):
        return sum([1 for x in self.conv_layers if type(x) == nn.MaxPool2d])

    def height_after_conv(self, height):
        return height // (2 ** self.num_max_pools)

    def width_after_conv(self, width):
        return width // (2 ** self.num_max_pools)

    def get_model_name(self):
        return self.architecture


if __name__ == "__main__":
    model = VGG_Net(in_channels=1, num_classes=10,
                    architecture="VGG16", height=64, width=2584)
    print(model)
    x = torch.randn(1, 1, 64, 2584)
    print(model(x).shape)
