import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=(1, 1),
                    stride=1,
                    padding=0,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        out = self.relu(out)
        return out

class CNN(nn.Module):
    def __init__(self, num_classes=2) -> None:
        super(CNN, self).__init__()
        self.model = nn.Sequential(
            ResidualBlock(in_channels=1, out_channels=8),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            ResidualBlock(in_channels=8, out_channels=24),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        last_conv_out_channels = 24
        self.output = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=last_conv_out_channels, out_features=num_classes)
        )

    def forward(self, x):
        x = self.model(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)

        output = self.output(x)
        return output