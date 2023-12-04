import torch.nn as nn

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size = 5, stride=1, padding="same", bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Mish(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.SELU = nn.SELU(inplace=True)

    def forward(self, x):
        x = self.maxpool(x)
        x = self.conv(x)
        x = self.SELU(x)

        return x