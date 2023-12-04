import torch.nn as nn
import torch

class DilatedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(DilatedConv, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Mish(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=3, dilation=3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Mish(inplace=True),
        )

    def forward(self, x):
        concat = torch.cat((self.conv1(x), self.conv2(x)), dim=1)
        return concat