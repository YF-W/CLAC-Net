import torch.nn as nn

class FD(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FD, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 5, stride=1, padding="same", dilation=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Mish(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)
