import torch.nn as nn

class FCKB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FCKB, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding="same", bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Mish(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SELU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x