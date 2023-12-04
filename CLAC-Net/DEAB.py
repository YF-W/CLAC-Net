import torch.nn as nn
from DA import DA
from MHSA import MHSA
from FD import FD

class DEAB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DEAB, self).__init__()
        self.DA = DA(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.MHSA = MHSA(out_channels, 32, 32)
        self.FD = FD(out_channels, out_channels)
        self.up = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.DA(x)
        x = self.pool(x)
        x = self.MHSA(x)
        x = self.FD(x)
        x = self.up(x)

        return x