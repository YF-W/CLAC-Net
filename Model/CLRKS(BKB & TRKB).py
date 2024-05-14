import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

from DoubleDilatedConv import DoubleDilatedConv
from FCKB import FCKB


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size = 5, stride= 1, padding="same", bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Mish(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size= 3, stride= 1, padding= 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.SELU = nn.SELU(inplace=True)

    def forward(self, x):
        x1 = self.up(x)
        x1 = self.conv(x1)
        add = self.SELU(x1)
        return add


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
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
        x1 = self.maxpool(x)
        x1 = self.conv(x1)
        add = self.SELU(x1)
        return add


class CLRKS(nn.Module):
    def __init__(self):
        super(CLRKS, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.DoubleDilatedConv_up = DoubleDilatedConv(64, 64)
        self.DoubleDilatedConv_mid = DoubleDilatedConv(128, 64)
        self.DoubleDilatedConv_low = DoubleDilatedConv(256, 64)
        self.UpBlock = UpBlock(128, 64)
        self.DownBlock = DownBlock(128, 256)
        self.FCKB = FCKB(384, 128)



    def forward(self, up, mid, low):
        #BKB concat DilatedConv
        mid = self.DoubleDilatedConv_mid(mid)

        #TRKB up&low layers
        up = self.DoubleDilatedConv_up(up)
        up_temp = up
        up = self.pool(up)
        up = up + mid
        up = self.DownBlock(up)

        low = self.DoubleDilatedConv_low(low)
        low_temp = low
        low = self.up(low)
        low = low + mid
        low = self.UpBlock(low)

        #BKB concat
        if mid.shape != up_temp.shape:
            up_temp = TF.resize(up_temp, size=mid.shape[2:])

        if mid.shape != low_temp.shape:
            low_temp = TF.resize(low_temp, size=mid.shape[2:])
        mid = torch.cat((up_temp, low_temp, mid), dim=1)
        mid = self.FCKB(mid)

        return up, mid, low