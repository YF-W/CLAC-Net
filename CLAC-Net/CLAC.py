import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from DEAB import DEAB
from DoubleConv import DoubleConv
from Dconv import DilatedConv
from FCKB import FCKB
from low_to_mid import UpBlock
from up_to_mid import DownBlock


"""
CLAC-Net

A Semantic Segmentation Network based on Cross-Layer Fusion and Asymmetric Connections

Authors: Ronghui Feng, Yuefei Wang
Affiliation: Chengdu University
Date: 2023.12

For research and clinical study only, commercial use is strictly prohibited
"""

class CLAC_Net(nn.Module):
    def __init__(
            self, in_channels, out_channels,
    ):
        super(CLAC_Net, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dcov1 = DoubleConv(in_channels, 64)
        self.dcov2 = DoubleConv(64, 128)
        self.dcov3 = DoubleConv(128, 256)
        self.dcov4 = DoubleConv(256, 512)

        self.DilatedConv2 = DilatedConv(128, 64)
        self.DilatedConv1 = DilatedConv(64, 64)
        self.DilatedConv3 = DilatedConv(256, 64)
        self.UpBlock = UpBlock(128, 256)
        self.up_skip = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.DownBlock = DownBlock(128, 64)

        self.FCKB = FCKB(384, 128)
        self.DEAB = DEAB(512, 1024)

        self.up = nn.ConvTranspose2d(1024, 1024, kernel_size=2, stride=2)

        self.dcov5 = DoubleConv(1024, 512)
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dcov6 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dcov7 = DoubleConv(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dcov8 = DoubleConv(128, 64)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.dcov1(x)
        skip_connection_up = x
        x = self.pool(x)
        x = self.dcov2(x)
        skip_connection_mid = x
        x = self.pool(x)
        x = self.dcov3(x)
        skip_connection_low = x
        x = self.pool(x)
        x = self.dcov4(x)

        skip_connection_mid = self.DilatedConv2(skip_connection_mid)

        skip_connection_up = self.DilatedConv1(skip_connection_up)
        skip_connectionI = skip_connection_up
        skip_connection_up = self.pool(skip_connection_up)
        skip_connection_up = skip_connection_up + skip_connection_mid
        skip_connection_up = self.UpBlock(skip_connection_up)

        skip_connection_low = self.DilatedConv3(skip_connection_low)
        skip_connectionII = skip_connection_low
        skip_connection_low = self.up_skip(skip_connection_low)
        skip_connection_low = skip_connection_low + skip_connection_mid
        skip_connection_low = self.DownBlock(skip_connection_low)

        skip_connectionI = TF.resize(skip_connectionI, size=skip_connection_mid.shape[2:])
        skip_connectionII = TF.resize(skip_connectionII, size=skip_connection_mid.shape[2:])
        skip_connection_mid = torch.cat((skip_connectionI, skip_connectionII, skip_connection_mid), dim=1)
        skip_connection_mid = self.FCKB(skip_connection_mid)

        x = self.DEAB(x)

        x = self.dcov5(x)
        x = self.up1(x)
        x = torch.cat((skip_connection_up, x), dim=1)
        x = self.dcov6(x)
        x = self.up2(x)
        x = torch.cat((skip_connection_mid, x), dim=1)
        x = self.dcov7(x)
        x = self.up3(x)
        x = torch.cat((skip_connection_low, x), dim=1)
        x = self.dcov8(x)

        return self.final_conv(x)




