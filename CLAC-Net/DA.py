import torch.nn as nn

class DA(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(DA, self).__init__()
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
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Mish(inplace=True),
        )

    def forward(self, x):
        DA = self.conv1(x) + self.conv2(x) + self.conv3(x)
        return DA