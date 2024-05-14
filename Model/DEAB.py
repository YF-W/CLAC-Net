import torch
import torch.nn as nn


class LargeConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LargeConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 5, stride=1, padding="same", dilation=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Mish(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class MHSA(nn.Module):
    def __init__(self, n_dims, width, height):
        super(MHSA, self).__init__()

        self.query = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.key = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.value = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.rel_h = nn.Parameter(torch.randn([1, n_dims, 1, height]), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn([1, n_dims, width, 1]), requires_grad=True)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        n_batch, C, width, height = x.size()
        q = self.query(x).view(n_batch, C, -1)
        k = self.key(x).view(n_batch, C, -1)
        v = self.value(x).view(n_batch, C, -1)
        content_content = torch.bmm(q.permute(0, 2, 1), k)

        content_position = (self.rel_h + self.rel_w).view(1, C, -1).permute(0, 2, 1)
        content_position = torch.matmul(content_position, q)

        energy = content_content + content_position
        attention = self.softmax(energy)

        out = torch.bmm(v, attention.permute(0, 2, 1))
        out = out.view(n_batch, C, width, height)

        return out


class MutiConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(MutiConv, self).__init__()
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
        concat = self.conv1(x) + self.conv2(x) + self.conv3(x)
        return concat


class DEAB(nn.Module):
    def __init__(
            self, in_channels, out_channels,
    ):
        super(DEAB, self).__init__()
        self.up = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.LargeConv = LargeConv(out_channels, out_channels)
        self.MutiConv = MutiConv(in_channels, out_channels)
        self.MHSA = MHSA(out_channels, 32, 32)

    def forward(self, x):
        x = self.MutiConv(x)
        x = self.pool(x)
        x = self.LargeConv(x)
        x = self.up(x)

        return x