import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm2d(channels, affine=True)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm2d(channels, affine=True)

    def forward(self, x):
        residual = x
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return residual + x


class LightSRNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, n_feats=32, n_blocks=8):
        super().__init__()
        self.input = nn.Conv2d(in_ch, n_feats, 3, padding=1)
        self.body = nn.Sequential(*[ResidualBlock(n_feats) for _ in range(n_blocks)])
        self.output_conv = nn.Conv2d(n_feats, 4, 3, padding=1)
        self.pixelshuffle = nn.PixelShuffle(2)
        self.final = nn.Conv2d(out_ch, out_ch, 3, padding=1)

    def forward(self, x):
        x = self.input(x)
        x = self.body(x)
        x = self.output_conv(x)
        x = self.pixelshuffle(x)
        x = self.final(x)
        return torch.clamp(x, 0.0, 1.0)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
