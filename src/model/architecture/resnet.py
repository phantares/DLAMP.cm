import torch.nn as nn


class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, dilation=1, groups=16, dropout=0.0):
        super().__init__()

        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.conv1 = nn.Conv3d(
            in_channels, out_channels, 3, padding=dilation, dilation=dilation
        )
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv3d(
            out_channels, out_channels, 3, padding=dilation, dilation=dilation
        )
        self.skip = (
            nn.Conv3d(in_channels, out_channels, 1)
            if out_channels != in_channels
            else nn.Identity()
        )
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)
        self.film1 = None
        self.film2 = None

    def forward(self, x, film_base=None):
        input = x

        x = self.norm1(x)
        if self.film1 is not None and film_base is not None:
            x = self.film1(x, film_base)
        x = self.act(x)
        x = self.conv1(x)

        x = self.norm2(x)
        if self.film2 is not None and film_base is not None:
            x = self.film2(x, film_base)
        x = self.act(x)
        x = self.drop(x)
        x = self.conv2(x)

        return x + self.skip(input)
