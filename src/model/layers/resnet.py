import torch.nn as nn

from . import FiLM


class ResNetBlock(nn.Module):

    def __init__(
        self,
        dim,
        dilation=1,
        groups=16,
        dropout=0.0,
        film_channel=0,
    ):
        super().__init__()

        self.norm1 = nn.GroupNorm(groups, dim)
        self.conv1 = nn.Conv3d(dim, dim, 3, padding=dilation, dilation=dilation)
        self.norm2 = nn.GroupNorm(groups, dim)
        self.conv2 = nn.Conv3d(dim, dim, 3, padding=dilation, dilation=dilation)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)

        if film_channel > 0:
            self.film = FiLM(film_channel, dim)
        else:
            self.film = None

    def forward(self, x, film_base=None):
        shortcut = x

        x = self.norm1(x)
        x = self.act(x)
        x = self.conv1(x)

        x = self.norm2(x)
        if self.film is not None and film_base is not None:
            x = self.film(x, film_base)
        x = self.act(x)
        x = self.drop(x)
        x = self.conv2(x)

        return x + shortcut
