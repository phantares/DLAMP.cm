import torch
import torch.nn as nn
from timm.layers import DropPath
from einops import rearrange

from . import MLP


class ConvNeXtLayer(nn.Module):
    def __init__(
        self,
        depth: int,
        dim: int,
        kernel=(1, 7, 7),
        layer_scale_init=1e-6,
        drop_path_rate=0.2,
    ):
        super().__init__()
        dpr = torch.linspace(0, drop_path_rate, depth).tolist() if depth > 1 else [0.0]
        self.blocks = nn.Sequential(
            *[
                ConvNeXtBlock(dim, kernel, layer_scale_init, dpr[i])
                for i in range(depth)
            ]
        )

    def forward(self, x):
        return self.blocks(x)


class ConvNeXtBlock(nn.Module):
    def __init__(
        self, dim: int, kernel=(1, 7, 7), layer_scale_init=1e-6, drop_path=0.0
    ):
        super().__init__()

        self.dwconv = nn.Conv3d(
            dim, dim, kernel, padding=tuple(i // 2 for i in kernel), groups=dim
        )
        self.norm = nn.LayerNorm(dim)
        self.mlp = MLP(dim, dim * 4, dim)
        self.gamma = nn.Parameter(
            layer_scale_init * torch.ones(dim) if layer_scale_init > 0.0 else None
        )
        self.drop = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor):
        input = x
        x = self.dwconv(x)

        x = rearrange(x, "b c z h w -> b z h w c")
        x = self.norm(x)
        x = self.mlp(x)
        if self.gamma is not None:
            x = x * self.gamma
        x = rearrange(x, "b z h w c -> b c z h w")

        x = input + self.drop(x)

        return x
