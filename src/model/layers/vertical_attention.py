import torch
from einops import rearrange
from torch import nn

from .mlp import MLP


class VerticalAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: int = 4,
        drop: float = 0.0,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=drop,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(
            dim,
            dim * mlp_ratio,
            dim,
        )

    def forward(
        self,
        x: torch.Tensor,
    ):
        B, C, Z, H, W = x.shape

        x = rearrange(x, "b c z h w -> (b h w) z c")

        h = self.norm1(x)
        h, _ = self.attn(
            h,
            h,
            h,
            need_weights=False,
        )

        x = x + h

        x = x + self.mlp(self.norm2(x))

        x = rearrange(
            x,
            "(b h w) z c -> b c z h w",
            b=B,
            h=H,
            w=W,
        )

        return x
