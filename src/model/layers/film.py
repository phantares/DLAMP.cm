import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


class FiLM(nn.Module):
    def __init__(self, film_channel, channel):
        super().__init__()

        self.gb_scalar = nn.Linear(film_channel, 2 * channel)
        self.gb_spatial = nn.Linear(film_channel, 2 * channel)

    def forward(self, x, film_scalar=None, film_spatial=None):
        if film_scalar is not None:
            sc_gamma, sc_beta = torch.chunk(self.gb_scalar(film_scalar), 2, dim=-1)
            x = (
                x * (1 + sc_gamma[:, :, None, None, None])
                + sc_beta[:, :, None, None, None]
            )

        if film_spatial is not None:
            B, C, Z, H, W = x.shape
            sp = F.interpolate(
                film_spatial, size=(H, W), mode="bilinear", align_corners=False
            )

            sp = rearrange(sp, "b c h w -> b h w c")
            sp_gamma, sp_beta = torch.chunk(self.gb_spatial(sp), 2, dim=-1)
            sp_gamma = rearrange(sp_gamma, "b h w c -> b c h w").unsqueeze(-3)
            sp_beta = rearrange(sp_beta, "b h w c -> b c h w").unsqueeze(-3)
            x = x * (1 + sp_gamma) + sp_beta

        return x
