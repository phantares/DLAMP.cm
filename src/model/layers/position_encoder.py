import torch
import torch.nn as nn
import math
from einops import repeat, rearrange


class PositionEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, data):
        device = data.device
        B, _, Z, H, W = data.shape

        x = repeat(
            torch.linspace(0, 1, W, device=device), "w -> b 1 z h w", b=B, z=Z, h=H
        )
        y = repeat(
            torch.linspace(0, 1, H, device=device), "h -> b 1 z h w", b=B, z=Z, w=W
        )
        z = repeat(
            torch.linspace(0, 1, Z, device=device), "z -> b 1 z h w", b=B, h=H, w=W
        )

        return torch.cat([data, z, y, x], dim=1)


class SinusoidalPositionEncoder(nn.Module):
    def __init__(self, lambda_min_km=5.0, lambda_max_km=1000.0):
        super().__init__()

        self.lambda_min = lambda_min_km
        self.lambda_max = lambda_max_km

    def forward(self, data, Lx_physical, Ly_physical):
        device = data.device
        B, C, Z, H, W = data.shape
        c = C // 6  # sin/cos pairs per axis

        lambdas = torch.logspace(
            math.log10(self.lambda_min),
            math.log10(self.lambda_max),
            steps=c,
            device=device,
        )
        k = 2 * math.pi / lambdas

        x = (torch.linspace(0, 1, W, device=device) * Lx_physical).unsqueeze(1)
        x_feat = repeat(self._get_pos_emb(x, k), "w c -> z h w c", z=Z, h=H)

        y = (torch.linspace(0, 1, H, device=device) * Ly_physical).unsqueeze(1)
        y_feat = repeat(self._get_pos_emb(y, k), "h c -> z h w c", z=Z, w=W)

        z = torch.linspace(0, 1, Z, device=device).unsqueeze(1)  # (n,1)
        z_feat = repeat(self._get_pos_emb(z, k), "z c -> z h w c", h=H, w=W)

        pos = torch.cat([z_feat, y_feat, x_feat], dim=-1)
        pos = rearrange(pos, "z h w c -> () c z h w")

        return data + pos

    def _get_pos_emb(self, coord, k):
        phase = coord * k
        return torch.cat([torch.sin(phase), torch.cos(phase)], dim=1)


class SinusoidalPositionEncoderTemp(nn.Module):
    def __init__(self, temperature=10000.0):
        super().__init__()

        self.temp = temperature

    def forward(self, data):
        device = data.device
        B, C, Z, H, W = data.shape
        c = C // 6  # sin/cos pairs per axis

        dim_t = torch.arange(c, device=device, dtype=data.dtype)
        dim_t = self.temp ** (2 * (dim_t // 2) / c)  # (c,)

        z_feat = repeat(self._get_pos_emb(Z, dim_t, device), "z c -> z h w c", h=H, w=W)
        y_feat = repeat(self._get_pos_emb(H, dim_t, device), "h c -> z h w c", z=Z, w=W)
        x_feat = repeat(self._get_pos_emb(W, dim_t, device), "w c -> z h w c", z=Z, h=H)

        pos = torch.cat([z_feat, y_feat, x_feat], dim=-1)
        pos = rearrange(pos, "z h w c -> () c z h w")

        return data + pos

    def _get_pos_emb(self, n, dim_t, device):
        t = torch.linspace(0, 1, n, device=device).unsqueeze(1)  # (n,1)
        emb = t * math.pi / dim_t

        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
