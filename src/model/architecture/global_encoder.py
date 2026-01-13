import torch
import torch.nn as nn
from einops import rearrange, repeat

from . import PositionEncoder, SinusoidalPositionEncoder, ConvNeXtLayer, MLP


class GlobalEncoder(nn.Module):
    def __init__(
        self,
        resolution,
        single_channel,
        upper_channel,
        use_map=True,
        map_channel=256,
        use_token=True,
        token_channel=768,
    ):
        super().__init__()

        self.resolution = resolution

        self.sfc_conv = nn.Conv2d(single_channel, 64, 1)
        self.up_conv = nn.Conv3d(upper_channel, 64, 1)
        self.pos_enc = PositionEncoder()
        self.conv = nn.Conv3d(64 + 3, 128, 1)

        self.stem = nn.Conv3d(128, 128, 3, padding=1, stride=(1, 2, 2))
        self.stage1 = ConvNeXtLayer(2, 128)
        self.down2 = nn.Conv3d(128, 256, 3, padding=1, stride=(1, 2, 2))
        self.stage2 = ConvNeXtLayer(2, 256)
        self.down3 = nn.Conv3d(256, 256, 3, padding=1, stride=(1, 2, 2))
        self.stage3 = ConvNeXtLayer(4, 256)
        self.down4 = nn.Conv3d(256, 256, 3, padding=1, stride=(1, 2, 2))
        self.proj = nn.Conv3d(256, map_channel, 1)
        self.scale_map = MLP(1, map_channel // 2, map_channel * 2)
        self.use_map = use_map

        self.use_token = use_token
        self.pool_adpt = nn.AdaptiveAvgPool3d((14, 10, 10))
        self.sin_pos_enc = SinusoidalPositionEncoder()
        self.k_proj = nn.Conv3d(map_channel, 240, 1)
        self.v_proj = nn.Conv3d(map_channel, 240, 1)
        self.q = nn.Parameter(torch.randn(1, 1, 240))
        self.scale = 240**-0.5
        self.softmax = nn.Softmax(dim=-1)
        self.pool_mlp = MLP(240, 512, token_channel)

    def forward(self, surface, upper):
        x = torch.cat(
            [self.sfc_conv(surface).unsqueeze(-3), self.up_conv(upper)], dim=-3
        )  # (B,64,Z+1,H,W)
        x = self.pos_enc(x)
        x = self.conv(x)  # (B,67,Z+1,H,W)

        x = self.stage1(self.stem(x))  # (B,128,Z+1,H/2,W/2)
        x = self.stage2(self.down2(x))  # (B,256,Z+1,H/4,W/4)
        x = self.stage3(self.down3(x))  # (B,256,Z+1,H/8,W/8)
        x = self.down4(x)
        global_map = self.proj(x)  # (B,C_map,Z+1,H/16,W/16)

        gb = self.scale_map(
            torch.full(
                (global_map.size(0), 1),
                self.resolution,
                device=global_map.device,
                dtype=global_map.dtype,
            )
        )
        gamma, beta = gb.chunk(2, dim=-1)
        global_map = (
            global_map * (1 + gamma[:, :, None, None, None])
            + beta[:, :, None, None, None]
        )

        global_token = None
        if self.use_token:
            canon = self.pool_adpt(global_map)
            k = rearrange(
                self.sin_pos_enc(
                    self.k_proj(canon),
                    upper.size(-2) * self.resolution,
                    upper.size(-1) * self.resolution,
                ),
                "b c z h w -> b (z h w) c",
            )  # (B,(Z+1)*H/16*W/16,240)
            v = rearrange(self.v_proj(canon), "b c z h w -> b (z h w) c")
            q = repeat(self.q, "1 1 c -> b 1 c", b=x.size(0))  # (B,1,240)
            attn = self.softmax(
                q @ k.transpose(-1, -2) * self.scale
            )  # (B,1,(Z+1)*H/16*W/16)
            token = attn @ v  # (B,1,240)
            global_token = self.pool_mlp(token.squeeze(1))  # (B,C_tk)

        if not self.use_map:
            global_map = None

        return global_token, global_map
