import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from . import ResBlock, MLP, FiLM, CrossAttention
from utils import calculate_factor, crop_center


class Unet(nn.Module):
    def __init__(
        self,
        target_horizontal_shape,
        input_resolution,
        target_resolution,
        surface_channel,
        upper_channel,
        out_channel,
        base_channel=128,
        token_channel=768,
        token_emb_channel=512,
        map_channel=256,
        kv_channel=128,
        position_emb_channel=128,
        time_emb_channel=128,
        resolution_emb_channel=128,
        include_sigma=False,
        sigma_emb_channel=32,
        drop=0.05,
    ):
        super().__init__()

        self.include_sigma = include_sigma
        self.res = input_resolution
        self.target_shape = target_horizontal_shape
        us_layer, self.us_scale = calculate_factor(
            input_resolution // target_resolution
        )
        if self.us_scale > 1:
            us_layer += 1

        c1, c2, c3 = base_channel, int(base_channel * 5 / 4), int(base_channel * 3 / 2)
        film_channel = base_channel * 2

        self.emb_glob = MLP(token_channel, token_emb_channel, token_emb_channel)
        self.emb_pos = MLP(2, 64, position_emb_channel)
        self.emb_time = MLP(4, 64, time_emb_channel)
        self.emb_res = MLP(1, 64, resolution_emb_channel)
        if include_sigma:
            self.emb_sigma = MLP(1, 64, sigma_emb_channel)
        else:
            sigma_emb_channel = 0
        self.film_mlp = MLP(
            token_emb_channel
            + position_emb_channel
            + time_emb_channel
            + resolution_emb_channel
            + sigma_emb_channel,
            512,
            film_channel,
        )

        self.sfc_conv = nn.Conv2d(surface_channel, 64, 1)
        self.up_conv = nn.Conv3d(upper_channel, 64, 1)
        self.enc0 = nn.Conv3d(64 * 2, c1, 3, padding=1)

        self.enc1 = ResBlock(c1, c1, dropout=drop)
        self.ds1 = nn.Conv3d(c1, c2, 3, stride=(1, 2, 2), padding=1)
        self.enc2 = ResBlock(c2, c2, dropout=drop)
        self.ds2 = nn.Conv3d(c2, c3, 3, stride=(1, 2, 2), padding=1)

        self.mid_dil = ResBlock(c2, c2, dilation=2, dropout=drop)
        self.mid = ResBlock(c3, c3, dropout=drop)

        self.us2 = nn.ConvTranspose3d(c3, c2, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec2 = ResBlock(c2 + c2, c2, dropout=drop)
        self.us1 = nn.ConvTranspose3d(c2, c1, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec1 = ResBlock(c1 + c1, c1, dropout=drop)
        c_out = c1

        self.upsample = nn.ModuleList()
        for n in range(us_layer):
            c_in = c1 // 2**n
            c_out = c1 // 2 ** (n + 1)

            self.upsample.append(
                nn.ConvTranspose3d(c_in, c_out, kernel_size=(1, 2, 2), stride=(1, 2, 2))
            )

            dec = ResBlock(c_out, c_out, dropout=drop)
            self._add_film(dec, film_channel)
            self.upsample.append(dec)

        if self.us_scale > 1:
            self.refine = ResBlock(c_out, c_out, dropout=drop)
            self._add_film(self.refine, film_channel)

        self.output = nn.Conv3d(c_out, out_channel, kernel_size=1)

        for block in [
            self.enc1,
            self.enc2,
            self.mid_dil,
            self.mid,
            self.dec2,
            self.dec1,
        ]:
            self._add_film(block, film_channel)

        self.global_reduce_map = nn.Conv3d(map_channel, 64, 1)
        self.global_reduce_token = nn.Conv3d(map_channel, kv_channel, 1)
        self.global_pool_token = nn.AdaptiveAvgPool3d((7, 10, 10))

        self.xattn2 = CrossAttention(c2, kv_channel, 128, 4)
        self.xattn_mid = CrossAttention(c3, kv_channel, 128, 4)

    def forward(
        self,
        input_surface,
        input_upper,
        global_token,
        global_map,
        position,
        time,
        sigma=None,
    ):
        B, C, Z, H, W = input_upper.shape

        h_glob = self.emb_glob(global_token)
        h_pos = self.emb_pos(position)
        h_time = self.emb_time(time)
        h_res = self.emb_res(
            torch.full(
                (input_surface.size(0), 1),
                self.res,
                device=input_surface.device,
                dtype=input_surface.dtype,
            )
        )
        feats = torch.cat([h_glob, h_pos, h_time, h_res], dim=-1)
        if sigma is not None and self.include_sigma:
            h_sigma = self.emb_sigma(sigma)
            feats = torch.cat([feats, h_sigma], dim=-1)
        film_base = self.film_mlp(feats)

        global_map_tokens = rearrange(
            self.global_pool_token(self.global_reduce_token(global_map)),
            "b c z h w -> b (z h w) c",
        )

        x = torch.cat(
            [self.sfc_conv(input_surface).unsqueeze(-3), self.up_conv(input_upper)],
            dim=-3,
        )  # (B,64,Z+1,H,W)
        x = torch.cat([x, self.global_reduce_map(global_map)], dim=1)
        e0 = self.enc0(x)  # (B,c1,Z+1,H,W)

        e1 = self.enc1(e0, film_base)
        d1 = self.ds1(e1)  # (B,c2,Z+1,H/2,W/2)

        e2 = self.enc2(d1, film_base)  # (B,c2,Z+1,H/2,W/2)
        e2 = self.xattn2(e2, global_map_tokens)

        if e2.size(-1) > 5 and e2.size(-2) > 5:
            d2 = self.ds2(e2)  # (B,c3,Z+1,H/4,H/4)

            mid = self.mid(d2, film_base)  # (B,c3,Z+1,H/4,H/4)
            mid = self.xattn_mid(mid, global_map_tokens)

            u2 = self.us2(mid)  # (B,c2,Z+1,H/2,H/2)
            u2 = crop_center(u2, e2)
            u2 = self.dec2(torch.cat([u2, e2], dim=1), film_base)

            u1 = self.us1(u2)  # (B,c1,Z+1,H,W)

        else:
            mid = self.mid_dil(e2)
            u1 = self.us1(mid)

        u1 = crop_center(u1, e1)
        u1 = self.dec1(torch.cat([u1, e1], dim=1), film_base)

        for layer in self.upsample:
            u1 = layer(u1)
        if self.us_scale > 1:
            u1 = F.interpolate(u1, (u1.size(-3), *self.target_shape))
            u1 = self.refine(u1)

        out = self.output(u1)  # (B,Cout,Z+1,H,W)

        return out[:, :, 1:, :, :]

    def _add_film(self, block, channel):
        block.film1 = FiLM(channel, block.norm1.num_channels)
        block.film2 = FiLM(channel, block.norm2.num_channels)
