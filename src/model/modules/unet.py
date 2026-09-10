import torch
from hydra.utils import instantiate
from model.layers import MLP, VerticalAttention
from torch import nn


class UNet(nn.Module):
    def __init__(
        self,
        layer_cfg,
        single_channel,
        upper_channel,
        out_channel,
        num_params,
        base_channel=128,
        use_mask=False,
        mask_mode="concat",  # "concat" | "separate"
        use_vert_attn=False,
        include_time=False,
        time_emb_channel=128,
        include_sigma=False,
        sigma_emb_channel=32,
    ):
        super().__init__()

        c1, c2, c3 = base_channel, int(base_channel * 5 / 4), int(base_channel * 3 / 2)

        film_channel = base_channel * 2

        self.include_sigma = include_sigma
        if include_sigma:
            self.emb_sigma = nn.Sequential(
                MLP(1, 64, sigma_emb_channel), MLP(sigma_emb_channel, 512, film_channel)
            )

        self.include_time = include_time
        if include_time:
            self.emb_time = nn.Sequential(
                nn.Conv2d(4, time_emb_channel, 3, padding=1),
                nn.SiLU(),
                nn.Conv2d(time_emb_channel, film_channel, 1),
            )

        self.layer_factory = instantiate(layer_cfg, _partial_=True)

        self.sfc_conv = nn.Conv2d(single_channel, 64, 1)
        self.up_conv = nn.Conv3d(upper_channel, 64, 1)
        emb_channel = 64
        self.enc0 = nn.Conv3d(emb_channel, c1, 3, padding=1, padding_mode="replicate")

        self.enc1 = self.layer_factory(dim=c1, film_channel=film_channel)
        self.ds1 = nn.Conv3d(c1, c2, (1, 2, 2), stride=(1, 2, 2), padding=0)
        self.enc2 = self.layer_factory(dim=c2, film_channel=film_channel)
        self.ds2 = nn.Conv3d(c2, c3, (1, 2, 2), stride=(1, 2, 2), padding=0)

        self.mid = self.layer_factory(dim=c3, film_channel=film_channel)

        self.use_attn = use_vert_attn
        if use_vert_attn:
            self.attn = VerticalAttention(c3, num_heads=8, mlp_ratio=4)

        self.us2 = nn.ConvTranspose3d(
            c3,
            c2,
            kernel_size=(1, 2, 2),
            stride=(1, 2, 2),
            padding=0,
        )
        self.dec2_reduce = nn.Conv3d(c2 + c2, c2, kernel_size=1)
        self.dec2 = self.layer_factory(dim=c2, film_channel=film_channel)
        self.us1 = nn.ConvTranspose3d(
            c2,
            c1,
            kernel_size=(1, 2, 2),
            stride=(1, 2, 2),
            padding=0,
        )
        self.dec1_reduce = nn.Conv3d(c1 + c1, c1, kernel_size=1)
        self.dec1 = self.layer_factory(dim=c1, film_channel=film_channel)

        self.use_mask = use_mask
        self.mask_mode = mask_mode
        regress_channel = c1
        if use_mask:
            self.mask = nn.Sequential(
                nn.Conv3d(c1, out_channel, kernel_size=1), nn.Sigmoid()
            )
            if mask_mode == "concat":
                regress_channel = regress_channel + out_channel
        self.regress = nn.Conv3d(
            regress_channel, out_channel * num_params, kernel_size=1
        )

    def forward(
        self,
        input_surface,
        input_upper,
        time=None,
        sigma=None,
    ):

        if time is not None and self.include_time:
            film_time = self.emb_time(time)
        if sigma is not None and self.include_sigma:
            film_sigma = self.emb_sigma(sigma)

        x = torch.cat(
            [self.sfc_conv(input_surface).unsqueeze(-3), self.up_conv(input_upper)],
            dim=-3,
        )  # (B,64,Z+1,H,W)
        e0 = self.enc0(x)  # (B,c1,Z+1,H,W)

        e1 = self.enc1(e0)
        d1 = self.ds1(e1)  # (B,c2,Z+1,H/2,W/2)

        e2 = self.enc2(d1)  # (B,c2,Z+1,H/2,W/2)

        d2 = self.ds2(e2)  # (B,c3,Z+1,H/4,H/4)

        mid = self.mid(d2)  # (B,c3,Z+1,H/4,H/4)
        if self.use_attn:
            mid = self.attn(mid)

        u2 = self.us2(mid)  # (B,c2,Z+1,H/2,H/2)
        u2 = torch.cat([u2, e2], dim=1)
        u2 = self.dec2_reduce(u2)
        u2 = self.dec2(u2)

        u1 = self.us1(u2)  # (B,c1,Z+1,H,W)
        u1 = torch.cat([u1, e1], dim=1)
        u1 = self.dec1_reduce(u1)
        u1 = self.dec1(u1)

        out = {}
        if self.use_mask:
            mask = self.mask(u1)
            out["mask"] = mask[:, :, 1:, :, :]

            if self.mask_mode == "concat":
                u1 = torch.cat([u1, mask], dim=1)

        regress = self.regress(u1)  # (B,Cout,Z+1,H,W)
        out["regress"] = regress[:, :, 1:, :, :]

        return out
