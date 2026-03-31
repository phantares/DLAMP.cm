import torch
import torch.nn as nn
from hydra.utils import instantiate

from model.layers import MLP


class UNet(nn.Module):
    def __init__(
        self,
        layer_cfg,
        single_channel,
        upper_channel,
        out_channel,
        base_channel=128,
        use_film=False,
        use_token=False,
        token_channel=768,
        token_emb_channel=512,
        time_emb_channel=128,
        include_sigma=False,
        sigma_emb_channel=32,
    ):
        super().__init__()

        c1, c2, c3 = base_channel, int(base_channel * 5 / 4), int(base_channel * 3 / 2)

        self.use_film = use_film
        if use_film:
            film_channel = base_channel * 2

            self.emb_time = MLP(4, 64, time_emb_channel)

            self.use_token = use_token
            if use_token:
                self.emb_glob = MLP(token_channel, token_emb_channel, token_emb_channel)
            else:
                token_emb_channel = 0

            self.include_sigma = include_sigma
            if include_sigma:
                self.emb_sigma = MLP(1, 64, sigma_emb_channel)
            else:
                sigma_emb_channel = 0

            self.film_mlp = MLP(
                token_emb_channel + time_emb_channel + sigma_emb_channel,
                512,
                film_channel,
            )
        else:
            film_channel = 0

        self.layer_factory = instantiate(layer_cfg, _partial_=True)

        self.sfc_conv = nn.Conv2d(single_channel, 64, 1)
        self.up_conv = nn.Conv3d(upper_channel, 64, 1)
        emb_channel = 64
        self.enc0 = nn.Conv3d(emb_channel, c1, 3, padding=1)

        self.enc1 = self.layer_factory(dim=c1, film_channel=film_channel)
        self.ds1 = nn.Conv3d(c1, c2, (1, 2, 2), stride=(1, 2, 2), padding=0)
        self.enc2 = self.layer_factory(dim=c2, film_channel=film_channel)
        self.ds2 = nn.Conv3d(c2, c3, (1, 2, 2), stride=(1, 2, 2), padding=0)

        self.mid = self.layer_factory(dim=c3, film_channel=film_channel)

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

        self.output = nn.Conv3d(c1, out_channel, kernel_size=1)

    def forward(
        self,
        input_surface,
        input_upper,
        time,
        global_token=None,
        sigma=None,
    ):

        if self.use_film:
            feats = self.emb_time(time)

            if global_token is not None and self.use_token:
                h_glob = self.emb_glob(global_token)
                feats = torch.cat([feats, h_glob], dim=-1)

            if sigma is not None and self.include_sigma:
                h_sigma = self.emb_sigma(sigma)
                feats = torch.cat([feats, h_sigma], dim=-1)

            film_base = self.film_mlp(feats)

        else:
            film_base = None

        x = torch.cat(
            [self.sfc_conv(input_surface).unsqueeze(-3), self.up_conv(input_upper)],
            dim=-3,
        )  # (B,64,Z+1,H,W)
        e0 = self.enc0(x)  # (B,c1,Z+1,H,W)

        e1 = self.enc1(e0, film_base)
        d1 = self.ds1(e1)  # (B,c2,Z+1,H/2,W/2)

        e2 = self.enc2(d1, film_base)  # (B,c2,Z+1,H/2,W/2)

        d2 = self.ds2(e2)  # (B,c3,Z+1,H/4,H/4)

        mid = self.mid(d2, film_base)  # (B,c3,Z+1,H/4,H/4)

        u2 = self.us2(mid)  # (B,c2,Z+1,H/2,H/2)
        u2 = torch.cat([u2, e2], dim=1)
        u2 = self.dec2_reduce(u2)
        u2 = self.dec2(u2, film_base)

        u1 = self.us1(u2)  # (B,c1,Z+1,H,W)
        u1 = torch.cat([u1, e1], dim=1)
        u1 = self.dec1_reduce(u1)
        u1 = self.dec1(u1, film_base)

        out = self.output(u1)  # (B,Cout,Z+1,H,W)

        return out[:, :, 1:, :, :]
