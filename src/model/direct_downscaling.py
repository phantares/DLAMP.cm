import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.v2 import functional as Ftrans


from utils import RandomCropper, interpolate_z
from model.architecture import (
    GlobalEncoder,
    project_global_to_roi,
    Unet,
)


class DirectDownscaling(L.LightningModule):
    def __init__(
        self,
        global_grid,
        resolution_input,
        resolution_target,
        column_km,
        crop_number,
        single_channel,
        upper_channel,
        output_channel,
        z_input,
        z_target,
        P_mean,
        P_std,
        sigma_data,
        sigma_min=None,
        sigma_max=None,
    ):
        super().__init__()

        column_grid = column_km // resolution_input
        target_grid = column_km // resolution_target

        sigma_min = sigma_min if sigma_min is not None else 0
        sigma_max = sigma_max if sigma_max is not None else float("inf")

        self.save_hyperparameters(ignore=["crop_number"])

        self.hparams.column_grid = column_grid
        self.hparams.target_grid = target_grid

        self.crop_number = crop_number

        self.global_encoder = GlobalEncoder(
            resolution=resolution_input,
            single_channel=single_channel,
            upper_channel=upper_channel,
        )
        self.cropper = RandomCropper((column_grid, column_grid))
        self.unet = Unet(
            target_horizontal_shape=(target_grid, target_grid),
            input_resolution=resolution_input,
            target_resolution=resolution_target,
            single_channel=single_channel,
            upper_channel=upper_channel + output_channel,
            out_channel=output_channel,
            include_sigma=True,
        )

        example_batch = 5
        self.example_input_array = {
            "surface": torch.randn(
                example_batch, single_channel, global_grid, global_grid
            ),
            "upper": torch.randn(
                example_batch,
                upper_channel,
                len(z_input),
                global_grid,
                global_grid,
            ),
            "time": torch.repeat_interleave(
                torch.randn(example_batch, 4), crop_number, dim=0
            ),
            "noise": torch.randn(
                example_batch * crop_number,
                output_channel,
                len(z_target),
                target_grid,
                target_grid,
            ),
            "sigma": (
                torch.randn(example_batch * crop_number, 1, 1, 1, 1) * P_std + P_mean
            )
            .exp()
            .clamp(min=sigma_min, max=sigma_max),
        }

    def forward(self, single, upper, time, noise, sigma):
        c_skip = self.hparams.sigma_data**2 / (sigma**2 + self.hparams.sigma_data**2)
        c_out = (
            sigma
            * self.hparams.sigma_data
            / (sigma**2 + self.hparams.sigma_data**2).sqrt()
        )
        c_in = 1 / (self.hparams.sigma_data**2 + sigma**2).sqrt()
        c_noise = sigma.log() / 4

        global_token, global_map = self.global_encoder(single, upper)
        global_token = global_token.repeat_interleave(self.crop_number, dim=0)
        global_map = global_map.repeat_interleave(self.crop_number, dim=0)

        column_surface, column_upper, cx, cy = self.cropper.crop(
            single, upper, crop_number=self.crop_number
        )
        column_upper_all = interpolate_z(
            column_upper, self.hparams.z_input, self.hparams.z_target
        )
        self.top = cx * 2 - (self.hparams.column_grid - 1)
        self.left = cy * 2 - (self.hparams.column_grid - 1)

        noise_lr = F.adaptive_avg_pool3d(
            noise, (noise.size(-3), self.hparams.column_grid, self.hparams.column_grid)
        )
        input_upper = torch.cat([column_upper_all, noise_lr * c_in], dim=1)

        global_map = project_global_to_roi(
            global_map,
            cy,
            cx,
            output_size=self.hparams.column_grid,
            window_size=self.hparams.column_grid,
            global_size=self.hparams.global_grid,
        )
        global_map_upper = interpolate_z(
            global_map[:, :, 1:, :, :], self.hparams.z_input, self.hparams.z_target
        )
        global_map_hr = torch.cat(
            [global_map[:, :, 0:1, :, :], global_map_upper], dim=-3
        )

        shaffle = torch.randperm(input_upper.size(0))
        F_x = self.unet(
            column_surface[shaffle, ...],
            input_upper[shaffle, ...],
            global_token[shaffle, ...],
            global_map_hr[shaffle, ...],
            torch.stack([cx[shaffle, ...], cy[shaffle, ...]], dim=1),
            time[shaffle, ...],
            sigma=c_noise[shaffle, ...].flatten(1),
        )

        output = c_skip * noise + c_out * F_x
        print(output.dtype)

        return output

    def general_step(self, single, upper, time, target):
        B = target.shape[0]

        rnd_normal = torch.randn([B, 1, 1, 1, 1], device=target.device)
        sigma = (rnd_normal * self.hparams.P_std + self.hparams.P_mean).exp()
        sigma = sigma.clamp(min=self.hparams.sigma_min, max=self.hparams.sigma_max)
        weight = (sigma**2 + self.hparams.sigma_data**2) / (
            sigma * self.hparams.sigma_data
        ) ** 2
        noise = torch.randn_like(target) * sigma

        output = self(
            single=single, upper=upper, time=time, noise=target + noise, sigma=sigma
        )

        target = target.repeat_interleave(self.crop_number, dim=0)
        cropped_target = []
        for b in range(target.shape[0]):
            cropped_target.append(
                Ftrans.crop(
                    target,
                    self.top[b],
                    self.left[b],
                    self.hparams.column_grid,
                    self.hparams.column_grid,
                )
            )

        loss = torch.mean(
            weight * nn.MSE(output, torch.tensor(cropped_target, dtype=target.dtype))
        )

        return loss, output

    def training_step(self, batch, batch_idx):
        single, upper, time, target = batch
        loss, output = self.general_step(single, upper, time, target)

        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3)
