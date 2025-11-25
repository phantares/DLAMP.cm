import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        column_km,
        resolution_input,
        resolution_target,
        single_channel,
        upper_channel,
        output_channel,
        crop_number,
        z_input,
        z_target,
        P_mean,
        P_std,
        sigma_data,
        sigma_min=None,
        sigma_max=None,
    ):
        super().__init__()

        self.global_grid = global_grid
        self.column_grid = column_km // resolution_input
        self.target_grid = column_km // resolution_target

        self.global_encoder = GlobalEncoder(
            resolution=resolution_input,
            single_channel=single_channel,
            upper_channel=upper_channel,
        )
        self.cropper = RandomCropper((self.column_grid, self.column_grid))
        self.unet = Unet(
            target_horizontal_shape=(self.target_grid, self.target_grid),
            input_resolution=resolution_input,
            target_resolution=resolution_target,
            single_channel=single_channel,
            upper_channel=upper_channel + output_channel,
            out_channel=output_channel,
            include_sigma=True,
        )

        self.z_input = z_input
        self.z_target = z_target

        self.crop_number = crop_number

        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min if sigma_min else 0
        self.sigma_max = sigma_max if sigma_max else float("inf")

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
                self.target_grid,
                self.target_grid,
            ),
            "sigma": (
                torch.randn(example_batch * crop_number, 1, 1, 1, 1) * P_std + P_mean
            )
            .exp()
            .clamp(min=self.sigma_min, max=self.sigma_max),
        }

    def forward(self, surface, upper, time, noise, sigma):
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
        c_noise = sigma.log() / 4

        global_token, global_map = self.global_encoder(surface, upper)
        global_token = global_token.repeat_interleave(self.crop_number, dim=0)
        global_map = global_map.repeat_interleave(self.crop_number, dim=0)

        column_surface, column_upper, cx, cy = self.cropper.crop(
            surface, upper, crop_number=self.crop_number
        )
        column_upper_all = interpolate_z(column_upper, self.z_input, self.z_target)

        noise_lr = F.adaptive_avg_pool3d(
            noise, (noise.size(-3), self.column_grid, self.column_grid)
        )
        input_upper = torch.cat([column_upper_all, noise_lr * c_in], dim=1)

        global_map = project_global_to_roi(
            global_map,
            cy,
            cx,
            output_size=self.column_grid,
            window_size=self.column_grid,
            global_size=self.global_grid,
        )
        global_map_upper = interpolate_z(
            global_map[:, :, 1:, :, :], self.z_input, self.z_target
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

    def training_step(self, batch, batch_idx):
        single, upper, target = batch
        output = self(single, upper)
        loss = nn.MSE(target, output)

        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3)
