import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.v2 import RandomCrop, functional as Ftrans


from utils import interpolate_z
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
        target_var,
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
        self.factor = resolution_input / resolution_target

        self.global_encoder = GlobalEncoder(
            resolution=resolution_input,
            single_channel=single_channel,
            upper_channel=upper_channel,
        )
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
            "single": torch.randn(
                example_batch, single_channel, global_grid, global_grid
            ),
            "upper": torch.randn(
                example_batch,
                upper_channel,
                len(z_input),
                global_grid,
                global_grid,
            ),
            "time": torch.randn(example_batch, 4),
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
            "column_top": torch.randn(example_batch * crop_number),
            "column_left": torch.randn(example_batch * crop_number),
        }

    def forward(
        self, single, upper, time, noise, sigma, column_top, column_left, shaffle=False
    ):
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
        time = time.repeat_interleave(self.crop_number, dim=0)

        single = single.repeat_interleave(self.crop_number, dim=0)
        upper = upper.repeat_interleave(self.crop_number, dim=0)
        column_single = []
        column_upper = []
        for b in range(single.shape[0]):
            column_single.append(
                Ftrans.crop(
                    single[b],
                    int(column_top[b]),
                    int(column_left[b]),
                    self.hparams.column_grid,
                    self.hparams.column_grid,
                )
            )
            column_upper.append(
                Ftrans.crop(
                    upper[b],
                    int(column_top[b]),
                    int(column_left[b]),
                    self.hparams.column_grid,
                    self.hparams.column_grid,
                )
            )
        column_single = torch.stack(column_single)
        column_upper = torch.stack(column_upper)

        column_upper_all = interpolate_z(
            column_upper, self.hparams.z_input, self.hparams.z_target
        )

        noise_lr = F.adaptive_avg_pool3d(
            noise, (noise.size(-3), self.hparams.column_grid, self.hparams.column_grid)
        )
        input_upper = torch.cat([column_upper_all, noise_lr * c_in], dim=1)

        global_map = project_global_to_roi(
            global_map,
            column_left,
            column_top,
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

        if shaffle:
            indices = torch.randperm(input_upper.size(0))
        else:
            indices = torch.arange(input_upper.size(0), device=column_single.device)
        F_x = self.unet(
            column_single[indices, ...],
            input_upper[indices, ...],
            global_token[indices, ...],
            global_map_hr[indices, ...],
            torch.stack(
                [column_top[indices, ...], column_left[indices, ...]],
                dim=1,
            ),
            time[indices, ...],
            sigma=c_noise[indices, ...].flatten(1),
        )

        output = c_skip * noise + c_out * F_x

        return output

    def general_step(self, single, upper, time, target):
        device = single.device
        dtype = single.dtype

        tops = []
        lefts = []
        column_target = []
        for b in range(single.shape[0]):
            for _ in range(self.crop_number):
                top, left, h, w = RandomCrop.get_params(
                    single[b], (self.hparams.column_grid, self.hparams.column_grid)
                )
                column_target.append(
                    Ftrans.crop(
                        target[b],
                        int(top * self.factor),
                        int(left * self.factor),
                        self.hparams.target_grid,
                        self.hparams.target_grid,
                    )
                )

                tops.append(top)
                lefts.append(left)

        tops = torch.tensor(tops, dtype=dtype, device=device)
        lefts = torch.tensor(lefts, dtype=dtype, device=device)
        column_target = torch.stack(column_target).to(dtype)

        rnd_normal = torch.randn(
            [column_target.shape[0], 1, 1, 1, 1], device=target.device
        )
        sigma = (rnd_normal * self.hparams.P_std + self.hparams.P_mean).exp()
        sigma = sigma.clamp(min=self.hparams.sigma_min, max=self.hparams.sigma_max)
        weight = (sigma**2 + self.hparams.sigma_data**2) / (
            sigma * self.hparams.sigma_data
        ) ** 2
        noise = torch.randn_like(column_target) * sigma

        output = self(
            single=single,
            upper=upper,
            time=time,
            noise=column_target + noise,
            sigma=sigma,
            column_top=tops,
            column_left=lefts,
            shaffle=True,
        )

        loss = torch.mean(weight * nn.MSELoss(reduction="none")(output, column_target))

        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-5)

    def training_step(self, batch, batch_idx):
        single, upper, time, target = batch
        loss = self.general_step(single, upper, time, target)

        return loss

    def validation_step(self, batch, batch_idx):
        single, upper, time, target = batch
        loss = self.general_step(single, upper, time, target)

        return loss
