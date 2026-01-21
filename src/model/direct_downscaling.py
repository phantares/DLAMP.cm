import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.v2 import functional as Ftrans
from einops import rearrange, repeat


from utils import interpolate_z, crop_column
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
        enable_global_encoder=True,
        use_global_token=True,
        use_global_map_embed=True,
        use_global_map_cross_attn=True,
    ):
        super().__init__()

        column_grid = column_km // resolution_input
        target_grid = column_km // resolution_target

        sigma_min = sigma_min if sigma_min is not None else 0
        sigma_max = sigma_max if sigma_max is not None else float("inf")

        self.save_hyperparameters()

        self.hparams.column_grid = column_grid
        self.hparams.target_grid = target_grid

        self.enable_global_encoder = enable_global_encoder
        self.use_global_map = use_global_map_embed or use_global_map_cross_attn

        if enable_global_encoder and (use_global_token or self.use_global_map):
            self.global_encoder = GlobalEncoder(
                resolution=resolution_input,
                single_channel=single_channel,
                upper_channel=upper_channel,
                use_map=self.use_global_map,
                use_token=use_global_token,
            )
        else:
            self.enable_global_encoder = False
            self.use_global_map = False

        self.unet = Unet(
            target_horizontal_shape=(target_grid, target_grid),
            input_resolution=resolution_input,
            target_resolution=resolution_target,
            single_channel=single_channel,
            upper_channel=upper_channel + output_channel,
            out_channel=output_channel,
            use_token=enable_global_encoder and use_global_token,
            use_map_embed=enable_global_encoder and use_global_map_embed,
            use_map_cross_attn=enable_global_encoder and use_global_map_cross_attn,
            include_sigma=True,
        )

        example_batch = 2
        example_crop = 3
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
                example_batch,
                example_crop,
                output_channel,
                len(z_target),
                target_grid,
                target_grid,
            ),
            "sigma": (
                torch.randn(example_batch, example_crop, 1, 1, 1, 1) * P_std + P_mean
            )
            .exp()
            .clamp(min=sigma_min, max=sigma_max),
            "column_top": torch.randn(example_batch, example_crop),
            "column_left": torch.randn(example_batch, example_crop),
        }

    def forward(
        self, single, upper, time, noise, sigma, column_top, column_left, shaffle=False
    ):
        crop_number = noise.shape[1]
        noise = rearrange(noise, "b n c z h w -> (b n) c z h w")
        sigma = rearrange(sigma, "b n 1 1 1 1 -> (b n) 1 1 1 1")
        time = repeat(time, "b c -> (b n) c", n=crop_number)

        c_skip = self.hparams.sigma_data**2 / (sigma**2 + self.hparams.sigma_data**2)
        c_out = (
            sigma
            * self.hparams.sigma_data
            / (sigma**2 + self.hparams.sigma_data**2).sqrt()
        )
        c_in = 1 / (self.hparams.sigma_data**2 + sigma**2).sqrt()
        c_noise = sigma.log() / 4

        if self.enable_global_encoder:
            global_token, global_map = self.global_encoder(single, upper)

            if global_token is not None:
                global_token = repeat(global_token, "b c -> (b n) c", n=crop_number)
            if global_map is not None:
                global_map = repeat(
                    global_map, "b c z h w -> (b n) c z h w", n=crop_number
                )

        else:
            global_token = None
            global_map = None

        column_single = crop_column(
            single,
            column_top,
            column_left,
            (self.hparams.column_grid, self.hparams.column_grid),
            output_shape=(self.hparams.target_grid, self.hparams.target_grid),
            mode="nearest",
            align_corners=True,
        )
        column_single = rearrange(column_single, "b n c h w -> (b n) c h w")

        column_upper = crop_column(
            upper,
            column_top,
            column_left,
            (self.hparams.column_grid, self.hparams.column_grid),
            output_shape=(self.hparams.target_grid, self.hparams.target_grid),
            mode="nearest",
            align_corners=True,
        )
        column_upper = rearrange(column_upper, "b n c z h w -> (b n) c z h w")

        column_upper_all = interpolate_z(
            column_upper, self.hparams.z_input, self.hparams.z_target
        )
        input_upper = torch.cat([column_upper_all, noise * c_in], dim=1)

        column_top = rearrange(column_top, "b n -> (b n)")
        column_left = rearrange(column_left, "b n -> (b n)")

        if global_map is not None and self.use_global_map:
            global_map = project_global_to_roi(
                global_map,
                column_left,
                column_top,
                output_size=self.hparams.target_grid,
                window_size=self.hparams.column_grid,
                global_size=self.hparams.global_grid,
            )
            global_map_upper = interpolate_z(
                global_map[:, :, 1:, :, :], self.hparams.z_input, self.hparams.z_target
            )

            global_map_hr = torch.cat(
                [global_map[:, :, 0:1, :, :], global_map_upper], dim=-3
            )

        else:
            global_map_hr = None

        if shaffle:
            indices = torch.randperm(input_upper.size(0))
        else:
            indices = torch.arange(input_upper.size(0), device=column_single.device)

        F_x = self.unet(
            input_surface=column_single[indices, ...],
            input_upper=input_upper[indices, ...],
            position=torch.stack(
                [column_top[indices, ...], column_left[indices, ...]],
                dim=1,
            ),
            time=time[indices, ...],
            global_token=(
                global_token[indices, ...] if global_token is not None else None
            ),
            global_map=(
                global_map_hr[indices, ...] if global_map_hr is not None else None
            ),
            sigma=c_noise[indices, ...].flatten(1),
        )

        output = c_skip * noise + c_out * F_x
        output = rearrange(output, "(b n) c z h w -> b n c z h w", n=crop_number)

        return output

    def general_step(self, single, upper, time, target, column_top, column_left):
        rnd_normal = torch.randn(
            [target.shape[0], column_top.shape[1], 1, 1, 1, 1], device=target.device
        )
        sigma = (rnd_normal * self.hparams.P_std + self.hparams.P_mean).exp()
        sigma = sigma.clamp(min=self.hparams.sigma_min, max=self.hparams.sigma_max)
        weight = (sigma**2 + self.hparams.sigma_data**2) / (
            sigma * self.hparams.sigma_data
        ) ** 2
        noise = torch.randn_like(target) * sigma

        output = self(
            single=single,
            upper=upper,
            time=time,
            noise=target + noise,
            sigma=sigma,
            column_top=column_top,
            column_left=column_left,
            shaffle=True,
        )

        loss_var = nn.MSELoss(reduction="none")(output, target).mean(dim=(0, 1, -1, -2))
        loss = torch.mean(weight * loss_var)

        return loss, loss_var

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-5)

    def training_step(self, batch, batch_idx):
        single, upper, time, target, column_top, column_left = batch
        loss, loss_var = self.general_step(
            single, upper, time, target, column_top, column_left
        )

        self.log(
            "total_train",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self._log_loss_var(
            loss_var, self.hparams.target_var, self.hparams.z_target, "train"
        )

        return loss

    def validation_step(self, batch, batch_idx):
        single, upper, time, target, column_top, column_left = batch
        loss, loss_var = self.general_step(
            single, upper, time, target, column_top, column_left
        )

        self.log(
            "total_val",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self._log_loss_var(
            loss_var, self.hparams.target_var, self.hparams.z_target, "val"
        )

        return loss

    def _log_loss_var(self, loss, variable_name, level, stage):
        for z, lev in enumerate(level):
            for n, var in enumerate(variable_name):
                self.log(
                    f"{stage}/{var}{lev}",
                    loss[n, z],
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )
