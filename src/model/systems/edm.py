import lightning as L
import torch
import torch.nn as nn
from einops import rearrange, repeat
from hydra.utils import instantiate
from omegaconf import OmegaConf

from utils import interpolate_z, crop_column


class EDM(L.LightningModule):
    def __init__(
        self,
        network_cfg,
        layer_cfg,
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
    ):
        super().__init__()

        network_cfg = OmegaConf.to_container(network_cfg, resolve=True)
        layer_cfg = OmegaConf.to_container(layer_cfg, resolve=True)

        column_grid = column_km // resolution_input
        target_grid = column_km // resolution_target

        sigma_min = sigma_min if sigma_min is not None else 0
        sigma_max = sigma_max if sigma_max is not None else float("inf")

        self.save_hyperparameters(ignore=["column_km"])

        self.hparams.column_grid = column_grid
        self.hparams.target_grid = target_grid

        unet_factory = instantiate(network_cfg, _partial_=True)
        self.unet = unet_factory(
            layer_cfg=layer_cfg,
            target_horizontal_shape=(target_grid, target_grid),
            target_resolution=resolution_target,
            single_channel=single_channel,
            upper_channel=upper_channel + output_channel,
            out_channel=output_channel,
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

        if shaffle:
            indices = torch.randperm(input_upper.size(0))
        else:
            indices = torch.arange(input_upper.size(0), device=column_single.device)

        F_x = self.unet(
            input_surface=column_single[indices, ...],
            input_upper=input_upper[indices, ...],
            time=time[indices, ...],
            sigma=c_noise[indices, ...].flatten(1),
        )

        output = c_skip * noise + c_out * F_x
        output = rearrange(output, "(b n) c z h w -> b n c z h w", n=crop_number)

        return output

    def general_step(
        self, single, upper, time, target, sigma, column_top, column_left, shaffle=False
    ):
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
            shaffle=shaffle,
        )

        loss_var = nn.MSELoss(reduction="none")(output, target).mean(dim=(0, 1, -1, -2))
        loss = torch.mean(weight * loss_var)

        return loss, loss_var

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-5)

    def training_step(self, batch, batch_idx):
        single, upper, time, target, column_top, column_left = batch

        rnd_normal = torch.randn(
            [target.shape[0], target.shape[1], 1, 1, 1, 1], device=target.device
        )
        sigma = (rnd_normal * self.hparams.P_std + self.hparams.P_mean).exp()
        sigma = sigma.clamp(min=self.hparams.sigma_min, max=self.hparams.sigma_max)

        loss, loss_var = self.general_step(
            single, upper, time, target, sigma, column_top, column_left, shaffle=True
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

        v_gen = torch.Generator(device=target.device).manual_seed(42 + batch_idx)
        rnd_normal = torch.randn(
            [target.shape[0], target.shape[1], 1, 1, 1, 1],
            device=target.device,
            generator=v_gen,
        )
        sigma = (rnd_normal * self.hparams.P_std + self.hparams.P_mean).exp()
        sigma = sigma.clamp(min=self.hparams.sigma_min, max=self.hparams.sigma_max)

        loss, loss_var = self.general_step(
            single, upper, time, target, sigma, column_top, column_left
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
