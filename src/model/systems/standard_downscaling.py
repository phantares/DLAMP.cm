import lightning as L
import torch
import torch.nn as nn
from einops import rearrange, repeat
from hydra.utils import instantiate
from omegaconf import OmegaConf

from utils import interpolate_z, crop_column


class StandardDownscaling(L.LightningModule):
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
    ):
        super().__init__()

        network_cfg = OmegaConf.to_container(network_cfg, resolve=True)
        layer_cfg = OmegaConf.to_container(layer_cfg, resolve=True)

        column_grid = column_km // resolution_input
        target_grid = column_km // resolution_target

        self.save_hyperparameters(ignore=["column_km"])

        self.column_grid = column_grid
        self.target_grid = target_grid

        self.use_global = network_cfg["use_global"]
        if self.use_global:
            self.global_encoder = instantiate(
                network_cfg["global_encoder"],
                single_channel=single_channel,
                upper_channel=upper_channel,
            )

        unet_factory = instantiate(network_cfg["unet"], _partial_=True)
        self.unet = unet_factory(
            layer_cfg=layer_cfg,
            target_horizontal_shape=(target_grid, target_grid),
            single_channel=single_channel,
            upper_channel=upper_channel,
            out_channel=output_channel,
            use_token=self.use_global,
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
            "column_top": torch.randn(example_batch, example_crop),
            "column_left": torch.randn(example_batch, example_crop),
        }

    def forward(self, single, upper, time, column_top, column_left, shaffle=False):
        crop_number = column_top.shape[1]

        time = repeat(time, "b c -> (b n) c", n=crop_number)

        global_token = None
        if self.use_global:
            global_token = self.global_encoder(single, upper)
            global_token = repeat(global_token, "b c -> (b n) c", n=crop_number)

        column_single = crop_column(
            single,
            column_top,
            column_left,
            (self.column_grid, self.column_grid),
            output_shape=(self.target_grid, self.target_grid),
            mode="nearest",
            align_corners=True,
        )
        column_single = rearrange(column_single, "b n c h w -> (b n) c h w")

        column_upper = crop_column(
            upper,
            column_top,
            column_left,
            (self.column_grid, self.column_grid),
            output_shape=(self.target_grid, self.target_grid),
            mode="nearest",
            align_corners=True,
        )
        column_upper = rearrange(column_upper, "b n c z h w -> (b n) c z h w")

        input_upper = interpolate_z(
            column_upper, self.hparams.z_input, self.hparams.z_target
        )

        if shaffle:
            indices = torch.randperm(input_upper.size(0))
        else:
            indices = torch.arange(input_upper.size(0), device=column_single.device)

        output = self.unet(
            input_surface=column_single[indices, ...],
            input_upper=input_upper[indices, ...],
            time=time[indices, ...],
            global_token=(
                global_token[indices, ...] if global_token is not None else None
            ),
        )

        output = rearrange(output, "(b n) c z h w -> b n c z h w", n=crop_number)

        return output

    def general_step(
        self, single, upper, time, target, column_top, column_left, shaffle=False
    ):

        output = self(
            single=single,
            upper=upper,
            time=time,
            column_top=column_top,
            column_left=column_left,
            shaffle=shaffle,
        )

        loss_var = nn.MSELoss(reduction="none")(output, target).mean(dim=(0, 1, -1, -2))
        loss = torch.mean(loss_var)

        return loss, loss_var

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-5)

    def training_step(self, batch, batch_idx):
        single, upper, time, target, column_top, column_left = batch

        loss, loss_var = self.general_step(
            single, upper, time, target, column_top, column_left, shaffle=True
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

    def generate_sample(self, batch):
        single, upper, time, target, column_top, column_left = batch

        return self(
            single=single[0:1],
            upper=upper[0:1],
            time=time[0:1],
            column_top=column_top[0:1, 0:1],
            column_left=column_left[0:1, 0:1],
        )
