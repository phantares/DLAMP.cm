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
        resolution_input,
        resolution_target,
        single_channel,
        upper_channel,
        output_channel,
        z_input,
        z_target,
        target_var,
        output_mode="regress",
        use_mask=False,
        loss_cfg=None,
        loss_weight={},
        optimizer_cfg=None,
        scheduler_cfg=None,
    ):
        super().__init__()

        self.loss_fn = instantiate(loss_cfg, reduction="none")

        if isinstance(network_cfg, dict):
            network_cfg = OmegaConf.create(network_cfg)
        if isinstance(layer_cfg, dict):
            layer_cfg = OmegaConf.create(layer_cfg)

        network_cfg = OmegaConf.to_container(network_cfg, resolve=True)
        layer_cfg = OmegaConf.to_container(layer_cfg, resolve=True)

        self.save_hyperparameters()

        self.use_global = network_cfg["use_global"]
        if self.use_global:
            self.global_encoder = instantiate(
                network_cfg["global_encoder"],
                single_channel=single_channel,
                upper_channel=upper_channel,
            )

        match output_mode:
            case "regress":
                num_params = 1
            case "norm":
                num_params = 2
        unet_factory = instantiate(network_cfg["unet"], _partial_=True)
        self.unet = unet_factory(
            layer_cfg=layer_cfg,
            single_channel=single_channel,
            upper_channel=upper_channel,
            out_channel=output_channel,
            num_params=num_params,
            use_token=self.use_global,
            use_mask=use_mask,
        )

        example_batch = 2
        example_crop = 3
        example_full_grid = 224
        example_column_km = 96.0
        self.example_input_array = {
            "single": torch.randn(
                example_batch, single_channel, example_full_grid, example_full_grid
            ),
            "upper": torch.randn(
                example_batch,
                upper_channel,
                len(z_input),
                example_full_grid,
                example_full_grid,
            ),
            "time": torch.randn(example_batch, 4, example_full_grid, example_full_grid),
            "column_km": torch.tensor([example_column_km]),
            "column_bottom": torch.randn(example_batch, example_crop),
            "column_left": torch.randn(example_batch, example_crop),
        }

        self.test_outputs = {"regress": []} | ({"mask": []} if use_mask else {})

    def forward(
        self, single, upper, time, column_km, column_bottom, column_left, shuffle=False
    ):
        crop_number = column_bottom.shape[1]
        column_grid = torch.round(column_km // self.hparams.resolution_input)
        target_grid = torch.round(column_km // self.hparams.resolution_target)

        global_token = None
        if self.use_global:
            global_token = self.global_encoder(single, upper)
            global_token = repeat(global_token, "b c -> (b n) c", n=crop_number)

        time = crop_column(
            time,
            column_bottom,
            column_left,
            (column_grid, column_grid),
            output_shape=(target_grid, target_grid),
            mode="nearest",
        )
        time = rearrange(time, "b n c h w -> (b n) c h w")

        column_single = crop_column(
            single,
            column_bottom,
            column_left,
            (column_grid, column_grid),
            output_shape=(target_grid, target_grid),
            mode="nearest",
        )
        column_single = rearrange(column_single, "b n c h w -> (b n) c h w")

        column_upper = crop_column(
            upper,
            column_bottom,
            column_left,
            (column_grid, column_grid),
            output_shape=(target_grid, target_grid),
            mode="nearest",
        )
        column_upper = rearrange(column_upper, "b n c z h w -> (b n) c z h w")

        input_upper = interpolate_z(
            column_upper, self.hparams.z_input, self.hparams.z_target
        )

        if shuffle:
            indices = torch.randperm(input_upper.size(0), device=column_single.device)
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

        if shuffle:
            inv_indices = torch.empty_like(indices)
            inv_indices[indices] = torch.arange(len(indices), device=indices.device)

            for k, v in output.items():
                output[k] = v[inv_indices, ...]

        for k, v in output.items():
            output[k] = rearrange(v, "(b n) c z h w -> b n c z h w", n=crop_number)

        return output

    def general_step(
        self,
        single,
        upper,
        time,
        target,
        column_bottom,
        column_left,
        shuffle=False,
        compute_loss_var=True,
    ):
        column_km = (
            torch.tensor([target["regress"].shape[-1] * self.hparams.resolution_target])
            .to(target["regress"].dtype)
            .to(target["regress"].device)
        )

        output = self(
            single=single,
            upper=upper,
            time=time,
            column_km=column_km,
            column_bottom=column_bottom,
            column_left=column_left,
            shuffle=shuffle,
        )

        loss = {}
        if self.hparams.use_mask:
            loss["mask"] = nn.BCELoss()(output["mask"], target["mask"])

            mask_target = target["mask"]
            raw_loss = self.loss_fn(output["regress"], target["regress"])
            masked_loss = raw_loss * mask_target

            loss_sum = rearrange(masked_loss, "b n c z h w -> b n (c z h w)").sum(
                dim=-1
            )
            mask_sum = rearrange(mask_target, "b n c z h w -> b n (c z h w)").sum(
                dim=-1
            )

            valid = mask_sum > 0
            if not valid.any():
                loss["regress"] = torch.tensor(
                    0.0, device=mask_sum.device, requires_grad=self.training
                )
            else:
                loss["regress"] = loss_sum[valid].sum() / mask_sum[valid].sum()

            weight_mask = self.hparams.loss_weight.get("mask", 1.0)
            weight_regress = self.hparams.loss_weight.get("regress", 1.0)
            loss["total"] = (
                weight_mask * loss["mask"] + weight_regress * loss["regress"]
            )

            output_result = torch.where(output["mask"] > 0.5, output["regress"], -1.0)

        else:
            loss["total"] = self.loss_fn(output["regress"], target["regress"]).mean()
            output_result = output["regress"]

        if compute_loss_var:
            loss_var = self.loss_fn(output_result, target["regress"]).mean(
                dim=(0, 1, -1, -2)
            )
        else:
            loss_var = None

        return loss, loss_var, output

    def configure_optimizers(self):
        optimizer = instantiate(self.hparams.optimizer_cfg, params=self.parameters())

        schedulers = [
            instantiate(s_cfg, optimizer=optimizer)
            for s_cfg in self.hparams.scheduler_cfg.schedulers
        ]
        scheduler = instantiate(
            self.hparams.scheduler_cfg, optimizer=optimizer, schedulers=schedulers
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def training_step(self, batch, batch_idx):
        single, upper, time, target_regress, column_bottom, column_left = batch

        target = {"regress": target_regress}
        if self.hparams.use_mask:
            target["mask"] = (target_regress > -1.0).float()

        loss, loss_var, _ = self.general_step(
            single, upper, time, target, column_bottom, column_left, shuffle=True
        )

        self._log_loss_var(
            loss, loss_var, self.hparams.target_var, self.hparams.z_target, "train"
        )

        return loss["total"]

    def validation_step(self, batch, batch_idx):
        single, upper, time, target_regress, column_bottom, column_left = batch

        target = {"regress": target_regress}
        if self.hparams.use_mask:
            target["mask"] = (target_regress > -1.0).float()

        loss, loss_var, _ = self.general_step(
            single, upper, time, target, column_bottom, column_left
        )

        self._log_loss_var(
            loss, loss_var, self.hparams.target_var, self.hparams.z_target, "val"
        )

        return loss["total"]

    def test_step(self, batch, batch_idx):
        single, upper, time, target_regress, column_bottom, column_left = batch

        target = {"regress": target_regress}
        if self.hparams.use_mask:
            target["mask"] = (target_regress > -1.0).float()

        loss, _, output = self.general_step(
            single,
            upper,
            time,
            target,
            column_bottom,
            column_left,
            compute_loss_var=False,
        )

        loss = {k: v.detach() for k, v in loss.items()}
        self.log(
            f"total_test",
            loss["total"].item(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        output = {k: v.detach().float().cpu() for k, v in output.items()}
        [self.test_outputs[k].append(v) for k, v in output.items()]

    def _log_loss_var(self, loss, loss_var, variable_name, level, stage):
        if self.hparams.use_mask:
            self.log(
                f"mask_{stage}",
                loss["mask"],
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
            self.log(
                f"regress_{stage}",
                loss["regress"],
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )

        self.log(
            f"total_{stage}",
            loss["total"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        for z, lev in enumerate(level):
            for n, var in enumerate(variable_name):
                self.log(
                    f"{stage}/{var}{lev}",
                    loss_var[n, z],
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )

    def generate_sample(self, batch):
        single, upper, time, target, column_bottom, column_left = batch
        column_km = (
            torch.tensor([target.shape[-1] * self.hparams.resolution_target])
            .to(target.dtype)
            .to(device=target.device)
        )

        output = self(
            single=single[0:1],
            upper=upper[0:1],
            time=time[0:1],
            column_km=column_km,
            column_bottom=column_bottom[0:1, 0:1],
            column_left=column_left[0:1, 0:1],
        )

        if self.hparams.use_mask:
            output["regress"] = torch.where(
                output["mask"] > 0.5, output["regress"], -1.0
            )

        return output["regress"]
