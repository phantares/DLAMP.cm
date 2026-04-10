import lightning as L
import torch
import torch.nn as nn
from einops import rearrange, repeat
from hydra.utils import instantiate
from omegaconf import OmegaConf

from utils import interpolate_z, crop_column


class EDMDownscaling(L.LightningModule):
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
        P_mean,
        P_std,
        sigma_data,
        sigma_min=None,
        sigma_max=None,
        use_mask=False,
        loss_weight={},
    ):
        super().__init__()

        network_cfg = OmegaConf.to_container(network_cfg, resolve=True)
        layer_cfg = OmegaConf.to_container(layer_cfg, resolve=True)

        sigma_min = sigma_min if sigma_min is not None else 0
        sigma_max = sigma_max if sigma_max is not None else float("inf")

        self.save_hyperparameters()

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
            single_channel=single_channel,
            upper_channel=upper_channel + output_channel,
            out_channel=output_channel,
            use_token=self.use_global,
            include_sigma=True,
            use_mask=use_mask,
        )

        example_batch = 2
        example_crop = 3
        example_full_grid = 224
        example_column_km = 96.0
        example_target_grid = int(example_column_km // resolution_target)
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
            "noise": torch.randn(
                example_batch,
                example_crop,
                output_channel,
                len(z_target),
                example_target_grid,
                example_target_grid,
            ),
            "sigma": (
                torch.randn(example_batch, example_crop, 1, 1, 1, 1) * P_std + P_mean
            )
            .exp()
            .clamp(min=sigma_min, max=sigma_max),
            "column_km": torch.tensor([example_column_km]),
            "column_bottom": torch.randn(example_batch, example_crop),
            "column_left": torch.randn(example_batch, example_crop),
        }

    def forward(
        self,
        single,
        upper,
        time,
        noise,
        sigma,
        column_km,
        column_bottom,
        column_left,
        shuffle=False,
    ):
        crop_number = column_bottom.shape[1]
        column_grid = torch.round(column_km // self.hparams.resolution_input)
        target_grid = torch.round(column_km // self.hparams.resolution_target)

        noise = rearrange(noise, "b n c z h w -> (b n) c z h w")
        sigma = rearrange(sigma, "b n 1 1 1 1 -> (b n) 1 1 1 1")

        c_skip = self.hparams.sigma_data**2 / (sigma**2 + self.hparams.sigma_data**2)
        c_out = (
            sigma
            * self.hparams.sigma_data
            / (sigma**2 + self.hparams.sigma_data**2).sqrt()
        )
        c_in = 1 / (self.hparams.sigma_data**2 + sigma**2).sqrt()
        c_noise = sigma.log() / 4

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

        column_upper_all = interpolate_z(
            column_upper, self.hparams.z_input, self.hparams.z_target
        )
        input_upper = torch.cat([column_upper_all, noise * c_in], dim=1)

        if shuffle:
            indices = torch.randperm(input_upper.size(0), device=column_single.device)
        else:
            indices = torch.arange(input_upper.size(0), device=column_single.device)

        F_x = self.unet(
            input_surface=column_single[indices, ...],
            input_upper=input_upper[indices, ...],
            time=time[indices, ...],
            global_token=(
                global_token[indices, ...] if global_token is not None else None
            ),
            sigma=c_noise[indices, ...].flatten(1),
        )

        output = {k: v for k, v in F_x.items()}
        output["regress"] = (
            c_skip[indices, ...] * noise[indices, ...]
            + c_out[indices, ...] * F_x["regress"]
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
        sigma,
        column_bottom,
        column_left,
        shuffle=False,
    ):
        weight = (sigma**2 + self.hparams.sigma_data**2) / (
            sigma * self.hparams.sigma_data
        ) ** 2
        noise = torch.randn_like(target["regress"]) * sigma
        column_km = (
            torch.tensor([target["regress"].shape[-1] * self.hparams.resolution_target])
            .to(target["regress"].dtype)
            .to(target["regress"].device)
        )

        output = self(
            single=single,
            upper=upper,
            time=time,
            noise=target["regress"] + noise,
            sigma=sigma,
            column_km=column_km,
            column_bottom=column_bottom,
            column_left=column_left,
            shuffle=shuffle,
        )

        loss = {}
        if self.hparams.use_mask:
            loss["mask"] = nn.BCEWithLogitsLoss()(output["mask"], target["mask"])

            mask_target = target["mask"]
            raw_loss = weight * (output["regress"] - target["regress"]) ** 2
            masked_loss = raw_loss * mask_target

            loss_sum = rearrange(masked_loss, "b n c z h w -> b n (c z h w)").sum(
                dim=-1
            )
            mask_sum = rearrange(mask_target, "b n c z h w -> b n (c z h w)").sum(
                dim=-1
            )

            loss["regress"] = torch.mean(loss_sum / (mask_sum + 1e-10))

            weight_mask = self.hparams.loss_weight.get("mask", 1.0)
            weight_regress = self.hparams.loss_weight.get("regress", 1.0)
            loss["total"] = (
                weight_mask * loss["mask"] + weight_regress * loss["regress"]
            )

            mask_output = (output["mask"] > 0).float()
            output_result = output["regress"] * mask_output

        else:
            loss["total"] = torch.mean(
                weight * (output["regress"] - target["regress"]) ** 2
            )
            output_result = output["regress"]

        loss_var = nn.MSELoss(reduction="none")(output_result, target["regress"]).mean(
            dim=(0, 1, -1, -2)
        )

        return loss, loss_var

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-5)

    def training_step(self, batch, batch_idx):
        single, upper, time, target_regress, column_bottom, column_left = batch

        target = {"regress": target_regress}
        if self.hparams.use_mask:
            target["mask"] = (target_regress > 0).float()

        rnd_normal = torch.randn(
            [target_regress.shape[0], target_regress.shape[1], 1, 1, 1, 1],
            device=target_regress.device,
        )
        sigma = (rnd_normal * self.hparams.P_std + self.hparams.P_mean).exp()
        sigma = sigma.clamp(min=self.hparams.sigma_min, max=self.hparams.sigma_max)

        loss, loss_var = self.general_step(
            single, upper, time, target, sigma, column_bottom, column_left, shuffle=True
        )

        self._log_loss_var(
            loss, loss_var, self.hparams.target_var, self.hparams.z_target, "train"
        )

        return loss["total"]

    def validation_step(self, batch, batch_idx):
        single, upper, time, target_regress, column_bottom, column_left = batch

        target = {"regress": target_regress}
        if self.hparams.use_mask:
            target["mask"] = (target_regress > 0).float()

        v_gen = torch.Generator(device=target_regress.device).manual_seed(
            42 + batch_idx
        )
        rnd_normal = torch.randn(
            [target_regress.shape[0], target_regress.shape[1], 1, 1, 1, 1],
            device=target_regress.device,
            generator=v_gen,
        )
        sigma = (rnd_normal * self.hparams.P_std + self.hparams.P_mean).exp()
        sigma = sigma.clamp(min=self.hparams.sigma_min, max=self.hparams.sigma_max)

        loss, loss_var = self.general_step(
            single, upper, time, target, sigma, column_bottom, column_left
        )

        self._log_loss_var(
            loss, loss_var, self.hparams.target_var, self.hparams.z_target, "val"
        )

        return loss["total"]

    def _log_loss_var(self, loss, loss_var, variable_name, level, stage):
        if self.hparams.use_mask:
            self.log(
                f"mask_{stage}",
                loss["mask"],
                on_step=stage != "test",
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
            self.log(
                f"regress_{stage}",
                loss["regress"],
                on_step=stage != "test",
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )

        self.log(
            f"total_{stage}",
            loss["total"],
            on_step=stage != "test",
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

        v_gen = torch.Generator(device=self.device).manual_seed(42)
        sigma = (
            torch.exp(torch.tensor(self.hparams.P_mean))
            .to(self.device)
            .view(1, 1, 1, 1, 1, 1)
        )
        noise = (
            torch.randn(target[0:1, 0:1].shape, generator=v_gen, device=self.device)
            * sigma
        )
        column_km = (
            torch.tensor([target.shape[-1] * self.hparams.resolution_target])
            .to(target.dtype)
            .to(device=target.device)
        )

        output = self(
            single=single[0:1],
            upper=upper[0:1],
            time=time[0:1],
            noise=target[0:1, 0:1] + noise,
            sigma=sigma,
            column_km=column_km,
            column_bottom=column_bottom[0:1, 0:1],
            column_left=column_left[0:1, 0:1],
        )

        if self.hparams.use_mask:
            mask_output = (output["mask"] > 0).float()
            output["regress"] = output["regress"] * mask_output

        return output["regress"]
