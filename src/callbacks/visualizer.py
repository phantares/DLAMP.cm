import lightning as L
import torch
import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from utils import ScalerPipe


class VisualizerCallback(L.Callback):
    def __init__(self, stats_file, z_levels, target_var, log_every_n_epochs=1):
        super().__init__()

        with open(stats_file, "r") as f:
            self.stats = json.load(f)

        self.z_levels = z_levels
        self.target_var = target_var
        self.log_every_n_epochs = log_every_n_epochs

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if batch_idx == 0:
            single, upper, time, target, column_top, column_left = batch

            with torch.no_grad():
                v_gen = torch.Generator(device=pl_module.device).manual_seed(42)
                sigma_val = torch.exp(torch.tensor(pl_module.hparams.P_mean)).to(
                    pl_module.device
                )
                sigma = sigma_val.view(1, 1, 1, 1, 1, 1)
                noise = (
                    torch.randn(
                        target[0:1, 0:1].shape, generator=v_gen, device=pl_module.device
                    )
                    * sigma
                )

                pred = pl_module(
                    single=single[0:1],
                    upper=upper[0:1],
                    time=time[0:1],
                    noise=target[0:1, 0:1] + noise,
                    sigma=sigma,
                    column_top=column_top[0:1, 0:1],
                    column_left=column_left[0:1, 0:1],
                )

                target = target[0, 0].cpu()
                pred = pred[0, 0].cpu()
                for v, variable in enumerate(self.target_var):
                    for k, lev in enumerate(self.z_levels):
                        scaler = ScalerPipe(self.stats.get(f"{variable}{int(lev)}"))
                        invt_tar = scaler.inverse_transform(target[v, k])
                        invt_pred = scaler.inverse_transform(pred[v, k])

                        target[v, k] = invt_tar
                        pred[v, k] = invt_pred

            self.last_val_sample = {
                "target": target.cpu(),
                "pred": pred.cpu(),
            }

    def on_validation_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.log_every_n_epochs != 0 or not hasattr(
            self, "last_val_sample"
        ):
            return

        target = self.last_val_sample["target"]
        pred = self.last_val_sample["pred"]

        fig = plt.figure(figsize=(len(self.target_var) * 4.5, 12))
        fig.suptitle(f"Epoch {trainer.current_epoch}", fontsize=20)
        outer_gs = gridspec.GridSpec(1, len(self.target_var), figure=fig)

        for i, var_name in enumerate(self.target_var):
            inner_gs = gridspec.GridSpecFromSubplotSpec(
                3,
                2,
                subplot_spec=outer_gs[i],
                width_ratios=[1, 0.05],
            )

            t_sum = target[i].sum(dim=0)
            p_sum = pred[i].sum(dim=0)
            v_min, v_max = t_sum.min(), t_sum.max()

            ax0 = fig.add_subplot(inner_gs[0, 0])
            ax0.imshow(t_sum, cmap="viridis", vmin=v_min, vmax=v_max)
            ax0.set_title(f"Target: {var_name}", fontsize=14)
            ax0.axis("off")

            ax1 = fig.add_subplot(inner_gs[1, 0])
            im = ax1.imshow(p_sum, cmap="viridis", vmin=v_min, vmax=v_max)
            ax1.set_title(f"Pred: {var_name}", fontsize=14)
            ax1.axis("off")

            cax = fig.add_subplot(inner_gs[0:2, 1])
            fig.colorbar(im, cax=cax, shrink=0.8)

            t_prof = target[i].mean(dim=(-1, -2)).numpy()
            p_prof = pred[i].mean(dim=(-1, -2)).numpy()

            ax2 = fig.add_subplot(inner_gs[2, :])
            ax2.plot(t_prof, self.z_levels, "k-", label="Target", linewidth=2)
            ax2.plot(p_prof, self.z_levels, "r--", label="Pred", linewidth=2)
            ax2.invert_yaxis()
            ax2.set_title(f"Vertical Profile: {var_name}", fontsize=14)
            ax2.grid(True, alpha=0.3)
            ax2.legend(fontsize=10)

        if trainer.logger:
            if hasattr(trainer.logger.experiment, "log"):  # WandB
                import wandb

                trainer.logger.log_image(
                    key="Downscaling_Diagnostics", images=[wandb.Image(fig)]
                )
            elif hasattr(trainer.logger.experiment, "add_figure"):  # TensorBoard
                trainer.logger.experiment.add_figure(
                    "Diagnostics", fig, global_step=trainer.global_step
                )

        plt.close(fig)
        del self.last_val_sample
