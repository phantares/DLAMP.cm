import lightning as L
import torch
import matplotlib.pyplot as plt


class VisualizerCallback(L.Callback):
    def __init__(self, z_levels, target_var, log_every_n_epochs=1):
        super().__init__()
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

            self.last_val_sample = {
                "target": target[0, 0].cpu(),
                "pred": pred[0, 0].cpu(),
                "epoch": trainer.current_epoch,
            }

    def on_validation_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.log_every_n_epochs != 0 or not hasattr(
            self, "last_val_sample"
        ):
            return

        target = self.last_val_sample["target"]
        pred = self.last_val_sample["pred"]

        fig, axes = plt.subplots(3, 5, figsize=(22, 15))
        fig.suptitle(f"Epoch {trainer.current_epoch}", fontsize=20)

        for i, var_name in enumerate(self.target_var):
            t_sum = target[i].sum(dim=0)
            p_sum = pred[i].sum(dim=0)
            v_min, v_max = t_sum.min(), t_sum.max()

            axes[0, i].imshow(t_sum, cmap="viridis", vmin=v_min, vmax=v_max)
            axes[0, i].set_title(f"Target: {var_name}", fontsize=14)
            axes[0, i].axis("off")

            im = axes[1, i].imshow(p_sum, cmap="viridis", vmin=v_min, vmax=v_max)
            axes[1, i].set_title(f"Pred: {var_name}", fontsize=14)
            axes[1, i].axis("off")
            plt.colorbar(
                im, ax=axes[1, i], orientation="horizontal", shrink=0.8, pad=0.05
            )

            t_prof = target[i].mean(dim=(-1, -2)).numpy()
            p_prof = pred[i].mean(dim=(-1, -2)).numpy()

            axes[2, i].plot(t_prof, self.z_levels, "k-", label="Target", linewidth=2)
            axes[2, i].plot(p_prof, self.z_levels, "r--", label="Pred", linewidth=2)
            axes[2, i].invert_yaxis()
            axes[2, i].set_title(f"Vertical: {var_name}", fontsize=14)
            axes[2, i].grid(True, alpha=0.3)
            if i == 0:
                axes[2, i].set_ylabel("P")
            axes[2, i].legend(fontsize=10)

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

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
