from pathlib import Path
from dotenv import dotenv_values
import hydra
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import torch
import wandb

from dataset import DataManager
from model import DirectDownscaling


@hydra.main(version_base=None, config_path="config", config_name="train")
def main(cfg) -> None:
    env = dotenv_values(".env")
    if cfg.dtype == "float64":
        dtype = torch.float64
    else:
        dtype = torch.float32
    torch.set_default_dtype(dtype)

    experiment_name = cfg.experiment.name
    print(f"Training experiment: {experiment_name}")

    cfg.dataset.res.stats_file = Path(env.get("STATS_DIR")) / cfg.dataset.res.stats_file
    datamodule = DataManager(
        input_dir=Path(env.get("INPUT_DIR")),
        dtype=dtype,
        **cfg.dataset,
    )

    model = DirectDownscaling(
        global_grid=cfg.dataset.res.global_grid,
        resolution_input=cfg.dataset.res.resolution_input,
        resolution_target=cfg.dataset.res.resolution_target,
        column_km=cfg.dataset.res.column_km,
        crop_number=cfg.dataset.res.crop_number,
        **cfg.model.architecture,
        single_channel=len(cfg.dataset.var.input_single)
        + len(cfg.dataset.var.input_static),
        upper_channel=len(cfg.dataset.var.input_upper),
        output_channel=len(cfg.dataset.var.target),
        z_input=cfg.dataset.var.z_input,
        z_target=cfg.dataset.var.z_target,
        target_var=cfg.dataset.var.target,
    )
    model = model.to(dtype)

    logger = WandbLogger(
        project="DLAMP.cm",
        name=experiment_name,
        save_dir=Path("logs", experiment_name),
        log_model=False,
        offline=True,
    )

    callbacks = ModelCheckpoint(
        dirpath=Path("checkpoints", experiment_name),
        filename="{epoch}-{step}-{val_loss:.6f}",
        save_top_k=3,
        monitor="val/total",
        mode="min",
    )
    trainer = Trainer(
        max_epochs=10,
        logger=logger,
        callbacks=callbacks,
        # strategy="ddp",  # if you use multi-GPU
        accelerator="gpu",
    )

    trainer.fit(model, datamodule)


if __name__ == "__main__":
    main()
