from pathlib import Path
from dotenv import dotenv_values
import hydra
from lightning import Trainer

# from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import torch

from dataset import DataManager
from model import DirectDownscaling
from callbacks import VisualizerCallback


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

    logger = hydra.utils.instantiate(cfg.logger)

    visualizer = VisualizerCallback(
        z_levels=cfg.dataset.var.z_target,
        target_var=cfg.dataset.var.target,
        log_every_n_epochs=1,
    )

    model_checkpoint = ModelCheckpoint(
        dirpath=Path("checkpoints", experiment_name),
        filename="{epoch}-{step}-{total_val_epoch:.6f}",
        save_top_k=3,
        monitor="total_val_epoch",
        mode="min",
    )

    gpu_count = torch.cuda.device_count()
    strategy = "ddp" if gpu_count > 1 else "auto"
    trainer = Trainer(
        logger=logger,
        callbacks=[model_checkpoint, visualizer],
        accelerator="gpu",
        strategy=strategy,
        devices=gpu_count,
    )

    trainer.fit(model, datamodule)


if __name__ == "__main__":
    main()
