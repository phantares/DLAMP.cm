from pathlib import Path
from dotenv import dotenv_values
import hydra
from lightning import Trainer
import torch
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../config", config_name="train")
def main(cfg) -> None:
    env = dotenv_values(".env")

    if cfg.dtype == "float64":
        dtype = torch.float64
    else:
        dtype = torch.float32
    torch.set_default_dtype(dtype)

    experiment_name = cfg.experiment.name
    print(f"Training experiment: {experiment_name}")

    datamodule = hydra.utils.instantiate(
        cfg.dataset,
        input_dir=Path(env.get("INPUT_DIR")),
        dtype=dtype,
    )

    model = hydra.utils.instantiate(
        cfg.model.system,
        single_channel=len(cfg.dataset.var.input_single)
        + len(cfg.dataset.var.input_static),
        upper_channel=len(cfg.dataset.var.input_upper),
        output_channel=len(cfg.dataset.var.target),
    )
    model = model.to(dtype)

    logger = hydra.utils.instantiate(cfg.logger)

    callbacks = []
    for _, cb_conf in cfg.callbacks.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            callbacks.append(hydra.utils.instantiate(cb_conf))

    gpu_count = torch.cuda.device_count()
    strategy = "ddp" if gpu_count > 1 else "auto"
    trainer = Trainer(
        logger=logger,
        callbacks=callbacks,
        accelerator="gpu",
        strategy=strategy,
        devices=gpu_count,
    )

    trainer.fit(model, datamodule)


if __name__ == "__main__":
    main()
